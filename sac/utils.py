import numpy as np
from collections import defaultdict


def deterministic_bisimulation(P, R, *, atol=0.0):
    """
    Compute the coarsest bisimulation partition for a finite MDP.

    Parameters
    ----------
    P : np.ndarray, shape (S, A, S)
        Transition probabilities P[s, a, s_next].
    R : np.ndarray, shape (S, A)
        Immediate rewards R[s, a].
    atol : float, default 0.0
        Numerical tolerance for equality checks. Use 0.0 for strict exact
        bisimulation; a small >0 can be helpful if P/R are floating-point.

    Returns
    -------
    blocks : list[list[int]]
        Equivalence classes of states (partition of {0..S-1}).
    label : np.ndarray, shape (S,)
        Block id for each state.
    """
    S, A, S2 = P.shape
    assert S == S2, "P must be (S, A, S)"
    assert R.shape == (S, A), "R must be (S, A)"

    # Helper to hash/compare vectors with tolerance
    def _key_array(x):
        if atol == 0.0:
            # exact match: tuple of exact floats
            return tuple(x.tolist())
        else:
            # quantize to tolerance to stabilize hashing under tiny fp noise
            return tuple(np.round(x / max(atol, 1e-12)).astype(np.int64).tolist())

    # Start with a single block containing all states
    label = np.zeros(S, dtype=int)

    # Precompute: for each (s, a), the transition vector to states
    # We'll turn it into distribution over *blocks* as blocks change
    # Refinement loop
    changed = True
    while changed:
        changed = False
        new_label = np.full(S, -1, dtype=int)
        # Within each current block, split by "signatures"
        # A state's signature under a partition is:
        #   for each action a:
        #     (reward[s,a], distribution over current blocks from (s,a))
        # Concatenate over actions.
        block_to_states = defaultdict(list)
        for s in range(S):
            block_to_states[label[s]].append(s)

        next_block_id = 0
        for _, states in block_to_states.items():
            # Build signatures for states in this block
            sig_to_states = defaultdict(list)
            for s in states:
                parts = []
                for a in range(A):
                    # reward component
                    parts.append(("R", a, _key_array(np.array([R[s, a]]))))
                    # transition-to-blocks component
                    # Sum P(s'|s,a) over s' belonging to each current block k
                    # Create vector of length = number of current blocks present
                    # We'll build a stable mapping from block ids used in 'label'
                    # to a compact range [0..K-1] by enumerating seen block ids.
                    # Simpler: gather by existing 'label' maximum + 1 length.
                    K = label.max() + 1
                    mass_per_block = np.zeros(K, dtype=float)
                    # Accumulate probability mass per current block
                    # (vectorized gather would be faster; explicit loop keeps it clear)
                    for sp in range(S):
                        mass_per_block[label[sp]] += P[s, a, sp]
                    parts.append(("P", a, _key_array(mass_per_block)))
                sig_to_states[tuple(parts)].append(s)

            # If a block splits into multiple signatures, refinement occurred
            if len(sig_to_states) > 1:
                changed = True

            # Assign new labels per signature
            for _, group in sig_to_states.items():
                for s in group:
                    new_label[s] = next_block_id
                next_block_id += 1

        label = new_label

    # Build blocks output
    K = label.max() + 1
    blocks = [[] for _ in range(K)]
    for s, k in enumerate(label):
        blocks[k].append(s)
    return blocks, label


def joint_state_action_abstraction(P, R, *, atol=0.0):
    """
    Compute the coarsest exact MDP homomorphism (joint state+action abstraction).

    Inputs
    ------
    P : (S, A, S) ndarray
        Transition probabilities P[s, a, s'].
    R : (S, A) ndarray
        Immediate expected rewards R[s, a].
    atol : float, default 0.0
        Tolerance for equality (use 0 for strict exactness; small >0 for fp noise).

    Returns
    -------
    state_blocks : list[list[int]]
        Partition of states; each inner list is an abstract state (block of base states).
    sa_blocks : list[list[tuple[int,int]]]
        Partition of state-actions; each inner list is an abstract action class
        (pairs (s,a)) that are equivalent.
    state_label : (S,) ndarray[int]
        Abstract-state id for each base state s.
    sa_label : (S, A) ndarray[int]
        Abstract SA-class id for each base (s,a).
    P_tilde : (K, A_b_max, K) ndarray
        Abstract transitions. Rows beyond each state's actual abstract-action count
        are zero. K = #abstract states; A_b_max = max #abstract actions over blocks.
    R_tilde : (K, A_b_max) ndarray
        Abstract rewards aligned with P_tilde.
    action_index_per_block : list[dict[int,int]]
        For block b: a dict mapping global sa_class_id -> local abstract-action index (0..A_b-1).
        Use: a_tilde = action_index_per_block[b][sa_label[s,a]] for any s in block b.
    """
    S, A, S2 = P.shape
    assert S == S2 and R.shape == (S, A)

    def key_array(x):
        if atol == 0.0:
            return tuple(x.tolist())
        return tuple(np.round(x / max(atol, 1e-12)).astype(np.int64).tolist())

    # Initialize all states in one block; SA classes uninitialized (-1)
    state_label = np.zeros(S, dtype=int)
    sa_label = -np.ones((S, A), dtype=int)

    def mass_to_blocks(current_state_label):
        K = current_state_label.max() + 1
        M = np.zeros((S, A, K), dtype=np.float64)
        for s in range(S):
            for a in range(A):
                for sp in range(S):
                    M[s, a, current_state_label[sp]] += P[s, a, sp]
        return M

    changed = True
    while changed:
        changed = False
        # Step 1: refine SA classes given current state blocks
        M = mass_to_blocks(state_label)  # (S, A, K)
        sig2class = {}
        next_id = 0
        new_sa_label = np.empty_like(sa_label)

        for s in range(S):
            for a in range(A):
                sig = (key_array(np.array([R[s, a]])), key_array(M[s, a]))
                if sig not in sig2class:
                    sig2class[sig] = next_id
                    next_id += 1
                new_sa_label[s, a] = sig2class[sig]

        if not np.array_equal(new_sa_label, sa_label):
            changed = True
            sa_label = new_sa_label

        # Step 2: refine state blocks from SA classes (treat actions as unlabeled)
        # For state s, signature is the *set* of SA class ids it offers (order-invariant).
        # Using set (presence) rather than counts ensures a valid g(s,·) mapping exists.
        sig2block = {}
        next_block = 0
        new_state_label = np.empty_like(state_label)
        for s in range(S):
            # presence bitset-ish signature via sorted tuple of unique classes
            sig = tuple(sorted(set(sa_label[s, :].tolist())))
            if sig not in sig2block:
                sig2block[sig] = next_block
                next_block += 1
            new_state_label[s] = sig2block[sig]

        if not np.array_equal(new_state_label, state_label):
            changed = True
            state_label = new_state_label

    # Build output partitions
    K = state_label.max() + 1
    state_blocks = [[] for _ in range(K)]
    for s in range(S):
        state_blocks[state_label[s]].append(s)

    C = sa_label.max() + 1
    sa_blocks = [[] for _ in range(C)]
    for s in range(S):
        for a in range(A):
            sa_blocks[sa_label[s, a]].append((s, a))

    # For each abstract state block b, determine which SA classes appear there,
    # and create a local index mapping for abstract actions of b.
    action_index_per_block = []
    max_actions = 0
    for b in range(K):
        classes = sorted({sa_label[s, a] for s in state_blocks[b] for a in range(A)})
        mapping = {cls: i for i, cls in enumerate(classes)}
        action_index_per_block.append(mapping)
        max_actions = max(max_actions, len(classes))

    # Construct abstract MDP (P_tilde, R_tilde)
    P_tilde = np.zeros((K, max_actions, K), dtype=np.float64)
    R_tilde = np.zeros((K, max_actions), dtype=np.float64)

    for b in range(K):
        reps = state_blocks[b]
        assert len(reps) > 0
        s0 = reps[0]  # any representative state in block b
        mapping = action_index_per_block[b]

        # Sanity: the set of SA classes must be identical for all states in the block
        base_set = {sa_label[s0, a] for a in range(A)}
        for s in reps[1:]:
            assert {sa_label[s, a] for a in range(A)} == base_set, \
                "Inconsistent SA-classes within a state block (should not happen at fixpoint)."

        # For each SA class available in this block, pick any (s,a) in the block with that class
        inv_map = defaultdict(list)
        for a in range(A):
            inv_map[sa_label[s0, a]].append((s0, a))

        for cls, idx in mapping.items():
            # pick representative (s,a) for this abstract action
            s_rep, a_rep = inv_map[cls][0]
            # reward
            R_tilde[b, idx] = R[s_rep, a_rep]
            # transitions to abstract states
            for sp in range(S):
                bp = state_label[sp]
                P_tilde[b, idx, bp] += P[s_rep, a_rep, sp]

        # Optional consistency checks (remove if speed matters):
        # All (s,a) in this cls within this block must produce the same R and P_tilde row
        # due to construction; if you like, assert here.

    return (state_blocks, sa_blocks, state_label, sa_label,
            P_tilde, R_tilde, action_index_per_block)



# --------------------------
# Minimal example & sanity check
if __name__ == "__main__":
    # 4 states, 2 actions
    S, A = 4, 2
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))

    # States 0 and 1 will be bisimilar; 2 and 3 distinct
    # Reward structure:
    R[0, :] = [1.0, 0.5]
    R[1, :] = [1.0, 0.5]
    R[2, :] = [0.0, 0.0]
    R[3, :] = [2.0, 2.0]

    # Transitions:
    # 0 and 1: under both actions, 0.7->2 and 0.3->3 (identical)
    for s in [0, 1]:
        for a in range(A):
            P[s, a, 2] = 0.7
            P[s, a, 3] = 0.3
    # 2: self-loop
    for a in range(A):
        P[2, a, 2] = 1.0
    # 3: self-loop
    for a in range(A):
        P[3, a, 3] = 1.0

    blocks, label = deterministic_bisimulation(P, R)
    print("Blocks:", blocks)  # Expect something like [[0,1],[2],[3]]
    print("Labels:", label)

   