"""
Utility functions for LLM tree navigation evaluation and RSA analysis.

This module contains:
1. Graph structure utilities (loading, parsing, navigation)
2. RSA (Representational Similarity Analysis) functions
3. MDP transition matrix building functions
4. Optimal policy computation using Dijkstra's algorithm
"""

import json
import re
import numpy as np
from pathlib import Path
from collections import defaultdict
import heapq

# Optional imports for visualization (not required for core functionality)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False


# =============================================================================
# Graph Structure Utilities
# =============================================================================

def load_graph_structure(tree_representations_dir: str = "tree_representations"):
    """
    Load graph structure from the adjacency list file.

    Supports both old format (node -> [neighbors]) and new spatial format
    (node -> {direction: neighbor}).

    Args:
        tree_representations_dir: Directory containing tree representation files

    Returns:
        adjacency_graph: Dict[int, List[int]] - mapping from node to neighbors
        target_node: int - the goal node
        start_node: int - the start node (if available)
        direction_graph: Dict[int, Dict[str, int]] - mapping from node to {direction: neighbor}
            (only for spatial maze format, None for old format)
    """
    adj_file = Path(tree_representations_dir) / "4_adjacency_list.txt"
    if not adj_file.exists():
        raise FileNotFoundError(
            f"Adjacency list file not found: {adj_file}\n"
            f"Please run generate_tree.py first to create the graph representations."
        )

    with open(adj_file, 'r') as f:
        content = f.read()

    # Extract the JSON adjacency list (skip the task description header)
    # Look for a '{' that starts a line (actual JSON data, not examples in description)
    lines = content.split('\n')
    json_start_line = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for a line that is just '{' - this is the start of actual JSON
        if stripped == '{':
            json_start_line = i
            break

    if json_start_line is None:
        raise ValueError("Could not find JSON data in adjacency list file")

    # Reconstruct JSON from that line onwards
    json_lines = []
    brace_count = 0
    for line in lines[json_start_line:]:
        json_lines.append(line)
        brace_count += line.count('{') - line.count('}')
        if brace_count == 0 and json_lines:
            break

    json_str = '\n'.join(json_lines)

    # Parse the adjacency list
    raw_adjacency = json.loads(json_str)

    # Detect format: new spatial format has dict values, old format has list values
    first_value = next(iter(raw_adjacency.values()))
    is_spatial_format = isinstance(first_value, dict)

    if is_spatial_format:
        # New spatial format: {node: {direction: neighbor, ...}, ...}
        direction_graph = {int(k): v for k, v in raw_adjacency.items()}
        # Convert to simple adjacency list (node -> [neighbors])
        adjacency_graph = {
            node: list(directions.values())
            for node, directions in direction_graph.items()
        }
    else:
        # Old format: {node: [neighbors], ...}
        adjacency_graph = {int(k): v for k, v in raw_adjacency.items()}
        direction_graph = None

    # Extract target node from the file
    target_match = re.search(r'Target Node:\s*(\d+)', content)
    if target_match:
        target_node = int(target_match.group(1))
    else:
        raise ValueError("Could not find target node in adjacency list file")

    # Extract start node (may not exist in old format)
    start_match = re.search(r'Start Node:\s*(\d+)', content)
    start_node = int(start_match.group(1)) if start_match else None

    return adjacency_graph, target_node, start_node, direction_graph


def get_representation_files(tree_representations_dir: str = "tree_representations"):
    """
    Get all representation files.

    Supports both old tree format and new spatial maze format filenames.
    Prefers new format when both exist.

    Args:
        tree_representations_dir: Directory containing tree representation files

    Returns:
        Dict mapping representation name to file path
    """
    rep_dir = Path(tree_representations_dir)
    if not rep_dir.exists():
        raise FileNotFoundError(
            f"Please run generate_tree.py first to create {tree_representations_dir}"
        )

    # Define file mappings with preference for new spatial format
    # Format: (preferred_name, preferred_file, fallback_name, fallback_file)
    file_mappings = [
        ("edges_directed", "1_edges_directed.txt", "edges_ordered", "1_edges_ordered.txt"),
        ("edges_random", "2_edges_random.txt", None, None),
        ("ascii_maze", "3_ascii_maze.txt", "ascii_tree", "3_ascii_tree.txt"),
        ("adjacency_list", "4_adjacency_list.txt", None, None),
        ("relation_triples", "5_relation_triples.txt", None, None),
        ("path_mapping", "6_path_mapping.txt", None, None),
    ]

    files = {}
    for pref_name, pref_file, fallback_name, fallback_file in file_mappings:
        pref_path = rep_dir / pref_file
        if pref_path.exists():
            files[pref_name] = pref_path
        elif fallback_file:
            fallback_path = rep_dir / fallback_file
            if fallback_path.exists():
                files[fallback_name] = fallback_path

    return files


def get_all_start_nodes(adjacency_graph: dict, target_node: int):
    """
    Get all possible start nodes (all nodes except target).

    Args:
        adjacency_graph: Dict mapping node to neighbors
        target_node: The goal node to exclude

    Returns:
        List of valid start nodes
    """
    all_nodes = list(adjacency_graph.keys())
    return [n for n in all_nodes if n != target_node]


def get_neighbors(adjacency_graph: dict, current_node_id: int):
    """
    Get all neighbor node IDs from current node.

    Args:
        adjacency_graph: Dict mapping node to neighbors
        current_node_id: Current node ID

    Returns:
        List of neighbor node IDs
    """
    return adjacency_graph.get(current_node_id, [])


def get_neighbors_with_directions(direction_graph: dict, current_node_id: int):
    """
    Get neighbors with their directions from current node.

    Args:
        direction_graph: Dict mapping node to {direction: neighbor}
        current_node_id: Current node ID

    Returns:
        Dict mapping direction to neighbor node ID, or empty dict if not found
    """
    if direction_graph is None:
        return {}
    return direction_graph.get(current_node_id, {})


def parse_node_id(response_text: str):
    """
    Parse the model's response to extract target node ID.

    Args:
        response_text: Raw text response from the model

    Returns:
        int or None: Parsed node ID, or None if parsing failed
    """
    response_text = response_text.strip()

    # Try to extract a number from the response
    patterns = [
        r'(?:move to node|go to node|to node|node)\s*[:\-]?\s*(\d+)',
        r'(?:move to|go to|to)\s+(\d+)',
        r'^(?:node\s+)?(\d+)$',
        r'\b(\d+)\b',
    ]

    for pattern in patterns:
        match = re.search(pattern, response_text, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                continue

    return None


# =============================================================================
# MDP Transition Matrix Building Functions
# =============================================================================

def build_graph_transition_matrices(adjacency_graph: dict, target_node: int):
    """
    Build transition and cost matrices from graph structure for MDP analysis.

    The graph is treated as a deterministic MDP where:
    - States (S): All nodes in the graph
    - Actions (A): Moving to neighbor i (variable per node, padded to max degree)
    - Transitions (P): Deterministic - P[node, action_i, neighbor_i] = 1.0
    - Costs (C): Unit cost for each step (1.0), inf for invalid actions

    Args:
        adjacency_graph: Dict[node_id, List[neighbor_ids]]
        target_node: int - the goal node

    Returns:
        P: (S, A_max, S) - transition matrix (deterministic, 0/1 entries)
        C: (S, A_max) - cost matrix (1.0 for valid moves, inf for invalid)
        R: (S, A_max) - reward matrix (1.0 for reaching target, 0.0 otherwise)
        node_to_idx: Dict[node_id, int] - mapping from node IDs to matrix indices
        idx_to_node: Dict[int, node_id] - reverse mapping
        action_meanings: Dict[node_id, Dict[action_idx, neighbor_id]] - what each action does
    """
    # Create mappings between node IDs and matrix indices
    all_nodes = sorted(adjacency_graph.keys())
    node_to_idx = {node: idx for idx, node in enumerate(all_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    S = len(all_nodes)
    # Find maximum degree (number of neighbors) across all nodes
    A_max = max(len(neighbors) for neighbors in adjacency_graph.values())

    # Initialize matrices
    P = np.zeros((S, A_max, S), dtype=np.float64)
    C = np.full((S, A_max), np.inf, dtype=np.float64)  # Invalid actions have inf cost
    R = np.zeros((S, A_max), dtype=np.float64)

    # Track what each action means for each node
    action_meanings = {}

    target_idx = node_to_idx[target_node]

    for node, neighbors in adjacency_graph.items():
        node_idx = node_to_idx[node]
        action_meanings[node] = {}

        # Skip transitions from target node (terminal state)
        if node == target_node:
            continue

        for action_idx, neighbor in enumerate(neighbors):
            neighbor_idx = node_to_idx[neighbor]

            # Set transition
            P[node_idx, action_idx, neighbor_idx] = 1.0

            # Set cost (unit cost for valid moves)
            C[node_idx, action_idx] = 1.0

            # Set reward (1.0 if this action reaches the target)
            if neighbor == target_node:
                R[node_idx, action_idx] = 1.0

            # Record action meaning
            action_meanings[node][action_idx] = neighbor

    return P, C, R, node_to_idx, idx_to_node, action_meanings


def compute_mdp_homomorphism_labels(P: np.ndarray, R: np.ndarray, atol: float = 0.0):
    """
    Compute MDP homomorphism labels using joint state-action abstraction.

    This groups nodes that have equivalent state-action structure in the MDP.
    States in the same homomorphism class have identical transition and reward structure.

    Args:
        P: (S, A, S) transition matrix
        R: (S, A) reward matrix
        atol: Numerical tolerance for equality checks

    Returns:
        state_label: (S,) ndarray - MDP homomorphism class for each state
        state_blocks: List[List[int]] - groups of states with same structure
    """
    S, A, S2 = P.shape
    assert S == S2, "P must be (S, A, S)"
    assert R.shape == (S, A), "R must be (S, A)"

    def key_array(x):
        if atol == 0.0:
            return tuple(x.tolist())
        return tuple(np.round(x / max(atol, 1e-12)).astype(np.int64).tolist())

    # Initialize all states in one block
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
        M = mass_to_blocks(state_label)
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

        # Step 2: refine state blocks from SA classes
        sig2block = {}
        next_block = 0
        new_state_label = np.empty_like(state_label)
        for s in range(S):
            sig = tuple(sorted(set(sa_label[s, :].tolist())))
            if sig not in sig2block:
                sig2block[sig] = next_block
                next_block += 1
            new_state_label[s] = sig2block[sig]

        if not np.array_equal(new_state_label, state_label):
            changed = True
            state_label = new_state_label

    # Build state blocks output
    K = state_label.max() + 1
    state_blocks = [[] for _ in range(K)]
    for s in range(S):
        state_blocks[state_label[s]].append(s)

    return state_label, state_blocks


# =============================================================================
# Optimal Policy Computation (Dijkstra's Algorithm)
# =============================================================================

def _deterministic_next(P: np.ndarray, atol: float = 1e-12):
    """
    Return next[s,a] = s' (int) if deterministic, else -1.

    Args:
        P: (S, A, S) transition matrix
        atol: Tolerance for checking determinism

    Returns:
        nxt: (S, A) array of next states (-1 if undefined)
    """
    S, A, S2 = P.shape
    assert S == S2
    nxt = -np.ones((S, A), dtype=int)

    for s in range(S):
        for a in range(A):
            idx = np.flatnonzero(P[s, a] > 1 - atol)
            if len(idx) == 1 and np.allclose(P[s, a, idx[0]], 1.0, atol=atol):
                nxt[s, a] = int(idx[0])
            else:
                # Allow explicit terminals to be self-loops w/ prob 1
                if np.allclose(P[s, a].sum(), 0.0, atol=atol):
                    nxt[s, a] = -1  # no transition defined

    return nxt


def dijkstra_policy(P: np.ndarray, C: np.ndarray, goal_mask: np.ndarray, atol: float = 1e-12):
    """
    Compute optimal policy using Dijkstra's algorithm for deterministic MDPs.

    Args:
        P: (S, A, S) transition probabilities (deterministic 0/1)
        C: (S, A) nonnegative costs
        goal_mask: (S,) bool, True for goal/terminal states
        atol: Tolerance for determinism check

    Returns:
        J: (S,) optimal cost-to-go (inf if goal unreachable)
        pi: (S,) int optimal action per state (-1 if terminal or no feasible action)
    """
    S, A, S2 = P.shape
    assert C.shape == (S, A)
    assert S == S2
    nxt = _deterministic_next(P, atol=atol)

    # Build reverse graph: edges sp -> s with cost c(s,a) whenever nxt[s,a]=sp
    rev_adj = [[] for _ in range(S)]
    for s in range(S):
        for a in range(A):
            sp = nxt[s, a]
            if sp >= 0:
                rev_adj[sp].append((s, a, C[s, a]))

    # Multi-source Dijkstra from all goals on the reverse graph
    J = np.full(S, np.inf, dtype=float)
    pi = -np.ones(S, dtype=int)
    pq = []

    # J(goal) = 0 and no action needed there
    for g in np.flatnonzero(goal_mask):
        J[g] = 0.0
        heapq.heappush(pq, (0.0, g))

    while pq:
        dist_u, u = heapq.heappop(pq)
        if dist_u > J[u]:
            continue
        # Relax predecessors (s --a,c--> u)
        for s, a, c in rev_adj[u]:
            new_cost = c + J[u]
            if new_cost < J[s]:
                J[s] = new_cost
                pi[s] = a
                heapq.heappush(pq, (new_cost, s))

    # Terminal states: keep pi=-1
    pi[goal_mask] = -1
    return J, pi


def compute_optimal_policy_labels(
    P: np.ndarray,
    C: np.ndarray,
    target_node_idx: int,
    idx_to_node: dict
):
    """
    Compute optimal policy labels - groups nodes by their optimal action.

    Args:
        P: (S, A, S) transition matrix
        C: (S, A) cost matrix
        target_node_idx: Index of target node in matrix
        idx_to_node: Mapping from matrix index to node ID

    Returns:
        node_to_policy_label: Dict[node_id, int] - optimal action for each node
        node_to_cost: Dict[node_id, float] - cost-to-go for each node
        pi: (S,) optimal policy array
    """
    S = P.shape[0]

    # Create goal mask (only the target node is the goal)
    goal_mask = np.zeros(S, dtype=bool)
    goal_mask[target_node_idx] = True

    # Run Dijkstra
    J, pi = dijkstra_policy(P, C, goal_mask)

    # Convert to node-based dictionaries
    node_to_policy_label = {}
    node_to_cost = {}

    for idx in range(S):
        node = idx_to_node[idx]
        if pi[idx] >= 0:  # Has a valid optimal action
            node_to_policy_label[node] = int(pi[idx])
        node_to_cost[node] = J[idx] if J[idx] < np.inf else None

    return node_to_policy_label, node_to_cost, pi


# =============================================================================
# Depth Computation
# =============================================================================

def compute_depth_labels(adjacency_graph: dict, start_node: int = None, target_node: int = None):
    """
    Compute depth labels for each node (distance from root).

    If start_node is provided, computes depth from start_node.
    Otherwise, if target_node is provided, computes depth from target_node.
    Otherwise, uses the node with the smallest ID as root.

    Args:
        adjacency_graph: Dict[node_id, List[neighbor_ids]]
        start_node: Optional start node to use as root
        target_node: Optional target node to use as root if start_node not provided

    Returns:
        node_to_depth: Dict[node_id, int] - depth for each node
    """
    from collections import deque

    # Determine root node
    if start_node is not None:
        root = start_node
    elif target_node is not None:
        root = target_node
    else:
        root = min(adjacency_graph.keys())

    # BFS to compute depths
    node_to_depth = {root: 0}
    queue = deque([root])
    visited = {root}

    while queue:
        current = queue.popleft()
        current_depth = node_to_depth[current]

        for neighbor in adjacency_graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                node_to_depth[neighbor] = current_depth + 1
                queue.append(neighbor)

    return node_to_depth


# =============================================================================
# RSA (Representational Similarity Analysis) Functions
# =============================================================================

def compute_rsa_matrix(representations: np.ndarray, mean_center: bool = True):
    """
    Compute RSA (cosine similarity) matrix from representations.

    Args:
        representations: (N, D) array of N representations with D dimensions
        mean_center: Whether to mean-center representations before computing similarity

    Returns:
        rsa_matrix: (N, N) cosine similarity matrix
    """
    if not HAS_PLOTTING:
        raise ImportError("sklearn is required for RSA analysis. Install with: pip install scikit-learn")

    if mean_center:
        representations = representations - representations.mean(axis=0)

    rsa_matrix = cosine_similarity(representations)
    return rsa_matrix


def plot_rsa_matrix(
    representations: np.ndarray,
    title: str,
    save_path: str = None,
    mean_center: bool = True
):
    """
    Compute and plot RSA matrix with cosine similarity.

    Args:
        representations: (N, D) array of representations for N nodes
        title: Plot title
        save_path: Where to save the figure (optional)
        mean_center: Whether to mean-center representations

    Returns:
        rsa_matrix: (N, N) cosine similarity matrix
    """
    if not HAS_PLOTTING:
        raise ImportError("matplotlib and sklearn are required for plotting. "
                          "Install with: pip install matplotlib scikit-learn seaborn")

    rsa_matrix = compute_rsa_matrix(representations, mean_center=mean_center)

    plt.figure(figsize=(10, 8))
    plt.imshow(rsa_matrix, cmap='viridis')
    plt.colorbar(label='Cosine Similarity')
    plt.title(title)
    plt.xlabel('State Index')
    plt.ylabel('State Index')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()

    return rsa_matrix


def calculate_participation_ratio(representations: np.ndarray):
    """
    Calculate participation ratio (effective dimensionality) of representations.

    The participation ratio measures how many dimensions are effectively used
    by the representations. A higher value indicates more distributed representations.

    Args:
        representations: (N, D) array of representations

    Returns:
        participation_ratio: float - effective dimensionality
    """
    cov_matrix = np.cov(representations, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)  # Ensure real values

    # Participation ratio = (sum of eigenvalues)^2 / sum of eigenvalues^2
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return participation_ratio


def get_mean_off_diagonal(rsa_matrix: np.ndarray, return_se: bool = False):
    """
    Calculate mean of off-diagonal elements in RSA matrix.

    This gives the global similarity - average similarity between all pairs of states.

    Args:
        rsa_matrix: (N, N) similarity matrix
        return_se: If True, also return standard error

    Returns:
        mean_off_diagonal: float - average pairwise similarity
        se: float - standard error (only if return_se=True)
    """
    n = rsa_matrix.shape[0]
    # Extract off-diagonal elements
    mask = ~np.eye(n, dtype=bool)
    off_diagonal_values = rsa_matrix[mask]

    mean_off_diagonal = np.mean(off_diagonal_values)

    if return_se:
        se = np.std(off_diagonal_values, ddof=1) / np.sqrt(len(off_diagonal_values))
        return mean_off_diagonal, se
    return mean_off_diagonal


def get_within_label_similarity(
    representations: np.ndarray,
    node_to_label: dict,
    node_to_idx: dict,
    mean_center: bool = True,
    return_se: bool = False
):
    """
    Calculate weighted mean similarity within nodes sharing the same label.

    This measures how similar representations are within each equivalence class
    (e.g., nodes with same MDP homomorphism label or same optimal action).

    Args:
        representations: (N, D) array of representations
        node_to_label: Dict[node_id, label] mapping nodes to their labels
        node_to_idx: Dict[node_id, index] mapping node IDs to row indices
        mean_center: Whether to mean-center before computing similarity
        return_se: If True, also return standard error

    Returns:
        weighted_mean: float - weighted average within-label similarity
        se: float - standard error (only if return_se=True)
    """
    if mean_center:
        global_mean = representations.mean(axis=0)
    else:
        global_mean = 0

    sims_same_label = []
    counts_same_label = []
    all_within_label_sims = []  # Collect all pairwise similarities for SE

    for label in set(node_to_label.values()):
        nodes_with_label = [node for node, lbl in node_to_label.items() if lbl == label]

        # Need at least 2 nodes to compute pairwise similarity
        if len(nodes_with_label) < 2:
            continue

        # Get indices for nodes with this label that exist in our mapping
        indices = []
        for node in nodes_with_label:
            if node in node_to_idx:
                indices.append(node_to_idx[node])

        if len(indices) < 2:
            continue

        reprs_subset = representations[indices]

        # Compute similarity matrix for this subset
        if mean_center:
            reprs_centered = reprs_subset - global_mean
        else:
            reprs_centered = reprs_subset

        sim_matrix = cosine_similarity(reprs_centered)
        sim_mean = get_mean_off_diagonal(sim_matrix)

        sims_same_label.append(sim_mean)
        counts_same_label.append(len(indices))

        # Collect all off-diagonal pairwise similarities for SE calculation
        if return_se:
            n = sim_matrix.shape[0]
            mask = ~np.eye(n, dtype=bool)
            all_within_label_sims.extend(sim_matrix[mask].tolist())

    # Calculate weighted mean
    if counts_same_label:
        weights = np.array(counts_same_label) / np.sum(counts_same_label)
        weighted_mean = np.average(sims_same_label, weights=weights)
    else:
        weighted_mean = 0.0

    if return_se:
        if all_within_label_sims:
            all_sims = np.array(all_within_label_sims)
            se = np.std(all_sims, ddof=1) / np.sqrt(len(all_sims))
        else:
            se = 0.0
        return weighted_mean, se

    return weighted_mean


def create_label_rsa_matrices(
    node_to_mdp_label: dict,
    node_to_policy_label: dict,
    node_to_idx: dict,
    num_nodes: int
):
    """
    Create binary RSA matrices for MDP and policy labels.

    These matrices indicate which pairs of nodes share the same label.

    Args:
        node_to_mdp_label: Dict[node_id, int] - MDP homomorphism labels
        node_to_policy_label: Dict[node_id, int] - optimal policy labels
        node_to_idx: Dict[node_id, int] - node to matrix index mapping
        num_nodes: Total number of nodes

    Returns:
        rsa_mdp: (N, N) binary matrix - 1 if same MDP label
        rsa_policy: (N, N) binary matrix - 1 if same policy label
        product_rsa: (N, N) binary matrix - 1 if same MDP AND policy label
        mask: (N, N) binary matrix - 1 - product_rsa (for filtering analysis)
    """
    rsa_mdp = np.zeros((num_nodes, num_nodes))
    rsa_policy = np.zeros((num_nodes, num_nodes))

    # Get list of nodes in index order
    idx_to_node = {v: k for k, v in node_to_idx.items()}

    for i in range(num_nodes):
        for j in range(num_nodes):
            node_i = idx_to_node.get(i)
            node_j = idx_to_node.get(j)

            if node_i is None or node_j is None:
                continue

            # Check MDP label similarity
            if (node_i in node_to_mdp_label and node_j in node_to_mdp_label and
                node_to_mdp_label[node_i] == node_to_mdp_label[node_j]):
                rsa_mdp[i, j] = 1

            # Check policy label similarity
            if (node_i in node_to_policy_label and node_j in node_to_policy_label and
                node_to_policy_label[node_i] == node_to_policy_label[node_j]):
                rsa_policy[i, j] = 1

    # Product RSA: 1 if both MDP and policy labels match
    product_rsa = rsa_mdp * rsa_policy

    # Mask: exclude pairs that share both labels
    mask = 1 - product_rsa

    return rsa_mdp, rsa_policy, product_rsa, mask


def compute_similarity_metrics(
    representations: np.ndarray,
    node_to_mdp_label: dict,
    node_to_policy_label: dict,
    node_to_idx: dict,
    node_to_depth: dict = None
):
    """
    Compute all similarity metrics for a set of representations.

    Args:
        representations: (N, D) array of representations
        node_to_mdp_label: MDP homomorphism labels
        node_to_policy_label: Optimal policy labels
        node_to_idx: Node to index mapping
        node_to_depth: Optional depth labels for each node

    Returns:
        metrics: dict with keys:
            - global_similarity: mean off-diagonal similarity
            - global_similarity_se: standard error of global similarity
            - within_mdp_label_similarity: weighted mean within MDP groups
            - within_mdp_label_similarity_se: standard error
            - within_policy_label_similarity: weighted mean within policy groups
            - within_policy_label_similarity_se: standard error
            - within_depth_similarity: weighted mean within same depth (if node_to_depth provided)
            - within_depth_similarity_se: standard error (if node_to_depth provided)
            - participation_ratio: effective dimensionality
    """
    # Compute RSA matrix
    rsa_matrix = compute_rsa_matrix(representations, mean_center=True)

    # Global similarity with SE
    global_sim, global_sim_se = get_mean_off_diagonal(rsa_matrix, return_se=True)

    # Within-label similarities with SE
    within_mdp_sim, within_mdp_se = get_within_label_similarity(
        representations, node_to_mdp_label, node_to_idx, mean_center=True, return_se=True
    )
    within_policy_sim, within_policy_se = get_within_label_similarity(
        representations, node_to_policy_label, node_to_idx, mean_center=True, return_se=True
    )

    # Within-depth similarity (if depth labels provided)
    if node_to_depth is not None:
        within_depth_sim, within_depth_se = get_within_label_similarity(
            representations, node_to_depth, node_to_idx, mean_center=True, return_se=True
        )
    else:
        within_depth_sim, within_depth_se = 0.0, 0.0

    # Participation ratio
    pr = calculate_participation_ratio(representations)

    return {
        'global_similarity': float(global_sim),
        'global_similarity_se': float(global_sim_se),
        'within_mdp_label_similarity': float(within_mdp_sim),
        'within_mdp_label_similarity_se': float(within_mdp_se),
        'within_policy_label_similarity': float(within_policy_sim),
        'within_policy_label_similarity_se': float(within_policy_se),
        'within_depth_similarity': float(within_depth_sim),
        'within_depth_similarity_se': float(within_depth_se),
        'participation_ratio': float(pr)
    }


def plot_similarity_comparison(
    layer_metrics: dict,
    save_path: str = None,
    title_prefix: str = ""
):
    """
    Create scatter plot comparing global and within-label similarities across layers.

    Replicates the visualization style from rsa_20251210.ipynb.

    Args:
        layer_metrics: Dict[layer_name, dict] with similarity metrics per layer
        save_path: Where to save the figure (optional)
        title_prefix: Prefix for plot titles
    """
    if not HAS_PLOTTING:
        raise ImportError("matplotlib and seaborn are required for plotting.")

    color_1 = '#009988'  # Teal for global
    color_2 = '#EE7733'  # Orange for within-label

    layer_names = list(layer_metrics.keys())
    x = np.arange(len(layer_names))
    offset = 0.15

    fig, axs = plt.subplots(1, 2, figsize=(6, 2.5), sharey=True)

    # Extract metrics
    global_sims = [layer_metrics[l]['global_similarity'] for l in layer_names]
    mdp_sims = [layer_metrics[l]['within_mdp_label_similarity'] for l in layer_names]
    policy_sims = [layer_metrics[l]['within_policy_label_similarity'] for l in layer_names]

    # Left plot: MDP abstraction
    axs[0].scatter(x, global_sims, color=color_1, marker='s', s=30, label='Global', alpha=0.8)
    axs[0].scatter(x, mdp_sims, color=color_2, marker='s', s=30, label='Within MDP label', alpha=0.8)
    for i in range(len(layer_names)):
        axs[0].plot([x[i], x[i]], [global_sims[i], mdp_sims[i]],
                    color='gray', alpha=0.3, linewidth=1)
    axs[0].set_title(f'{title_prefix}MDP Abstraction')
    axs[0].set_ylabel('Cosine Similarity')
    axs[0].legend(frameon=False, fontsize=8)

    # Right plot: Policy abstraction
    axs[1].scatter(x, global_sims, color=color_1, marker='s', s=30, label='Global', alpha=0.8)
    axs[1].scatter(x, policy_sims, color=color_2, marker='s', s=30, label='Within policy label', alpha=0.8)
    for i in range(len(layer_names)):
        axs[1].plot([x[i], x[i]], [global_sims[i], policy_sims[i]],
                    color='gray', alpha=0.3, linewidth=1)
    axs[1].set_title(f'{title_prefix}Policy Abstraction')
    axs[1].legend(frameon=False, fontsize=8)

    # Format both axes
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(layer_names, fontsize=8, rotation=45, ha='right')
        ax.set_ylim(-0.1, 1.0)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150, transparent=True)
        plt.close()
    else:
        plt.show()

    return fig


# =============================================================================
# Main Analysis Orchestrator
# =============================================================================

def analyze_llm_representations(
    layer_representations: dict,
    node_to_mdp_label: dict,
    node_to_policy_label: dict,
    node_to_idx: dict,
    node_to_depth: dict = None,
    save_dir: str = None,
    representation_name: str = ""
):
    """
    Main analysis function - computes all metrics and generates plots.

    Args:
        layer_representations: Dict[layer_name, Dict[node_id, np.ndarray]]
            Hidden states for each layer and node
        node_to_mdp_label: MDP homomorphism labels for each node
        node_to_policy_label: Optimal policy labels for each node
        node_to_idx: Mapping from node IDs to matrix indices
        node_to_depth: Optional depth labels for each node
        save_dir: Directory to save plots (optional)
        representation_name: Name of the graph representation for titles

    Returns:
        results: Dict with analysis results for each layer
    """
    results = {}

    for layer_name, node_reprs in layer_representations.items():
        # Stack representations into (N, D) array in index order
        num_nodes = len(node_to_idx)

        # Get dimension from first representation
        first_node = next(iter(node_reprs.keys()))
        repr_dim = node_reprs[first_node].shape[-1]

        representations = np.zeros((num_nodes, repr_dim))
        for node_id, repr_vec in node_reprs.items():
            if node_id in node_to_idx:
                idx = node_to_idx[node_id]
                representations[idx] = repr_vec.flatten()

        # Compute similarity metrics
        metrics = compute_similarity_metrics(
            representations,
            node_to_mdp_label,
            node_to_policy_label,
            node_to_idx,
            node_to_depth
        )

        results[layer_name] = metrics

        # Generate and save RSA matrix plot if save_dir provided
        if save_dir:
            save_path = Path(save_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            plot_rsa_matrix(
                representations,
                title=f"{representation_name} - {layer_name}",
                save_path=str(save_path / f"rsa_matrix_{layer_name}.png")
            )

            metrics['rsa_matrix_path'] = str(save_path / f"rsa_matrix_{layer_name}.png")

    # Generate comparison plot
    if save_dir and len(results) > 1:
        plot_similarity_comparison(
            results,
            save_path=str(Path(save_dir) / "similarity_comparison.png"),
            title_prefix=f"{representation_name} - "
        )

    return results
