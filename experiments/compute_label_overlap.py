"""
Compute overlap between MDP symmetry labels and policy symmetry labels.

For each environment, builds two binary N×N matrices:
  M_mdp[i,j] = 1 if states i and j are MDP-equivalent
  M_pol[i,j] = 1 if states i and j take the same action under the policy

The primary overlap metric is the fraction of 1s in the elementwise product
over the upper triangle (i.e. A / (A+B+C+D) where A=both-1, D=both-0).

Environments:
  - CartPole-v1  (PPO and DQN, sampled states)
  - Meister maze (DQN and PPO, full state enumeration, Dijkstra policy)

Usage:
    python experiments/compute_label_overlap.py
"""

import os
import sys
import csv
import warnings
import numpy as np
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

CARTPOLE_PPO_MODEL = REPO_ROOT / "results" / "cartpole" / "ppo_model.zip"
CARTPOLE_DQN_MODEL = REPO_ROOT / "results" / "cartpole" / "dqn_model.zip"

MEISTER_DQN_MAP       = REPO_ROOT / "sac" / "maps" / "meister_maze.txt"
MEISTER_DQN_MAP_YAML  = REPO_ROOT / "sac" / "maps" / "test_meister_maze.yaml"
MEISTER_PPO_MAP       = REPO_ROOT / "sac" / "maps" / "meister_trimmed.txt"
MEISTER_PPO_MAP_YAML  = REPO_ROOT / "sac" / "maps" / "test_meister_trimmed.yaml"
MEISTER_DQN_CKPT      = REPO_ROOT / "sac" / "results" / "2025-06-10-21-13" / "dqn_model_9950.pth"
MEISTER_PPO_CKPT      = REPO_ROOT / "sac" / "seb_runs" / "best_ppo" / "ppo_model_4000000.pth"

OUTPUT_DIR = REPO_ROOT / "results" / "label_overlap"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Core overlap computation
# ---------------------------------------------------------------------------

def compute_overlap_metrics(M_mdp: np.ndarray, M_pol: np.ndarray,
                            valid_mask: np.ndarray = None) -> dict:
    """
    Given two binary N×N matrices, compute overlap metrics over the upper
    triangle (k=1). An optional boolean valid_mask (same shape, upper-tri)
    restricts which pairs are counted.

    The primary metric is:
        overlap = A / (A+B+C+D)
              where A = both 1, B = mdp-only, C = pol-only, D = both 0

    This equals the fraction of 1s in the elementwise product M_mdp * M_pol
    over all valid pairs.
    """
    n = M_mdp.shape[0]
    idx = np.triu_indices(n, k=1)

    mdp_flat = M_mdp[idx].astype(int)
    pol_flat = M_pol[idx].astype(int)

    if valid_mask is not None:
        mask = valid_mask
    else:
        mask = np.ones(len(mdp_flat), dtype=bool)

    mdp_v = mdp_flat[mask]
    pol_v = pol_flat[mask]

    A = int(np.sum(mdp_v & pol_v))
    B = int(np.sum(mdp_v & ~pol_v.astype(bool)))
    C = int(np.sum(~mdp_v.astype(bool) & pol_v))
    D = int(np.sum(~mdp_v.astype(bool) & ~pol_v.astype(bool)))
    total = A + B + C + D

    overlap   = A / total if total > 0 else float("nan")
    jaccard   = A / (A + B + C) if (A + B + C) > 0 else float("nan")
    precision = A / (A + C) if (A + C) > 0 else float("nan")
    recall    = A / (A + B) if (A + B) > 0 else float("nan")

    # Cohen's kappa
    p_o = (A + D) / total if total > 0 else float("nan")
    p_e = ((A + B) / total) * ((A + C) / total) + \
          ((C + D) / total) * ((B + D) / total) if total > 0 else float("nan")
    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) != 0 else float("nan")

    return dict(
        A=A, B=B, C=C, D=D, total=total,
        mdp_sym=A + B,
        pol_sym=A + C,
        overlap=overlap,
        jaccard=jaccard,
        precision=precision,
        recall=recall,
        kappa=kappa,
    )


def print_overlap_table(env: str, model: str, metrics: dict):
    print(f"\n{'='*55}")
    print(f"  {env}  |  {model}")
    print(f"{'='*55}")
    print(f"  Valid pairs   : {metrics['total']:,}")
    print(f"  MDP sym (A+B) : {metrics['mdp_sym']:,}")
    print(f"  Pol sym (A+C) : {metrics['pol_sym']:,}")
    print(f"  A (both=1)    : {metrics['A']:,}")
    print(f"  B (MDP only)  : {metrics['B']:,}")
    print(f"  C (pol only)  : {metrics['C']:,}")
    print(f"  D (both=0)    : {metrics['D']:,}")
    print(f"  Overlap (A/N) : {metrics['overlap']:.4f}")
    print(f"  Jaccard       : {metrics['jaccard']:.4f}")
    print(f"  Precision     : {metrics['precision']:.4f}")
    print(f"  Recall        : {metrics['recall']:.4f}")
    print(f"  Cohen's kappa : {metrics['kappa']:.4f}")


# ---------------------------------------------------------------------------
# CartPole
# ---------------------------------------------------------------------------

def run_cartpole(seed: int = 42) -> list[dict]:
    import gymnasium as gym
    import torch
    from stable_baselines3 import PPO, DQN
    from experiments.cartpole.train_inverted_pendulum import (
        CartPoleControl, SymmetryAnalyzer, generate_dataset
    )

    print("\n" + "="*55)
    print("CARTPOLE")
    print("="*55)

    env = gym.make("CartPole-v1")
    lqr = CartPoleControl(env)
    analyzer = SymmetryAnalyzer(lqr)

    states = generate_dataset(env, n_states=10000, seed=seed)
    print(f"Sampled {len(states)} states (including reflections)")

    rows = []
    for algo, model_path in [("PPO", CARTPOLE_PPO_MODEL), ("DQN", CARTPOLE_DQN_MODEL)]:
        print(f"\n--- CartPole {algo} ---")
        if algo == "PPO":
            model = PPO.load(str(model_path), env=env)
        else:
            model = DQN.load(str(model_path), env=env)

        matrices = analyzer.generate_ground_truth_matrices(states, model=model)
        M_mdp = matrices["state_sym"].astype(int)
        M_pol = matrices["model_sym"].astype(int)

        metrics = compute_overlap_metrics(M_mdp, M_pol)
        print_overlap_table("CartPole", algo, metrics)
        rows.append({"env": "CartPole", "model": algo, **metrics})

    env.close()
    return rows


# ---------------------------------------------------------------------------
# Meister maze
# ---------------------------------------------------------------------------

def build_P_R(pos_env):
    """Build transition and reward matrices for a positional key-door env."""
    state_id = {s: i for i, s in enumerate(pos_env.positional_state_space)}
    id_state = {i: s for i, s in enumerate(pos_env.positional_state_space)}
    S = len(pos_env.positional_state_space)
    A = len(pos_env.action_space)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    reward_positions = list(pos_env._rewards.keys())

    for state in pos_env.positional_state_space:
        si = state_id[state]
        if state not in reward_positions:
            for action in pos_env.action_space:
                pos_env.reset_environment(train=True)
                pos_env.move_agent_to(state)
                reward, new_state = pos_env.step(action)
                ni = state_id[new_state[:2]]
                P[si][action][ni] = 1
                R[si][action] = reward

    return P, R, state_id, id_state


def load_dqn_model(sample_state, num_actions, ckpt_path):
    from sac.models import dqn
    model = dqn.DQN(
        sample_state=sample_state,
        num_actions=num_actions,
        learning_rate=0.0,
        discount_factor=0.99,
        exploration_rate=0.01,
        exploration_decay=1.0,
        batch_size=64,
        target_update_frequency=1000,
        replay_buffer_size=10000,
        burnin=1000,
        convolutional=True,
        optimistic_init=False,
    )
    import torch
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model._net.load_state_dict(ckpt["model_state_dict"])
    model._net.to("cpu").eval()
    return model


def load_ppo_model(sample_state, num_actions, ckpt_path):
    from sac.models import ppo
    model = ppo.PPO(
        sample_state=sample_state,
        num_actions=num_actions,
        batch_size=64,
        replay_buffer_size=10000,
        learning_rate=3e-4,
        discount_factor=0.99,
        gae_lambda=0.95,
        target_kl=0.03,
        clip_coef=0.2,
        vf_coef=1.0,
        ent_coef=0.05,
        max_grad_norm=0.5,
        convolutional=True,
        layer_norm=False,
        weight_decay=0.01,
        optimistic_init=True,
    )
    import torch
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model._net.load_state_dict(ckpt["model_state_dict"])
    model._net.to("cpu").eval()
    return model


def get_dqn_policy(model, env, state_shape):
    """Return array of greedy actions for all states in positional_state_space."""
    import torch
    policy = []
    for state in env.positional_state_space:
        env.move_agent_to(state)
        obs = env.get_state_representation()
        obs_t = torch.FloatTensor(obs.reshape((1,) + state_shape))
        with torch.no_grad():
            qvals = model._net.fc3(
                torch.relu(model._net.fc2(
                    torch.relu(model._net.fc1(
                        model._net.conv2(torch.relu(model._net.conv1(obs_t)))
                        .view(obs_t.size(0), -1)
                    ))
                ))
            )
            action = torch.argmax(qvals, dim=1).item()
        policy.append(action)
    return np.array(policy)


def get_ppo_policy(model, env, state_shape):
    """Return array of greedy actions for all states in positional_state_space."""
    import torch
    policy = []
    for state in env.positional_state_space:
        env.move_agent_to(state)
        obs = env.get_state_representation()
        obs_t = torch.FloatTensor(obs.reshape((1,) + state_shape))
        with torch.no_grad():
            net = model._net
            x = torch.relu(net.conv2(torch.relu(net.conv1(obs_t))))
            x = torch.relu(net.fc2(torch.relu(net.fc1(x.view(x.size(0), -1)))))
            actor_repr = torch.relu(net.pi_fc(x))
            action = torch.argmax(net.pi(actor_repr), dim=1).item()
        policy.append(action)
    return np.array(policy)


def run_meister() -> list[dict]:
    from key_door import key_door_env, visualisation_env
    from sac.utils import joint_state_action_abstraction, dijkstra_policy

    print("\n" + "="*55)
    print("MEISTER MAZE")
    print("="*55)

    rows = []

    configs = [
        ("DQN", MEISTER_DQN_MAP, MEISTER_DQN_MAP_YAML),
        ("PPO", MEISTER_PPO_MAP, MEISTER_PPO_MAP_YAML),
    ]

    for model_name, map_path, yaml_path in configs:
        print(f"\n--- Meister {model_name} ---")

        # Positional env for MDP computation
        pos_env = key_door_env.KeyDoorEnv(
            map_ascii_path=str(map_path),
            map_yaml_path=str(yaml_path),
            representation="agent_position",
            episode_timeout=200,
        )

        print("  Building P, R matrices...")
        P, R, state_id, id_state = build_P_R(pos_env)
        S = len(pos_env.positional_state_space)
        A_n = len(pos_env.action_space)

        # MDP labels via joint state-action abstraction
        print("  Computing MDP abstraction (joint_state_action_abstraction)...")
        _, _, state_label, _, _, _, _ = joint_state_action_abstraction(P, R)

        # Dijkstra policy
        print("  Computing Dijkstra policy...")
        reward_positions = list(pos_env._rewards.keys())
        goal_mask = np.zeros(S, dtype=bool)
        for xy in reward_positions:
            goal_mask[state_id[xy]] = True
        C_cost = np.full((S, A_n), 1.0)
        C_cost[P.sum(axis=2) == 0.0] = np.inf
        _, pi_dijkstra = dijkstra_policy(P, C_cost, goal_mask)
        # States unreachable or at goal get pi=-1; treat as separate label
        dijkstra_pol = np.array([pi_dijkstra[i] for i in range(S)])

        # Build binary matrices
        # MDP: same abstraction label
        M_mdp = (state_label[:, None] == state_label[None, :]).astype(int)

        # Policy: same Dijkstra action (exclude states with pi=-1 from both matrices)
        valid_states = (dijkstra_pol >= 0)  # states with a defined optimal action
        M_pol = (dijkstra_pol[:, None] == dijkstra_pol[None, :]).astype(int)

        # valid_mask for upper triangle: both states must have a valid Dijkstra action
        n = S
        idx = np.triu_indices(n, k=1)
        valid_mask = valid_states[idx[0]] & valid_states[idx[1]]

        metrics = compute_overlap_metrics(M_mdp, M_pol, valid_mask=valid_mask)
        print_overlap_table("Meister", model_name, metrics)
        rows.append({"env": f"Meister ({model_name})", "model": model_name, **metrics})

    return rows


# ---------------------------------------------------------------------------
# Save to CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fieldnames = ["env", "model", "total", "mdp_sym", "pol_sym", "A", "B", "C", "D",
                  "overlap", "jaccard", "precision", "recall", "kappa"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"\nSaved CSV: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_rows = []

    all_rows += run_cartpole(seed=42)
    all_rows += run_meister()

    csv_path = OUTPUT_DIR / "label_overlap.csv"
    save_csv(all_rows, csv_path)

    print("\n\nSUMMARY TABLE")
    print(f"{'Env':<28} {'Model':<18} {'Total':>10} {'Overlap':>8} {'Jaccard':>8} {'Prec':>7} {'Rec':>7} {'Kappa':>8}")
    print("-" * 100)
    for r in all_rows:
        print(f"{r['env']:<28} {r['model']:<18} {r['total']:>10,} "
              f"{r['overlap']:>8.4f} {r['jaccard']:>8.4f} "
              f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['kappa']:>8.4f}")
