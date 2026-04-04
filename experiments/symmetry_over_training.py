"""
Track within-symmetry cosine similarity of representations at intermediate
training checkpoints for DQN and PPO on the Meister maze, alongside reward.

For each checkpoint and each layer:
  - Global similarity   : mean off-diagonal cosine similarity (all pairs)
  - Within-MDP sim      : mean cosine similarity for MDP-bisimulation-equivalent pairs
  - Within-policy sim   : mean cosine similarity for Dijkstra-policy-equivalent pairs

Produces a 4-panel plot per model:
  [reward | global sim | within-MDP sim | within-policy sim]
and saves results to CSV.

Usage:
    python experiments/symmetry_over_training.py
"""

import sys
import csv
import warnings
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from key_door import key_door_env, visualisation_env
from sac.models import dqn as dqn_module
from sac.models import ppo as ppo_module
from sac.utils import joint_state_action_abstraction, dijkstra_policy

OUTPUT_DIR = REPO_ROOT / "results" / "symmetry_over_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DQN_MAP       = REPO_ROOT / "sac" / "maps" / "meister_maze.txt"
DQN_MAP_YAML  = REPO_ROOT / "sac" / "maps" / "test_meister_maze.yaml"
PPO_MAP       = REPO_ROOT / "sac" / "maps" / "meister_trimmed.txt"
PPO_MAP_YAML  = REPO_ROOT / "sac" / "maps" / "test_meister_trimmed.yaml"
DQN_CKPT_DIR  = REPO_ROOT / "sac" / "results" / "2025-06-10-21-13"
PPO_CKPT_DIR  = REPO_ROOT / "sac" / "seb_runs" / "best_ppo"

# Subsample DQN checkpoints: every 500 episodes (~20 points) plus final.
DQN_EPISODES = list(range(0, 10000, 500)) + [9950]
# PPO has 8 checkpoints; use all.
PPO_STEPS    = [512000, 1024000, 1536000, 2048000, 2560000, 3072000, 3584000, 4000000]

DQN_LAYERS   = ["conv", "fc1", "fc2"]
PPO_LAYERS   = ["conv", "shared", "actor", "critic"]

LAYER_COLORS = {
    "conv":   "#4477AA",
    "fc1":    "#EE6677",
    "fc2":    "#228833",
    "shared": "#EE6677",
    "actor":  "#228833",
    "critic": "#CCBB44",
}


# ---------------------------------------------------------------------------
# Environment / MDP helpers
# ---------------------------------------------------------------------------

def build_env(map_path, yaml_path, representation="agent_position"):
    return key_door_env.KeyDoorEnv(
        map_ascii_path=str(map_path),
        map_yaml_path=str(yaml_path),
        representation=representation,
        episode_timeout=200,
    )


def build_P_R(pos_env):
    state_id = {s: i for i, s in enumerate(pos_env.positional_state_space)}
    id_state = {i: s for i, s in enumerate(pos_env.positional_state_space)}
    S = len(pos_env.positional_state_space)
    A = len(pos_env.action_space)
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    for state in pos_env.positional_state_space:
        si = state_id[state]
        if state not in pos_env._rewards:
            for action in pos_env.action_space:
                pos_env.reset_environment(train=True)
                pos_env.move_agent_to(state)
                reward, new_state = pos_env.step(action)
                P[si][action][state_id[new_state[:2]]] = 1
                R[si][action] = reward
    return P, R, state_id, id_state


def compute_mdp_and_dijkstra(pos_env, P, R, state_id):
    S = len(pos_env.positional_state_space)
    _, _, state_label, _, _, _, _ = joint_state_action_abstraction(P, R)

    goal_mask = np.zeros(S, dtype=bool)
    for xy in pos_env._rewards.keys():
        goal_mask[state_id[xy]] = True
    C = np.full((S, len(pos_env.action_space)), 1.0)
    C[P.sum(axis=2) == 0] = np.inf
    _, pi = dijkstra_policy(P, C, goal_mask)
    pi = np.array(pi)
    valid = pi >= 0          # states with a defined Dijkstra action

    return state_label, pi, valid


# ---------------------------------------------------------------------------
# Representation extraction (batched, CPU)
# ---------------------------------------------------------------------------

def get_all_dqn_reprs(model, pix_env, state_shape):
    model._net.eval()
    net = model._net
    obs_list = []
    for state in pix_env.positional_state_space:
        pix_env.move_agent_to(state)
        obs_list.append(pix_env.get_state_representation().reshape(state_shape))
    states_t = torch.FloatTensor(np.stack(obs_list))
    with torch.no_grad():
        x    = torch.relu(net.conv2(torch.relu(net.conv1(states_t))))
        conv = x.view(x.size(0), -1)
        fc1  = torch.relu(net.fc1(conv))
        fc2  = torch.relu(net.fc2(fc1))
    return {"conv": conv.numpy(), "fc1": fc1.numpy(), "fc2": fc2.numpy()}


def get_all_ppo_reprs(model, pix_env, state_shape):
    model._net.eval()
    net = model._net
    obs_list = []
    for state in pix_env.positional_state_space:
        pix_env.move_agent_to(state)
        obs_list.append(pix_env.get_state_representation().reshape(state_shape))
    states_t = torch.FloatTensor(np.stack(obs_list))
    with torch.no_grad():
        x      = torch.relu(net.conv2(torch.relu(net.conv1(states_t))))
        conv   = x.view(x.size(0), -1)
        shared = torch.relu(net.fc2(torch.relu(net.fc1(conv))))
        actor  = torch.relu(net.pi_fc(shared))
        critic = torch.relu(net.v_fc(shared))
    return {"conv": conv.numpy(), "shared": shared.numpy(),
            "actor": actor.numpy(), "critic": critic.numpy()}


# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def _centred_cosine(reprs: np.ndarray) -> np.ndarray:
    return cosine_similarity(reprs - reprs.mean(axis=0))


def global_cosine_sim(reprs: np.ndarray) -> float:
    """Mean off-diagonal cosine similarity (mean-centred)."""
    sim = _centred_cosine(reprs)
    n = sim.shape[0]
    return float((sim.sum() - np.trace(sim)) / (n * (n - 1)))


def within_label_cosine_sim(reprs: np.ndarray, labels: np.ndarray) -> float:
    """Weighted mean cosine similarity within label groups (mean-centred, upper-tri)."""
    sim = _centred_cosine(reprs)
    sims_w, counts = [], []
    for lbl in np.unique(labels):
        idx = np.where(labels == lbl)[0]
        if len(idx) < 2:
            continue
        sub = sim[np.ix_(idx, idx)]
        ut = sub[np.triu_indices(len(idx), k=1)]
        sims_w.append(ut.mean())
        counts.append(len(ut))
    if not counts:
        return float("nan")
    return float(np.average(sims_w, weights=np.array(counts) / sum(counts)))


def compute_layer_metrics(reprs_dict, layers, mdp_labels, dijkstra_labels, valid_mask):
    """Return dict of {layer_global, layer_mdp, layer_policy} for all layers."""
    row = {}
    for layer in layers:
        r = reprs_dict[layer]
        row[f"{layer}_global"] = global_cosine_sim(r)
        row[f"{layer}_mdp"]    = within_label_cosine_sim(r, mdp_labels)
        row[f"{layer}_policy"] = within_label_cosine_sim(r[valid_mask], dijkstra_labels[valid_mask])
    return row


# ---------------------------------------------------------------------------
# Reward helpers
# ---------------------------------------------------------------------------

def load_dqn_rewards():
    d = np.load(str(DQN_CKPT_DIR / "training_stats.npz"))
    return d["test_episode_rewards"]   # shape (200,)


def dqn_reward_at(ep, rewards, window=3):
    if ep == 0:
        return float("nan")
    idx = ep // 50 - 1
    return float(rewards[max(0, idx-window):min(len(rewards), idx+window+1)].mean())


def load_ppo_rewards():
    d = np.load(str(PPO_CKPT_DIR / "final_training_stats.npz"), allow_pickle=True)
    return d["test_episode_returns"]   # shape (782,)


def ppo_reward_at(step, rewards, window=2):
    idx = step // (1024 * 5)
    return float(rewards[max(0, idx-window):min(len(rewards), idx+window+1)].mean())


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_dqn(sample_state, num_actions, ckpt_path):
    model = dqn_module.DQN(
        sample_state=sample_state, num_actions=num_actions,
        learning_rate=0.0, discount_factor=0.99,
        exploration_rate=0.01, exploration_decay=1.0,
        batch_size=64, target_update_frequency=1000,
        replay_buffer_size=10000, burnin=1000,
        convolutional=True, optimistic_init=False,
    )
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model._net.load_state_dict(ckpt["model_state_dict"])
    model._net.to("cpu").eval()
    return model


def load_ppo(sample_state, num_actions, ckpt_path):
    model = ppo_module.PPO(
        sample_state=sample_state, num_actions=num_actions,
        batch_size=64, replay_buffer_size=1024,
        learning_rate=3e-4, discount_factor=0.99,
        gae_lambda=0.95, target_kl=0.03,
        clip_coef=0.2, vf_coef=1.0, ent_coef=0.05,
        max_grad_norm=0.5, convolutional=True,
        layer_norm=False, weight_decay=0.01,
        optimistic_init=True,
    )
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    model._net.load_state_dict(ckpt["model_state_dict"])
    model._net.to("cpu").eval()
    return model


# ---------------------------------------------------------------------------
# Per-model analysis loops
# ---------------------------------------------------------------------------

def analyse_checkpoints(ckpt_list, load_fn, repr_fn,
                        pix_env, state_shape,
                        mdp_labels, dijkstra_labels, valid_mask,
                        layers, reward_fn, x_key):
    records = []
    for x_val, ckpt_path in ckpt_list:
        if not ckpt_path.exists():
            print(f"  Missing {ckpt_path.name}, skipping")
            continue
        print(f"  {x_key}={x_val}", end=" ", flush=True)
        sample = pix_env.reset_environment(train=False)
        model  = load_fn(sample, len(pix_env.action_space), ckpt_path)
        reprs  = repr_fn(model, pix_env, state_shape)
        metrics = compute_layer_metrics(reprs, layers, mdp_labels, dijkstra_labels, valid_mask)
        metrics[x_key]   = x_val
        metrics["reward"] = reward_fn(x_val)
        records.append(metrics)
    print()
    return records


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(records, layers, x_key, x_label, model_name, save_name):
    xs      = [r[x_key]   for r in records]
    rewards = [r["reward"] for r in records]

    fig, axes = plt.subplots(4, 1, figsize=(4, 7), sharex=True)

    # Panel 1: reward
    axes[0].plot(xs, rewards, color="black", linewidth=1.5, marker="o", markersize=3)
    axes[0].set_ylabel("Test reward", fontsize=9)
    axes[0].set_title(f"{model_name} — symmetry scores over training", fontsize=10)

    panel_info = [
        (1, "global",  "Global cosine sim"),
        (2, "mdp",     "Within-MDP cosine sim"),
        (3, "policy",  "Within-policy cosine sim"),
    ]
    for panel_idx, suffix, ylabel in panel_info:
        ax = axes[panel_idx]
        for layer in layers:
            ys = [r[f"{layer}_{suffix}"] for r in records]
            ax.plot(xs, ys, label=layer, color=LAYER_COLORS[layer],
                    linewidth=1.5, marker="o", markersize=3)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=7, frameon=False)

    axes[-1].set_xlabel(x_label, fontsize=9)
    for ax in axes:
        ax.tick_params(labelsize=8)
        sns.despine(ax=ax)

    plt.tight_layout()
    path = OUTPUT_DIR / save_name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    png_path = path.with_suffix(".png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}, {png_path}")


# ---------------------------------------------------------------------------
# CSV save
# ---------------------------------------------------------------------------

def save_csv(records, layers, x_key, path):
    if not records:
        return
    suffixes   = ["global", "mdp", "policy"]
    layer_cols = [f"{l}_{s}" for l in layers for s in suffixes]
    fieldnames = [x_key, "reward"] + layer_cols
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({k: row.get(k, "") for k in fieldnames})
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ---- DQN ----------------------------------------------------------------
    print("\n" + "="*55)
    print("DQN — building environment and labels")
    print("="*55)

    dqn_pos_env = build_env(DQN_MAP, DQN_MAP_YAML)
    P_dqn, R_dqn, sid_dqn, _ = build_P_R(dqn_pos_env)
    dqn_mdp_labels, dqn_dijkstra, dqn_valid = compute_mdp_and_dijkstra(
        dqn_pos_env, P_dqn, R_dqn, sid_dqn)

    dqn_pix_env = visualisation_env.VisualisationEnv(
        build_env(DQN_MAP, DQN_MAP_YAML, "pixel"))
    dqn_state_shape = dqn_pix_env.reset_environment(train=False).shape[1:]
    dqn_test_rewards = load_dqn_rewards()

    dqn_ckpts = [(ep, DQN_CKPT_DIR / f"dqn_model_{ep}.pth") for ep in DQN_EPISODES]

    print("Analysing DQN checkpoints...")
    dqn_records = analyse_checkpoints(
        dqn_ckpts, load_dqn, get_all_dqn_reprs,
        dqn_pix_env, dqn_state_shape,
        dqn_mdp_labels, dqn_dijkstra, dqn_valid,
        DQN_LAYERS,
        reward_fn=lambda ep: dqn_reward_at(ep, dqn_test_rewards),
        x_key="episode",
    )
    plot_results(dqn_records, DQN_LAYERS,
                 x_key="episode", x_label="Training episode",
                 model_name="DQN", save_name="dqn_symmetry_over_training.pdf")
    save_csv(dqn_records, DQN_LAYERS, "episode",
             OUTPUT_DIR / "dqn_symmetry_over_training.csv")

    # ---- PPO ----------------------------------------------------------------
    print("\n" + "="*55)
    print("PPO — building environment and labels")
    print("="*55)

    ppo_pos_env = build_env(PPO_MAP, PPO_MAP_YAML)
    P_ppo, R_ppo, sid_ppo, _ = build_P_R(ppo_pos_env)
    ppo_mdp_labels, ppo_dijkstra, ppo_valid = compute_mdp_and_dijkstra(
        ppo_pos_env, P_ppo, R_ppo, sid_ppo)

    ppo_pix_env = visualisation_env.VisualisationEnv(
        build_env(PPO_MAP, PPO_MAP_YAML, "pixel"))
    ppo_state_shape = ppo_pix_env.reset_environment(train=False).shape[1:]
    ppo_test_rewards = load_ppo_rewards()

    ppo_ckpts = [(step, PPO_CKPT_DIR / f"ppo_model_{step}.pth") for step in PPO_STEPS]

    print("Analysing PPO checkpoints...")
    ppo_records = analyse_checkpoints(
        ppo_ckpts, load_ppo, get_all_ppo_reprs,
        ppo_pix_env, ppo_state_shape,
        ppo_mdp_labels, ppo_dijkstra, ppo_valid,
        PPO_LAYERS,
        reward_fn=lambda step: ppo_reward_at(step, ppo_test_rewards),
        x_key="step",
    )
    plot_results(ppo_records, PPO_LAYERS,
                 x_key="step", x_label="Training step",
                 model_name="PPO", save_name="ppo_symmetry_over_training.pdf")
    save_csv(ppo_records, PPO_LAYERS, "step",
             OUTPUT_DIR / "ppo_symmetry_over_training.csv")
