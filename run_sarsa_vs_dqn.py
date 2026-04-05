"""
Deep SARSA vs DQN comparison on meister_maze.

Runs Deep SARSA with hyperparameters chosen to match the existing DQN baseline
(sac/results/2025-06-10-21-13/), then produces a side-by-side comparison plot.

Usage:
    python run_sarsa_vs_dqn.py

Results are written to results/sarsa_vs_dqn/<timestamp>/ and a comparison
plot is saved there at the end.

DQN reference run: sac/results/2025-06-10-21-13/
  lr=0.0003, eps_decay=0.995, opt=10.0, tuf=50, rbs=50000, burnin=5000,
  bs=64, n_ep=10000, timeout=200, convolutional=True, pixel representation
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime

from key_door import key_door_env, visualisation_env
from sac.models import deep_sarsa
from sac.trainers import episodic_trainer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAC_DIR = os.path.join(SCRIPT_DIR, "sac")
MAPS_DIR = os.path.join(SAC_DIR, "maps")
DQN_RESULTS_DIR = os.path.join(SAC_DIR, "results", "2025-06-10-21-13")

MAP_PATH = os.path.join(MAPS_DIR, "meister_maze.txt")
TRAIN_YAML = os.path.join(MAPS_DIR, "meister_maze.yaml")      # random starts
TEST_YAML = os.path.join(MAPS_DIR, "test_meister_maze.yaml")  # fixed start [2,22]

# ---------------------------------------------------------------------------
# Deep SARSA hyperparameters — matched to and adapted from DQN baseline
#
# == Matched from DQN ==
#   gamma=0.99        same
#   optimistic_init   10.0 same — essential for exploration without replay;
#                        every (s,a) starts at Q=10 >> true reward ~1,
#                        so the agent seeks out unvisited states automatically
#   tuf=50            same — target network cadence
#   n_episodes=10000  same budget
#
# == Adapted for SARSA ==
#   lr=0.0001         DQN uses 0.0003 but with batch=64 from a 50k buffer;
#                        online 1-step updates are ~64x noisier so use lower end
#   eps_decay=0.99999 DQN uses 0.995 — replay buffer maintains distributional
#                        diversity even at low eps; SARSA is on-policy so must
#                        keep exploring longer.
#                        At 10k ep x ~200 steps = 2M steps, eps hits the 0.01
#                        floor at ~460k steps (episode ~2300), then exploits.
#   timeout=500       DQN's replay lets it learn from partial trajectories;
#                        SARSA must reach the reward in a single rollout
# ---------------------------------------------------------------------------
SARSA_CONFIG = dict(
    learning_rate=0.0001,
    discount_factor=0.99,
    exploration_rate=1.0,
    exploration_decay=0.99999,
    target_update_frequency=50,
    convolutional=True,
    optimistic_init=10.0,
    weight_decay=0.0,
)
NUM_EPISODES = 10000
EPISODE_TIMEOUT = 500

# ---------------------------------------------------------------------------
# Trainer settings
# ---------------------------------------------------------------------------
SEED = 42
REPRESENTATION = "pixel"
TEST_FREQUENCY = 50
VIZ_FREQUENCY = 500
SAVE_FREQUENCY = 500


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def make_envs():
    train_env = key_door_env.KeyDoorEnv(
        map_ascii_path=MAP_PATH,
        map_yaml_path=TRAIN_YAML,
        representation=REPRESENTATION,
        episode_timeout=EPISODE_TIMEOUT,
    )
    train_env = visualisation_env.VisualisationEnv(train_env)

    test_env = key_door_env.KeyDoorEnv(
        map_ascii_path=MAP_PATH,
        map_yaml_path=TEST_YAML,
        representation=REPRESENTATION,
        episode_timeout=EPISODE_TIMEOUT,
    )
    test_env = visualisation_env.VisualisationEnv(test_env)
    return train_env, test_env


def plot_comparison(dqn_stats, sarsa_stats, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Deep SARSA vs DQN — meister_maze", fontsize=12)

    dqn_test = dqn_stats["test_episode_rewards"]
    sarsa_test = sarsa_stats["test_episode_rewards"]
    dqn_x = np.arange(len(dqn_test)) * TEST_FREQUENCY
    sarsa_x = np.arange(len(sarsa_test)) * TEST_FREQUENCY

    ax = axes[0]
    ax.plot(dqn_x, dqn_test, alpha=0.6, label="DQN (lr=3e-4, opt=10, rbs=50k)", color="tab:blue")
    ax.plot(sarsa_x, sarsa_test, alpha=0.6, label="Deep SARSA (lr=1e-4, opt=10, online)", color="tab:orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Test reward (greedy policy)")
    ax.set_title("Test reward")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    window = 100

    def smooth(x):
        if len(x) < window:
            return x
        return np.convolve(x, np.ones(window) / window, mode="valid")

    ax.plot(smooth(dqn_stats["episode_rewards"]), alpha=0.8,
            label=f"DQN (w={window})", color="tab:blue")
    ax.plot(smooth(sarsa_stats["episode_rewards"]), alpha=0.8,
            label=f"Deep SARSA (w={window})", color="tab:orange")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Train reward")
    ax.set_title(f"Training reward (smoothed, window={window})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot: {out_path}")


def print_summary(label: str, stats):
    test_r = stats["test_episode_rewards"]
    train_r = stats["episode_rewards"]
    first_hit = next((i * TEST_FREQUENCY for i, v in enumerate(test_r) if v > 0), None)
    print(f"\n  {label}")
    print(f"    Episodes:           {len(train_r)}")
    print(f"    First nonzero test: episode {first_hit}")
    print(f"    Tests > 0:          {(test_r > 0).sum()} / {len(test_r)}")
    print(f"    Max test reward:    {test_r.max():.4f}")
    print(f"    Final test reward:  {test_r[-1]:.4f}")


if __name__ == "__main__":
    set_seed(SEED)

    # --- Load existing DQN stats ---
    dqn_stats_path = os.path.join(DQN_RESULTS_DIR, "training_stats.npz")
    if not os.path.exists(dqn_stats_path):
        raise FileNotFoundError(
            f"DQN stats not found at {dqn_stats_path}. "
            "Update DQN_RESULTS_DIR to point at the right run."
        )
    dqn_stats = np.load(dqn_stats_path)
    print_summary("DQN baseline (pre-run)", dqn_stats)

    # --- Setup output directory ---
    timestamp = datetime.now().strftime("%Y-%d-%m-%H-%M")
    out_dir = os.path.join(SCRIPT_DIR, "results", "sarsa_vs_dqn", timestamp)
    os.makedirs(os.path.join(out_dir, "rollouts"), exist_ok=True)

    with open(os.path.join(out_dir, "sarsa_config.txt"), "w") as f:
        for k, v in SARSA_CONFIG.items():
            f.write(f"{k}: {v}\n")
        f.write(f"num_episodes: {NUM_EPISODES}\n")
        f.write(f"episode_timeout: {EPISODE_TIMEOUT}\n")
        f.write(f"seed: {SEED}\n")

    # --- Build model ---
    train_env, test_env = make_envs()
    sample_state = train_env.reset_environment()
    num_actions = len(train_env.action_space)

    print(f"\nState shape: {sample_state.shape}")
    print(f"Num actions: {num_actions}")

    model = deep_sarsa.DeepSARSA(
        sample_state=sample_state,
        num_actions=num_actions,
        **SARSA_CONFIG,
    )

    # --- Train ---
    print(f"\n{'='*60}")
    print("Training Deep SARSA")
    for k, v in SARSA_CONFIG.items():
        print(f"  {k}: {v}")
    print(f"  num_episodes:  {NUM_EPISODES}")
    print(f"  timeout:       {EPISODE_TIMEOUT}")
    print(f"{'='*60}")

    episodic_trainer.train(
        model=model,
        train_env=train_env,
        test_env=test_env,
        num_episodes=NUM_EPISODES,
        episode_timeout=EPISODE_TIMEOUT,
        test_frequency=TEST_FREQUENCY,
        save_model_frequency=SAVE_FREQUENCY,
        visualisation_frequency=VIZ_FREQUENCY,
        experiment_dir=out_dir,
        early_stop_episodes=0,
    )

    sarsa_stats = np.load(os.path.join(out_dir, "training_stats.npz"))
    print_summary("Deep SARSA (just trained)", sarsa_stats)

    plot_path = os.path.join(out_dir, "comparison.png")
    plot_comparison(dqn_stats, sarsa_stats, plot_path)

    print(f"\nResults written to: {out_dir}")
