"""
Train PPO or DQN models for CartPole and analyze neural symmetry.

Usage:
    python train_inverted_pendulum.py --algorithm ppo
    python train_inverted_pendulum.py --algorithm dqn
"""

import argparse
import os
import pickle
import random
import warnings
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from scipy import linalg
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.evaluation import evaluate_policy

# Suppress warnings
warnings.filterwarnings('ignore')

SEED = 42


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CartPoleControl:
    """LQR controller for CartPole environment."""

    def __init__(self, env):
        # CartPole-v1 Parameters
        self.g = env.unwrapped.gravity
        self.mc = env.unwrapped.masscart
        self.mp = env.unwrapped.masspole
        self.lp = env.unwrapped.length
        self.mt = self.mc + self.mp

        # Linearize Dynamics (Exact linearization around upright)
        denom = self.lp * (4.0/3.0 - self.mp / self.mt)
        factor = 1.0 / denom

        # State: [x, x_dot, theta, theta_dot]
        self.A = np.zeros((4, 4))
        self.A[0, 1] = 1
        self.A[2, 3] = 1
        self.A[1, 2] = -(self.mp * self.lp / self.mt) * self.g * factor
        self.A[3, 2] = self.g * factor

        self.B = np.zeros((4, 1))
        self.B[1, 0] = (1.0 / self.mt) + (self.mp * self.lp / self.mt) * (1.0 / self.mt) * factor
        self.B[3, 0] = - (1.0 / self.mt) * factor

        # LQR Costs
        self.Q = np.diag([0.1, 0.01, 10.0, 0.1])
        self.R = np.array([[0.1]])

        # Solve Riccati
        P = linalg.solve_continuous_are(self.A, self.B, self.Q, self.R)
        self.K = np.dot(linalg.inv(self.R), np.dot(self.B.T, P))

    def get_action(self, state):
        u = -np.dot(self.K, state)
        return 1 if u > 0 else 0


class SymmetryAnalyzer:
    """Analyzer for checking symmetry in states and actions."""

    def __init__(self, lqr_controller):
        self.lqr = lqr_controller

    def check_state_symmetry(self, s1, s2, threshold=0.1):
        dist = np.linalg.norm(s1 - s2)
        dist_reflected = np.linalg.norm(s1 - (-s2))
        return (dist < threshold) or (dist_reflected < threshold)

    def check_lqr_action_symmetry(self, s1, s2):
        return self.lqr.get_action(s1) == self.lqr.get_action(s2)

    def check_model_action_symmetry(self, model, s1, s2):
        with torch.no_grad():
            a1, _ = model.predict(s1, deterministic=True)
            a2, _ = model.predict(s2, deterministic=True)
        return a1 == a2

    def generate_ground_truth_matrices(self, states, model=None):
        n = len(states)
        m_state = np.zeros((n, n))
        m_lqr = np.zeros((n, n))
        m_model = np.zeros((n, n))
        m_random = np.random.randint(0, 2, size=(n, n))  # Random binary matrix

        print("Generating ground truth matrices...")

        # Precompute all actions for efficiency
        lqr_actions = np.array([self.lqr.get_action(s) for s in states])

        if model:
            print("  Computing model actions...", end=' ', flush=True)
            with torch.no_grad():
                model_actions = np.array([model.predict(s, deterministic=True)[0] for s in states])
            print("done")

        print("  Computing symmetry matrices...", end=' ', flush=True)
        for i in range(n):
            for j in range(n):
                if self.check_state_symmetry(states[i], states[j]):
                    m_state[i, j] = 1
                if lqr_actions[i] == lqr_actions[j]:
                    m_lqr[i, j] = 1
                if model and model_actions[i] == model_actions[j]:
                    m_model[i, j] = 1
        print("done")

        return {
            "state_sym": m_state,
            "lqr_sym": m_lqr,
            "model_sym": m_model,
            "random_sym": m_random
        }


class NetworkProbe:
    """Probe for extracting and analyzing neural network activations."""

    def __init__(self, module):
        self.activations = defaultdict(list)
        self.hooks = []
        self.module = module

    def register_hooks(self):
        self.remove_hooks()
        count = 0
        for name, layer in self.module.named_modules():
            if isinstance(layer, (nn.Linear, nn.ReLU, nn.Tanh)):
                handle = layer.register_forward_hook(self._get_hook(name))
                self.hooks.append(handle)
                count += 1
        return count

    def _get_hook(self, name):
        def hook(model, input, output):
            if isinstance(output, torch.Tensor):
                flat = output.detach().cpu().numpy().reshape(output.shape[0], -1)
                self.activations[name].append(flat)
        return hook

    def clear_activations(self):
        self.activations = defaultdict(list)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def get_activation_similarity_matrix(self, layer_name, max_samples=None):
        if layer_name not in self.activations:
            return None

        try:
            data = np.concatenate(self.activations[layer_name], axis=0)
        except ValueError:
            return None

        if max_samples is not None and data.shape[0] > max_samples:
            data = data[:max_samples]

        data = data + np.random.normal(0, 1e-9, data.shape)
        rsm = np.corrcoef(data)
        return np.nan_to_num(rsm)


def analyze_similarity_groups(behavioral_matrix, neural_matrix):
    """
    Compare average neural similarity for behaviorally similar vs dissimilar states.
    """
    # Get upper triangular indices
    n = behavioral_matrix.shape[0]
    triu_indices = np.triu_indices(n, k=1)

    behavioral_flat = behavioral_matrix[triu_indices]
    neural_flat = neural_matrix[triu_indices]

    # Split into similar (1) and dissimilar (0) groups based on behavioral matrix
    similar_mask = behavioral_flat == 1
    dissimilar_mask = behavioral_flat == 0

    similar_neural = neural_flat[similar_mask]
    dissimilar_neural = neural_flat[dissimilar_mask]

    results = {
        'similar_mean': np.mean(similar_neural) if len(similar_neural) > 0 else np.nan,
        'similar_std': np.std(similar_neural) if len(similar_neural) > 0 else np.nan,
        'dissimilar_mean': np.mean(dissimilar_neural) if len(dissimilar_neural) > 0 else np.nan,
        'dissimilar_std': np.std(dissimilar_neural) if len(dissimilar_neural) > 0 else np.nan,
        'similar_count': len(similar_neural),
        'dissimilar_count': len(dissimilar_neural)
    }

    # Compute statistical test
    if len(similar_neural) > 0 and len(dissimilar_neural) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(similar_neural, dissimilar_neural)
        results['t_stat'] = t_stat
        results['p_val'] = p_val

    return results


def generate_dataset(env, n_states=10000, seed=None):
    """Generate dataset of states for analysis."""
    print("Generating Analysis Dataset...")

    # Set seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        env.action_space.seed(seed)

    sampled_states = []
    env_seed = seed if seed is not None else 0
    while len(sampled_states) < n_states:
        s, _ = env.reset(seed=env_seed)
        env_seed += 1
        sampled_states.append(s)
        for _ in range(100):
            s, _, _, _, _ = env.step(env.action_space.sample())
            sampled_states.append(s)

    sampled_indices = np.random.choice(len(sampled_states), size=500, replace=False)
    sampled_states = [sampled_states[i] for i in sampled_indices]

    # Add reflected states
    for s in sampled_states[:]:
        sampled_states.append(-s)

    sampled_states = np.array(sampled_states)
    print(f"Dataset size: {len(sampled_states)} states")
    return sampled_states


def train_and_analyze(algorithm, env, sampled_states, analyzer):
    """Train a model and analyze its symmetry properties."""

    policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[256, 256])
    ppo_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))

    # Create model
    if algorithm == "dqn":
        model = DQN(
            "MlpPolicy",
            env,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=SEED,
            learning_rate=2.3e-3,
            buffer_size=100000,
            learning_starts=1000,
            target_update_interval=10,
            exploration_fraction=0.16,
            exploration_final_eps=0.04,
            train_freq=256,
            gradient_steps=128,
            max_grad_norm=10.0,
            batch_size=64
        )
        steps = 50000
    elif algorithm == "ppo":
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=ppo_kwargs,
            verbose=0,
            seed=SEED,
            learning_rate=3e-4
        )
        steps = 50000
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    # Train
    print(f"\n{'='*30}\nTraining {algorithm.upper()} ({steps} steps)...\n{'='*30}")
    model.learn(total_timesteps=steps)

    # Evaluate
    mean, std = evaluate_policy(model, env, n_eval_episodes=20, deterministic=True)
    print(f"Performance -> Mean Reward: {mean:.2f} +/- {std:.2f}")

    # Probe network
    probe = NetworkProbe(model.policy)
    n_hooks = probe.register_hooks()
    print(f"Hooks registered: {n_hooks}")
    probe.clear_activations()

    # Forward pass through network
    obs_tensor = torch.as_tensor(sampled_states, device=model.device, dtype=torch.float32)
    with torch.no_grad():
        if algorithm == "dqn":
            model.q_net(obs_tensor)
        else:
            # Activate both policy and value networks
            model.policy.get_distribution(obs_tensor)
            model.policy.predict_values(obs_tensor)

    # Generate ground truth matrices
    gt = analyzer.generate_ground_truth_matrices(sampled_states, model)

    # Analyze layers
    # Each entry is (similar_mean, dissimilar_mean, similar_std, dissimilar_std, similar_count, dissimilar_count)
    layer_scores = {"state": [], "lqr": [], "model": []}
    layer_names = []

    all_layers = list(probe.activations.keys())
    if algorithm == "dqn":
        relevant_layers = [l for l in all_layers if "q_net" in l]
    else:
        relevant_layers = [l for l in all_layers if "policy_net" in l or "action_net" in l or "value_net" in l]

    if not relevant_layers:
        relevant_layers = all_layers

    print(f"Analyzing {len(relevant_layers)} layers: {[l.split('.')[-1] for l in relevant_layers]}")

    n_samples = len(sampled_states)

    for idx, layer in enumerate(relevant_layers):
        print(f"  Processing layer {idx+1}/{len(relevant_layers)}: {layer.split('.')[-1]}...", end=' ', flush=True)
        rsm = probe.get_activation_similarity_matrix(layer, max_samples=n_samples)
        if rsm is None:
            print("skipped (no activations)")
            continue

        results_state = analyze_similarity_groups(gt["state_sym"], rsm)
        results_lqr = analyze_similarity_groups(gt["lqr_sym"], rsm)
        results_model = analyze_similarity_groups(gt["model_sym"], rsm)

        # Store (similar_mean, dissimilar_mean, similar_std, dissimilar_std, similar_count, dissimilar_count)
        layer_scores["state"].append((
            results_state['similar_mean'], results_state['dissimilar_mean'],
            results_state['similar_std'], results_state['dissimilar_std'],
            results_state['similar_count'], results_state['dissimilar_count']
        ))
        layer_scores["lqr"].append((
            results_lqr['similar_mean'], results_lqr['dissimilar_mean'],
            results_lqr['similar_std'], results_lqr['dissimilar_std'],
            results_lqr['similar_count'], results_lqr['dissimilar_count']
        ))
        layer_scores["model"].append((
            results_model['similar_mean'], results_model['dissimilar_mean'],
            results_model['similar_std'], results_model['dissimilar_std'],
            results_model['similar_count'], results_model['dissimilar_count']
        ))
        layer_names.append(layer)
        print("done")

    probe.remove_hooks()

    return {
        'model': model,
        'layer_names': layer_names,
        'layer_scores': layer_scores,
        'mean_reward': mean,
        'std_reward': std
    }


def main():
    parser = argparse.ArgumentParser(description='Train CartPole model with symmetry analysis')
    parser.add_argument('--algorithm', type=str, required=True, choices=['ppo', 'dqn'],
                        help='Algorithm to use (ppo or dqn)')
    parser.add_argument('--output-dir', type=str, default='../../results',
                        help='Directory to save models and results')
    parser.add_argument('--seed', type=int, default=SEED,
                        help='Random seed')

    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize environment
    env = gym.make("CartPole-v1")

    # Initialize controllers
    lqr = CartPoleControl(env)
    analyzer = SymmetryAnalyzer(lqr)

    # Generate dataset
    sampled_states = generate_dataset(env, n_states=10000, seed=args.seed)

    # Train and analyze
    results = train_and_analyze(args.algorithm, env, sampled_states, analyzer)

    # Save model
    model_path = os.path.join(args.output_dir, f'{args.algorithm}_model.zip')
    results['model'].save(model_path)
    print(f"\nModel saved to: {model_path}")

    # Save results (without the model object)
    results_to_save = {
        'algorithm': args.algorithm,
        'layer_names': results['layer_names'],
        'layer_scores': results['layer_scores'],
        'mean_reward': results['mean_reward'],
        'std_reward': results['std_reward']
    }

    results_path = os.path.join(args.output_dir, f'{args.algorithm}_results.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved to: {results_path}")

    env.close()
    print("\nTraining complete!")


if __name__ == '__main__':
    main()
