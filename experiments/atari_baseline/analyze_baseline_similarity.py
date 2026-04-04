#!/usr/bin/env python3
"""
Baseline Similarity Analysis for Atari Agents
==============================================

This script computes baseline representational similarity across layers for
DQN and QR-DQN agents on Atari environments. No symmetry analysis is performed;
we simply compute the average pairwise similarity across random states at each
layer to see how representations evolve through the network.

Usage:
    python analyze_baseline_similarity.py [OPTIONS]

Options:
    --envs ENV1,ENV2,...            Comma-separated list of environments (default: Pong)
    --num-states N                  Number of states to sample (default: 500)
    --output-dir DIR                Output directory (default: results/)
    --seed SEED                     Random seed (default: 42)
    --no-plots                      Skip generating plots
    --activation-processing METHOD  Processing method: none, pooling, pca, random_projection
    --pca-components N              Number of PCA components (default: 100)
    --rp-dimension N                Random projection dimension (default: 100)
    --rp-repetitions N              Random projection repetitions (default: 10)

Examples:
    # Analyze Pong only
    python analyze_baseline_similarity.py --envs Pong

    # Analyze multiple environments
    python analyze_baseline_similarity.py --envs Pong,Breakout,SpaceInvaders

    # Use random projection for faster processing
    python analyze_baseline_similarity.py --activation-processing random_projection

    # Use spatial pooling for conv layers
    python analyze_baseline_similarity.py --activation-processing pooling
"""

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import ale_py
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from huggingface_sb3 import load_from_hub
from sb3_contrib import QRDQN
from sklearn.decomposition import PCA
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

warnings.filterwarnings('ignore')
sys.modules['gym'] = gym


# Environment configurations
ATARI_ENVS = {
    'Pong': 'PongNoFrameskip-v4',
    'Breakout': 'BreakoutNoFrameskip-v4',
    'SpaceInvaders': 'SpaceInvadersNoFrameskip-v4',
    'Enduro': 'EnduroNoFrameskip-v4',
    'Qbert': 'QbertNoFrameskip-v4',
}

# HuggingFace model repos
HF_REPOS = {
    'dqn': 'sb3/dqn-{env}',
    'qrdqn': 'sb3/qrdqn-{env}',
}


class ActivationExtractor:
    """Extract activations from neural network layers."""

    def __init__(self, model, model_type: str):
        self.model = model
        self.model_type = model_type
        self.activations = {}
        self.hooks = []
        self.layer_order = []
        # Detect device from model's policy parameters
        self.device = next(model.policy.parameters()).device

    def register_hooks(self):
        """Register forward hooks on all relevant layers."""
        self.hooks.clear()
        self.layer_order.clear()

        def make_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu().numpy()
            return hook

        # Get the Q-network (DQN uses q_net, QR-DQN uses quantile_net)
        if hasattr(self.model, 'q_net'):
            net = self.model.q_net
        elif hasattr(self.model, 'quantile_net'):
            net = self.model.quantile_net
        elif hasattr(self.model, 'policy'):
            # For QR-DQN, the quantile_net is inside policy
            if hasattr(self.model.policy, 'quantile_net'):
                net = self.model.policy.quantile_net
            else:
                net = self.model.policy
        else:
            raise ValueError("Could not find Q-network in model")

        # Register hooks on feature extractor CNN
        if hasattr(net, 'features_extractor') and hasattr(net.features_extractor, 'cnn'):
            cnn = net.features_extractor.cnn
            for i, layer in enumerate(cnn):
                if isinstance(layer, (nn.Conv2d, nn.ReLU, nn.Flatten)):
                    name = f'cnn_{i}_{layer.__class__.__name__}'
                    hook = layer.register_forward_hook(make_hook(name))
                    self.hooks.append(hook)
                    self.layer_order.append(name)

        # Register hooks on feature extractor linear
        if hasattr(net, 'features_extractor') and hasattr(net.features_extractor, 'linear'):
            linear = net.features_extractor.linear
            for i, layer in enumerate(linear):
                if isinstance(layer, (nn.Linear, nn.ReLU)):
                    name = f'linear_{i}_{layer.__class__.__name__}'
                    hook = layer.register_forward_hook(make_hook(name))
                    self.hooks.append(hook)
                    self.layer_order.append(name)

        # Register hooks on Q-network head (DQN)
        if hasattr(net, 'q_net'):
            q_net = net.q_net
            if isinstance(q_net, nn.Sequential):
                for i, layer in enumerate(q_net):
                    if isinstance(layer, (nn.Linear, nn.ReLU)):
                        name = f'q_net_{i}_{layer.__class__.__name__}'
                        hook = layer.register_forward_hook(make_hook(name))
                        self.hooks.append(hook)
                        self.layer_order.append(name)

        # Register hooks on quantile network head (QR-DQN)
        if hasattr(net, 'quantile_net'):
            quantile_net = net.quantile_net
            if isinstance(quantile_net, nn.Sequential):
                for i, layer in enumerate(quantile_net):
                    if isinstance(layer, (nn.Linear, nn.ReLU)):
                        name = f'quantile_net_{i}_{layer.__class__.__name__}'
                        hook = layer.register_forward_hook(make_hook(name))
                        self.hooks.append(hook)
                        self.layer_order.append(name)

        print(f"Registered {len(self.hooks)} hooks: {self.layer_order}")

    def extract(self, states: np.ndarray) -> Dict[str, np.ndarray]:
        """Extract activations for a batch of states."""
        self.activations.clear()

        # Convert to tensor and move to device
        # NOTE: Do NOT normalize by 255 here - NatureCNN does that internally
        states_tensor = torch.FloatTensor(states).to(self.device)

        with torch.no_grad():
            if hasattr(self.model, 'q_net'):
                self.model.q_net(states_tensor)
            elif hasattr(self.model, 'quantile_net'):
                self.model.quantile_net(states_tensor)
            elif hasattr(self.model.policy, 'quantile_net'):
                # QR-DQN: quantile_net is inside policy
                self.model.policy.quantile_net(states_tensor)
            else:
                self.model.policy(states_tensor)

        return dict(self.activations)

    def cleanup(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def create_env(env_name: str):
    """Create Atari environment with standard wrappers."""
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)
    return env


def load_model(model_type: str, env_name: str, env):
    """Load a model from HuggingFace hub."""
    model_classes = {'dqn': DQN, 'qrdqn': QRDQN}

    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "optimize_memory_usage": False,
        "handle_timeout_termination": False,
    }

    repo_id = HF_REPOS[model_type].format(env=env_name)
    filename = f"{model_type}-{env_name}.zip"

    print(f"Loading {model_type.upper()} from {repo_id}...")
    try:
        path = load_from_hub(repo_id=repo_id, filename=filename)
        model = model_classes[model_type].load(path, env=env, custom_objects=custom_objects)
        print(f"  Loaded successfully")
        return model
    except Exception as e:
        print(f"  Failed to load: {e}")
        return None


def collect_states(model, env, num_states: int, verbose: bool = True,
                   oversample_factor: int = 10, seed: int = 42) -> np.ndarray:
    """
    Collect states by running the model in the environment.

    To avoid high correlations from consecutive frames, we collect more states
    than needed and randomly subsample.
    """
    # Collect more states than needed to get diverse samples
    total_to_collect = num_states * oversample_factor
    all_states = []
    obs = env.reset()

    if verbose:
        print(f"Collecting {total_to_collect} states (will subsample to {num_states})...")

    while len(all_states) < total_to_collect:
        all_states.append(obs[0].copy())  # obs is (1, 4, 84, 84)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        if done[0]:
            obs = env.reset()

    # Randomly subsample to get diverse states (not consecutive frames)
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(all_states), size=min(num_states, len(all_states)), replace=False)
    indices = np.sort(indices)  # Keep temporal order for debugging
    states = [all_states[i] for i in indices]

    if verbose:
        print(f"  Subsampled {len(states)} states from {len(all_states)} collected")

    return np.array(states)


def compute_layer_similarity(activations: np.ndarray) -> Tuple[float, float]:
    """
    Compute average pairwise correlation for a layer.

    Args:
        activations: 2D array of shape (n_states, n_features)

    Returns:
        Tuple of (mean, std) of pairwise correlations
    """
    n_states = activations.shape[0]

    corr_matrix = np.corrcoef(activations)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    upper_tri = np.triu_indices(n_states, k=1)
    corr_vals = corr_matrix[upper_tri]

    return (float(np.mean(corr_vals)), float(np.std(corr_vals)))


def process_activations(
    activations: np.ndarray,
    method: str = 'none',
    pca_components: int = 100,
    rp_dimension: int = 100,
    rp_repetitions: int = 10,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Process activations before computing similarity.

    Args:
        activations: Raw activations of shape (n_states, ...)
        method: Processing method ('none', 'pooling', 'pca', 'random_projection')
        pca_components: Number of PCA components
        rp_dimension: Dimension for random projection
        rp_repetitions: Number of random projection repetitions to average
        rng: Random number generator for random projection

    Returns:
        Processed activations of shape (n_states, n_features)
    """
    n_states = activations.shape[0]

    if method == 'none':
        return activations.reshape(n_states, -1)

    elif method == 'pooling':
        # Spatial pooling for conv layers (mean over spatial dimensions)
        if len(activations.shape) == 4:  # (batch, channels, H, W)
            # Global average pooling
            pooled = activations.mean(axis=(2, 3))
            return pooled
        else:
            return activations.reshape(n_states, -1)

    elif method == 'pca':
        flat = activations.reshape(n_states, -1)
        n_components = min(pca_components, flat.shape[0], flat.shape[1])
        if n_components < 2:
            return flat
        pca = PCA(n_components=n_components)
        return pca.fit_transform(flat)

    elif method == 'random_projection':
        flat = activations.reshape(n_states, -1)
        n_features = flat.shape[1]

        if rng is None:
            rng = np.random.default_rng()

        target_dim = min(rp_dimension, n_features)

        # Average over multiple random projections for stability
        projected_sum = np.zeros((n_states, target_dim))
        for _ in range(rp_repetitions):
            # Gaussian random projection matrix
            proj_matrix = rng.standard_normal((n_features, target_dim)) / np.sqrt(target_dim)
            projected_sum += flat @ proj_matrix

        return projected_sum / rp_repetitions

    else:
        raise ValueError(f"Unknown processing method: {method}")


def analyze_model(
    model,
    model_type: str,
    env,
    num_states: int,
    verbose: bool = True,
    activation_processing: str = 'none',
    pca_components: int = 100,
    rp_dimension: int = 100,
    rp_repetitions: int = 10,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Analyze a single model: collect states, extract activations, compute similarities.
    """
    if model is None:
        return {}

    # Collect states
    states = collect_states(model, env, num_states, verbose, seed=seed)

    # Extract activations
    extractor = ActivationExtractor(model, model_type)
    extractor.register_hooks()

    # Process in batches to avoid memory issues
    batch_size = 100
    all_activations = defaultdict(list)

    for i in range(0, len(states), batch_size):
        batch = states[i:i+batch_size]
        batch_activations = extractor.extract(batch)
        for layer_name, acts in batch_activations.items():
            all_activations[layer_name].append(acts)

    # Concatenate batches
    layer_activations = {}
    for layer_name in extractor.layer_order:
        if layer_name in all_activations:
            layer_activations[layer_name] = np.concatenate(all_activations[layer_name], axis=0)

    extractor.cleanup()

    # Compute per-layer similarities
    results = {
        'model_type': model_type,
        'num_states': num_states,
        'layer_order': extractor.layer_order,
        'layer_similarities': {},
        'activation_processing': activation_processing,
    }

    if verbose:
        print(f"Computing layer similarities (processing: {activation_processing})...")

    rng = np.random.default_rng(seed)

    for layer_name in extractor.layer_order:
        if layer_name not in layer_activations:
            continue
        acts = layer_activations[layer_name]
        original_shape = acts.shape

        # Process activations before computing similarity
        acts_processed = process_activations(
            acts,
            method=activation_processing,
            pca_components=pca_components,
            rp_dimension=rp_dimension,
            rp_repetitions=rp_repetitions,
            rng=rng
        )

        corr_mean, corr_std = compute_layer_similarity(acts_processed)
        results['layer_similarities'][layer_name] = {
            'mean': corr_mean,
            'std': corr_std,
            'shape': original_shape,
            'processed_shape': acts_processed.shape,
        }
        if verbose:
            print(f"  {layer_name}: corr={corr_mean:.4f} ± {corr_std:.4f}")

    return results


def plot_similarity_comparison(
    results: Dict[str, Dict[str, Any]],
    env_name: str,
    save_path: Optional[str] = None
):
    """Plot layer-wise similarity comparison between models."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    colors = {'dqn': '#1f77b4', 'qrdqn': '#ff7f0e'}
    markers = {'dqn': 'o', 'qrdqn': 's'}

    for model_type, model_results in results.items():
        if not model_results:
            continue

        layer_order = model_results['layer_order']
        similarities = model_results['layer_similarities']

        x = list(range(len(layer_order)))
        y = [similarities[l]['mean'] for l in layer_order if l in similarities]
        yerr = [similarities[l]['std'] for l in layer_order if l in similarities]

        ax.errorbar(
            x[:len(y)], y, yerr=yerr,
            label=model_type.upper(),
            color=colors.get(model_type, 'gray'),
            marker=markers.get(model_type, 'o'),
            markersize=6,
            capsize=2,
            linewidth=1.5,
        )

    ax.set_xlabel('Layer Index', fontsize=10)
    ax.set_ylabel('Correlation', fontsize=10)
    ax.set_title(f'Layer-wise Representational Similarity: {env_name}', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()


def plot_multi_env_comparison(
    all_results: Dict[str, Dict[str, Dict[str, Any]]],
    save_path: Optional[str] = None
):
    """Plot comparison across multiple environments."""
    n_envs = len(all_results)
    if n_envs == 0:
        return

    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)

    colors = {'dqn': '#1f77b4', 'qrdqn': '#ff7f0e'}
    markers = {'dqn': 'o', 'qrdqn': 's'}

    for idx, (env_name, env_results) in enumerate(all_results.items()):
        ax = axes[0, idx]

        for model_type, model_results in env_results.items():
            if not model_results:
                continue

            layer_order = model_results['layer_order']
            similarities = model_results['layer_similarities']

            x = list(range(len(layer_order)))
            y = [similarities[l]['mean'] for l in layer_order if l in similarities]
            yerr = [similarities[l]['std'] for l in layer_order if l in similarities]

            ax.errorbar(
                x[:len(y)], y, yerr=yerr,
                label=model_type.upper(),
                color=colors.get(model_type, 'gray'),
                marker=markers.get(model_type, 'o'),
                markersize=6,
                capsize=2,
                linewidth=1.5,
            )

        ax.set_xlabel('Layer Index', fontsize=10)
        ax.set_ylabel('Correlation', fontsize=10)
        ax.set_title(env_name, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle('Layer-wise Representational Similarity Across Atari Environments', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved multi-env plot to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Baseline similarity analysis for Atari agents',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--envs', default='Pong',
                        help='Comma-separated environment names (default: Pong)')
    parser.add_argument('--num-states', type=int, default=500,
                        help='Number of states to sample (default: 500)')
    parser.add_argument('--output-dir', default='results',
                        help='Output directory (default: results/)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--activation-processing', default='none',
                        choices=['none', 'pooling', 'pca', 'random_projection'],
                        help='Activation processing method (default: none)')
    parser.add_argument('--pca-components', type=int, default=100,
                        help='Number of PCA components (default: 100)')
    parser.add_argument('--rp-dimension', type=int, default=100,
                        help='Random projection dimension (default: 100)')
    parser.add_argument('--rp-repetitions', type=int, default=10,
                        help='Random projection repetitions (default: 10)')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parse environments
    env_names = [e.strip() for e in args.envs.split(',')]
    print(f"Environments to analyze: {env_names}")
    print(f"States per model: {args.num_states}")
    print(f"Activation processing: {args.activation_processing}")
    if args.activation_processing == 'pca':
        print(f"  PCA components: {args.pca_components}")
    elif args.activation_processing == 'random_projection':
        print(f"  RP dimension: {args.rp_dimension}, repetitions: {args.rp_repetitions}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'plots').mkdir(exist_ok=True)
    (output_dir / 'data').mkdir(exist_ok=True)

    all_results = {}

    for env_name in env_names:
        if env_name not in ATARI_ENVS:
            print(f"Unknown environment: {env_name}, skipping...")
            continue

        env_id = ATARI_ENVS[env_name]
        print(f"\n{'='*60}")
        print(f"Analyzing {env_name} ({env_id})")
        print('='*60)

        # Create environment
        env = create_env(env_id)

        env_results = {}

        for model_type in ['dqn', 'qrdqn']:
            print(f"\n--- {model_type.upper()} ---")

            # Load model
            model = load_model(model_type, env_id, env)

            if model is not None:
                # Analyze model
                results = analyze_model(
                    model, model_type, env, args.num_states,
                    activation_processing=args.activation_processing,
                    pca_components=args.pca_components,
                    rp_dimension=args.rp_dimension,
                    rp_repetitions=args.rp_repetitions,
                    seed=args.seed
                )
                env_results[model_type] = results
            else:
                env_results[model_type] = {}

        env.close()
        all_results[env_name] = env_results

        # Plot individual environment comparison
        if not args.no_plots and any(env_results.values()):
            plot_path = output_dir / 'plots' / f'{env_name}_similarity.png'
            plot_similarity_comparison(env_results, env_name, str(plot_path))

    # Plot multi-environment comparison
    if not args.no_plots and len(all_results) > 1:
        plot_path = output_dir / 'plots' / 'multi_env_comparison.png'
        plot_multi_env_comparison(all_results, str(plot_path))

    # Save results as JSON
    def to_json(obj):
        if isinstance(obj, dict):
            return {k: to_json(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [to_json(x) for x in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    results_file = output_dir / 'data' / 'baseline_similarity_results.json'
    with open(results_file, 'w') as f:
        json.dump(to_json(all_results), f, indent=2)
    print(f"\nResults saved to {results_file}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)

    for env_name, env_results in all_results.items():
        print(f"\n{env_name}:")
        for model_type, model_results in env_results.items():
            if not model_results:
                print(f"  {model_type.upper()}: Not available")
                continue

            similarities = model_results['layer_similarities']
            if similarities:
                first_layer = list(similarities.keys())[0]
                last_layer = list(similarities.keys())[-1]
                print(f"  {model_type.upper()}:")
                print(f"    First layer ({first_layer}): "
                      f"corr={similarities[first_layer]['mean']:.4f} ± {similarities[first_layer]['std']:.4f}")
                print(f"    Last layer ({last_layer}): "
                      f"corr={similarities[last_layer]['mean']:.4f} ± {similarities[last_layer]['std']:.4f}")

    print(f"\nAnalysis complete!")
    print(f"Results: {output_dir.absolute()}")


if __name__ == '__main__':
    main()
