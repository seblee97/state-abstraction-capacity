#!/usr/bin/env python3
"""
Pong Symmetry Analysis
======================

This script analyzes state and representational symmetries in trained Pong agents.
It extracts neural activations, computes representational similarity analysis (RSA),
and compares state symmetries with neural representations.

The analysis includes:
- State symmetry detection in game states
- Neural activation extraction from different model layers
- Representational Similarity Analysis (RSA) comparing state and neural patterns
- Value-based analysis of state representations
- Action consistency analysis across similar states

Usage:
    python analyze_pong_symmetry.py [OPTIONS]

Options:
    --model MODEL_NAME              Specific model to analyze (ppo, dqn, a2c, qrdqn, or "all")
    --output-dir DIR                Directory to save results and plots (default: results/)
    --download-models               Download models from HuggingFace if not present
    --num-states N                  Number of states to sample for analysis (default: 500)
    --activation-processing METHOD  Activation processing method: none (default), pooling, pca, or random_projection
    --pca-components N              Number of PCA components when using pca (default: 100)
    --rp-dimension N                Dimension for random projection (default: 100)
    --rp-repetitions N              Number of random projection repetitions to average (default: 10)
    --symmetric-pair-ratio RATIO    Target ratio of symmetric pairs (0.0-1.0, default: None for natural)
    --no-plots                      Skip generating plots
    --seed SEED                     Random seed for reproducibility

The script automatically analyzes both state symmetry (naive state-based) and policy symmetry
(action-based) for each model, and generates side-by-side comparison plots.

Examples:
    # Analyze all models (both state and policy symmetry)
    python analyze_pong_symmetry.py

    # Analyze only PPO model
    python analyze_pong_symmetry.py --model ppo

    # Use spatial pooling for conv layers (makes layers comparable)
    python analyze_pong_symmetry.py --activation-processing pooling

    # Use PCA for dimensionality reduction
    python analyze_pong_symmetry.py --activation-processing pca --pca-components 50

    # Use random projection (faster than PCA)
    python analyze_pong_symmetry.py --activation-processing random_projection --rp-dimension 100 --rp-repetitions 10

    # Download models and analyze with custom output directory
    python analyze_pong_symmetry.py --download-models --output-dir my_results/

    # Analyze DQN with 200 sampled states and pooling
    python analyze_pong_symmetry.py --model dqn --num-states 200 --activation-processing pooling

    # Ensure 30% of sampled states are symmetric pairs (stratified sampling)
    python analyze_pong_symmetry.py --symmetric-pair-ratio 0.3

    # Combine pooling and stratified sampling for better layer comparison
    python analyze_pong_symmetry.py --activation-processing pooling --symmetric-pair-ratio 0.5

    # Augment dataset with synthetically constructed symmetric pairs (recommended)
    # This creates flipped frames for each state, guaranteeing many symmetric pairs
    python analyze_pong_symmetry.py --augment-symmetric

    # Combine augmentation with pooling for best results
    python analyze_pong_symmetry.py --augment-symmetric --activation-processing pooling
"""

# Standard library imports
import argparse
import json
import os
import pickle
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

# Third-party imports
import ale_py
import cv2
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from huggingface_sb3 import load_from_hub
from sb3_contrib import QRDQN
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr, ttest_ind
from sklearn.decomposition import PCA
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

# Configure warnings and gym compatibility
warnings.filterwarnings('ignore')
sys.modules['gym'] = gym


# ============================================================================
# Constants
# ============================================================================

PONG_RAM_INDEX = {
    "ball_x":   49,
    "ball_y":   54,
    "enemy_y":  50,
    "player_y": 51,
}

BALL_X_MIN = 50
BALL_X_MAX = 208
BALL_Y_MIN = 44
BALL_Y_MAX = 207
PLAYER_Y_MIN = 38
PLAYER_Y_MAX = 203
ENEMY_Y_MIN = 0
ENEMY_Y_MAX = 208

BALL_X_MID = (BALL_X_MAX + BALL_X_MIN) / 2
BALL_Y_MID = (BALL_Y_MAX + BALL_Y_MIN) / 2

PLAYER_Y_MID = (PLAYER_Y_MAX + PLAYER_Y_MIN) / 2
ENEMY_Y_MID = (ENEMY_Y_MAX + ENEMY_Y_MIN) / 2


# ============================================================================
# Classes
# ============================================================================

class PongSymmetryAnalyzer:
    """
    A class for analyzing symmetries in Pong environments using RAM observations.
    """
    
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the Pong environment with RAM observations.
        
        Args:
            render_mode: Rendering mode for the environment (None, "human", etc.)
        """
        # Store sampled states
        self.sampled_states: List[Dict[str, Any]] = []
        
    def sample_states(
        self, 
        num_episodes: int = 5, 
        max_steps_per_episode: int = 1000,
        model: Optional[Any] = None,
        max_states: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample states from the environment using either a model or random policy.
        
        Args:
            num_episodes: Number of episodes to run
            max_steps_per_episode: Maximum steps per episode
            model: Optional model with select_action method. If None, uses random policy
            max_states: Maximum number of states to collect. If None, no limit
            
        Returns:
            List of logical game states
        """
        self.sampled_states = []
        
        for ep in range(num_episodes):
            ram, info = self.env.reset()
            prev_state = None
            
            for t in range(max_steps_per_episode):
                # Convert RAM to logical state
                state = ram_to_logic_state(ram, prev_state=prev_state)
                self.sampled_states.append(state)
                prev_state = state
                
                # Check if we've reached the maximum number of states
                if max_states is not None and len(self.sampled_states) >= max_states:
                    return self.sampled_states
                
                # Select action using model or random policy
                if model is not None:
                    # Assume model has a select_action method
                    if hasattr(model, 'select_action'):
                        action = model.select_action(ram)
                    elif hasattr(model, 'predict'):
                        action = model.predict(ram)
                    else:
                        # Try calling the model directly
                        action = model(ram)
                else:
                    # Random policy
                    action = self.env.action_space.sample()
                
                # Take step in environment
                ram, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                
                if done:
                    break
                    
        return self.sampled_states
    
    def naive_symmetry(self, state_1: Dict[str, Any], state_2: Dict[str, Any], tolerance: int = 1) -> int:
        """
        Check if two states are symmetric under naive symmetry assumption.
        Ignores scores and opponent paddle position, bins coordinates.

        States are considered symmetric when everything is reflected across the horizontal midline.
        Identical states are marked separately and should be filtered out from analysis.

        Args:
            state_1: First game state
            state_2: Second game state
            tolerance: Tolerance in pixels for distance equality (default: 1)

        Returns:
            1 if states are symmetric, 0 if dissimilar, -1 if identical (to be excluded)
        """
        # Check if ball x positions and x velocities match
        ballx = state_1["ball_x"] == state_2["ball_x"]
        balldx = state_1["ball_dx"] == state_2["ball_dx"]

        if not ballx or not balldx:
            return 0

        bally = state_1["ball_y"] == state_2["ball_y"]
        balldy = state_1["ball_dy"] == state_2["ball_dy"]
        playery = state_1["player_y"] == state_2["player_y"]

        # Filter out identical states - mark as -1 so they're excluded from both groups
        if bally and balldy and playery:
            return -1

        # Case 2: Ball off screen, paddle positions symmetric (reflected across midpoint)
        if state_1["ball_y"] is None and state_2["ball_y"] is None:
            # Check if paddles are reflections of each other across the midpoint (with tolerance)
            player_dist_1 = abs(state_1["player_y"] - PLAYER_Y_MID)
            player_dist_2 = abs(state_2["player_y"] - PLAYER_Y_MID)

            if abs(player_dist_1 - player_dist_2) <= tolerance:
                # Also check they're on opposite sides (different signs of deviation)
                if (state_1["player_y"] - PLAYER_Y_MID) * (state_2["player_y"] - PLAYER_Y_MID) < 0:
                    return 1

        # Case 3: Symmetric reflection about horizontal midline
        if (state_1["ball_y"] is not None and state_2["ball_y"] is not None):
            # Ball and player should be reflections across their respective midlines
            # Check equal distances from midpoint (with tolerance)
            ball_dist_1 = abs(state_1["ball_y"] - BALL_Y_MID)
            ball_dist_2 = abs(state_2["ball_y"] - BALL_Y_MID)
            player_dist_1 = abs(state_1["player_y"] - PLAYER_Y_MID)
            player_dist_2 = abs(state_2["player_y"] - PLAYER_Y_MID)

            ball_dist_equal = abs(ball_dist_1 - ball_dist_2) <= tolerance
            player_dist_equal = abs(player_dist_1 - player_dist_2) <= tolerance

            # Check they're on opposite sides (different signs)
            ball_opposite_sides = (state_1["ball_y"] - BALL_Y_MID) * (state_2["ball_y"] - BALL_Y_MID) < 0
            player_opposite_sides = (state_1["player_y"] - PLAYER_Y_MID) * (state_2["player_y"] - PLAYER_Y_MID) < 0

            # Ball dy should be opposite
            ball_dy_opposite = state_1["ball_dy"] == -state_2["ball_dy"]

            if ball_dist_equal and player_dist_equal and ball_opposite_sides and player_opposite_sides and ball_dy_opposite:
                return 1

        return 0
    
    def generate_similarity_matrix(
        self,
        states: Optional[List[Dict[str, Any]]] = None,
        symmetry_function: Optional[Callable] = None,
        actions: Optional[List[Any]] = None,
        use_actions: bool = False
    ) -> np.ndarray:
        """
        Generate a similarity matrix for the given states using a symmetry function.

        Args:
            states: List of states to compare. If None, uses self.sampled_states
            symmetry_function: Function to check symmetry between two states.
                             If None, uses self.naive_symmetry
            actions: Optional list of actions corresponding to each state (for policy symmetry)
            use_actions: If True, use action-based symmetry comparison

        Returns:
            Binary similarity matrix where entry (i,j) is 1 if states i and j are symmetric
        """
        if states is None:
            states = self.sampled_states

        if symmetry_function is None:
            symmetry_function = self.naive_symmetry

        if len(states) == 0:
            raise ValueError("No states provided for similarity matrix generation")

        matrix = np.zeros((len(states), len(states)), dtype=int)

        if use_actions and actions is not None:
            # Policy symmetry: compare actions directly
            for i in range(len(states)):
                for j in range(len(states)):
                    matrix[i][j] = int(symmetry_function(actions[i], actions[j]))
        else:
            # State symmetry: compare states
            for i, state_i in enumerate(states):
                for j, state_j in enumerate(states):
                    matrix[i][j] = int(symmetry_function(state_i, state_j))

        return matrix
    
    def get_similarity_stats(self, similarity_matrix: np.ndarray) -> Dict[str, float]:
        """
        Compute statistics about the similarity matrix.

        Args:
            similarity_matrix: Similarity matrix with values: 1 (symmetric), 0 (dissimilar), -1 (identical/excluded)

        Returns:
            Dictionary with statistics about symmetries
        """
        n = similarity_matrix.shape[0]

        # Get upper triangle indices (excluding diagonal)
        triu_indices = np.triu_indices(n, k=1)
        upper_triangle = similarity_matrix[triu_indices]

        # Count different pair types
        symmetric_pairs = np.sum(upper_triangle == 1)
        dissimilar_pairs = np.sum(upper_triangle == 0)
        excluded_pairs = np.sum(upper_triangle == -1)
        total_pairs = len(upper_triangle)

        return {
            "total_states": n,
            "total_pairs": total_pairs,
            "symmetric_pairs": int(symmetric_pairs),
            "dissimilar_pairs": int(dissimilar_pairs),
            "excluded_pairs": int(excluded_pairs),
            "symmetry_ratio": symmetric_pairs / (symmetric_pairs + dissimilar_pairs) if (symmetric_pairs + dissimilar_pairs) > 0 else 0.0,
            "diagonal_sum": np.sum(np.diag(similarity_matrix)),
        }
    
    def close(self):
        """Close the environment."""
        # self.env.close()
        pass


class PolicySymmetryAnalyzer(PongSymmetryAnalyzer):
    """
    Child class that defines 'symmetry' based on a model's policy.
    Two states are 'symmetric' (similar) if the model chooses the exact same action for both.
    """
    def __init__(self, render_mode: Optional[str] = None):
        """
        Initialize the PolicySymmetryAnalyzer.

        Args:
            render_mode: Rendering mode for the environment (None, "human", etc.)
        """
        # Call parent init but don't create environment (we won't use it for policy symmetry)
        self.PONG_RAM_INDEX = {
            "ball_x": 49,
            "ball_y": 54,
            "enemy_y": 50,
            "player_y": 51,
        }

        self.BALL_X_MIN = 50
        self.BALL_X_MAX = 208
        self.BALL_Y_MIN = 44
        self.BALL_Y_MAX = 207
        self.PLAYER_Y_MIN = 38
        self.PLAYER_Y_MAX = 203
        self.ENEMY_Y_MIN = 0
        self.ENEMY_Y_MAX = 208

        self.BALL_X_MID = (self.BALL_X_MAX - self.BALL_X_MIN) / 2
        self.BALL_Y_MID = (self.BALL_Y_MAX - self.BALL_Y_MIN) / 2

        self.PLAYER_Y_MID = (self.PLAYER_Y_MAX - self.PLAYER_Y_MIN) / 2
        self.ENEMY_Y_MID = (self.ENEMY_Y_MAX - self.ENEMY_Y_MIN) / 2

        self.sampled_states = []

    def get_policy_equivalence_func(self) -> Callable:
        """
        Returns a comparison function that checks if 'model'
        takes the same action in two different states.
        """
        def equivalence(action_1: Any, action_2: Any) -> bool:
            return action_1 == action_2

        return equivalence

    def generate_similarity_matrix(
        self,
        states: Optional[List[Dict[str, Any]]] = None,
        symmetry_function: Optional[Callable] = None,
        actions: Optional[List[Any]] = None,
        use_actions: bool = False
    ) -> np.ndarray:
        """
        Generate similarity matrix using policy-based symmetry.

        Args:
            states: List of states (required even for action-based comparison)
            symmetry_function: Ignored, uses policy equivalence function
            actions: List of actions taken at each state
            use_actions: Automatically set to True for policy symmetry

        Returns:
            Binary similarity matrix based on action equivalence
        """
        symmetry_function = self.get_policy_equivalence_func()
        return super().generate_similarity_matrix(
            states=states,
            symmetry_function=symmetry_function,
            actions=actions,
            use_actions=True
        )


# --- 1. The Expert Agent (Tuned for Robustness) ---
# Experimental classes (HardcodedPongAgent, ActionOptimalSymmetryAnalyzer) have been
# moved to experimental_pong_agents.py for future reference

class ModelActivationExtractor:
    """
    Extract activations from different layers of trained RL models.
    Handles various model architectures robustly.
    """
    
    def __init__(self, model, model_type='ppo'):
        self.model = model
        self.model_type = model_type.lower()
        self.activations = {}
        self.hooks = []
        
    def register_hooks(self, layer_names=None):
        """Register forward hooks to capture activations with robust architecture handling."""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu().numpy()
                elif isinstance(output, (list, tuple)) and len(output) > 0:
                    # Handle cases where output is a tuple/list
                    if isinstance(output[0], torch.Tensor):
                        self.activations[name] = output[0].detach().cpu().numpy()
            return hook
            
        # Get the policy/q-network
        if hasattr(self.model, 'policy'):
            net = self.model.policy
            net_type = 'policy'
        elif hasattr(self.model, 'q_net'):
            net = self.model.q_net
            net_type = 'q_net'
        else:
            net = self.model
            net_type = 'unknown'
            
        print(f"Network type: {net_type}, Architecture: {type(net)}")
        
        # Handle different network architectures
        feature_extractors_found = []
        
        # Method 1: Handle ActorCriticCnnPolicy (PPO/A2C) - separate feature extractors
        if hasattr(net, 'pi_features_extractor') and net.pi_features_extractor is not None:
            self._register_cnn_hooks(net.pi_features_extractor, 'policy_features', hook_fn, layer_names)
            feature_extractors_found.append('policy_features')
            
        if hasattr(net, 'vf_features_extractor') and net.vf_features_extractor is not None:
            self._register_cnn_hooks(net.vf_features_extractor, 'value_features', hook_fn, layer_names)
            feature_extractors_found.append('value_features')
        
        # Method 2: Handle shared features_extractor (PPO/A2C/DQN)
        if hasattr(net, 'features_extractor') and net.features_extractor is not None:
            self._register_cnn_hooks(net.features_extractor, 'shared_features', hook_fn, layer_names)
            feature_extractors_found.append('shared_features')
        
        # Method 3: Handle DQN QNetwork structure
        if self.model_type in ['dqn', 'qrdqn']:
            if hasattr(self.model, 'q_net') and self.model.q_net is not None:
                q_net = self.model.q_net
                
                # Check if q_net has features_extractor (DQN structure)
                if hasattr(q_net, 'features_extractor') and q_net.features_extractor is not None:
                    self._register_cnn_hooks(q_net.features_extractor, 'q_features', hook_fn, layer_names)
                    feature_extractors_found.append('q_features')
                
                # Register hooks for q_net layers (final Q-value layers)
                if hasattr(q_net, 'q_net') and q_net.q_net is not None:
                    self._register_final_layers(q_net.q_net, 'q_net', hook_fn, layer_names)
                elif isinstance(q_net, nn.Sequential):
                    self._register_final_layers(q_net, 'q_net', hook_fn, layer_names)
        
        if not feature_extractors_found:
            print("Warning: Could not find any feature extractors in model architecture")
            print("Model structure:")
            print(net)
            # Try to auto-discover layers
            self._auto_discover_layers(net, hook_fn, layer_names)
            
        # Register hooks for final layers (action, value, q-networks)
        self._register_final_layer_hooks(net, hook_fn, layer_names)
        
        print(f"Total hooks registered: {len(self.hooks)}")
        print(f"Feature extractors found: {feature_extractors_found}")
        
    def _register_cnn_hooks(self, feature_extractor, prefix, hook_fn, layer_names):
        """Register hooks for a CNN feature extractor."""
        # Handle NatureCNN structure (both DQN and PPO use this)
        if hasattr(feature_extractor, 'cnn'):
            cnn_layers = feature_extractor.cnn
            print(f"Found CNN layers in {prefix} with {len(cnn_layers)} layers")
            
            for i, layer in enumerate(cnn_layers):
                if isinstance(layer, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten)):
                    name = f'{prefix}_cnn_{i}_{layer.__class__.__name__}'
                    if layer_names is None or name in layer_names:
                        hook = layer.register_forward_hook(hook_fn(name))
                        self.hooks.append(hook)
                        print(f"  Registered hook for {name}")
        
        # Handle linear layers in feature extractor
        if hasattr(feature_extractor, 'linear'):
            linear_layers = feature_extractor.linear
            print(f"Found linear layers in {prefix} with {len(linear_layers)} layers")
            
            for i, layer in enumerate(linear_layers):
                if isinstance(layer, (nn.Linear, nn.ReLU)):
                    name = f'{prefix}_linear_{i}_{layer.__class__.__name__}'
                    if layer_names is None or name in layer_names:
                        hook = layer.register_forward_hook(hook_fn(name))
                        self.hooks.append(hook)
                        print(f"  Registered hook for {name}")
    
    def _register_final_layers(self, layer_sequence, prefix, hook_fn, layer_names):
        """Register hooks for final layers (q_net, action_net, value_net)."""
        if isinstance(layer_sequence, nn.Sequential):
            for i, layer in enumerate(layer_sequence):
                if isinstance(layer, (nn.Linear, nn.ReLU)):
                    name = f'{prefix}_{i}_{layer.__class__.__name__}'
                    if layer_names is None or name in layer_names:
                        hook = layer.register_forward_hook(hook_fn(name))
                        self.hooks.append(hook)
                        print(f"  Registered hook for {name}")
        elif isinstance(layer_sequence, (nn.Linear, nn.ReLU)):
            # Single layer
            name = f'{prefix}_{layer_sequence.__class__.__name__}'
            if layer_names is None or name in layer_names:
                hook = layer_sequence.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"  Registered hook for {name}")
    
    def _register_final_layer_hooks(self, net, hook_fn, layer_names):
        """Register hooks for final action and value layers."""
        # Action network (policy head)
        if hasattr(net, 'action_net') and net.action_net is not None:
            name = f'action_net_{net.action_net.__class__.__name__}'
            if layer_names is None or name in layer_names:
                hook = net.action_net.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"  Registered hook for {name}")
                
        # Value network (critic head)
        if hasattr(net, 'value_net') and net.value_net is not None:
            name = f'value_net_{net.value_net.__class__.__name__}'
            if layer_names is None or name in layer_names:
                hook = net.value_net.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
                print(f"  Registered hook for {name}")
        
        # Q-network (DQN head) - check if it's at the top level
        if hasattr(net, 'q_net') and net.q_net is not None and not hasattr(net, 'features_extractor'):
            # This is likely the final q_net layer
            self._register_final_layers(net.q_net, 'q_output', hook_fn, layer_names)
        
        # For PPO/A2C: look for mlp_extractor (though this one is empty in your case)
        if hasattr(net, 'mlp_extractor') and net.mlp_extractor is not None:
            mlp = net.mlp_extractor
            
            # Policy MLP
            if hasattr(mlp, 'policy_net') and mlp.policy_net is not None and len(mlp.policy_net) > 0:
                for i, layer in enumerate(mlp.policy_net):
                    if isinstance(layer, (nn.Linear, nn.ReLU)):
                        name = f'mlp_policy_{i}_{layer.__class__.__name__}'
                        if layer_names is None or name in layer_names:
                            hook = layer.register_forward_hook(hook_fn(name))
                            self.hooks.append(hook)
                            print(f"  Registered hook for {name}")
            
            # Value MLP  
            if hasattr(mlp, 'value_net') and mlp.value_net is not None and len(mlp.value_net) > 0:
                for i, layer in enumerate(mlp.value_net):
                    if isinstance(layer, (nn.Linear, nn.ReLU)):
                        name = f'mlp_value_{i}_{layer.__class__.__name__}'
                        if layer_names is None or name in layer_names:
                            hook = layer.register_forward_hook(hook_fn(name))
                            self.hooks.append(hook)
                            print(f"  Registered hook for {name}")
    
    def _auto_discover_layers(self, net, hook_fn, layer_names):
        """Auto-discover layers if standard patterns fail."""
        print("Attempting auto-discovery of layers...")
        
        def register_if_target_layer(module, name):
            if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU, nn.MaxPool2d, nn.Flatten)):
                hook_name = f'auto_{name}_{module.__class__.__name__}'
                if layer_names is None or hook_name in layer_names:
                    hook = module.register_forward_hook(hook_fn(hook_name))
                    self.hooks.append(hook)
                    print(f"  Auto-discovered: {hook_name}")
        
        # Walk through all named modules
        for name, module in net.named_modules():
            if name:  # Skip the root module (empty name)
                register_if_target_layer(module, name.replace('.', '_'))
    
    def extract_activations(self, states):
        """Extract activations for a batch of states."""
        self.activations.clear()
        
        # Convert states to tensor
        if isinstance(states, np.ndarray):
            states_tensor = torch.FloatTensor(states)
        else:
            states_tensor = torch.FloatTensor(np.array(states))
            
        # Ensure correct shape (batch_size, channels, height, width)
        if len(states_tensor.shape) == 3:
            states_tensor = states_tensor.unsqueeze(0)
        if states_tensor.shape[-1] == 3:  # If channels last
            states_tensor = states_tensor.permute(0, 3, 1, 2)
            
        # Normalize to [0, 1] if needed
        if states_tensor.max() > 1.0:
            states_tensor = states_tensor / 255.0
            
        with torch.no_grad():
            try:
                if self.model_type in ['ppo', 'a2c']:
                    # For policy gradient methods - activate both policy and value networks
                    self.model.policy.get_distribution(states_tensor)
                    self.model.policy.predict_values(states_tensor)
                elif self.model_type in ['dqn', 'qrdqn']:
                    # For Q-learning methods
                    if hasattr(self.model, 'q_net'):
                        self.model.q_net(states_tensor)
                    else:
                        # Fallback
                        self.model.policy(states_tensor)
                else:
                    # Generic fallback
                    if hasattr(self.model, 'policy'):
                        self.model.policy(states_tensor)
                    elif hasattr(self.model, 'q_net'):
                        self.model.q_net(states_tensor)
                    else:
                        self.model(states_tensor)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                return {}
                
        return dict(self.activations)
    
    def cleanup(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()


class ModelEvaluatorWithRSA:
    """
    Evaluate trained RL models on Pong and collect activations for RSA analysis.

    This class combines model evaluation functionality with RSA (Representational
    Similarity Analysis) by collecting neural activations during rollouts.
    """

    def __init__(self, analyzer, env_name: str = "PongNoFrameskip-v4", render: bool = False,
                 activation_processing: str = 'none', pca_components: int = 100,
                 rp_dimension: int = 100, rp_repetitions: int = 10,
                 symmetric_pair_ratio: Optional[float] = None,
                 augment_symmetric: bool = False,
                 symmetric_top_rows: int = 13,
                 symmetric_bottom_rows: int = 6,
                 symmetric_midline_threshold: float = 5.0):
        self.env_name = env_name
        self.render = render
        self.evaluation_results = {}
        self.analyzer = analyzer
        self.rsa_results = {}
        self.activation_processing = activation_processing
        self.pca_components = pca_components
        self.rp_dimension = rp_dimension
        self.rp_repetitions = rp_repetitions
        self.symmetric_pair_ratio = symmetric_pair_ratio
        self.augment_symmetric = augment_symmetric
        self.symmetric_top_rows = symmetric_top_rows
        self.symmetric_bottom_rows = symmetric_bottom_rows
        self.symmetric_midline_threshold = symmetric_midline_threshold

    def create_evaluation_env(self):
        """Create environment for evaluation matching the training setup."""
        from stable_baselines3.common.env_util import make_atari_env
        from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage
        
        # Create the same environment setup as used in training
        if self.render:
            env = make_atari_env(
                self.env_name, 
                n_envs=1, 
                env_kwargs={"render_mode": "human"}
            )
        else:
            env = make_atari_env(
                self.env_name, 
                n_envs=1
            )
            
        # Apply the same wrappers as in training
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)
        
        return env
        
    def evaluate_model(
        self,
        model,
        model_name: str,
        num_episodes: int = 10,
        max_episode_length: int = 10000,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate a single model over multiple episodes.
        
        Args:
            model: Trained RL model
            model_name: Name identifier for the model
            num_episodes: Number of evaluation episodes
            max_episode_length: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            verbose: Print progress
            
        Returns:
            Dictionary with evaluation metrics
        """
        env = self.create_evaluation_env()
        
        episode_rewards = []
        episode_lengths = []
        wins = 0
        losses = 0
        
        if verbose:
            print(f"\nEvaluating {model_name} model...")
            print(f"Episodes: {num_episodes}, Max length: {max_episode_length}")
            
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            
            start_time = time.time()
            
            for step in range(max_episode_length):
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Take step
                obs, reward, done, info = env.step(action)
                
                # Extract scalar values from vectorized environment
                episode_reward += reward[0]  # reward is array of length 1
                episode_length += 1
                
                if done[0]:  # done is array of length 1
                    break
                    
            episode_time = time.time() - start_time
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Determine win/loss (in Pong, positive reward means winning)
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
                
            if verbose:
                print(f"  Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}, Time={episode_time:.2f}s")
                
        env.close()
        
        # Calculate statistics
        results = {
            'model_name': model_name,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'wins': wins,
            'losses': losses,
            'draws': num_episodes - wins - losses,
            'win_rate': wins / num_episodes,
            'loss_rate': losses / num_episodes
        }
        
        self.evaluation_results[model_name] = results
        
        if verbose:
            print(f"\n{model_name.upper()} Results:")
            print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
            print(f"  Win Rate: {results['win_rate']:.2%}")
            print(f"  Mean Episode Length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            
        return results
        
    def evaluate_all_models(
        self,
        models: Dict,
        num_episodes: int = 10,
        max_episode_length: int = 10000,
        deterministic: bool = True
    ) -> Dict:
        """
        Evaluate all provided models.
        
        Args:
            models: Dictionary of {model_name: model} pairs
            num_episodes: Number of episodes per model
            max_episode_length: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            
        Returns:
            Dictionary with all evaluation results
        """
        print(f"Evaluating {len(models)} models on {self.env_name}")
        print("=" * 50)
        
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_model(
                    model=model,
                    model_name=model_name,
                    num_episodes=num_episodes,
                    max_episode_length=max_episode_length,
                    deterministic=deterministic
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                traceback.print_exc()
                continue
                
        return all_results
        
    def compare_models(self, results: Dict = None) -> None:
        """
        Print comparison table of model performances.
        
        Args:
            results: Results dictionary, uses self.evaluation_results if None
        """
        if results is None:
            results = self.evaluation_results
            
        if not results:
            print("No evaluation results available. Run evaluate_all_models first.")
            return
            
        print("\n" + "=" * 80)
        print("MODEL PERFORMANCE COMPARISON")
        print("=" * 80)
        print(f"{'Model':<10} {'Mean Reward':<12} {'Win Rate':<10} {'Mean Length':<12} {'Episodes':<10}")
        print("-" * 80)
        
        # Sort by mean reward
        sorted_results = sorted(results.items(), key=lambda x: x[1]['mean_reward'], reverse=True)
        
        for model_name, result in sorted_results:
            print(f"{model_name:<10} {result['mean_reward']:>8.2f} ± {result['std_reward']:>4.2f} "
                  f"{result['win_rate']:>8.1%}   {result['mean_length']:>8.1f} ± {result['std_length']:>4.1f} "
                  f"{result['num_episodes']:>8d}")
                  
    def plot_performance(self, results: Dict = None, save_path: str = None) -> None:
        """
        Plot model performance comparison.
        
        Args:
            results: Results dictionary, uses self.evaluation_results if None
            save_path: Path to save the plot
        """
        if results is None:
            results = self.evaluation_results
            
        if not results:
            print("No evaluation results available.")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        model_names = list(results.keys())
        mean_rewards = [results[name]['mean_reward'] for name in model_names]
        std_rewards = [results[name]['std_reward'] for name in model_names]
        win_rates = [results[name]['win_rate'] for name in model_names]
        mean_lengths = [results[name]['mean_length'] for name in model_names]
        
        # Mean rewards
        axes[0, 0].bar(model_names, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Mean Episode Reward')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # Win rates
        axes[0, 1].bar(model_names, win_rates, alpha=0.7, color='green')
        axes[0, 1].set_title('Win Rate')
        axes[0, 1].set_ylabel('Win Rate')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Episode lengths
        axes[1, 0].bar(model_names, mean_lengths, alpha=0.7, color='orange')
        axes[1, 0].set_title('Mean Episode Length')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Reward distributions (box plot)
        reward_data = [results[name]['episode_rewards'] for name in model_names]
        axes[1, 1].boxplot(reward_data, labels=model_names)
        axes[1, 1].set_title('Reward Distribution')
        axes[1, 1].set_ylabel('Episode Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()
        
    def single_episode_demo(
        self,
        model,
        model_name: str,
        render: bool = True,
        deterministic: bool = True,
        max_steps: int = 5000
    ) -> Dict:
        """
        Run a single episode demonstration with optional rendering.
        
        Args:
            model: Trained RL model
            model_name: Name of the model
            render: Whether to render the episode
            deterministic: Whether to use deterministic policy
            max_steps: Maximum steps in the episode
            
        Returns:
            Dictionary with episode information
        """
        # Create environment with the same setup as evaluation
        env = self.create_evaluation_env()
        
        print(f"Running demo for {model_name} (render={render})")
        
        obs = env.reset()
        episode_reward = 0
        step_count = 0
        actions_taken = []
        rewards_per_step = []
        
        start_time = time.time()
        
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            actions_taken.append(action[0] if isinstance(action, np.ndarray) else action)
            
            obs, reward, done, info = env.step(action)
            
            episode_reward += reward[0]
            rewards_per_step.append(reward[0])
            step_count += 1
            
            if render:
                time.sleep(0.01)  # Slow down for viewing
                
            if done[0]:
                break
                
        episode_time = time.time() - start_time
        env.close()
        
        result = {
            'model_name': model_name,
            'total_reward': episode_reward,
            'episode_length': step_count,
            'episode_time': episode_time,
            'actions_taken': actions_taken,
            'rewards_per_step': rewards_per_step
        }
        
        print(f"Demo completed:")
        print(f"  Total Reward: {episode_reward}")
        print(f"  Episode Length: {step_count} steps")
        print(f"  Duration: {episode_time:.2f} seconds")
        print(f"  Result: {'Won' if episode_reward > 0 else 'Lost' if episode_reward < 0 else 'Draw'}")
        
        return result

    def evaluate_model_with_rsa(
        self,
        model,
        model_name: str,
        num_episodes: int = 10,
        max_episode_length: int = 10000,
        deterministic: bool = True,
        verbose: bool = True,
        collect_rsa: bool = True,
        max_states_for_rsa: int = 500
    ) -> Dict:
        """
        Evaluate model and simultaneously collect activations for RSA analysis.
        
        Args:
            model: Trained RL model
            model_name: Name identifier for the model
            num_episodes: Number of evaluation episodes
            max_episode_length: Maximum steps per episode
            deterministic: Whether to use deterministic policy
            verbose: Print progress
            collect_rsa: Whether to collect activations for RSA
            max_states_for_rsa: Maximum states to sample for RSA analysis
            
        Returns:
            Dictionary with evaluation metrics and RSA results
        """
        env = self.create_evaluation_env()
        
        # Standard evaluation metrics
        episode_rewards = []
        episode_lengths = []
        wins = 0
        losses = 0
        
        # RSA collection - collect ALL states first
        all_pixel_states = []  # All states in exact model format
        all_logical_states = []  # All corresponding logical states
        all_actions = []  # Actions taken at each state
        all_activations_by_step = {}  # Store activations by step, then by layer
        all_rl_context = {}
        
        # Set up activation extractor if needed
        extractor = None
        if collect_rsa:
            extractor = ModelActivationExtractor(model, model_name)
            extractor.register_hooks()
            if verbose:
                print(f"Registered hooks for {len(extractor.hooks)} layers")
        
        if verbose:
            print(f"\nEvaluating {model_name} model with RSA collection...")
            print(f"Episodes: {num_episodes}, Max length: {max_episode_length}")
            print(f"Will sample up to {max_states_for_rsa} states from all collected states for RSA")
            
        for episode in range(num_episodes):
            obs = env.reset()
            episode_reward = 0
            episode_length = 0
            prev_logical = None
            
            start_time = time.time()
            
            for step in range(max_episode_length):
                # Always collect states for RSA if enabled - we'll sample later
                if collect_rsa:
                    # Store the exact state that will be fed to the model
                    # obs[0] is shape (4, 84, 84) - exactly what the model sees
                    all_pixel_states.append(obs[0].copy())
                    
                    # Get corresponding logical state from RAM
                    try:
                        # Extract RAM from the underlying environment
                        underlying_env = env.envs[0].unwrapped
                        ram = underlying_env.ale.getRAM()
                        logical_state = ram_to_logic_state(ram, prev_state=prev_logical)
                        all_logical_states.append(logical_state)
                        prev_logical = logical_state
                    except Exception as e:
                        if verbose and len(all_pixel_states) == 1:  # Only warn once
                            print(f"Warning: Could not extract RAM: {e}")
                        # Remove the pixel state we just added if RAM extraction fails
                        all_pixel_states.pop()
                        continue  # Skip activation collection for this step
                
                # Clear activations before predict call to ensure we only get this step's activations
                if collect_rsa and extractor is not None:
                    extractor.activations.clear()
                
                # Get action from model (this triggers activation collection)
                if collect_rsa:
                    rl_context = self._extract_rl_context(model, obs, deterministic, model_name)
                    action = rl_context['chosen_action']
                    # Store action and RL context for this step
                    all_actions.append(action[0] if isinstance(action, np.ndarray) else action)
                    step_idx = len(all_logical_states) - 1  # Current step index
                    all_rl_context[step_idx] = rl_context
                else:
                    action, _ = model.predict(obs, deterministic=deterministic)
                
                # Collect activations if we're doing RSA
                if collect_rsa and extractor is not None and extractor.activations:
                    step_idx = len(all_logical_states) - 1  # Current step index
                    all_activations_by_step[step_idx] = {}
                    
                    # Store activations from the forward pass we just made
                    for layer_name, activation in extractor.activations.items():
                        # The activation should be for a single sample (batch size 1)
                        # Take the first (and only) sample from the batch
                        if len(activation.shape) > 1 and activation.shape[0] == 1:
                            all_activations_by_step[step_idx][layer_name] = activation[0]  # Remove batch dimension
                        else:
                            all_activations_by_step[step_idx][layer_name] = activation
                
                # Take step in environment
                obs, reward, done, info = env.step(action)
                
                # Update episode metrics
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    break
                    
            episode_time = time.time() - start_time
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Determine win/loss
            if episode_reward > 0:
                wins += 1
            elif episode_reward < 0:
                losses += 1
                
            if verbose:
                print(f"  Episode {episode + 1}: Reward={episode_reward:.1f}, Length={episode_length}, "
                      f"Total States Collected={len(all_logical_states)}, Time={episode_time:.2f}s")
                
        env.close()
        
        # Standard evaluation results
        evaluation_results = {
            'model_name': model_name,
            'num_episodes': num_episodes,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'std_length': np.std(episode_lengths),
            'wins': wins,
            'losses': losses,
            'draws': num_episodes - wins - losses,
            'win_rate': wins / num_episodes,
            'loss_rate': losses / num_episodes
        }
        
        # RSA analysis with sampling
        rsa_results = {}
        if collect_rsa and len(all_logical_states) > 1:
            total_states = len(all_logical_states)
            
            if verbose:
                print(f"\nCollected {total_states} total states during evaluation")
                print(f"Sampling {min(max_states_for_rsa, total_states)} states for RSA analysis...")
            
            # Sample states for RSA analysis (with optional symmetric pair bias)
            if verbose and self.symmetric_pair_ratio is not None:
                print(f"Using stratified sampling with target symmetric ratio: {self.symmetric_pair_ratio:.2%}")

            sampled_indices, sampling_stats = sample_states_with_symmetry_bias(
                all_states=all_logical_states,
                all_actions=all_actions,
                analyzer=self.analyzer,
                max_states=max_states_for_rsa,
                target_symmetric_ratio=self.symmetric_pair_ratio,
                seed=42
            )

            if verbose:
                if sampling_stats["strategy"] == "all_states":
                    print(f"Using all {total_states} states for RSA analysis")
                elif sampling_stats["strategy"] == "random":
                    print(f"Randomly sampled {len(sampled_indices)} states from {total_states} total states")
                elif sampling_stats["strategy"] == "stratified_symmetric":
                    print(f"Stratified sampling complete:")
                    print(f"  Achieved symmetric ratio: {sampling_stats['actual_symmetric_ratio']:.2%}")
                    print(f"  Total symmetric pairs in sample: {sampling_stats['sampled_symmetric_pairs']}")
            
            # Extract sampled states and activations
            sampled_logical_states = [all_logical_states[i] for i in sampled_indices]
            sampled_pixel_states = [all_pixel_states[i] for i in sampled_indices]
            sampled_actions = [all_actions[i] for i in sampled_indices]
            sampled_rl_context = [all_rl_context[i] for i in sampled_indices if i in all_rl_context]

            # Reorganize activations by layer for sampled indices
            sampled_activations = {}
            for idx_pos, original_idx in enumerate(sampled_indices):
                if original_idx in all_activations_by_step:
                    step_activations = all_activations_by_step[original_idx]
                    for layer_name, activation in step_activations.items():
                        if layer_name not in sampled_activations:
                            sampled_activations[layer_name] = []
                        sampled_activations[layer_name].append(activation)

            # Optionally augment with synthetic symmetric pairs
            symmetric_pairs = []
            if self.augment_symmetric:
                if verbose:
                    print(f"\nAugmenting dataset with synthetic symmetric pairs...")

                # Build activations_by_step dict for sampled states (mapping new indices)
                sampled_activations_by_step = {}
                for idx_pos, original_idx in enumerate(sampled_indices):
                    if original_idx in all_activations_by_step:
                        sampled_activations_by_step[idx_pos] = all_activations_by_step[original_idx]

                # Augment the dataset (we ignore augmented_actions since we'll get actual model actions)
                (augmented_logical, augmented_pixels, _,
                 _, _, symmetric_pairs) = augment_with_symmetric_pairs(
                    logical_states=sampled_logical_states,
                    pixel_states=sampled_pixel_states,
                    actions=sampled_actions,
                    activations_by_step=sampled_activations_by_step,
                    filter_near_midline=True,
                    midline_threshold=self.symmetric_midline_threshold,
                    top_rows=self.symmetric_top_rows,
                    bottom_rows=self.symmetric_bottom_rows
                )

                # Get activations for synthetic states by running them through the model
                if verbose:
                    print(f"  Computing activations for {len(symmetric_pairs)} synthetic states...")

                # Use the extractor to get activations for flipped frames
                synthetic_activations_list = []
                synthetic_actual_actions = []  # Store the model's actual actions on flipped frames
                for orig_idx, synth_idx in symmetric_pairs:
                    # Get the flipped pixel state
                    flipped_pixels = augmented_pixels[synth_idx]

                    # Run through model to get activations
                    extractor.activations.clear()

                    # Prepare observation in the format the model expects
                    # flipped_pixels is (4, 84, 84), model.predict expects (1, 4, 84, 84) numpy array
                    obs_np = flipped_pixels[np.newaxis, ...].astype(np.float32)

                    # Forward pass to get activations AND the model's actual action
                    with torch.no_grad():
                        action, _ = model.predict(obs_np, deterministic=True)
                        # Store the actual action the model takes on the flipped frame
                        actual_action = action[0] if isinstance(action, np.ndarray) else action
                        synthetic_actual_actions.append(actual_action)

                    # Store activations for this synthetic state
                    synth_activations = {}
                    for layer_name, activation in extractor.activations.items():
                        if len(activation.shape) > 1 and activation.shape[0] == 1:
                            synth_activations[layer_name] = activation[0]
                        else:
                            synth_activations[layer_name] = activation
                    synthetic_activations_list.append(synth_activations)

                # Update sampled states and activations with augmented data
                sampled_logical_states = augmented_logical
                sampled_pixel_states = augmented_pixels

                # For actions: keep original actions, but replace synthetic actions with
                # the MODEL'S ACTUAL actions on flipped frames (important for policy symmetry)
                # augmented_actions has: [original_actions..., mirrored_actions...]
                # We want: [original_actions..., model_actual_actions_on_flipped...]
                n_original_actions = len(sampled_actions)  # Number of original states before augmentation
                sampled_actions = list(sampled_actions)  # Keep original actions
                sampled_actions.extend(synthetic_actual_actions)  # Add model's actual actions on flipped frames

                # Append synthetic activations to sampled_activations
                for synth_activations in synthetic_activations_list:
                    for layer_name, activation in synth_activations.items():
                        if layer_name in sampled_activations:
                            sampled_activations[layer_name].append(activation)

                if verbose:
                    print(f"  Augmented dataset now has {len(sampled_logical_states)} states")
                    print(f"  Guaranteed symmetric pairs: {len(symmetric_pairs)}")

            if verbose:
                print(f"Performing RSA analysis with {len(sampled_logical_states)} sampled states...")

            # Generate state similarity matrix
            analyzer = self.analyzer
            state_matrix = analyzer.generate_similarity_matrix(
                states=sampled_logical_states,
                actions=sampled_actions
            )
            analyzer.close()

            if verbose:
                stats = analyzer.get_similarity_stats(state_matrix)
                print("State similarity stats:")
                for key, value in stats.items():
                    print(f"  {key}: {value}")
            
            # Analyze each layer
            for layer_name, layer_activations in sampled_activations.items():
                if len(layer_activations) != len(sampled_logical_states):
                    if verbose:
                        print(f"Warning: Mismatch in {layer_name}: {len(layer_activations)} activations vs {len(sampled_logical_states)} states")
                    continue

                try:
                    # Convert to numpy array
                    activations_array = np.array(layer_activations)

                    if verbose:
                        print(f"  Processing {layer_name} (shape: {activations_array.shape})")

                    # Compute RSA matrix
                    rsa_matrix = compute_rsa_matrix(
                        activations_array,
                        processing_method=self.activation_processing,
                        pca_components=self.pca_components,
                        rp_dimension=self.rp_dimension,
                        rp_repetitions=self.rp_repetitions
                    )

                    # State symmetry analysis: compare constructed symmetric pairs vs random non-symmetric pairs
                    if symmetric_pairs:
                        state_symmetry_analysis = analyze_state_symmetry_with_constructed_pairs(
                            symmetric_pairs=symmetric_pairs,
                            logical_states=sampled_logical_states,
                            rsa_matrix=rsa_matrix,
                            analyzer=analyzer,
                            seed=42
                        )
                    else:
                        # Fallback to old method if no augmentation
                        group_analysis = analyze_similarity_groups(state_matrix, rsa_matrix)
                        state_symmetry_analysis = {
                            'symmetric_mean': group_analysis.get('similar_mean', np.nan),
                            'symmetric_std': group_analysis.get('similar_std', np.nan),
                            'random_mean': group_analysis.get('dissimilar_mean', np.nan),
                            'random_std': group_analysis.get('dissimilar_std', np.nan),
                            't_stat': group_analysis.get('t_stat', np.nan),
                            'p_val': group_analysis.get('p_val', np.nan),
                            'n_symmetric_pairs': group_analysis.get('similar_count', 0),
                            'n_random_pairs': group_analysis.get('dissimilar_count', 0)
                        }

                    rsa_results[layer_name] = {
                        'rsa_matrix': rsa_matrix,
                        'symmetry_analysis': state_symmetry_analysis,
                        'n_states': len(sampled_logical_states),
                        'n_total_states': total_states,
                        'activation_shape': activations_array.shape
                    }

                    if verbose:
                        analysis_type = "State Symmetry" if symmetric_pairs else "Policy Symmetry"
                        print(f"    {analysis_type} Analysis:")
                        print(f"      Similar pairs mean: {state_symmetry_analysis['symmetric_mean']:.4f} ± {state_symmetry_analysis['symmetric_std']:.4f}")
                        print(f"      Dissimilar pairs mean: {state_symmetry_analysis['random_mean']:.4f} ± {state_symmetry_analysis['random_std']:.4f}")
                        print(f"      N pairs: {state_symmetry_analysis['n_symmetric_pairs']} similar, {state_symmetry_analysis['n_random_pairs']} dissimilar")
                        if not np.isnan(state_symmetry_analysis['p_val']):
                            print(f"      T-test p-value: {state_symmetry_analysis['p_val']:.4f}")

                except Exception as e:
                    if verbose:
                        print(f"Error processing {layer_name}: {e}")
                    continue
            
            # Store results for reference
            rsa_results['state_matrix'] = state_matrix
            rsa_results['logical_states'] = sampled_logical_states
            rsa_results['pixel_states'] = sampled_pixel_states
            rsa_results['rl_context'] = sampled_rl_context
            rsa_results['total_states_collected'] = total_states
            rsa_results['sampled_indices'] = sampled_indices
            rsa_results['augmented_symmetric_pairs'] = symmetric_pairs  # List of (orig_idx, synth_idx) tuples
        
        # Cleanup
        if extractor is not None:
            extractor.cleanup()
        
        # Store results
        self.evaluation_results[model_name] = evaluation_results
        if rsa_results:
            self.rsa_results[model_name] = rsa_results
            
        if verbose:
            print(f"\n{model_name.upper()} Results:")
            print(f"  Mean Reward: {evaluation_results['mean_reward']:.2f} ± {evaluation_results['std_reward']:.2f}")
            print(f"  Win Rate: {evaluation_results['win_rate']:.2%}")
            print(f"  Mean Episode Length: {evaluation_results['mean_length']:.1f} ± {evaluation_results['std_length']:.1f}")
            if rsa_results:
                total_collected = rsa_results.get('total_states_collected', 0)
                sampled_count = len(rsa_results.get('logical_states', []))
                print(f"  RSA Analysis: {len(rsa_results)-5} layers analyzed with {sampled_count} states (from {total_collected} total)")
            
        return {
            'evaluation': evaluation_results,
            'rsa': rsa_results
        }

    def _extract_rl_context(self, model, obs, deterministic, model_name):
        """
        Extract RL-specific context information from model predictions.
        
        Returns:
            Dictionary containing action, values, probabilities, etc.
        """
        
        context = {}
        
        try:
            # Convert observation to tensor format
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.FloatTensor(obs)
            else:
                obs_tensor = torch.FloatTensor(np.array(obs))
            
            with torch.no_grad():
                if model_name.lower() in ['ppo', 'a2c']:
                    # For policy gradient methods
                    policy = model.policy
                    
                    # Get action and log probability
                    action, values, log_prob = policy(obs_tensor)
                    
                    # Get action probabilities
                    action_probs = torch.softmax(policy.action_net(policy.pi_features_extractor(obs_tensor)), dim=-1)
                    
                    context.update({
                        'chosen_action': action.cpu().numpy(),
                        'action_logprobs': log_prob.cpu().numpy(),
                        'action_probs': action_probs.cpu().numpy(),
                        'predicted_value': values.cpu().numpy(),
                        'model_type': 'policy_gradient'
                    })
                    
                elif model_name.lower() in ['dqn', 'qrdqn']:
                    # For Q-learning methods
                    q_values = model.q_net(obs_tensor)
                    
                    if deterministic:
                        action = torch.argmax(q_values, dim=-1)
                    else:
                        # Add some exploration noise for non-deterministic sampling
                        action = torch.multinomial(torch.softmax(q_values / 0.1, dim=-1), 1)
                    
                    context.update({
                        'chosen_action': action.cpu().numpy(),
                        'q_values': q_values.cpu().numpy(),
                        'max_q_value': torch.max(q_values, dim=-1)[0].cpu().numpy(),
                        'action_value': q_values.gather(-1, action.unsqueeze(-1)).cpu().numpy(),
                        'model_type': 'q_learning'
                    })
                    
                    # For QRDQN, also get quantile information
                    if model_name.lower() == 'qrdqn':
                        try:
                            quantile_values = model.quantile_net(model.q_net.features_extractor(obs_tensor))
                            context['quantile_values'] = quantile_values.cpu().numpy()
                            context['quantile_mean'] = torch.mean(quantile_values, dim=-1).cpu().numpy()
                            context['quantile_std'] = torch.std(quantile_values, dim=-1).cpu().numpy()
                        except:
                            pass  # Skip if quantile extraction fails
                
                # Add general context
                context.update({
                    'deterministic': deterministic,
                    'observation_shape': obs.shape if hasattr(obs, 'shape') else None,
                    'step_type': 'evaluation'
                })
                
        except Exception as e:
            # Fallback: use standard predict method
            action, _ = model.predict(obs, deterministic=deterministic)
            context = {
                'chosen_action': action,
                'model_type': 'unknown',
                'deterministic': deterministic,
                'error': str(e)
            }
        
        return context
    
    def evaluate_all_models_with_rsa(
        self,
        models: Dict,
        num_episodes: int = 10,
        max_episode_length: int = 10000,
        deterministic: bool = True,
        max_states_for_rsa: int = 500
    ) -> Dict:
        """
        Evaluate all models with RSA analysis.
        """
        print(f"Evaluating {len(models)} models on {self.env_name} with RSA analysis")
        print("=" * 70)
        
        all_results = {}
        
        for model_name, model in models.items():
            try:
                results = self.evaluate_model_with_rsa(
                    model=model,
                    model_name=model_name,
                    num_episodes=num_episodes,
                    max_episode_length=max_episode_length,
                    deterministic=deterministic,
                    collect_rsa=True,
                    max_states_for_rsa=max_states_for_rsa
                )
                all_results[model_name] = results
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
                traceback.print_exc()
                continue
                
        return all_results
    
    def compare_rsa_across_models(self):
        """
        Compare RSA results across all evaluated models.
        """
        if not self.rsa_results:
            print("No RSA results available. Run evaluate_all_models_with_rsa first.")
            return
            
        print("\n" + "=" * 100)
        print("RSA CORRELATION SUMMARY (ON-POLICY EVALUATION STATES)")
        print("=" * 100)
        print(f"{'Model':<10} {'Layer':<25} {'N States':<10} {'Pearson':<10} {'Spearman':<10} {'Sim Mean':<10} {'Dissim Mean':<12} {'P-value':<10}")
        print("-" * 100)
        
        for model_name, rsa_data in self.rsa_results.items():
            for layer_name, layer_results in rsa_data.items():
                if layer_name in ['state_matrix', 'logical_states', 'pixel_states', 'total_states_collected', 'sampled_indices', 'rl_context']:
                    continue
                    
                pearson = layer_results['pearson_corr']
                spearman = layer_results['spearman_corr']
                n_states = layer_results['n_states']
                group_analysis = layer_results['group_analysis']
                
                sim_mean = group_analysis['similar_mean']
                dissim_mean = group_analysis['dissimilar_mean']
                p_val = group_analysis.get('p_val', np.nan)
                
                print(f"{model_name:<10} {layer_name:<25} {n_states:<10} {pearson:<10.4f} {spearman:<10.4f} {sim_mean:<10.4f} {dissim_mean:<12.4f} {p_val:<10.4f}")

    def _analyze_action_consistency(self, rl_contexts, similar_pairs, dissimilar_pairs):
        """Analyze how consistent actions are within similar vs dissimilar state pairs."""
        
        def action_agreement(ctx1, ctx2):
            action1 = ctx1.get('chosen_action', [])
            action2 = ctx2.get('chosen_action', [])
            if hasattr(action1, '__iter__') and hasattr(action2, '__iter__'):
                return np.array_equal(action1, action2)
            return action1 == action2
        
        # Action agreement in similar pairs
        similar_agreements = []
        for i, j in similar_pairs:
            if i < len(rl_contexts) and j < len(rl_contexts):
                agreement = action_agreement(rl_contexts[i], rl_contexts[j])
                similar_agreements.append(agreement)
        
        # Action agreement in dissimilar pairs (sample to avoid huge computation)
        dissimilar_agreements = []
        sample_size = min(len(dissimilar_pairs), len(similar_pairs) * 2)  # Sample for balance
        sampled_dissimilar = np.random.choice(len(dissimilar_pairs), sample_size, replace=False)
        
        for idx in sampled_dissimilar:
            i, j = dissimilar_pairs[idx]
            if i < len(rl_contexts) and j < len(rl_contexts):
                agreement = action_agreement(rl_contexts[i], rl_contexts[j])
                dissimilar_agreements.append(agreement)
        
        similar_rate = np.mean(similar_agreements) if similar_agreements else 0
        dissimilar_rate = np.mean(dissimilar_agreements) if dissimilar_agreements else 0
        
        print(f"Action Consistency:")
        print(f"  Similar states: {similar_rate:.3f} ({np.sum(similar_agreements)}/{len(similar_agreements)})")
        print(f"  Dissimilar states: {dissimilar_rate:.3f} ({np.sum(dissimilar_agreements)}/{len(dissimilar_agreements)})")
        print(f"  Difference: {similar_rate - dissimilar_rate:.3f}")
    
    def _analyze_value_patterns(self, rl_contexts, similar_pairs, dissimilar_pairs):
        """Analyze value prediction patterns in similar vs dissimilar states."""
        
        # Extract values based on model type
        values = []
        for ctx in rl_contexts:
            if 'predicted_value' in ctx:
                val = ctx['predicted_value']
                values.append(val if isinstance(val, (int, float)) else val.item() if hasattr(val, 'item') else float(val))
            elif 'max_q_value' in ctx:
                val = ctx['max_q_value']
                values.append(val if isinstance(val, (int, float)) else val.item() if hasattr(val, 'item') else float(val))
            else:
                values.append(np.nan)
        
        if not values or all(np.isnan(values)):
            print("No value predictions available for analysis")
            return
        
        # Value differences in similar pairs
        similar_value_diffs = []
        for i, j in similar_pairs:
            if i < len(values) and j < len(values) and not (np.isnan(values[i]) or np.isnan(values[j])):
                diff = abs(values[i] - values[j])
                similar_value_diffs.append(diff)
        
        # Value differences in dissimilar pairs (sample)
        dissimilar_value_diffs = []
        sample_size = min(len(dissimilar_pairs), len(similar_pairs) * 2)
        sampled_dissimilar = np.random.choice(len(dissimilar_pairs), sample_size, replace=False)
        
        for idx in sampled_dissimilar:
            i, j = dissimilar_pairs[idx]
            if i < len(values) and j < len(values) and not (np.isnan(values[i]) or np.isnan(values[j])):
                diff = abs(values[i] - values[j])
                dissimilar_value_diffs.append(diff)
        
        if similar_value_diffs and dissimilar_value_diffs:
            similar_mean = np.mean(similar_value_diffs)
            dissimilar_mean = np.mean(dissimilar_value_diffs)
            
            print(f"Value Prediction Differences:")
            print(f"  Similar states: {similar_mean:.3f} ± {np.std(similar_value_diffs):.3f}")
            print(f"  Dissimilar states: {dissimilar_mean:.3f} ± {np.std(dissimilar_value_diffs):.3f}")
            print(f"  Ratio (similar/dissimilar): {similar_mean/dissimilar_mean:.3f}")
            
            # Statistical test
            from scipy.stats import ttest_ind
            t_stat, p_val = ttest_ind(similar_value_diffs, dissimilar_value_diffs)
            print(f"  T-test: t={t_stat:.3f}, p={p_val:.4f}")

# ============================================================================
# Helper Functions
# ============================================================================

def ram_to_logic_state(
    ram: np.ndarray,
    prev_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Minimal logic-level state for ALE Pong from a 128-byte RAM vector.

    Returns:
        {
            "ball_x":   int or None,
            "ball_y":   int or None,
            "ball_dx":  int,
            "ball_dy":  int,
            "player_y": int or None,
            "enemy_y":  int or None,
        }
    """
    # Raw positions from RAM (as in OCAtari)
    ball_x_raw   = int(ram[PONG_RAM_INDEX["ball_x"]])
    ball_y_raw   = int(ram[PONG_RAM_INDEX["ball_y"]])
    enemy_y_raw  = int(ram[PONG_RAM_INDEX["enemy_y"]])
    player_y_raw = int(ram[PONG_RAM_INDEX["player_y"]])

    # OCAtari condition for “ball exists”
    ball_exists = (ball_y_raw != 0) and (ball_x_raw > 49)

    # If you want to treat "no ball" explicitly:
    ball_x = ball_x_raw if ball_exists else None
    ball_y = ball_y_raw if ball_exists else None

    # Paddles basically always exist when game is running;
    # if you want to mirror OCAtari, you could add checks similar to ram[50] / ram[51] ranges.
    player_y = player_y_raw
    enemy_y  = enemy_y_raw

    # Velocity via finite differences
    if prev_state is not None and prev_state.get("ball_x") is not None and ball_x is not None:
        ball_dx = ball_x - prev_state["ball_x"]
        ball_dy = ball_y - prev_state["ball_y"]
    else:
        ball_dx = 0
        ball_dy = 0

    return {
        "ball_x":   ball_x,
        "ball_y":   ball_y,
        "ball_dx":  ball_dx,
        "ball_dy":  ball_dy,
        "player_y": player_y,
        "enemy_y":  enemy_y,
    }


def construct_symmetric_logical_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Construct the symmetric (vertically reflected) version of a logical Pong state.

    The symmetry reflects everything across the horizontal midline of the screen.
    This means:
    - ball_y is reflected across BALL_Y_MID
    - player_y is reflected across PLAYER_Y_MID
    - enemy_y is reflected across ENEMY_Y_MID
    - ball_dy (vertical velocity) is negated
    - ball_x and ball_dx remain unchanged (horizontal position/velocity)

    Args:
        state: Original logical game state dictionary

    Returns:
        New state dictionary representing the vertically mirrored state
    """
    symmetric = state.copy()

    # Reflect ball_y across midline: new_y = 2 * mid - old_y
    if state["ball_y"] is not None:
        symmetric["ball_y"] = int(2 * BALL_Y_MID - state["ball_y"])

    # Reflect player_y across midline
    if state["player_y"] is not None:
        symmetric["player_y"] = int(2 * PLAYER_Y_MID - state["player_y"])

    # Reflect enemy_y across midline
    if state["enemy_y"] is not None:
        symmetric["enemy_y"] = int(2 * ENEMY_Y_MID - state["enemy_y"])

    # Flip vertical velocity (ball moving up becomes moving down and vice versa)
    symmetric["ball_dy"] = -state["ball_dy"]

    # ball_x and ball_dx stay the same (horizontal symmetry axis)
    return symmetric


def flip_pixel_frame_gameplay_area(pixel_state: np.ndarray, top_rows: int = 13, bottom_rows: int = 6) -> np.ndarray:
    """
    Flip the gameplay area of a preprocessed Pong frame, preserving top and bottom regions.

    This creates a vertically symmetric version of the frame by flipping only the
    gameplay area while keeping the score display (top) and bottom border intact.

    Args:
        pixel_state: Pixel state array of shape (C, H, W) where C=4 for frame-stacked
                    Atari observations (4 grayscale frames of 84x84)
        top_rows: Number of rows at the top to preserve (score region).
                  Default is 13 for standard 84x84 Atari preprocessing.
        bottom_rows: Number of rows at the bottom to preserve.
                     Default is 6 for standard 84x84 Atari preprocessing.

    Returns:
        Frame with gameplay area flipped vertically, top and bottom preserved
    """
    flipped = pixel_state.copy()
    if bottom_rows == 0:
        end_idx = None
    else:
        end_idx = -bottom_rows

    if pixel_state.ndim == 3:
        # Shape is (C, H, W) - flip along the height axis (axis=1), preserving top/bottom rows
        flipped[:, top_rows:end_idx, :] = np.flip(pixel_state[:, top_rows:end_idx, :], axis=1)
    elif pixel_state.ndim == 2:
        # Shape is (H, W) - flip along the height axis (axis=0), preserving top/bottom rows
        flipped[top_rows:end_idx, :] = np.flip(pixel_state[top_rows:end_idx, :], axis=0)
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {pixel_state.shape}")
    return flipped


def mirror_pong_action(action: int) -> int:
    """
    Mirror a Pong action for the symmetric state.

    Pong action space (standard Atari):
    - 0: NOOP (no operation)
    - 1: FIRE (starts game)
    - 2: RIGHT (move paddle up)
    - 3: LEFT (move paddle down)
    - 4: RIGHTFIRE
    - 5: LEFTFIRE

    Under vertical reflection, UP becomes DOWN and vice versa.

    Args:
        action: Original action integer

    Returns:
        Mirrored action integer
    """
    ACTION_MIRROR = {
        0: 0,  # NOOP -> NOOP
        1: 1,  # FIRE -> FIRE
        2: 3,  # RIGHT (up) -> LEFT (down)
        3: 2,  # LEFT (down) -> RIGHT (up)
        4: 5,  # RIGHTFIRE -> LEFTFIRE
        5: 4,  # LEFTFIRE -> RIGHTFIRE
    }
    return ACTION_MIRROR.get(action, action)


def augment_with_symmetric_pairs(
    logical_states: List[Dict[str, Any]],
    pixel_states: List[np.ndarray],
    actions: List[int],
    activations_by_step: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    rl_context: Optional[Dict[int, Dict[str, Any]]] = None,
    filter_near_midline: bool = True,
    midline_threshold: float = 5.0,
    top_rows: int = 13,
    bottom_rows: int = 6
) -> Tuple[List[Dict[str, Any]], List[np.ndarray], List[int],
           Optional[Dict[int, Dict[str, np.ndarray]]], Optional[Dict[int, Dict[str, Any]]],
           List[Tuple[int, int]]]:
    """
    Augment a dataset by constructing synthetic symmetric pairs for each state.

    For each original state, creates its vertically mirrored version by:
    1. Reflecting the logical state across the horizontal midline
    2. Flipping the gameplay area of the pixel frame (preserving top/bottom regions)
    3. Mirroring the action (UP <-> DOWN)

    This guarantees that the augmented dataset has a high number of symmetric pairs.

    Args:
        logical_states: List of logical game states
        pixel_states: List of pixel observations (shape: N x C x H x W, typically 4x84x84)
        actions: List of actions taken at each state
        activations_by_step: Optional dict mapping step_idx -> {layer_name: activation}
        rl_context: Optional dict mapping step_idx -> RL context info
        filter_near_midline: If True, skip states too close to the midline
                            (their symmetric version would be nearly identical)
        midline_threshold: Minimum distance from midline for ball_y and player_y
        top_rows: Number of rows to preserve at top of frame (score region)
        bottom_rows: Number of rows to preserve at bottom of frame

    Returns:
        Tuple of:
        - augmented_logical_states: Original + synthetic states (2N if no filtering)
        - augmented_pixel_states: Original + flipped pixel states
        - augmented_actions: Original + mirrored actions
        - augmented_activations: Original activations only (synthetics have no activations)
        - augmented_rl_context: Original RL context only (synthetics have no context)
        - symmetric_pairs: List of (original_idx, synthetic_idx) tuples
    """
    n_original = len(logical_states)

    augmented_logical = list(logical_states)
    augmented_pixels = list(pixel_states)
    augmented_actions = list(actions)
    augmented_activations = dict(activations_by_step) if activations_by_step else {}
    augmented_rl_context = dict(rl_context) if rl_context else {}

    symmetric_pairs = []
    skipped_near_midline = 0

    for i in range(n_original):
        state = logical_states[i]

        # Optionally filter out states too close to midline
        if filter_near_midline:
            ball_y = state.get("ball_y")
            player_y = state.get("player_y")

            # Skip if ball is near midline (symmetric would be almost identical)
            if ball_y is not None and abs(ball_y - BALL_Y_MID) < midline_threshold:
                skipped_near_midline += 1
                continue

            # Skip if player paddle is near midline
            if player_y is not None and abs(player_y - PLAYER_Y_MID) < midline_threshold:
                skipped_near_midline += 1
                continue

        # Create symmetric logical state
        symmetric_state = construct_symmetric_logical_state(state)

        # Create flipped pixel state (preserving top/bottom regions)
        symmetric_pixels = flip_pixel_frame_gameplay_area(pixel_states[i], top_rows=top_rows, bottom_rows=bottom_rows)

        # Mirror the action
        symmetric_action = mirror_pong_action(actions[i])

        # Add to augmented lists
        synthetic_idx = len(augmented_logical)
        augmented_logical.append(symmetric_state)
        augmented_pixels.append(symmetric_pixels)
        augmented_actions.append(symmetric_action)

        # Record the symmetric pair (original_idx, synthetic_idx)
        symmetric_pairs.append((i, synthetic_idx))

    print(f"  Symmetric augmentation: {n_original} original states")
    print(f"  Skipped {skipped_near_midline} states near midline")
    print(f"  Created {len(symmetric_pairs)} synthetic symmetric pairs")
    print(f"  Total states after augmentation: {len(augmented_logical)}")

    return (augmented_logical, augmented_pixels, augmented_actions,
            augmented_activations, augmented_rl_context, symmetric_pairs)


def download_models(env_name: str = "PongNoFrameskip-v4"):
    """Download pre-trained models from Hugging Face."""
    models = {}
    env = make_atari_env(env_name, n_envs=1, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    custom_objects = {
        "observation_space": env.observation_space,
        "action_space": env.action_space
    }

    def try_load_model(name, model_class, repo_id, filename, extra_objects=None):
        """Helper to load a model with fallback strategies."""
        try:
            print(f"Downloading {name.upper()} model...")
            path = load_from_hub(repo_id=repo_id, filename=filename)
            objects = {**custom_objects, **(extra_objects or {})}
            model = model_class.load(path, env=env, custom_objects=objects)
            models[name] = model
            print(f"{name.upper()} model downloaded successfully")
        except Exception as e:
            print(f"Failed to download {name.upper()} model: {e}")

    # Download all models (DQN/QRDQN need replay buffer compatibility settings)
    replay_buffer_fix = {"optimize_memory_usage": False, "handle_timeout_termination": False}

    try_load_model('ppo', PPO, f"sb3/ppo-{env_name}", f"ppo-{env_name}.zip")
    try_load_model('a2c', A2C, f"sb3/a2c-{env_name}", f"a2c-{env_name}.zip")
    try_load_model('dqn', DQN, "sb3/dqn-PongNoFrameskip-v4", "dqn-PongNoFrameskip-v4.zip",
                   extra_objects=replay_buffer_fix)
    try_load_model('qrdqn', QRDQN, "sb3/qrdqn-PongNoFrameskip-v4", "qrdqn-PongNoFrameskip-v4.zip",
                   extra_objects=replay_buffer_fix)

    env.close()
    return models

def sample_states_with_symmetry_bias(
    all_states: List,
    all_actions: List,
    analyzer,
    max_states: int,
    target_symmetric_ratio: Optional[float] = None,
    seed: int = 42
) -> Tuple[List[int], Dict[str, Any]]:
    """
    Sample states with optional bias toward symmetric pairs.

    Args:
        all_states: List of all collected states
        all_actions: List of all collected actions
        analyzer: Symmetry analyzer to identify symmetric pairs
        max_states: Maximum number of states to sample
        target_symmetric_ratio: Target ratio of symmetric pairs (0.0-1.0). If None, uses random sampling.
        seed: Random seed for reproducibility

    Returns:
        Tuple of (sampled_indices, sampling_stats)
    """
    np.random.seed(seed)
    total_states = len(all_states)

    # If we don't need to sample or no target ratio specified, use simple random sampling
    if total_states <= max_states or target_symmetric_ratio is None:
        if total_states <= max_states:
            sampled_indices = list(range(total_states))
            strategy = "all_states"
        else:
            sampled_indices = sorted(np.random.choice(total_states, size=max_states, replace=False))
            strategy = "random"

        return sampled_indices, {"strategy": strategy, "total_states": total_states}

    # Step 1: Build a quick similarity lookup by checking all pairs
    # This is expensive but necessary to identify symmetric pairs
    print(f"  Building symmetry map for {total_states} states...")
    symmetric_pairs = []  # List of (i, j) tuples where i < j

    # Get symmetry function from analyzer
    if hasattr(analyzer, 'get_policy_equivalence_func'):
        # Policy symmetry analyzer
        symmetry_func = analyzer.get_policy_equivalence_func()
    else:
        # State symmetry analyzer
        symmetry_func = analyzer.get_state_equivalence_func()

    # Find all symmetric pairs
    for i in range(total_states):
        for j in range(i + 1, total_states):
            if i != j:
                # Check if states are symmetric
                if hasattr(analyzer, 'get_policy_equivalence_func'):
                    # Policy symmetry needs actions
                    equiv = symmetry_func(all_states[i], all_states[j],
                                         all_actions[i], all_actions[j])
                else:
                    # State symmetry
                    equiv = symmetry_func(all_states[i], all_states[j])

                if equiv == 1:  # Symmetric
                    symmetric_pairs.append((i, j))

    print(f"  Found {len(symmetric_pairs)} symmetric pairs out of {total_states * (total_states - 1) // 2} possible pairs")

    # Step 2: Stratified sampling
    # We want to select states such that the pairwise similarity ratio matches target
    # target_symmetric_ratio refers to the ratio of symmetric pairs in the final similarity matrix

    # Strategy: Greedily select states that maximize symmetric pair coverage
    selected_states = set()
    covered_symmetric_pairs = []

    # Sort symmetric pairs by how many other symmetric pairs they enable
    pair_scores = {}
    for pair in symmetric_pairs:
        score = sum(1 for other_pair in symmetric_pairs
                   if pair[0] in other_pair or pair[1] in other_pair)
        pair_scores[pair] = score

    sorted_pairs = sorted(symmetric_pairs, key=lambda p: pair_scores[p], reverse=True)

    # Greedily add states from symmetric pairs
    for pair in sorted_pairs:
        i, j = pair
        if len(selected_states) >= max_states:
            break

        # Add both states if we have room
        if len(selected_states) < max_states - 1:
            selected_states.add(i)
            selected_states.add(j)
            covered_symmetric_pairs.append(pair)
        elif i not in selected_states and len(selected_states) < max_states:
            selected_states.add(i)
        elif j not in selected_states and len(selected_states) < max_states:
            selected_states.add(j)

    # Fill remaining slots with random non-selected states if needed
    remaining_states = set(range(total_states)) - selected_states
    if len(selected_states) < max_states and remaining_states:
        additional_needed = max_states - len(selected_states)
        additional_states = np.random.choice(list(remaining_states),
                                            size=min(additional_needed, len(remaining_states)),
                                            replace=False)
        selected_states.update(additional_states)

    sampled_indices = sorted(list(selected_states))

    # Calculate actual symmetric ratio in sampled set
    sampled_symmetric_pairs = sum(1 for i, j in symmetric_pairs
                                  if i in selected_states and j in selected_states)
    actual_pairs = len(sampled_indices) * (len(sampled_indices) - 1) // 2
    actual_ratio = sampled_symmetric_pairs / actual_pairs if actual_pairs > 0 else 0

    sampling_stats = {
        "strategy": "stratified_symmetric",
        "total_states": total_states,
        "sampled_states": len(sampled_indices),
        "target_symmetric_ratio": target_symmetric_ratio,
        "actual_symmetric_ratio": actual_ratio,
        "total_symmetric_pairs": len(symmetric_pairs),
        "sampled_symmetric_pairs": sampled_symmetric_pairs,
        "actual_sampled_pairs": actual_pairs,
    }

    print(f"  Sampled {len(sampled_indices)} states with {sampled_symmetric_pairs} symmetric pairs")
    print(f"  Actual symmetric pair ratio: {actual_ratio:.2%} (target: {target_symmetric_ratio:.2%})")

    return sampled_indices, sampling_stats

def process_activations(activations, method='none', n_components=100,
                        rp_dimension=100, rp_repetitions=10, rng=None):
    """
    Process activations before RSA computation to make layers comparable.

    Args:
        activations: Numpy array of activations with shape (n_states, ...)
        method: Processing method - 'none', 'pooling', 'pca', or 'random_projection'
        n_components: Number of PCA components (only used when method='pca')
        rp_dimension: Dimension for random projection (only used when method='random_projection')
        rp_repetitions: Number of random projection repetitions (only used when method='random_projection')
        rng: Random number generator for reproducibility (optional)

    Returns:
        For non-random-projection methods: Processed activations as 2D array (n_states, features)
        For random_projection: Tuple of (activations, rp_params dict) or (activations, None) if skipped
    """
    # Handle different activation shapes
    if len(activations.shape) == 2:
        # Already flat (e.g., from fully connected layers)
        activations_flat = activations
    elif len(activations.shape) == 4:
        # Conv layer: (n_states, channels, height, width)
        if method == 'pooling':
            # Global average pooling over spatial dimensions
            activations_flat = activations.mean(axis=(2, 3))
            print(f"    Applied spatial pooling: {activations.shape} -> {activations_flat.shape}")
        else:
            # Flatten spatial dimensions
            activations_flat = activations.reshape(activations.shape[0], -1)
    else:
        # Flatten any other shape
        activations_flat = activations.reshape(activations.shape[0], -1)

    # Apply PCA if requested
    if method == 'pca':
        original_shape = activations_flat.shape
        n_samples, n_features = activations_flat.shape
        max_components = min(n_samples, n_features)

        # Only apply PCA if we have more features than target components AND enough samples
        if n_features > n_components and max_components >= n_components:
            pca = PCA(n_components=n_components)
            activations_flat = pca.fit_transform(activations_flat)
            variance_explained = pca.explained_variance_ratio_.sum()
            print(f"    Applied PCA: {original_shape} -> {activations_flat.shape} "
                  f"(explained variance: {variance_explained:.2%})")
        else:
            if max_components < n_components:
                print(f"    Skipped PCA: n_components={n_components} but max_components={max_components} "
                      f"(min(n_samples={n_samples}, n_features={n_features}))")
            else:
                print(f"    Skipped PCA: feature dim {n_features} <= {n_components}")

        return activations_flat

    # Apply random projection if requested
    if method == 'random_projection':
        original_shape = activations_flat.shape
        n_samples, n_features = activations_flat.shape

        # Only apply random projection if we have more features than target dimension
        if n_features > rp_dimension:
            if rng is None:
                rng = np.random.RandomState(42)

            print(f"    Applying random projection: {original_shape} -> ({n_samples}, {rp_dimension}) "
                  f"with {rp_repetitions} repetitions")

            # Return both activations and parameters as a tuple
            rp_params = {
                'rp_dimension': rp_dimension,
                'rp_repetitions': rp_repetitions,
                'rng': rng
            }
            return activations_flat, rp_params
        else:
            print(f"    Skipped random projection: feature dim {n_features} <= {rp_dimension}")
            return activations_flat, None

    # Default case for 'none' and 'pooling' methods
    return activations_flat

def compute_rsa_matrix(activations, processing_method='none', pca_components=100,
                       rp_dimension=100, rp_repetitions=10):
    """
    Compute RSA matrix from neural activations.

    Args:
        activations: Numpy array of activations
        processing_method: Method to process activations ('none', 'pooling', 'pca', 'random_projection')
        pca_components: Number of PCA components (only used when processing_method='pca')
        rp_dimension: Dimension for random projection (only used when processing_method='random_projection')
        rp_repetitions: Number of random projection repetitions to average (only used when processing_method='random_projection')

    Returns:
        RSA correlation matrix
    """
    # Process activations
    result = process_activations(
        activations, processing_method, pca_components,
        rp_dimension=rp_dimension, rp_repetitions=rp_repetitions
    )

    # Handle tuple return from process_activations
    if isinstance(result, tuple):
        activations_processed, rp_params = result
    else:
        activations_processed = result
        rp_params = None

    # If random projection, compute RSA matrix multiple times and average
    if processing_method == 'random_projection' and rp_params is not None:
        n_samples, n_features = activations_processed.shape
        rng = rp_params['rng']
        target_dim = rp_params['rp_dimension']
        n_repetitions = rp_params['rp_repetitions']

        # Accumulate RSA matrices across repetitions
        rsa_sum = None

        for rep in range(n_repetitions):
            # Generate random projection matrix: (n_features, target_dim)
            # Use Gaussian random projection (normalized)
            projection_matrix = rng.randn(n_features, target_dim) / np.sqrt(target_dim)

            # Project the activations
            projected = activations_processed @ projection_matrix

            # Compute RSA matrix for this repetition
            rsa_matrix = np.corrcoef(projected)

            # Accumulate
            if rsa_sum is None:
                rsa_sum = rsa_matrix
            else:
                rsa_sum += rsa_matrix

        # Average over repetitions
        return rsa_sum / n_repetitions

    else:
        # Standard processing (none, pooling, or pca)
        return np.corrcoef(activations_processed)

def compare_matrices(state_matrix, neural_matrix, method='pearson'):
    """Compare state similarity matrix with neural RSA matrix.

    Note: Excludes identical state pairs (marked as -1) from correlation computation.
    """
    triu_indices = np.triu_indices(state_matrix.shape[0], k=1)
    state_flat = state_matrix[triu_indices]
    neural_flat = neural_matrix[triu_indices]

    # Exclude identical pairs (marked as -1)
    valid_mask = state_flat != -1
    state_flat_filtered = state_flat[valid_mask]
    neural_flat_filtered = neural_flat[valid_mask]

    corr_funcs = {'pearson': pearsonr, 'spearman': spearmanr}
    if method not in corr_funcs:
        raise ValueError("Method must be 'pearson' or 'spearman'")

    return corr_funcs[method](state_flat_filtered, neural_flat_filtered)

def analyze_similarity_groups(state_matrix, neural_matrix):
    """Compare neural similarity for state-similar vs dissimilar states.

    Note: Excludes identical states (marked as -1 in state_matrix) from both groups.
    """
    assert state_matrix.shape == neural_matrix.shape, \
        f"Matrix shapes don't match: {state_matrix.shape} vs {neural_matrix.shape}"

    # Extract neural similarities, excluding identical states (-1)
    similar_neural = neural_matrix[state_matrix == 1]
    dissimilar_neural = neural_matrix[state_matrix == 0]
    excluded_neural = neural_matrix[state_matrix == -1]

    results = {
        'similar_mean': np.mean(similar_neural) if len(similar_neural) else np.nan,
        'similar_std': np.std(similar_neural) if len(similar_neural) else np.nan,
        'dissimilar_mean': np.mean(dissimilar_neural) if len(dissimilar_neural) else np.nan,
        'dissimilar_std': np.std(dissimilar_neural) if len(dissimilar_neural) else np.nan,
        'similar_count': len(similar_neural),
        'dissimilar_count': len(dissimilar_neural),
        'excluded_count': len(excluded_neural)
    }

    if len(similar_neural) and len(dissimilar_neural):
        t_stat, p_val = ttest_ind(similar_neural, dissimilar_neural)
        results.update({'t_stat': t_stat, 'p_val': p_val})

    return results


def analyze_state_symmetry_with_constructed_pairs(
    symmetric_pairs: List[Tuple[int, int]],
    logical_states: List[Dict[str, Any]],
    rsa_matrix: np.ndarray,
    analyzer,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Analyze state symmetry by comparing constructed symmetric pairs vs random non-symmetric pairs.

    This is the new simplified state symmetry analysis:
    1. For constructed symmetric pairs: extract their neural similarity from RSA matrix
    2. Sample equal number of random non-symmetric pairs (where naive_symmetry == 0)
    3. Compare the two distributions

    Args:
        symmetric_pairs: List of (original_idx, synthetic_idx) tuples from augmentation
        logical_states: List of all logical states (original + synthetic)
        rsa_matrix: Neural RSA matrix (correlation-based similarity)
        analyzer: PongSymmetryAnalyzer instance for naive_symmetry checks
        seed: Random seed for reproducibility

    Returns:
        Dictionary with:
        - symmetric_mean: Mean neural similarity for symmetric pairs
        - symmetric_std: Std of neural similarity for symmetric pairs
        - random_mean: Mean neural similarity for random non-symmetric pairs
        - random_std: Std of neural similarity for random non-symmetric pairs
        - t_stat: T-statistic from independent t-test
        - p_val: P-value from independent t-test
        - n_symmetric_pairs: Number of symmetric pairs
        - n_random_pairs: Number of random pairs sampled
    """
    if not symmetric_pairs:
        return {
            'symmetric_mean': np.nan,
            'symmetric_std': np.nan,
            'random_mean': np.nan,
            'random_std': np.nan,
            't_stat': np.nan,
            'p_val': np.nan,
            'n_symmetric_pairs': 0,
            'n_random_pairs': 0
        }

    # 1. Extract neural similarities for constructed symmetric pairs
    symmetric_similarities = []
    for orig_idx, synth_idx in symmetric_pairs:
        if orig_idx < rsa_matrix.shape[0] and synth_idx < rsa_matrix.shape[0]:
            similarity = rsa_matrix[orig_idx, synth_idx]
            symmetric_similarities.append(similarity)

    n_symmetric = len(symmetric_similarities)

    if n_symmetric == 0:
        return {
            'symmetric_mean': np.nan,
            'symmetric_std': np.nan,
            'random_mean': np.nan,
            'random_std': np.nan,
            't_stat': np.nan,
            'p_val': np.nan,
            'n_symmetric_pairs': 0,
            'n_random_pairs': 0
        }

    # 2. Find all non-symmetric pairs from original states only
    # Original states are indices 0 to n_original-1
    n_original = len(logical_states) - len(symmetric_pairs)  # Original states before augmentation

    # Collect all pairs where naive_symmetry returns 0 (not symmetric)
    non_symmetric_pairs = []
    for i in range(n_original):
        for j in range(i + 1, n_original):
            symm_result = analyzer.naive_symmetry(logical_states[i], logical_states[j])
            if symm_result == 0:  # Explicitly non-symmetric
                non_symmetric_pairs.append((i, j))

    # 3. Sample same number of random non-symmetric pairs
    rng = np.random.RandomState(seed)
    n_to_sample = min(n_symmetric, len(non_symmetric_pairs))

    if n_to_sample == 0:
        return {
            'symmetric_mean': np.mean(symmetric_similarities),
            'symmetric_std': np.std(symmetric_similarities),
            'random_mean': np.nan,
            'random_std': np.nan,
            't_stat': np.nan,
            'p_val': np.nan,
            'n_symmetric_pairs': n_symmetric,
            'n_random_pairs': 0
        }

    sampled_indices = rng.choice(len(non_symmetric_pairs), size=n_to_sample, replace=False)
    sampled_random_pairs = [non_symmetric_pairs[idx] for idx in sampled_indices]

    # Extract neural similarities for random non-symmetric pairs
    random_similarities = []
    for i, j in sampled_random_pairs:
        similarity = rsa_matrix[i, j]
        random_similarities.append(similarity)

    # 4. Compute statistics
    symmetric_mean = np.mean(symmetric_similarities)
    symmetric_std = np.std(symmetric_similarities)
    random_mean = np.mean(random_similarities)
    random_std = np.std(random_similarities)

    # T-test comparing the two groups
    t_stat, p_val = ttest_ind(symmetric_similarities, random_similarities)

    return {
        'symmetric_mean': symmetric_mean,
        'symmetric_std': symmetric_std,
        'random_mean': random_mean,
        'random_std': random_std,
        't_stat': t_stat,
        'p_val': p_val,
        'n_symmetric_pairs': n_symmetric,
        'n_random_pairs': len(random_similarities)
    }


# Plot style settings (matching plot_inverted_pendulum.py)
COLOR_SIMILAR = '#EE7733'  # Orange
COLOR_DISSIMILAR = '#009988'  # Teal


def plot_rsa_by_value_threshold(evaluator_rsa, model_name, threshold=None, save_path=None, ax=None, symmetry_type='state',
                                 show_ylabel=True, show_xlabel=True, show_title=True, filter_post_activation=False):
    """
    Plot RSA similarity split by state similarity and value difference threshold.

    Args:
        evaluator_rsa: ModelEvaluatorWithRSA instance with results
        model_name: Name of the model to plot ('ppo', 'dqn', 'qrdqn', etc.)
        threshold: Value difference threshold. If None, uses median difference
        save_path: Optional path to save the plot
        ax: Matplotlib axis to plot on. If None, creates new figure
        symmetry_type: Type of symmetry ('state' or 'policy')
        show_ylabel: Whether to show y-axis label
        show_xlabel: Whether to show x-axis label
        show_title: Whether to show title
        filter_post_activation: If True, only include post-activation (ReLU) layers and output heads
    """
    if model_name not in evaluator_rsa.rsa_results:
        print(f"No results found for model {model_name}")
        return

    rsa_data = evaluator_rsa.rsa_results[model_name]
    
    if 'rl_context' not in rsa_data or 'state_matrix' not in rsa_data:
        print(f"No RL context data for model {model_name}")
        return
    
    rl_contexts = rsa_data['rl_context']
    state_matrix = rsa_data['state_matrix']
    n_states = state_matrix.shape[0]  # Use matrix size, not rl_context size (may differ with augmentation)

    # Extract values based on model type (only for original states that have rl_context)
    values = []
    for i in range(n_states):
        if i < len(rl_contexts):
            ctx = rl_contexts[i]
            if 'predicted_value' in ctx:  # PPO/A2C
                val = ctx['predicted_value']
                values.append(val if isinstance(val, (int, float)) else val.item() if hasattr(val, 'item') else float(val))
            elif 'max_q_value' in ctx:  # DQN/QRDQN
                val = ctx['max_q_value']
                values.append(val if isinstance(val, (int, float)) else val.item() if hasattr(val, 'item') else float(val))
            else:
                values.append(np.nan)
        else:
            # Synthetic states don't have rl_context
            values.append(np.nan)

    # Check if we have value predictions
    has_values = values and not all(np.isnan(values))

    if not has_values:
        print(f"No value predictions available for {model_name}, creating simplified plot")
    
    # Calculate all pairwise value differences (only if values available)
    value_diffs = []
    similar_pairs = []
    dissimilar_pairs = []

    if has_values:
        for i in range(n_states):
            for j in range(i+1, n_states):
                # Skip identical pairs (marked as -1)
                if state_matrix[i, j] == -1:
                    continue

                if not (np.isnan(values[i]) or np.isnan(values[j])):
                    val_diff = abs(values[i] - values[j])
                    value_diffs.append(val_diff)

                    if state_matrix[i, j] == 1:
                        similar_pairs.append((i, j, val_diff))
                    else:  # state_matrix[i, j] == 0
                        dissimilar_pairs.append((i, j, val_diff))

        # Set threshold
        if threshold is None:
            threshold = np.median(value_diffs)
            print(f"Using median value difference as threshold: {threshold:.3f}")
        else:
            print(f"Using provided threshold: {threshold:.3f}")
    else:
        # No value-based splitting, just use state similarity
        threshold = 0.0
        for i in range(n_states):
            for j in range(i+1, n_states):
                # Skip identical pairs (marked as -1)
                if state_matrix[i, j] == -1:
                    continue
                elif state_matrix[i, j] == 1:
                    similar_pairs.append((i, j, 0.0))
                else:  # state_matrix[i, j] == 0
                    dissimilar_pairs.append((i, j, 0.0))
    
    # Get layer names and organize results
    layer_data = {}
    for key in rsa_data.keys():
        # Skip metadata entries
        if key in ['state_matrix', 'logical_states', 'pixel_states', 'total_states_collected',
                   'sampled_indices', 'rl_context', 'augmented_symmetric_pairs']:
            continue

        # For PPO models, skip shared_features and value_features layers
        if model_name.lower() == 'ppo':
            if 'shared_features' in key or 'value_features' in key:
                print(f"Skipping {key} for PPO model")
                continue
        
        if 'symmetry_analysis' not in rsa_data[key] and 'group_analysis' not in rsa_data[key]:
            continue

        layer_data[key] = rsa_data[key]

    # Filter to post-activation layers only if requested
    if filter_post_activation:
        filtered_layer_data = {}
        for key, value in layer_data.items():
            # Keep ReLU layers (post-activation) and output heads
            if 'ReLU' in key or 'action_net' in key or ('value_net' in key and 'features' not in key):
                filtered_layer_data[key] = value
        layer_data = filtered_layer_data

    if not layer_data:
        print(f"No layer data found for {model_name}")
        return

    # Calculate similarity means for each category
    if has_values:
        categories = {
            'similar_high_val_diff': [],
            'similar_low_val_diff': [],
            'dissimilar_high_val_diff': [],
            'dissimilar_low_val_diff': []
        }
    else:
        # Simplified categories when no value predictions available
        categories = {
            'similar': [],
            'dissimilar': []
        }

    category_stds = {key: [] for key in categories.keys()}
    layer_names = []

    for layer_name, layer_results in layer_data.items():
        if 'rsa_matrix' not in layer_results:
            continue

        rsa_matrix = layer_results['rsa_matrix']
        layer_names.append(layer_name)

        if has_values:
            # Calculate similarities for each category with value splits
            sim_high_vals = []
            sim_low_vals = []
            dissim_high_vals = []
            dissim_low_vals = []

            # Similar pairs split by value difference
            for i, j, val_diff in similar_pairs:
                if i < rsa_matrix.shape[0] and j < rsa_matrix.shape[1]:
                    similarity = rsa_matrix[i, j]
                    if val_diff >= threshold:
                        sim_high_vals.append(similarity)
                    else:
                        sim_low_vals.append(similarity)

            # Dissimilar pairs split by value difference
            for i, j, val_diff in dissimilar_pairs:
                if i < rsa_matrix.shape[0] and j < rsa_matrix.shape[1]:
                    similarity = rsa_matrix[i, j]
                    if val_diff >= threshold:
                        dissim_high_vals.append(similarity)
                    else:
                        dissim_low_vals.append(similarity)

            # Store means and stds
            categories['similar_high_val_diff'].append(np.mean(sim_high_vals) if sim_high_vals else np.nan)
            categories['similar_low_val_diff'].append(np.mean(sim_low_vals) if sim_low_vals else np.nan)
            categories['dissimilar_high_val_diff'].append(np.mean(dissim_high_vals) if dissim_high_vals else np.nan)
            categories['dissimilar_low_val_diff'].append(np.mean(dissim_low_vals) if dissim_low_vals else np.nan)

            category_stds['similar_high_val_diff'].append(np.std(sim_high_vals) if sim_high_vals else np.nan)
            category_stds['similar_low_val_diff'].append(np.std(sim_low_vals) if sim_low_vals else np.nan)
            category_stds['dissimilar_high_val_diff'].append(np.std(dissim_high_vals) if dissim_high_vals else np.nan)
            category_stds['dissimilar_low_val_diff'].append(np.std(dissim_low_vals) if dissim_low_vals else np.nan)
        else:
            # Simplified calculation without value splits
            sim_vals = []
            dissim_vals = []

            # Just split by state similarity
            for i, j, _ in similar_pairs:
                if i < rsa_matrix.shape[0] and j < rsa_matrix.shape[1]:
                    sim_vals.append(rsa_matrix[i, j])

            for i, j, _ in dissimilar_pairs:
                if i < rsa_matrix.shape[0] and j < rsa_matrix.shape[1]:
                    dissim_vals.append(rsa_matrix[i, j])

            # Store means and stds
            categories['similar'].append(np.mean(sim_vals) if sim_vals else np.nan)
            categories['dissimilar'].append(np.mean(dissim_vals) if dissim_vals else np.nan)

            category_stds['similar'].append(np.std(sim_vals) if sim_vals else np.nan)
            category_stds['dissimilar'].append(np.std(dissim_vals) if dissim_vals else np.nan)
    
    # Clean up layer names for display with simplified labels (L1, L2, ..., Act, Val)
    display_names = []
    layer_count = 0
    for name in layer_names:
        if "action_net" in name:
            display_names.append("Act")
        elif "value_net" in name and "features" not in name:
            display_names.append("Val")
        else:
            layer_count += 1
            display_names.append(f"L{layer_count}")

    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(2, 1.5))
        standalone = True
    else:
        standalone = False

    x_pos = np.arange(len(layer_names))

    # Simplified colors/styles matching plot_inverted_pendulum.py style
    # Only use similar/dissimilar categories (ignore value splits for cleaner plots)
    if has_values:
        # Aggregate similar and dissimilar categories (ignore value splits)
        similar_means = [(categories['similar_high_val_diff'][i] + categories['similar_low_val_diff'][i]) / 2
                        if not (np.isnan(categories['similar_high_val_diff'][i]) or np.isnan(categories['similar_low_val_diff'][i]))
                        else categories['similar_high_val_diff'][i] if not np.isnan(categories['similar_high_val_diff'][i])
                        else categories['similar_low_val_diff'][i]
                        for i in range(len(layer_names))]
        dissimilar_means = [(categories['dissimilar_high_val_diff'][i] + categories['dissimilar_low_val_diff'][i]) / 2
                           if not (np.isnan(categories['dissimilar_high_val_diff'][i]) or np.isnan(categories['dissimilar_low_val_diff'][i]))
                           else categories['dissimilar_high_val_diff'][i] if not np.isnan(categories['dissimilar_high_val_diff'][i])
                           else categories['dissimilar_low_val_diff'][i]
                           for i in range(len(layer_names))]
        similar_stds = [(category_stds['similar_high_val_diff'][i] + category_stds['similar_low_val_diff'][i]) / 2
                       if not (np.isnan(category_stds['similar_high_val_diff'][i]) or np.isnan(category_stds['similar_low_val_diff'][i]))
                       else category_stds['similar_high_val_diff'][i] if not np.isnan(category_stds['similar_high_val_diff'][i])
                       else category_stds['similar_low_val_diff'][i]
                       for i in range(len(layer_names))]
        dissimilar_stds = [(category_stds['dissimilar_high_val_diff'][i] + category_stds['dissimilar_low_val_diff'][i]) / 2
                          if not (np.isnan(category_stds['dissimilar_high_val_diff'][i]) or np.isnan(category_stds['dissimilar_low_val_diff'][i]))
                          else category_stds['dissimilar_high_val_diff'][i] if not np.isnan(category_stds['dissimilar_high_val_diff'][i])
                          else category_stds['dissimilar_low_val_diff'][i]
                          for i in range(len(layer_names))]
    else:
        similar_means = categories['similar']
        dissimilar_means = categories['dissimilar']
        similar_stds = category_stds['similar']
        dissimilar_stds = category_stds['dissimilar']

    # Plot similar states line
    ax.plot(x_pos, similar_means, '-', color=COLOR_SIMILAR, linewidth=0.8)
    # Add error shading for similar
    ax.fill_between(x_pos,
                   [m - s if not np.isnan(m) and not np.isnan(s) else m for m, s in zip(similar_means, similar_stds)],
                   [m + s if not np.isnan(m) and not np.isnan(s) else m for m, s in zip(similar_means, similar_stds)],
                   color=COLOR_SIMILAR, alpha=0.2)

    # Plot dissimilar states line
    ax.plot(x_pos, dissimilar_means, '-', color=COLOR_DISSIMILAR, linewidth=0.8)
    # Add error shading for dissimilar
    ax.fill_between(x_pos,
                   [m - s if not np.isnan(m) and not np.isnan(s) else m for m, s in zip(dissimilar_means, dissimilar_stds)],
                   [m + s if not np.isnan(m) and not np.isnan(s) else m for m, s in zip(dissimilar_means, dissimilar_stds)],
                   color=COLOR_DISSIMILAR, alpha=0.2)

    if show_xlabel:
        # Use simplified layer labels with rotation, no xlabel title
        ax.set_xticklabels(display_names, fontsize=8, rotation=45, ha='right')
    else:
        # Hide x tick labels for top row
        ax.set_xticklabels([])
    if show_ylabel:
        ax.set_ylabel("Cosine Sim.", fontsize=8)
    ax.set_xticks(x_pos)
    ax.tick_params(axis='both', labelsize=8)

    # Auto-adjust y-axis limits with some padding
    all_vals = [v for v in similar_means + dissimilar_means if not np.isnan(v)]
    all_stds = [s for s in similar_stds + dissimilar_stds if not np.isnan(s)]
    if all_vals:
        y_min = min(all_vals) - max(all_stds) if all_stds else min(all_vals)
        y_max = max(all_vals) + max(all_stds) if all_stds else max(all_vals)
        padding = (y_max - y_min) * 0.05
        ax.set_ylim(-0.2, 1.0 + padding)

    if show_title:
        ax.set_title(f'{model_name.upper()}', fontsize=8)

    # Only save/show if this is a standalone plot
    if standalone:
        sns.despine()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            plt.close()
        else:
            plt.show()

    # Print summary statistics
    print(f"\nSummary for {model_name.upper()} ({symmetry_type} symmetry):")
    similar_vals = [v for v in similar_means if not np.isnan(v)]
    dissimilar_vals = [v for v in dissimilar_means if not np.isnan(v)]
    if similar_vals:
        print(f"Similar States: mean={np.mean(similar_vals):.4f} ± {np.std(similar_vals):.4f}")
    if dissimilar_vals:
        print(f"Dissimilar States: mean={np.mean(dissimilar_vals):.4f} ± {np.std(dissimilar_vals):.4f}")


def plot_rsa_comparison(evaluator_state, evaluator_policy, model_name, save_path=None):
    """
    Plot RSA analysis for both state and policy symmetry side-by-side.

    Args:
        evaluator_state: ModelEvaluatorWithRSA instance with state symmetry results
        evaluator_policy: ModelEvaluatorWithRSA instance with policy symmetry results
        model_name: Name of the model to plot
        save_path: Path to save the combined plot
    """
    # Layout: 2 rows (policy symmetry, state symmetry) x 1 column
    fig, axes = plt.subplots(2, 1, figsize=(1.5, 3), sharex=True, sharey=True)

    # Row 0: Policy Symmetry (top, with title)
    plot_rsa_by_value_threshold(evaluator_policy, model_name, ax=axes[0], symmetry_type='policy',
                                 show_ylabel=True, show_xlabel=False, show_title=True)

    # Row 1: State Symmetry (bottom, with xlabel)
    plot_rsa_by_value_threshold(evaluator_state, model_name, ax=axes[1], symmetry_type='state',
                                 show_ylabel=True, show_xlabel=True, show_title=False)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot saved to {save_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_aggregate_comparison(evaluator_state_dict, evaluator_policy_dict, model_names, save_path=None):
    """
    Plot aggregate comparison of multiple algorithms (e.g., DQN + PPO) in the same layout
    and figure size as plot_inverted_pendulum.py.

    Layout: 2 rows (policy symmetry, state symmetry) x n_algorithms columns.
    Titles (algorithm names) appear only on the top row.

    Saves both full and simplified (post-activation only) versions as PDFs.

    Args:
        evaluator_state_dict: Dict mapping model_name -> ModelEvaluatorWithRSA instance with state symmetry results
        evaluator_policy_dict: Dict mapping model_name -> ModelEvaluatorWithRSA instance with policy symmetry results
        model_names: List of model names to plot (e.g., ['dqn', 'ppo'])
        save_path: Path to save the combined plot (base path, _combined.pdf and _combined_simplified.pdf will be added)
    """
    n_algorithms = len(model_names)

    print(f"\nPlotting aggregate comparison of {n_algorithms} algorithms: {', '.join(model_names)}...")

    def create_combined_plot(filter_post_activation=False):
        """Helper function to create combined plot (full or simplified)."""
        # Use 3.4 x 3 inches like inverted pendulum
        fig, axes = plt.subplots(2, n_algorithms, figsize=(3.4, 3),
                                  sharex=False, sharey=True)

        # Handle single algorithm case
        if n_algorithms == 1:
            axes = axes.reshape(2, 1)

        for col, model_name in enumerate(model_names):
            is_left_col = (col == 0)

            evaluator_state = evaluator_state_dict.get(model_name)
            evaluator_policy = evaluator_policy_dict.get(model_name)

            if evaluator_state is None or evaluator_policy is None:
                print(f"Warning: Missing evaluator for {model_name}, skipping...")
                continue

            # Row 0: Policy Symmetry (top row, with title, no x labels)
            plot_rsa_by_value_threshold(evaluator_policy, model_name, ax=axes[0, col], symmetry_type='policy',
                                         show_ylabel=is_left_col, show_xlabel=False, show_title=True,
                                         filter_post_activation=filter_post_activation)

            # Row 1: State Symmetry (bottom row, with xlabel)
            plot_rsa_by_value_threshold(evaluator_state, model_name, ax=axes[1, col], symmetry_type='state',
                                         show_ylabel=is_left_col, show_xlabel=True, show_title=False,
                                         filter_post_activation=filter_post_activation)

        sns.despine()
        plt.tight_layout()
        return fig

    # Save full combined plot as PDF
    fig_full = create_combined_plot(filter_post_activation=False)
    if save_path:
        base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
        full_path = f"{base_path}_combined.pdf"
        fig_full.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot (full) saved to: {full_path}")
        plt.close(fig_full)

        # Save simplified combined plot as PDF
        fig_simp = create_combined_plot(filter_post_activation=True)
        simp_path = f"{base_path}_combined_simplified.pdf"
        fig_simp.savefig(simp_path, dpi=300, bbox_inches='tight')
        print(f"Combined plot (simplified) saved to: {simp_path}")
        plt.close(fig_simp)
    else:
        plt.show()
        plt.close(fig_full)


# ============================================================================
# State Visualization Functions
# ============================================================================

def render_symmetric_state_pairs(evaluator_state, evaluator_policy, model_name: str, output_dir: Path,
                                  num_pairs: int = 10, suffix: str = ''):
    """
    Render and save pairs of symmetric states for visualization.

    Creates side-by-side visualizations of state pairs that are:
    1. Symmetric under policy (same action taken)
    2. Symmetric under MDP homomorphism (state symmetry)

    Args:
        evaluator_state: ModelEvaluatorWithRSA instance with state symmetry results
        evaluator_policy: ModelEvaluatorWithRSA instance with policy symmetry results
        model_name: Name of the model
        output_dir: Output directory path
        num_pairs: Number of pairs to render for each symmetry type
        suffix: Optional suffix for subdirectory name
    """
    # Create subdirectory for state visualizations
    viz_dir = output_dir / 'state_visualizations' / f'{model_name}{suffix}'
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nRendering symmetric state pairs for {model_name.upper()}...")
    print(f"Output directory: {viz_dir}")

    # Helper function to render a single frame from stacked observations
    def render_frame(pixel_state, frame_idx=-1):
        """
        Render a single frame from stacked pixel observations.
        pixel_state is shape (4, 84, 84) - 4 stacked grayscale frames.
        frame_idx=-1 means the most recent frame.
        """
        frame = pixel_state[frame_idx]  # Shape (84, 84)
        return frame

    # Helper function to create side-by-side comparison
    def save_pair_visualization(pixel_states, idx1, idx2, pair_num, symmetry_type, viz_dir):
        """Save a side-by-side visualization of a state pair."""
        fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))

        # Render the most recent frame from each stacked observation
        frame1 = render_frame(pixel_states[idx1])
        frame2 = render_frame(pixel_states[idx2])

        axes[0].imshow(frame1, cmap='gray')
        axes[0].axis('off')

        axes[1].imshow(frame2, cmap='gray')
        axes[1].axis('off')

        plt.tight_layout()

        save_path = viz_dir / f'{symmetry_type}_pair_{pair_num + 1:03d}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return save_path

    # Process state symmetry pairs
    if model_name in evaluator_state.rsa_results:
        state_rsa = evaluator_state.rsa_results[model_name]
        if 'state_matrix' in state_rsa and 'pixel_states' in state_rsa:
            state_matrix = state_rsa['state_matrix']
            pixel_states = state_rsa['pixel_states']

            # Check for augmented symmetric pairs first (these are guaranteed true pairs)
            augmented_pairs = state_rsa.get('augmented_symmetric_pairs', [])

            # Find symmetric pairs (state_matrix == 1)
            n_states = state_matrix.shape[0]
            similar_pairs = []
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    if state_matrix[i, j] == 1:
                        similar_pairs.append((i, j))

            # Prioritize augmented pairs if available
            if augmented_pairs:
                print(f"  State symmetry: Found {len(augmented_pairs)} augmented pairs, {len(similar_pairs)} total symmetric pairs in matrix")
                # Use augmented pairs first, then add other pairs if needed
                pairs_to_render = list(augmented_pairs[:num_pairs])
                if len(pairs_to_render) < num_pairs:
                    # Add non-augmented pairs
                    other_pairs = [p for p in similar_pairs if p not in augmented_pairs and (p[1], p[0]) not in augmented_pairs]
                    np.random.shuffle(other_pairs)
                    pairs_to_render.extend(other_pairs[:num_pairs - len(pairs_to_render)])
            elif similar_pairs:
                np.random.shuffle(similar_pairs)
                pairs_to_render = similar_pairs[:num_pairs]
            else:
                pairs_to_render = []

            if pairs_to_render:
                print(f"  State symmetry: Rendering {len(pairs_to_render)} pairs")

                for pair_num, (idx1, idx2) in enumerate(pairs_to_render):
                    save_pair_visualization(pixel_states, idx1, idx2, pair_num, 'state_symmetry', viz_dir)
            else:
                print(f"  State symmetry: No symmetric pairs found")
        else:
            print(f"  State symmetry: Missing state_matrix or pixel_states in results")
    else:
        print(f"  State symmetry: No results for {model_name}")

    # Process policy symmetry pairs
    if model_name in evaluator_policy.rsa_results:
        policy_rsa = evaluator_policy.rsa_results[model_name]
        if 'state_matrix' in policy_rsa and 'pixel_states' in policy_rsa:
            state_matrix = policy_rsa['state_matrix']
            pixel_states = policy_rsa['pixel_states']

            # Check for augmented symmetric pairs first
            augmented_pairs = policy_rsa.get('augmented_symmetric_pairs', [])

            # Find symmetric pairs (state_matrix == 1)
            n_states = state_matrix.shape[0]
            similar_pairs = []
            for i in range(n_states):
                for j in range(i + 1, n_states):
                    if state_matrix[i, j] == 1:
                        similar_pairs.append((i, j))

            # Prioritize augmented pairs if available
            if augmented_pairs:
                print(f"  Policy symmetry: Found {len(augmented_pairs)} augmented pairs, {len(similar_pairs)} total symmetric pairs in matrix")
                pairs_to_render = list(augmented_pairs[:num_pairs])
                if len(pairs_to_render) < num_pairs:
                    other_pairs = [p for p in similar_pairs if p not in augmented_pairs and (p[1], p[0]) not in augmented_pairs]
                    np.random.shuffle(other_pairs)
                    pairs_to_render.extend(other_pairs[:num_pairs - len(pairs_to_render)])
            elif similar_pairs:
                np.random.shuffle(similar_pairs)
                pairs_to_render = similar_pairs[:num_pairs]
            else:
                pairs_to_render = []

            if pairs_to_render:
                print(f"  Policy symmetry: Rendering {len(pairs_to_render)} pairs")

                for pair_num, (idx1, idx2) in enumerate(pairs_to_render):
                    save_pair_visualization(pixel_states, idx1, idx2, pair_num, 'policy_symmetry', viz_dir)
            else:
                print(f"  Policy symmetry: No symmetric pairs found")
        else:
            print(f"  Policy symmetry: Missing state_matrix or pixel_states in results")
    else:
        print(f"  Policy symmetry: No results for {model_name}")

    print(f"  Visualizations saved to: {viz_dir}")


# ============================================================================
# Results Save/Load Functions
# ============================================================================

def save_rsa_results(evaluator_state, evaluator_policy, model_name: str, output_dir: Path, suffix: str = ''):
    """
    Save RSA results to pickle file for fast re-plotting later.

    Args:
        evaluator_state: ModelEvaluatorWithRSA instance with state symmetry results
        evaluator_policy: ModelEvaluatorWithRSA instance with policy symmetry results
        model_name: Name of the model
        output_dir: Output directory path
        suffix: Optional suffix for filename (e.g., '_pooled', '_pca')
    """
    results_to_save = {
        'model_name': model_name,
        'state_rsa_results': evaluator_state.rsa_results.get(model_name, {}),
        'policy_rsa_results': evaluator_policy.rsa_results.get(model_name, {}),
    }

    pickle_path = output_dir / 'data' / f'{model_name}_rsa_results{suffix}.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"RSA results saved to: {pickle_path}")


def load_rsa_results(model_name: str, output_dir: Path, suffix: str = '') -> Optional[Dict]:
    """
    Load RSA results from pickle file.

    Args:
        model_name: Name of the model
        output_dir: Output directory path
        suffix: Optional suffix for filename (e.g., '_pooled', '_pca')

    Returns:
        Dict with 'state_rsa_results' and 'policy_rsa_results', or None if file not found
    """
    pickle_path = output_dir / 'data' / f'{model_name}_rsa_results{suffix}.pkl'
    if not pickle_path.exists():
        print(f"No saved results found at: {pickle_path}")
        return None

    with open(pickle_path, 'rb') as f:
        results = pickle.load(f)
    print(f"Loaded RSA results from: {pickle_path}")
    return results


class MockEvaluator:
    """Mock evaluator class that holds pre-loaded RSA results for plotting."""
    def __init__(self, rsa_results: Dict):
        self.rsa_results = rsa_results


# ============================================================================
# Main Analysis Pipeline
# ============================================================================

def print_section(title: str):
    """Print a section header."""
    print(f"\n{'='*80}\n{title}\n{'='*80}\n")

def setup_output_directory(output_dir: str) -> Path:
    """Create output directory structure for results and plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / 'plots').mkdir(exist_ok=True)
    (output_path / 'data').mkdir(exist_ok=True)
    return output_path


def load_single_model(model_name: str, env_name: str = "PongNoFrameskip-v4"):
    """Load a single trained model from local storage."""
    model_path = f"{model_name}-{env_name}"
    model_classes = {'ppo': PPO, 'dqn': DQN, 'a2c': A2C, 'qrdqn': QRDQN}

    try:
        if model_name not in model_classes:
            raise ValueError(f"Unknown model: {model_name}")
        model = model_classes[model_name].load(model_path)
        print(f"Loaded {model_name.upper()} model from {model_path}.zip")
        return model
    except Exception as e:
        print(f"Error loading {model_name} model: {e}")
        return None


def analyze_single_model(
    model_name: str,
    model,
    evaluator_state: 'ModelEvaluatorWithRSA',
    evaluator_policy: 'ModelEvaluatorWithRSA',
    num_states: int,
    output_dir: Path,
    generate_plots: bool = True,
    activation_processing: str = 'none'
) -> Dict[str, Any]:
    """Analyze a single model for both state and policy symmetries."""
    print_section(f"Analyzing {model_name.upper()} model")

    try:
        if model is None:
            print(f"Model {model_name} not available, skipping...")
            return {}

        results = {}

        # Run state symmetry analysis
        print(f"\n--- State Symmetry Analysis ---")
        results_state = evaluator_state.evaluate_model_with_rsa(
            model=model, model_name=model_name, max_states_for_rsa=num_states
        )
        results['state'] = results_state

        if results_state:
            print(f"\nState symmetry analysis complete for {model_name}")

        # Run policy symmetry analysis
        print(f"\n--- Policy Symmetry Analysis ---")
        results_policy = evaluator_policy.evaluate_model_with_rsa(
            model=model, model_name=model_name, max_states_for_rsa=num_states
        )
        results['policy'] = results_policy

        if results_policy:
            print(f"\nPolicy symmetry analysis complete for {model_name}")

        # Generate combined plot if requested
        if generate_plots and results_state and results_policy:
            try:
                # Add processing suffix to filename
                suffix = ''
                if activation_processing == 'pooling':
                    suffix = '_pooled'
                elif activation_processing == 'pca':
                    suffix = '_pca'
                elif activation_processing == 'random_projection':
                    suffix = '_rp'
                plot_path = output_dir / 'plots' / f'{model_name}_rsa_comparison{suffix}.png'
                plot_rsa_comparison(evaluator_state, evaluator_policy, model_name, save_path=str(plot_path))
            except Exception as e:
                print(f"Warning: Could not generate plot: {e}")
                traceback.print_exc()

        return results

    except Exception as e:
        print(f"Error analyzing {model_name}: {e}")
        traceback.print_exc()
        return {}


def main():
    """Main analysis pipeline."""
    parser = argparse.ArgumentParser(
        description='Analyze state and representational symmetries in Pong agents',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--model', default='all', choices=['all', 'ppo', 'dqn', 'a2c', 'qrdqn'],
                        help='Model to analyze (default: all)')
    parser.add_argument('--output-dir', default='../../results',
                        help='Output directory for results and plots (default: ../../results/)')
    parser.add_argument('--download-models', action='store_true',
                        help='Download models from HuggingFace')
    parser.add_argument('--num-states', type=int, default=500,
                        help='Number of states to sample for analysis (default: 500)')
    parser.add_argument('--no-plots', action='store_true',
                        help='Skip generating plots')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--activation-processing', default='none',
                        choices=['none', 'pooling', 'pca', 'random_projection'],
                        help='Method to process activations before RSA: none (default), pooling (spatial pooling), pca (dimensionality reduction), or random_projection (fast dimensionality reduction)')
    parser.add_argument('--pca-components', type=int, default=100,
                        help='Number of PCA components when using --activation-processing pca (default: 100)')
    parser.add_argument('--rp-dimension', type=int, default=100,
                        help='Dimension for random projection when using --activation-processing random_projection (default: 100)')
    parser.add_argument('--rp-repetitions', type=int, default=10,
                        help='Number of random projection repetitions to average when using --activation-processing random_projection (default: 10)')
    parser.add_argument('--symmetric-pair-ratio', type=float, default=None,
                        help='Target ratio of symmetric pairs in sampled states (0.0-1.0). If None, uses natural sampling (default: None)')
    parser.add_argument('--augment-symmetric', action='store_true',
                        help='Augment dataset with synthetically constructed symmetric pairs (flipped frames)')
    parser.add_argument('--symmetric-top-rows', type=int, default=14,
                        help='Number of rows to preserve at top when flipping frames for symmetric augmentation (default: 14)')
    parser.add_argument('--symmetric-bottom-rows', type=int, default=7,
                        help='Number of rows to preserve at bottom when flipping frames for symmetric augmentation (default: 7)')
    parser.add_argument('--symmetric-midline-threshold', type=float, default=5.0,
                        help='Minimum distance from midline for states to be augmented (default: 5.0)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip analysis and only regenerate plots from saved pickle results')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Setup output directory
    output_dir = setup_output_directory(args.output_dir)
    print(f"Output directory: {output_dir.absolute()}")

    # Determine suffix based on activation processing
    suffix = ''
    if args.activation_processing == 'pooling':
        suffix = '_pooled'
    elif args.activation_processing == 'pca':
        suffix = '_pca'
    elif args.activation_processing == 'random_projection':
        suffix = '_rp'

    # Determine which models to analyze
    models_to_analyze = ['ppo', 'dqn', 'a2c', 'qrdqn'] if args.model == 'all' else [args.model]

    # =========================================================================
    # Plot-only mode: Load saved results and regenerate plots
    # =========================================================================
    if args.plot_only:
        print("\n=== PLOT-ONLY MODE ===")
        print(f"Loading saved results from: {output_dir / 'data'}")
        print(f"Models to plot: {', '.join(models_to_analyze)}")

        evaluator_state_dict = {}
        evaluator_policy_dict = {}

        for model_name in models_to_analyze:
            loaded = load_rsa_results(model_name, output_dir, suffix)
            if loaded is None:
                print(f"Skipping {model_name} - no saved results found")
                continue

            # Create mock evaluators with loaded results
            state_evaluator = MockEvaluator({model_name: loaded['state_rsa_results']})
            policy_evaluator = MockEvaluator({model_name: loaded['policy_rsa_results']})

            evaluator_state_dict[model_name] = state_evaluator
            evaluator_policy_dict[model_name] = policy_evaluator

            # Generate individual comparison plot
            try:
                plot_path = output_dir / 'plots' / f'{model_name}_rsa_comparison{suffix}.png'
                plot_rsa_comparison(state_evaluator, policy_evaluator, model_name, save_path=str(plot_path))
            except Exception as e:
                print(f"Warning: Could not generate plot for {model_name}: {e}")
                traceback.print_exc()

            # Render symmetric state pair visualizations
            try:
                render_symmetric_state_pairs(state_evaluator, policy_evaluator, model_name, output_dir,
                                            num_pairs=10, suffix=suffix)
            except Exception as e:
                print(f"Warning: Could not render state pairs for {model_name}: {e}")
                traceback.print_exc()

        # Generate aggregate comparison plot for DQN + PPO
        aggregate_models = [m for m in ['dqn', 'ppo'] if m in evaluator_state_dict]
        if len(aggregate_models) >= 2:
            aggregate_path = output_dir / 'plots' / f'pong_dqn_ppo_comparison{suffix}.pdf'
            try:
                plot_aggregate_comparison(
                    evaluator_state_dict, evaluator_policy_dict,
                    aggregate_models, save_path=str(aggregate_path)
                )
            except Exception as e:
                print(f"Warning: Could not generate aggregate plot: {e}")
                traceback.print_exc()

        print(f"\nPlots saved to: {(output_dir / 'plots').absolute()}")
        print("\nPlot-only mode complete!")
        return

    # =========================================================================
    # Normal analysis mode
    # =========================================================================

    # Download or load models
    loaded_models = {}
    if args.download_models:
        print("\nDownloading models from HuggingFace...")
        try:
            loaded_models = download_models()
            print("Models downloaded successfully")
        except Exception as e:
            print(f"Warning: Could not download models: {e}\nContinuing with existing models...")

    # Initialize both analyzers and evaluators
    print("\nInitializing analyzers and evaluators...")
    print("Creating state symmetry analyzer...")
    analyzer_state = PongSymmetryAnalyzer()
    evaluator_state = ModelEvaluatorWithRSA(
        analyzer=analyzer_state,
        env_name="PongNoFrameskip-v4",
        render=False,
        activation_processing=args.activation_processing,
        pca_components=args.pca_components,
        rp_dimension=args.rp_dimension,
        rp_repetitions=args.rp_repetitions,
        symmetric_pair_ratio=args.symmetric_pair_ratio,
        augment_symmetric=args.augment_symmetric,
        symmetric_top_rows=args.symmetric_top_rows,
        symmetric_bottom_rows=args.symmetric_bottom_rows,
        symmetric_midline_threshold=args.symmetric_midline_threshold
    )

    print("Creating policy symmetry analyzer...")
    analyzer_policy = PolicySymmetryAnalyzer()
    # Policy symmetry does NOT use augmentation - it groups states by action
    evaluator_policy = ModelEvaluatorWithRSA(
        analyzer=analyzer_policy,
        env_name="PongNoFrameskip-v4",
        render=False,
        activation_processing=args.activation_processing,
        pca_components=args.pca_components,
        rp_dimension=args.rp_dimension,
        rp_repetitions=args.rp_repetitions,
        symmetric_pair_ratio=args.symmetric_pair_ratio,
        augment_symmetric=False,  # Policy symmetry never uses augmentation
        symmetric_top_rows=args.symmetric_top_rows,
        symmetric_bottom_rows=args.symmetric_bottom_rows,
        symmetric_midline_threshold=args.symmetric_midline_threshold
    )

    print(f"\nAnalyzing models: {', '.join(models_to_analyze)}")
    print(f"Number of states: {args.num_states}")
    print(f"Activation processing: {args.activation_processing}")
    if args.activation_processing == 'pca':
        print(f"PCA components: {args.pca_components}")
    elif args.activation_processing == 'random_projection':
        print(f"Random projection dimension: {args.rp_dimension}")
        print(f"Random projection repetitions: {args.rp_repetitions}")
    if args.symmetric_pair_ratio is not None:
        print(f"Symmetric pair ratio: {args.symmetric_pair_ratio:.2%} (stratified sampling)")
    elif args.augment_symmetric:
        print(f"Symmetric augmentation: ENABLED (constructing synthetic symmetric pairs)")
        print(f"  Top rows preserved: {args.symmetric_top_rows}")
        print(f"  Bottom rows preserved: {args.symmetric_bottom_rows}")
        print(f"  Midline threshold: {args.symmetric_midline_threshold}")
    else:
        print(f"Sampling strategy: Random (natural distribution)")
    print(f"Generate plots: {not args.no_plots}")
    print("Both state and policy symmetry will be analyzed for each model\n")

    # Analyze each model
    all_results = {}
    # Keep track of evaluators for aggregate plot
    evaluator_state_dict = {}
    evaluator_policy_dict = {}

    for model_name in models_to_analyze:
        model = loaded_models.get(model_name) or load_single_model(model_name)
        results = analyze_single_model(
            model_name=model_name,
            model=model,
            evaluator_state=evaluator_state,
            evaluator_policy=evaluator_policy,
            num_states=args.num_states,
            output_dir=output_dir,
            generate_plots=not args.no_plots,
            activation_processing=args.activation_processing
        )
        all_results[model_name] = results

        # Store evaluators and save results for later plotting
        if results:
            evaluator_state_dict[model_name] = evaluator_state
            evaluator_policy_dict[model_name] = evaluator_policy

            # Save RSA results to pickle for fast re-plotting
            try:
                save_rsa_results(evaluator_state, evaluator_policy, model_name, output_dir, suffix)
            except Exception as e:
                print(f"Warning: Could not save RSA results for {model_name}: {e}")

            # Render symmetric state pair visualizations
            try:
                render_symmetric_state_pairs(evaluator_state, evaluator_policy, model_name, output_dir,
                                            num_pairs=10, suffix=suffix)
            except Exception as e:
                print(f"Warning: Could not render state pairs for {model_name}: {e}")
                traceback.print_exc()

    # Generate aggregate comparison plot for DQN + PPO (if both are analyzed)
    if not args.no_plots:
        aggregate_models = [m for m in ['dqn', 'ppo'] if m in evaluator_state_dict]
        if len(aggregate_models) >= 2:
            aggregate_path = output_dir / 'plots' / f'pong_dqn_ppo_comparison{suffix}.pdf'
            try:
                plot_aggregate_comparison(
                    evaluator_state_dict, evaluator_policy_dict,
                    aggregate_models, save_path=str(aggregate_path)
                )
            except Exception as e:
                print(f"Warning: Could not generate aggregate plot: {e}")
                traceback.print_exc()

    # Save results to disk
    results_file = output_dir / 'data' / 'analysis_results.json'
    try:
        def to_json(obj):
            """Convert numpy/torch objects to JSON-serializable types."""
            if isinstance(obj, dict):
                return {k: to_json(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [to_json(x) for x in obj]
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer, np.floating)) or hasattr(obj, 'item'):
                return obj.item()
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            return str(obj)

        with open(results_file, 'w') as f:
            json.dump(to_json(all_results), f, indent=2)
        print(f"\nResults data saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results to JSON: {e}")

    # Print summary
    print_section("ANALYSIS SUMMARY")
    for model_name, results in all_results.items():
        if results:
            print(f"{model_name.upper()}:")
            if 'correlation' in results:
                print(f"  Correlation: {results['correlation']:.4f}")
            if 'p_value' in results:
                print(f"  P-value: {results['p_value']:.4e}")
            print()

    print(f"Results saved to: {output_dir.absolute()}")
    print(f"Plots saved to: {(output_dir / 'plots').absolute()}")
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
