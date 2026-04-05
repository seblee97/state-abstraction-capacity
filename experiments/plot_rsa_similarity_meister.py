"""
RSA Cosine Similarity Plotting and Heatmap Generation for Meister Maze

This script analyzes neural network representations using Representational
Similarity Analysis (RSA) with cosine similarity for DQN and PPO models
trained on the Meister maze environment.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Tuple, List, Optional
from collections import deque

from key_door import key_door_env, visualisation_env
from sac.models import dqn, ppo, deep_sarsa
from sac import utils
from sac.utils import dijkstra_policy

# Default save directory for all outputs
DEFAULT_SAVE_DIR = '../results/meister/'


def ensure_save_dir(save_dir: str = DEFAULT_SAVE_DIR) -> None:
    """Create the save directory if it doesn't exist."""
    os.makedirs(save_dir, exist_ok=True)


def get_save_path(filename: str, save_dir: str = DEFAULT_SAVE_DIR) -> str:
    """Get full save path, creating directory if needed."""
    ensure_save_dir(save_dir)
    return os.path.join(save_dir, filename)


# =============================================================================
# RSA Computation Functions
# =============================================================================

def compute_rsa_matrix(reprs: np.ndarray) -> np.ndarray:
    """Compute RSA matrix using cosine similarity with mean-centering."""
    reprs_centered = reprs - reprs.mean(axis=0)
    return cosine_similarity(reprs_centered)


def plot_rsa_matrix(
    reprs: np.ndarray,
    title: str = '',
    save_name: Optional[str] = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    show: bool = True,
    figsize: Tuple[float, float] = (1.5, 1.5),
    cmap: str = 'viridis',
    save_colorbar: bool = True
) -> np.ndarray:
    """Compute and plot RSA matrix using cosine similarity."""
    rsa_matrix = compute_rsa_matrix(reprs)

    # Plot RSA matrix without colorbar
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(rsa_matrix, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    plt.tight_layout()

    if save_name:
        save_path = get_save_path(save_name, save_dir)
        plt.savefig(save_path, bbox_inches='tight', dpi=300, pad_inches=0)
    if show:
        plt.show()
    else:
        plt.close()

    # Save colorbar separately (only once)
    if save_colorbar and save_name:
        colorbar_path = get_save_path('rsa_colorbar.pdf', save_dir)
        if not os.path.exists(colorbar_path):
            fig_cb, ax_cb = plt.subplots(figsize=(0.15, 1.5))
            cb = fig_cb.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(-1, 1)),
                                  cax=ax_cb)
            cb.ax.tick_params(labelsize=8)
            plt.savefig(colorbar_path, bbox_inches='tight', dpi=300)
            plt.close(fig_cb)

    return rsa_matrix


def calculate_participation_ratio(reprs: np.ndarray) -> float:
    """Calculate the participation ratio (effective dimensionality)."""
    cov_matrix = np.cov(reprs, rowvar=False)
    eigenvalues, _ = np.linalg.eig(cov_matrix)
    eigenvalues = np.real(eigenvalues)
    participation_ratio = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2)
    return participation_ratio


def get_mean_off_diagonal(rsa_matrix: np.ndarray) -> float:
    """Compute the mean of off-diagonal elements in an RSA matrix."""
    n = rsa_matrix.shape[0]
    off_diagonal_sum = np.sum(rsa_matrix) - np.sum(np.diag(rsa_matrix))
    num_off_diagonal_elements = n * (n - 1)
    return off_diagonal_sum / num_off_diagonal_elements


def get_within_label_similarity(
    reprs: np.ndarray,
    state_to_label: Dict,
    state_id_mapping: Dict
) -> float:
    """Compute weighted mean cosine similarity within label groups."""
    sims_same_label = []
    counts_same_label = []
    global_mean = reprs.mean(axis=0)

    for label in set(state_to_label.values()):
        states_with_label = [state for state, lbl in state_to_label.items() if lbl == label]
        if len(states_with_label) < 2:
            continue
        indices = [state_id_mapping[state] for state in states_with_label]
        if len(indices) < 2:
            continue
        reprs_subset = reprs[indices]
        sim_matrix = cosine_similarity(reprs_subset - global_mean)
        sim_mean = get_mean_off_diagonal(sim_matrix)
        sims_same_label.append(sim_mean)
        counts_same_label.append(len(states_with_label))

    if counts_same_label:
        weights = np.array(counts_same_label) / np.sum(counts_same_label)
        return np.average(sims_same_label, weights=weights)
    return 0.0


# =============================================================================
# Depth Computation by Direction Changes (Turns)
# =============================================================================

def compute_state_depth_by_turns(
    id_state_mapping: Dict[int, tuple],
    P: np.ndarray,
    state_id_mapping: Dict[tuple, int],
    start_pos: Tuple[int, int] = (2, 22)
) -> Dict[tuple, int]:
    """
    Compute depth for each state based on minimum number of direction changes
    (turns) needed to reach it from start position.

    Uses 0-1 BFS: continuing in same direction costs 0, changing direction costs 1.

    Parameters
    ----------
    id_state_mapping : Dict[int, tuple]
        Mapping from original state index to (x, y) coordinates
    P : np.ndarray
        Transition matrix of shape (S, A, S)
    state_id_mapping : Dict[tuple, int]
        Mapping from (x, y) coordinates to state index
    start_pos : Tuple[int, int]
        Starting position (x, y) for BFS

    Returns
    -------
    state_to_depth : Dict[tuple, int]
        Mapping from state coordinates to depth (number of turns from start)
    """
    state_to_depth = {}

    # Check if start position exists
    if start_pos not in state_id_mapping:
        print(f"Warning: start position {start_pos} not in state space!")
        print(f"Available positions (first 10): {list(state_id_mapping.keys())[:10]}")
        return state_to_depth

    start_idx = state_id_mapping[start_pos]
    num_actions = P.shape[1]

    # Track minimum turns to reach (state, direction) pairs
    # direction = -1 means "no direction yet" (at start)
    # Use dict: (state_idx, direction) -> min_turns
    min_turns = {}

    # 0-1 BFS using deque: add cost-0 transitions to front, cost-1 to back
    # State: (state_idx, last_direction, num_turns)
    queue = deque()

    # From start, we can go in any direction with 0 turns
    min_turns[(start_idx, -1)] = 0
    queue.append((start_idx, -1, 0))  # -1 means no direction yet

    while queue:
        current_idx, last_dir, turns = queue.popleft()

        # Skip if we've found a better path to this (state, direction) pair
        if (current_idx, last_dir) in min_turns and min_turns[(current_idx, last_dir)] < turns:
            continue

        # Update state_to_depth with minimum turns to reach this position
        current_pos = id_state_mapping[current_idx]
        if current_pos not in state_to_depth or state_to_depth[current_pos] > turns:
            state_to_depth[current_pos] = turns

        # Explore all actions
        for action in range(num_actions):
            next_states = np.where(P[current_idx, action] > 0)[0]
            for next_idx in next_states:
                if next_idx == current_idx:
                    continue  # Skip self-loops

                # Determine cost: 0 if same direction, 1 if turning
                if last_dir == -1:
                    # First move from start, no turn cost
                    new_turns = turns
                elif action == last_dir:
                    # Same direction, no turn cost
                    new_turns = turns
                else:
                    # Direction change, +1 turn
                    new_turns = turns + 1

                # Check if this is a better path
                if (next_idx, action) not in min_turns or min_turns[(next_idx, action)] > new_turns:
                    min_turns[(next_idx, action)] = new_turns

                    # 0-1 BFS: cost 0 goes to front, cost 1 goes to back
                    if last_dir == -1 or action == last_dir:
                        queue.appendleft((next_idx, action, new_turns))
                    else:
                        queue.append((next_idx, action, new_turns))

    return state_to_depth


def get_states_at_depth(
    state_to_depth: Dict[tuple, int],
    depth: int
) -> List[tuple]:
    """Get all states at a specific depth."""
    return [state for state, d in state_to_depth.items() if d == depth]


# =============================================================================
# Depth-wise Similarity Computation
# =============================================================================

def compute_similarity_at_depth(
    reprs: np.ndarray,
    state_to_depth: Dict[tuple, int],
    state_to_label: Dict[tuple, int],
    state_id_mapping: Dict[tuple, int],
    depth: int
) -> Tuple[float, float]:
    """
    Compute global and within-label similarity for states at a specific depth.

    Parameters
    ----------
    reprs : np.ndarray
        Representations for all states
    state_to_depth : Dict[tuple, int]
        Mapping from state to depth
    state_to_label : Dict[tuple, int]
        Mapping from state to label (MDP or policy)
    state_id_mapping : Dict[tuple, int]
        Mapping from state coordinates to index
    depth : int
        The depth level to analyze

    Returns
    -------
    global_sim : float
        Mean off-diagonal similarity for states at this depth
    within_label_sim : float
        Within-label similarity for states at this depth
    """
    # Get states at this depth
    states_at_depth = get_states_at_depth(state_to_depth, depth)

    if len(states_at_depth) < 2:
        return np.nan, np.nan

    # Get indices for these states
    indices = [state_id_mapping[state] for state in states_at_depth if state in state_id_mapping]

    if len(indices) < 2:
        return np.nan, np.nan

    # Get representations for states at this depth
    reprs_at_depth = reprs[indices]
    global_mean = reprs.mean(axis=0)  # Use global mean for centering

    # Compute RSA matrix for states at this depth
    reprs_centered = reprs_at_depth - global_mean
    rsa_matrix = cosine_similarity(reprs_centered)

    # Global similarity (mean off-diagonal)
    global_sim = get_mean_off_diagonal(rsa_matrix)

    # Within-label similarity
    # Filter state_to_label to only include states at this depth
    filtered_state_to_label = {state: state_to_label[state]
                               for state in states_at_depth
                               if state in state_to_label}

    # Create a local mapping for states at this depth
    local_state_id_mapping = {state: i for i, state in enumerate(states_at_depth) if state in state_id_mapping}

    within_label_sim = get_within_label_similarity(
        reprs_at_depth, filtered_state_to_label, local_state_id_mapping
    )

    return global_sim, within_label_sim


def compute_all_depth_similarities(
    reprs: np.ndarray,
    state_to_depth: Dict[tuple, int],
    state_to_mdp_label: Dict[tuple, int],
    state_to_policy_label: Dict[tuple, int],
    state_id_mapping: Dict[tuple, int]
) -> Tuple[List[int], List[float], List[float], List[float]]:
    """
    Compute similarity metrics across all depths.

    Returns
    -------
    depths : List[int]
        List of depth levels
    global_sims : List[float]
        Global similarity at each depth
    mdp_within_sims : List[float]
        Within-label similarity for MDP labels at each depth
    policy_within_sims : List[float]
        Within-label similarity for policy labels at each depth
    """
    max_depth = max(state_to_depth.values()) if state_to_depth else 0

    depths = []
    global_sims = []
    mdp_within_sims = []
    policy_within_sims = []

    for depth in range(max_depth + 1):
        states_at_depth = get_states_at_depth(state_to_depth, depth)
        if len(states_at_depth) < 2:
            continue

        global_sim, mdp_within = compute_similarity_at_depth(
            reprs, state_to_depth, state_to_mdp_label, state_id_mapping, depth
        )
        _, policy_within = compute_similarity_at_depth(
            reprs, state_to_depth, state_to_policy_label, state_id_mapping, depth
        )

        depths.append(depth)
        global_sims.append(global_sim)
        mdp_within_sims.append(mdp_within)
        policy_within_sims.append(policy_within)

    return depths, global_sims, mdp_within_sims, policy_within_sims


def get_local_global_sims(
    rsa_matrix: np.ndarray,
    state_to_label: Dict,
    state_id_mapping: Dict
) -> Tuple[float, float]:
    """Compute global and within-label (local) similarities."""
    mean_off_diag = get_mean_off_diagonal(rsa_matrix)
    # For within-label, we need to recompute from the RSA matrix
    # This is a simplified version - in the notebook they use reprs directly
    within_label_sim = get_within_label_similarity(rsa_matrix, state_to_label, state_id_mapping)
    return mean_off_diag, within_label_sim


# =============================================================================
# DQN Representation Extraction
# =============================================================================

def get_dqn_representations(model, states):
    """Extract representations from DQN model at each layer."""
    model._net.eval()
    network = model._net
    states = torch.tensor(states, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        x = torch.relu(network.conv1(states))
        x = torch.relu(network.conv2(x))
        conv_x = x.view(x.size(0), -1)
        x1 = torch.relu(network.fc1(conv_x))
        x2 = torch.relu(network.fc2(x1))
        qvals = network.fc3(x2)
        vals = torch.max(qvals, dim=1).values
        action = torch.argmax(qvals, dim=1)
    return conv_x, x1, x2, vals, action


def accumulate_dqn_representations(model, env, state_shape, save_dir=DEFAULT_SAVE_DIR, prefix='dqn', show=False):
    """Accumulate representations for all states and compute RSA matrices for DQN."""
    conv_reprs = []
    shared_reprs_1 = []
    shared_reprs_2 = []
    state_reprs = []
    values = []
    policy = []

    state_id_mapping = {state: i for i, state in enumerate(env.positional_state_space)}
    id_state_mapping = {i: state for i, state in enumerate(env.positional_state_space)}
    state_to_value = {id_state_mapping[i]: 0.0 for i in range(len(env.positional_state_space))}
    state_to_policy = {id_state_mapping[i]: 0.0 for i in range(len(env.positional_state_space))}

    for state in env.positional_state_space:
        env.move_agent_to(state)
        state_input = env.get_state_representation()
        shape = (1,) + state_shape
        state_input = torch.FloatTensor(state_input.reshape(shape)).to('cuda')
        conv, shared_1, shared_2, vals, actions = get_dqn_representations(model, state_input)
        conv_reprs.append(conv.cpu().numpy())
        shared_reprs_1.append(shared_1.cpu().numpy())
        shared_reprs_2.append(shared_2.cpu().numpy())
        state_reprs.append(state_input.cpu().numpy())
        values.append(vals.cpu().numpy())
        policy.append(actions.cpu().numpy())
        state_to_value[state] = vals.item()
        state_to_policy[state] = actions.item()

    conv_reprs = np.vstack(conv_reprs)
    shared_reprs_1 = np.vstack(shared_reprs_1)
    shared_reprs_2 = np.vstack(shared_reprs_2)
    state_reprs = np.vstack(state_reprs).reshape(len(state_reprs), -1)
    values = np.hstack(values)
    policy = np.hstack(policy)

    print(f"[{prefix}] Input State Shape: {state_reprs.shape}")
    print(f"[{prefix}] Conv Shape: {conv_reprs.shape}")
    print(f"[{prefix}] FC1 Shape: {shared_reprs_1.shape}, PR: {calculate_participation_ratio(shared_reprs_1):.2f}")
    print(f"[{prefix}] FC2 Shape: {shared_reprs_2.shape}, PR: {calculate_participation_ratio(shared_reprs_2):.2f}")
    print(f"[{prefix}] Values: min={np.min(values):.2f}, max={np.max(values):.2f}, mean={np.mean(values):.2f}")

    # Compute and plot RSA matrices
    state_rsa = plot_rsa_matrix(state_reprs, f"{prefix} Input State RSA",
                                 save_name=f"{prefix}_state_rsa.pdf", save_dir=save_dir, show=show)
    conv_rsa = plot_rsa_matrix(conv_reprs, f"{prefix} Conv RSA",
                                save_name=f"{prefix}_conv_rsa.pdf", save_dir=save_dir, show=show)
    shared_1_rsa = plot_rsa_matrix(shared_reprs_1, f"{prefix} FC1 RSA",
                                    save_name=f"{prefix}_fc1_rsa.pdf", save_dir=save_dir, show=show)
    shared_2_rsa = plot_rsa_matrix(shared_reprs_2, f"{prefix} FC2 RSA",
                                    save_name=f"{prefix}_fc2_rsa.pdf", save_dir=save_dir, show=show)

    # Plot value and policy heatmaps
    env.plot_heatmap_over_env(state_to_value, save_name=get_save_path(f'{prefix}_values.pdf', save_dir))
    env.plot_heatmap_over_env(state_to_value, save_name=get_save_path(f'{prefix}_values.png', save_dir))
    cmap = plt.get_cmap('viridis')
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.pdf', save_dir), colormap=cmap)
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.png', save_dir), colormap=cmap)

    return {
        'state': state_rsa,
        'conv': conv_rsa,
        'shared_1': shared_1_rsa,
        'shared_2': shared_2_rsa,
        'reprs': {
            'state': state_reprs,
            'conv': conv_reprs,
            'shared_1': shared_reprs_1,
            'shared_2': shared_reprs_2
        },
        'state_to_value': state_to_value,
        'state_to_policy': state_to_policy
    }


# =============================================================================
# PPO Representation Extraction
# =============================================================================

def get_ppo_representations(model, state):
    """Extract representations from PPO model at each layer."""
    model._net.eval()
    network = model._net
    with torch.no_grad():
        x = torch.relu(network.conv1(state))
        x = torch.relu(network.conv2(x))
        x_c = x.view(x.size(0), -1)
        x = torch.relu(network.fc1(x_c))
        x = torch.relu(network.fc2(x))
        actor_repr = torch.relu(network.pi_fc(x))
        critic_repr = torch.relu(network.v_fc(x))
        value = network.v(critic_repr)
        actions = network.pi(actor_repr)
        action = torch.argmax(actions, dim=1)
    return x_c, x, actor_repr, critic_repr, value, action


def accumulate_ppo_representations(model, env, state_shape, save_dir=DEFAULT_SAVE_DIR, prefix='ppo', show=False):
    """Accumulate representations for all states and compute RSA matrices for PPO."""
    conv_reprs = []
    shared_reprs = []
    actor_reprs = []
    critic_reprs = []
    state_reprs = []
    values = []
    policy = []

    for state in env.positional_state_space:
        env.move_agent_to(state)
        state_input = env.get_state_representation()
        shape = (1,) + state_shape
        state_input = torch.FloatTensor(state_input.reshape(shape)).to('cuda')
        conv, shared, actor, critic, value, actions = get_ppo_representations(model, state_input)
        conv_reprs.append(conv.cpu().numpy())
        shared_reprs.append(shared.cpu().numpy())
        actor_reprs.append(actor.cpu().numpy())
        critic_reprs.append(critic.cpu().numpy())
        state_reprs.append(state_input.cpu().numpy())
        values.append(value.cpu().numpy())
        policy.append(actions.cpu().numpy())

    conv_reprs = np.vstack(conv_reprs)
    shared_reprs = np.vstack(shared_reprs)
    actor_reprs = np.vstack(actor_reprs)
    critic_reprs = np.vstack(critic_reprs)
    state_reprs = np.vstack(state_reprs).reshape(len(state_reprs), -1)
    values = np.vstack(values)

    state_to_values = {state: values[i].squeeze() for i, state in enumerate(env.positional_state_space)}
    state_to_policy = {state: policy[i].squeeze() for i, state in enumerate(env.positional_state_space)}

    print(f"[{prefix}] Input State Shape: {state_reprs.shape}")
    print(f"[{prefix}] Conv Shape: {conv_reprs.shape}")
    print(f"[{prefix}] Shared Shape: {shared_reprs.shape}, PR: {calculate_participation_ratio(shared_reprs):.2f}")
    print(f"[{prefix}] Actor Shape: {actor_reprs.shape}, PR: {calculate_participation_ratio(actor_reprs):.2f}")
    print(f"[{prefix}] Critic Shape: {critic_reprs.shape}, PR: {calculate_participation_ratio(critic_reprs):.2f}")
    print(f"[{prefix}] Values: min={np.min(values):.2f}, max={np.max(values):.2f}, mean={np.mean(values):.2f}")

    # Compute and plot RSA matrices
    state_rsa = plot_rsa_matrix(state_reprs, f"{prefix} Input State RSA",
                                 save_name=f"{prefix}_state_rsa.pdf", save_dir=save_dir, show=show)
    conv_rsa = plot_rsa_matrix(conv_reprs, f"{prefix} Conv RSA",
                                save_name=f"{prefix}_conv_rsa.pdf", save_dir=save_dir, show=show)
    shared_rsa = plot_rsa_matrix(shared_reprs, f"{prefix} Shared RSA",
                                  save_name=f"{prefix}_shared_rsa.pdf", save_dir=save_dir, show=show)
    actor_rsa = plot_rsa_matrix(actor_reprs, f"{prefix} Actor RSA",
                                 save_name=f"{prefix}_actor_rsa.pdf", save_dir=save_dir, show=show)
    critic_rsa = plot_rsa_matrix(critic_reprs, f"{prefix} Critic RSA",
                                  save_name=f"{prefix}_critic_rsa.pdf", save_dir=save_dir, show=show)

    # Plot value and policy heatmaps
    env.plot_heatmap_over_env(state_to_values, save_name=get_save_path(f'{prefix}_values.pdf', save_dir))
    env.plot_heatmap_over_env(state_to_values, save_name=get_save_path(f'{prefix}_values.png', save_dir))
    cmap = plt.get_cmap('viridis')
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.pdf', save_dir), colormap=cmap)
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.png', save_dir), colormap=cmap)

    return {
        'state': state_rsa,
        'conv': conv_rsa,
        'shared': shared_rsa,
        'actor': actor_rsa,
        'critic': critic_rsa,
        'reprs': {
            'state': state_reprs,
            'conv': conv_reprs,
            'shared': shared_reprs,
            'actor': actor_reprs,
            'critic': critic_reprs
        },
        'state_to_value': state_to_values,
        'state_to_policy': state_to_policy
    }


# =============================================================================
# SARSA Representation Extraction
# =============================================================================

def get_sarsa_representations(model, states):
    """Extract representations from SARSA model at each layer."""
    model._net.eval()
    network = model._net
    states = torch.tensor(states, dtype=torch.float32).to('cuda')
    with torch.no_grad():
        x = torch.relu(network.conv1(states))
        x = torch.relu(network.conv2(x))
        conv_x = x.view(x.size(0), -1)
        x1 = torch.relu(network.fc1(conv_x))
        x2 = torch.relu(network.fc2(x1))
        qvals = network.fc3(x2)
        vals = torch.max(qvals, dim=1).values
        action = torch.argmax(qvals, dim=1)
    return conv_x, x1, x2, vals, action


def accumulate_sarsa_representations(model, env, state_shape, save_dir=DEFAULT_SAVE_DIR, prefix='sarsa', show=False):
    """Accumulate representations for all states and compute RSA matrices for SARSA."""
    conv_reprs = []
    shared_reprs_1 = []
    shared_reprs_2 = []
    state_reprs = []
    values = []
    policy = []

    state_id_mapping = {state: i for i, state in enumerate(env.positional_state_space)}
    id_state_mapping = {i: state for i, state in enumerate(env.positional_state_space)}
    state_to_value = {id_state_mapping[i]: 0.0 for i in range(len(env.positional_state_space))}
    state_to_policy = {id_state_mapping[i]: 0.0 for i in range(len(env.positional_state_space))}

    for state in env.positional_state_space:
        env.move_agent_to(state)
        state_input = env.get_state_representation()
        shape = (1,) + state_shape
        state_input = torch.FloatTensor(state_input.reshape(shape)).to('cuda')
        conv, shared_1, shared_2, vals, actions = get_sarsa_representations(model, state_input)
        conv_reprs.append(conv.cpu().numpy())
        shared_reprs_1.append(shared_1.cpu().numpy())
        shared_reprs_2.append(shared_2.cpu().numpy())
        state_reprs.append(state_input.cpu().numpy())
        values.append(vals.cpu().numpy())
        policy.append(actions.cpu().numpy())
        state_to_value[state] = vals.item()
        state_to_policy[state] = actions.item()

    conv_reprs = np.vstack(conv_reprs)
    shared_reprs_1 = np.vstack(shared_reprs_1)
    shared_reprs_2 = np.vstack(shared_reprs_2)
    state_reprs = np.vstack(state_reprs).reshape(len(state_reprs), -1)
    values = np.hstack(values)
    policy = np.hstack(policy)

    print(f"[{prefix}] Input State Shape: {state_reprs.shape}")
    print(f"[{prefix}] Conv Shape: {conv_reprs.shape}")
    print(f"[{prefix}] FC1 Shape: {shared_reprs_1.shape}, PR: {calculate_participation_ratio(shared_reprs_1):.2f}")
    print(f"[{prefix}] FC2 Shape: {shared_reprs_2.shape}, PR: {calculate_participation_ratio(shared_reprs_2):.2f}")
    print(f"[{prefix}] Values: min={np.min(values):.2f}, max={np.max(values):.2f}, mean={np.mean(values):.2f}")

    state_rsa = plot_rsa_matrix(state_reprs, f"{prefix} Input State RSA",
                                 save_name=f"{prefix}_state_rsa.pdf", save_dir=save_dir, show=show)
    conv_rsa = plot_rsa_matrix(conv_reprs, f"{prefix} Conv RSA",
                                save_name=f"{prefix}_conv_rsa.pdf", save_dir=save_dir, show=show)
    shared_1_rsa = plot_rsa_matrix(shared_reprs_1, f"{prefix} FC1 RSA",
                                    save_name=f"{prefix}_fc1_rsa.pdf", save_dir=save_dir, show=show)
    shared_2_rsa = plot_rsa_matrix(shared_reprs_2, f"{prefix} FC2 RSA",
                                    save_name=f"{prefix}_fc2_rsa.pdf", save_dir=save_dir, show=show)

    env.plot_heatmap_over_env(state_to_value, save_name=get_save_path(f'{prefix}_values.pdf', save_dir))
    env.plot_heatmap_over_env(state_to_value, save_name=get_save_path(f'{prefix}_values.png', save_dir))
    cmap = plt.get_cmap('viridis')
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.pdf', save_dir), colormap=cmap)
    env.plot_heatmap_over_env(state_to_policy, save_name=get_save_path(f'{prefix}_policy.png', save_dir), colormap=cmap)

    return {
        'state': state_rsa,
        'conv': conv_rsa,
        'shared_1': shared_1_rsa,
        'shared_2': shared_2_rsa,
        'reprs': {
            'state': state_reprs,
            'conv': conv_reprs,
            'shared_1': shared_reprs_1,
            'shared_2': shared_reprs_2
        },
        'state_to_value': state_to_value,
        'state_to_policy': state_to_policy
    }


# =============================================================================
# Comparison Plotting Functions
# =============================================================================

def plot_reference_rsa_matrix(
    values: np.ndarray,
    title: str,
    save_name: Optional[str] = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 10),
    cmap: str = 'binary'
) -> None:
    """Plot a binary reference RSA matrix."""
    plt.figure(figsize=figsize)
    plt.imshow(values, cmap=cmap, interpolation='nearest')
    plt.colorbar(label='Same value')
    plt.xlabel('Index')
    plt.ylabel('Index')
    plt.title(title)
    plt.tight_layout()

    if save_name:
        save_path = get_save_path(save_name, save_dir)
        plt.savefig(save_path, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()


def compute_binary_rsa_policy(policy: np.ndarray, n_states: int) -> np.ndarray:
    """Compute binary RSA matrix based on policy agreement."""
    rsa_policy = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if policy[i] == policy[j]:
                rsa_policy[i, j] = 1
    return rsa_policy


def compute_binary_rsa_labels(state_labels: np.ndarray, n_states: int) -> np.ndarray:
    """Compute binary RSA matrix based on state label agreement."""
    rsa_labels = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if state_labels[i] == state_labels[j]:
                rsa_labels[i, j] = 1
    return rsa_labels


def plot_dqn_ppo_comparison(
    dqn_hom_sims: Dict,
    dqn_init_hom_sims: Dict,
    dqn_policy_sims: Dict,
    dqn_init_policy_sims: Dict,
    ppo_hom_sims: Dict,
    ppo_init_hom_sims: Dict,
    ppo_policy_sims: Dict,
    ppo_init_policy_sims: Dict,
    sarsa_hom_sims: Optional[Dict] = None,
    sarsa_init_hom_sims: Optional[Dict] = None,
    sarsa_policy_sims: Optional[Dict] = None,
    sarsa_init_policy_sims: Optional[Dict] = None,
    save_dir: str = DEFAULT_SAVE_DIR,
    color_global: str = '#009988',
    color_local: str = '#EE7733'
) -> Tuple[plt.Figure, plt.Figure]:
    """Create comparison plots for DQN, PPO, and optionally SARSA models side by side."""
    dqn_labels = ['Pixels', 'Conv', 'FC1', 'FC2']
    dqn_keys = ['state', 'conv', 'shared_1', 'shared_2']
    ppo_labels = ['Pixels', 'Conv', 'FC1', 'Actor', 'Critic']
    ppo_keys = ['state', 'conv', 'shared', 'actor', 'critic']
    sarsa_labels = ['Pixels', 'Conv', 'FC1', 'FC2']
    sarsa_keys = ['state', 'conv', 'shared_1', 'shared_2']

    include_sarsa = sarsa_hom_sims is not None
    n_cols = 3 if include_sarsa else 2
    fig_width = 4.5 if include_sarsa else 3.0
    offset = 0.2

    def _scatter_model(ax, x, init_global, trained_global, init_local, trained_local, labels, legend=False):
        ax.scatter(x - offset, init_global, color=color_global, marker='o', s=15, alpha=0.8,
                   label='Global (init)' if legend else None)
        ax.scatter(x + offset, trained_global, color=color_global, marker='s', s=15, alpha=0.8,
                   label='Global (trained)' if legend else None)
        ax.scatter(x - offset, init_local, color=color_local, marker='o', s=15, alpha=0.8,
                   label='Within-label (init)' if legend else None)
        ax.scatter(x + offset, trained_local, color=color_local, marker='s', s=15, alpha=0.8,
                   label='Within-label (trained)' if legend else None)
        for i in range(len(x)):
            ax.plot([x[i] - offset, x[i] + offset], [init_global[i], trained_global[i]],
                    color_global, alpha=0.5, linewidth=1)
            ax.plot([x[i] - offset, x[i] + offset], [init_local[i], trained_local[i]],
                    color_local, alpha=0.5, linewidth=1)

    # Figure 1: MDP Abstractions
    fig1, axs1 = plt.subplots(1, n_cols, figsize=(fig_width, 1.2), sharey=True)

    x_dqn = np.arange(len(dqn_labels))
    _scatter_model(axs1[0], x_dqn,
                   [abs(dqn_init_hom_sims[k][0]) for k in dqn_keys],
                   [abs(dqn_hom_sims[k][0]) for k in dqn_keys],
                   [abs(dqn_init_hom_sims[k][1]) for k in dqn_keys],
                   [abs(dqn_hom_sims[k][1]) for k in dqn_keys],
                   dqn_labels)

    x_ppo = np.arange(len(ppo_labels))
    _scatter_model(axs1[1], x_ppo,
                   [abs(ppo_init_hom_sims[k][0]) for k in ppo_keys],
                   [abs(ppo_hom_sims[k][0]) for k in ppo_keys],
                   [abs(ppo_init_hom_sims[k][1]) for k in ppo_keys],
                   [abs(ppo_hom_sims[k][1]) for k in ppo_keys],
                   ppo_labels, legend=True)

    if include_sarsa:
        x_sarsa = np.arange(len(sarsa_labels))
        _scatter_model(axs1[2], x_sarsa,
                       [abs(sarsa_init_hom_sims[k][0]) for k in sarsa_keys],
                       [abs(sarsa_hom_sims[k][0]) for k in sarsa_keys],
                       [abs(sarsa_init_hom_sims[k][1]) for k in sarsa_keys],
                       [abs(sarsa_hom_sims[k][1]) for k in sarsa_keys],
                       sarsa_labels)

    axs1[0].set_ylabel('Cosine Sim.', fontsize=8)
    axs1[0].set_xticks(x_dqn)
    axs1[0].set_xticklabels(dqn_labels, fontsize=8, rotation=45)
    axs1[1].set_xticks(x_ppo)
    axs1[1].set_xticklabels(ppo_labels, fontsize=8, rotation=45)
    axs1[0].tick_params(axis='y', labelsize=8)
    axs1[1].tick_params(axis='y', labelsize=8)
    axs1[0].set_ylim(-0.1, 1.0)
    axs1[1].legend(frameon=False, fontsize=6, loc=(0.01, 0.4))
    if include_sarsa:
        axs1[2].set_xticks(x_sarsa)
        axs1[2].set_xticklabels(sarsa_labels, fontsize=8, rotation=45)
        axs1[2].tick_params(axis='y', labelsize=8)

    sns.despine()
    fig1.savefig(get_save_path('mdp_abstractions.pdf', save_dir), transparent=True, bbox_inches='tight')

    # Figure 2: Policy Abstractions
    fig2, axs2 = plt.subplots(1, n_cols, figsize=(fig_width, 1.2), sharey=True)

    _scatter_model(axs2[0], x_dqn,
                   [abs(dqn_init_policy_sims[k][0]) for k in dqn_keys],
                   [abs(dqn_policy_sims[k][0]) for k in dqn_keys],
                   [abs(dqn_init_policy_sims[k][1]) for k in dqn_keys],
                   [abs(dqn_policy_sims[k][1]) for k in dqn_keys],
                   dqn_labels)

    _scatter_model(axs2[1], x_ppo,
                   [abs(ppo_init_policy_sims[k][0]) for k in ppo_keys],
                   [abs(ppo_policy_sims[k][0]) for k in ppo_keys],
                   [abs(ppo_init_policy_sims[k][1]) for k in ppo_keys],
                   [abs(ppo_policy_sims[k][1]) for k in ppo_keys],
                   ppo_labels, legend=True)

    if include_sarsa:
        _scatter_model(axs2[2], x_sarsa,
                       [abs(sarsa_init_policy_sims[k][0]) for k in sarsa_keys],
                       [abs(sarsa_policy_sims[k][0]) for k in sarsa_keys],
                       [abs(sarsa_init_policy_sims[k][1]) for k in sarsa_keys],
                       [abs(sarsa_policy_sims[k][1]) for k in sarsa_keys],
                       sarsa_labels)

    axs2[0].set_ylabel('Cosine Sim.', fontsize=8)
    axs2[0].set_title('DQN', fontsize=10)
    axs2[1].set_title('PPO', fontsize=10)
    axs2[0].set_xticks(x_dqn)
    axs2[0].set_xticklabels(dqn_labels, fontsize=8, rotation=45)
    axs2[1].set_xticks(x_ppo)
    axs2[1].set_xticklabels(ppo_labels, fontsize=8, rotation=45)
    axs2[0].tick_params(axis='y', labelsize=8)
    axs2[1].tick_params(axis='y', labelsize=8)
    axs2[0].set_ylim(-0.1, 1.0)
    if include_sarsa:
        axs2[2].set_title('SARSA-λ', fontsize=10)
        axs2[2].set_xticks(x_sarsa)
        axs2[2].set_xticklabels(sarsa_labels, fontsize=8, rotation=45)
        axs2[2].tick_params(axis='y', labelsize=8)

    sns.despine()
    fig2.savefig(get_save_path('policy_abstractions.pdf', save_dir), transparent=True, bbox_inches='tight')

    return fig1, fig2


def plot_single_model_comparison(
    hom_sims: Dict,
    init_hom_sims: Dict,
    policy_sims: Dict,
    init_policy_sims: Dict,
    layer_labels: List[str],
    layer_keys: List[str],
    model_name: str,
    save_dir: str = DEFAULT_SAVE_DIR,
    color_global: str = '#009988',
    color_local: str = '#EE7733'
) -> plt.Figure:
    """Create side-by-side MDP and Policy abstraction plots for a single model."""
    global_hom = [abs(hom_sims[k][0]) for k in layer_keys]
    local_hom = [abs(hom_sims[k][1]) for k in layer_keys]
    init_global_hom = [abs(init_hom_sims[k][0]) for k in layer_keys]
    init_local_hom = [abs(init_hom_sims[k][1]) for k in layer_keys]

    global_policy = [abs(policy_sims[k][0]) for k in layer_keys]
    local_policy = [abs(policy_sims[k][1]) for k in layer_keys]
    init_global_policy = [abs(init_policy_sims[k][0]) for k in layer_keys]
    init_local_policy = [abs(init_policy_sims[k][1]) for k in layer_keys]

    x = np.arange(len(layer_labels))
    offset = 0.2

    fig, axs = plt.subplots(1, 2, figsize=(3.0, 1.2), sharey=True)

    # MDP abstraction
    axs[0].scatter(x - offset, init_global_hom, color=color_global, marker='o', s=15, label='Global (init)', alpha=0.8)
    axs[0].scatter(x + offset, global_hom, color=color_global, marker='s', s=15, label='Global (trained)', alpha=0.8)
    axs[0].scatter(x - offset, init_local_hom, color=color_local, marker='o', s=15, label='Within-label (init)', alpha=0.8)
    axs[0].scatter(x + offset, local_hom, color=color_local, marker='s', s=15, label='Within-label (trained)', alpha=0.8)

    # Policy abstraction
    axs[1].scatter(x - offset, init_global_policy, color=color_global, marker='o', s=15, alpha=0.8)
    axs[1].scatter(x + offset, global_policy, color=color_global, marker='s', s=15, alpha=0.8)
    axs[1].scatter(x - offset, init_local_policy, color=color_local, marker='o', s=15, alpha=0.8)
    axs[1].scatter(x + offset, local_policy, color=color_local, marker='s', s=15, alpha=0.8)

    # Connect init to trained with lines
    for i in range(len(layer_labels)):
        axs[0].plot([x[i] - offset, x[i] + offset], [init_global_hom[i], global_hom[i]], color_global, alpha=0.5, linewidth=1)
        axs[0].plot([x[i] - offset, x[i] + offset], [init_local_hom[i], local_hom[i]], color_local, alpha=0.5, linewidth=1)
        axs[1].plot([x[i] - offset, x[i] + offset], [init_global_policy[i], global_policy[i]], color_global, alpha=0.5, linewidth=1)
        axs[1].plot([x[i] - offset, x[i] + offset], [init_local_policy[i], local_policy[i]], color_local, alpha=0.5, linewidth=1)

    axs[0].set_ylabel('Cosine Sim.', fontsize=8)
    axs[0].set_title('MDP abstr.', fontsize=10)
    axs[1].set_title('Policy abstr.', fontsize=10)
    for ax in axs:
        ax.set_xticks(x)
        ax.set_xticklabels(layer_labels, fontsize=8, rotation=45)
    axs[0].tick_params(axis='y', labelsize=8)
    axs[0].set_ylim(-0.1, 1.0)
    axs[1].legend(frameon=False, fontsize=6, loc=(0.01, 0.4))

    sns.despine()
    fig.savefig(get_save_path(f'{model_name.lower()}_abstractions.pdf', save_dir), transparent=True, bbox_inches='tight')

    return fig


# =============================================================================
# Depth-wise Similarity Plotting Functions
# =============================================================================

def plot_similarity_vs_depth(
    reprs_dict: Dict[str, np.ndarray],
    state_to_depth: Dict[tuple, int],
    state_to_mdp_label: Dict[tuple, int],
    state_to_policy_label: Dict[tuple, int],
    state_id_mapping: Dict[tuple, int],
    layer_labels: List[str],
    layer_keys: List[str],
    model_name: str,
    save_dir: str = DEFAULT_SAVE_DIR,
    color_global: str = '#009988',
    color_mdp: str = '#EE7733',
    color_policy: str = '#CC3311',
    show: bool = False
) -> plt.Figure:
    """
    Plot similarity as a function of depth with separate subplots for each layer.

    Each subplot shows three curves:
    - Baseline global similarity
    - Within-label similarity for MDP labels
    - Within-label similarity for policy labels

    Parameters
    ----------
    reprs_dict : Dict[str, np.ndarray]
        Dictionary mapping layer keys to representation arrays
    state_to_depth : Dict[tuple, int]
        Mapping from state to depth in the abstract MDP
    state_to_mdp_label : Dict[tuple, int]
        Mapping from state to MDP abstraction label
    state_to_policy_label : Dict[tuple, int]
        Mapping from state to policy label
    state_id_mapping : Dict[tuple, int]
        Mapping from state coordinates to index
    layer_labels : List[str]
        Display names for layers
    layer_keys : List[str]
        Keys for accessing layers in reprs_dict
    model_name : str
        Name of the model (for title and filename)
    save_dir : str
        Directory to save the plot
    color_global : str
        Color for global similarity curve
    color_mdp : str
        Color for MDP within-label similarity curve
    color_policy : str
        Color for policy within-label similarity curve
    show : bool
        Whether to display the plot

    Returns
    -------
    fig : plt.Figure
        The matplotlib figure
    """
    n_layers = len(layer_keys)
    n_cols = min(n_layers, 3)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.4 * n_rows), sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for idx, (layer_key, layer_label) in enumerate(zip(layer_keys, layer_labels)):
        ax = axs[idx]
        reprs = reprs_dict[layer_key]

        depths, global_sims, mdp_within_sims, policy_within_sims = compute_all_depth_similarities(
            reprs, state_to_depth, state_to_mdp_label, state_to_policy_label, state_id_mapping
        )

        if len(depths) == 0:
            ax.set_title(f'{layer_label}\n(No data)', fontsize=9)
            continue

        # Plot curves
        ax.plot(depths, global_sims, 'o-', color=color_global, label='Global', markersize=4, linewidth=1.5)
        ax.plot(depths, mdp_within_sims, 's-', color=color_mdp, label='MDP within-label', markersize=4, linewidth=1.5)
        ax.plot(depths, policy_within_sims, '^-', color=color_policy, label='Policy within-label', markersize=4, linewidth=1.5)

        ax.set_title(layer_label, fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_ylim(-0.2, 1.0)
        ax.set_xticks([0, 2, 4, 6])

        # Only show xlabel on bottom row
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('Tree Depth', fontsize=8)

        # Only show ylabel on leftmost column
        if idx % n_cols == 0:
            ax.set_ylabel('Cosine Sim.', fontsize=8)

        # Only show legend on first subplot
        if idx == 0:
            ax.legend(frameon=False, fontsize=6, loc='upper right')

    # Hide unused subplots
    for idx in range(len(layer_keys), len(axs)):
        axs[idx].set_visible(False)

    sns.despine()
    plt.tight_layout()

    save_path = get_save_path(f'{model_name.lower()}_similarity_vs_depth.pdf', save_dir)
    fig.savefig(save_path, transparent=True, bbox_inches='tight')
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


def plot_similarity_vs_depth_combined(
    trained_reprs_dict: Dict[str, np.ndarray],
    init_reprs_dict: Dict[str, np.ndarray],
    state_to_depth: Dict[tuple, int],
    state_to_mdp_label: Dict[tuple, int],
    state_to_policy_label: Dict[tuple, int],
    state_id_mapping: Dict[tuple, int],
    layer_labels: List[str],
    layer_keys: List[str],
    model_name: str,
    save_dir: str = DEFAULT_SAVE_DIR,
    color_global: str = '#009988',
    color_mdp: str = '#EE7733',
    color_policy: str = '#CC3311',
    show: bool = False
) -> plt.Figure:
    """
    Plot similarity vs depth comparing trained and init models.

    Each subplot shows curves for both trained (solid) and init (dashed) models.
    """
    n_layers = len(layer_keys)
    n_cols = min(n_layers, 3)
    n_rows = (n_layers + n_cols - 1) // n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(1.5 * n_cols, 1.4 * n_rows), sharex=True, sharey=True, squeeze=False)
    axs = axs.flatten()

    for idx, (layer_key, layer_label) in enumerate(zip(layer_keys, layer_labels)):
        ax = axs[idx]

        # Trained model
        trained_reprs = trained_reprs_dict[layer_key]
        depths_t, global_t, mdp_t, policy_t = compute_all_depth_similarities(
            trained_reprs, state_to_depth, state_to_mdp_label, state_to_policy_label, state_id_mapping
        )

        # Init model
        init_reprs = init_reprs_dict[layer_key]
        depths_i, global_i, mdp_i, policy_i = compute_all_depth_similarities(
            init_reprs, state_to_depth, state_to_mdp_label, state_to_policy_label, state_id_mapping
        )

        if len(depths_t) == 0:
            ax.set_title(f'{layer_label}\n(No data)', fontsize=9)
            continue

        # Plot trained (solid lines)
        ax.plot(depths_t, global_t, 'o-', color=color_global, label='Global (trained)', markersize=4, linewidth=1.5)
        ax.plot(depths_t, mdp_t, 's-', color=color_mdp, label='MDP within (trained)', markersize=4, linewidth=1.5)
        ax.plot(depths_t, policy_t, '^-', color=color_policy, label='Policy within (trained)', markersize=4, linewidth=1.5)

        # Plot init (dashed lines)
        ax.plot(depths_i, global_i, 'o--', color=color_global, label='Global (init)', markersize=3, linewidth=1, alpha=0.6)
        ax.plot(depths_i, mdp_i, 's--', color=color_mdp, label='MDP within (init)', markersize=3, linewidth=1, alpha=0.6)
        ax.plot(depths_i, policy_i, '^--', color=color_policy, label='Policy within (init)', markersize=3, linewidth=1, alpha=0.6)

        ax.set_title(layer_label, fontsize=9)
        ax.tick_params(axis='both', labelsize=7)
        ax.set_ylim(-0.2, 1.0)
        ax.set_xticks([0, 2, 4, 6])

        # Only show xlabel on bottom row
        if idx >= n_cols * (n_rows - 1):
            ax.set_xlabel('Tree Depth', fontsize=8)

        # Only show ylabel on leftmost column
        if idx % n_cols == 0:
            ax.set_ylabel('Cosine Sim.', fontsize=8)

        if idx == 0:
            ax.legend(frameon=False, fontsize=5, loc='upper right', ncol=2)

    for idx in range(len(layer_keys), len(axs)):
        axs[idx].set_visible(False)

    sns.despine()
    plt.tight_layout()

    save_path = get_save_path(f'{model_name.lower()}_similarity_vs_depth_combined.pdf', save_dir)
    fig.savefig(save_path, transparent=True, bbox_inches='tight')
    print(f"Saved: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return fig


# =============================================================================
# Main Analysis Pipeline
# =============================================================================

def run_analysis(save_dir: str = DEFAULT_SAVE_DIR, show_plots: bool = False):
    """Run the full RSA analysis pipeline."""
    ensure_save_dir(save_dir)

    # -------------------------------------------------------------------------
    # Setup environments
    # -------------------------------------------------------------------------
    print("Setting up environments...")
    dqn_map_path = '../sac/maps/meister_maze.txt'
    dqn_map_yaml_path = '../sac/maps/test_meister_maze.yaml'
    ppo_map_path = '../sac/maps/meister_trimmed.txt'
    ppo_map_yaml_path = '../sac/maps/test_meister_trimmed.yaml'

    dqn_env = key_door_env.KeyDoorEnv(map_ascii_path=dqn_map_path, map_yaml_path=dqn_map_yaml_path,
                                       representation="pixel", episode_timeout=200)
    dqn_env = visualisation_env.VisualisationEnv(dqn_env)

    ppo_env = key_door_env.KeyDoorEnv(map_ascii_path=ppo_map_path, map_yaml_path=ppo_map_yaml_path,
                                       representation="pixel", episode_timeout=200)
    ppo_env = visualisation_env.VisualisationEnv(ppo_env)

    # Get state shapes
    dqn_sample_state = dqn_env.reset_environment(train=False)
    dqn_state_shape = dqn_sample_state.shape[1:]
    dqn_num_actions = len(dqn_env.action_space)

    ppo_sample_state = ppo_env.reset_environment(train=False)
    ppo_state_shape = ppo_sample_state.shape[1:]
    ppo_num_actions = len(ppo_env.action_space)

    # -------------------------------------------------------------------------
    # Plot environment
    # -------------------------------------------------------------------------
    print("Plotting environment...")
    dqn_env_image = dqn_env._env._env_skeleton(rewards="state", agent="state", cue="state")
    white_pixels = np.where(np.all(dqn_env_image == [1, 1, 1], axis=-1))
    black_pixels = np.where(np.all(dqn_env_image == [0, 0, 0], axis=-1))
    dqn_env_image[white_pixels] = [0.0, 0.0, 0.0]
    dqn_env_image[black_pixels] = [1.0, 1.0, 1.0]

    plt.figure()
    plt.imshow(dqn_env_image, origin="lower", aspect='equal')
    plt.axis('off')
    plt.savefig(get_save_path('environment.png', save_dir), bbox_inches='tight', pad_inches=0)
    plt.savefig(get_save_path('environment.pdf', save_dir), bbox_inches='tight', pad_inches=0)
    if show_plots:
        plt.show()
    else:
        plt.close()

    # -------------------------------------------------------------------------
    # Setup positional environments and compute MDP abstractions
    # -------------------------------------------------------------------------
    print("Computing MDP abstractions...")
    dqn_pos_env = key_door_env.KeyDoorEnv(map_ascii_path=dqn_map_path, map_yaml_path=dqn_map_yaml_path,
                                           representation="agent_position", episode_timeout=200)
    ppo_pos_env = key_door_env.KeyDoorEnv(map_ascii_path=ppo_map_path, map_yaml_path=ppo_map_yaml_path,
                                           representation="agent_position", episode_timeout=200)

    dqn_state_id_mapping = {state: i for i, state in enumerate(dqn_pos_env.positional_state_space)}
    dqn_id_state_mapping = {i: state for i, state in enumerate(dqn_pos_env.positional_state_space)}
    ppo_state_id_mapping = {state: i for i, state in enumerate(ppo_pos_env.positional_state_space)}
    ppo_id_state_mapping = {i: state for i, state in enumerate(ppo_pos_env.positional_state_space)}

    S_dqn = len(dqn_pos_env.positional_state_space)
    A_dqn = len(dqn_pos_env.action_space)
    S_ppo = len(ppo_pos_env.positional_state_space)
    A_ppo = len(ppo_pos_env.action_space)

    P_dqn = np.zeros((S_dqn, A_dqn, S_dqn))
    R_dqn = np.zeros((S_dqn, A_dqn))
    P_ppo = np.zeros((S_ppo, A_ppo, S_ppo))
    R_ppo = np.zeros((S_ppo, A_ppo))

    dqn_reward_positions = list(dqn_pos_env._rewards.keys())
    ppo_reward_positions = list(ppo_pos_env._rewards.keys())

    for state in dqn_pos_env.positional_state_space:
        state_index = dqn_state_id_mapping[state]
        if state not in dqn_reward_positions:
            for action in dqn_pos_env.action_space:
                dqn_pos_env.reset_environment(train=True)
                dqn_pos_env.move_agent_to(state)
                reward, new_state = dqn_pos_env.step(action)
                new_state_index = dqn_state_id_mapping[new_state[:2]]
                P_dqn[state_index][action][new_state_index] = 1
                R_dqn[state_index][action] = reward

    for state in ppo_pos_env.positional_state_space:
        state_index = ppo_state_id_mapping[state]
        if state not in ppo_reward_positions:
            for action in ppo_pos_env.action_space:
                ppo_pos_env.reset_environment(train=True)
                ppo_pos_env.move_agent_to(state)
                reward, new_state = ppo_pos_env.step(action)
                new_state_index = ppo_state_id_mapping[new_state[:2]]
                P_ppo[state_index][action][new_state_index] = 1
                R_ppo[state_index][action] = reward

    # Compute state abstractions
    dqn_state_blocks, dqn_sa_blocks, dqn_state_label, dqn_sa_label, dqn_P_tilde, dqn_R_tilde, _ = utils.joint_state_action_abstraction(P_dqn, R_dqn)
    ppo_state_blocks, ppo_sa_blocks, ppo_state_label, ppo_sa_label, ppo_P_tilde, ppo_R_tilde, _ = utils.joint_state_action_abstraction(P_ppo, R_ppo)

    dqn_state_to_label = {dqn_id_state_mapping[i]: dqn_state_label[i] for i in range(len(dqn_state_label))}
    ppo_state_to_label = {ppo_id_state_mapping[i]: ppo_state_label[i] for i in range(len(ppo_state_label))}

    # Compute tree depth by number of direction changes (turns) from start
    print("Computing tree depth by turns...")
    # Get start positions from environment objects
    dqn_start_pos = tuple(dqn_pos_env._starting_xy)
    ppo_start_pos = tuple(ppo_pos_env._starting_xy)
    print(f"DQN start position: {dqn_start_pos}")
    print(f"PPO start position: {ppo_start_pos}")
    dqn_state_to_depth = compute_state_depth_by_turns(dqn_id_state_mapping, P_dqn, dqn_state_id_mapping, dqn_start_pos)
    ppo_state_to_depth = compute_state_depth_by_turns(ppo_id_state_mapping, P_ppo, ppo_state_id_mapping, ppo_start_pos)

    dqn_max_depth = max(dqn_state_to_depth.values()) if dqn_state_to_depth else 0
    ppo_max_depth = max(ppo_state_to_depth.values()) if ppo_state_to_depth else 0
    print(f"DQN: max tree depth (turns) = {dqn_max_depth}")
    print(f"PPO: max tree depth (turns) = {ppo_max_depth}")

    # Plot tree depth heatmap
    dqn_env.plot_heatmap_over_env(dqn_state_to_depth, save_name=get_save_path('tree_depth.pdf', save_dir))
    dqn_env.plot_heatmap_over_env(dqn_state_to_depth, save_name=get_save_path('tree_depth.png', save_dir))

    # Plot original state index (to verify state ordering)
    dqn_env.plot_heatmap_over_env(dqn_state_id_mapping, save_name=get_save_path('original_state_index.pdf', save_dir))
    dqn_env.plot_heatmap_over_env(dqn_state_id_mapping, save_name=get_save_path('original_state_index.png', save_dir))

    # Save colorbar for original state index
    fig_cb, ax_cb = plt.subplots(figsize=(2.0, 0.2))
    values = list(dqn_state_id_mapping.values())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    plt.savefig(get_save_path('original_state_index_colorbar.pdf', save_dir), dpi=300, bbox_inches='tight')
    plt.savefig(get_save_path('original_state_index_colorbar.png', save_dir), dpi=300, bbox_inches='tight')
    plt.close(fig_cb)

    # Plot state aggregation
    dqn_env.plot_heatmap_over_env(dqn_state_to_label, save_name=get_save_path('state_aggregation.pdf', save_dir))
    dqn_env.plot_heatmap_over_env(dqn_state_to_label, save_name=get_save_path('state_aggregation.png', save_dir))

    # -------------------------------------------------------------------------
    # Compute optimal policy using Dijkstra
    # -------------------------------------------------------------------------
    print("Computing optimal policies...")
    dqn_goal_mask = np.zeros(S_dqn, dtype=bool)
    for xy in dqn_reward_positions:
        dqn_goal_mask[dqn_state_id_mapping[xy]] = True
    ppo_goal_mask = np.zeros(S_ppo, dtype=bool)
    for xy in ppo_reward_positions:
        ppo_goal_mask[ppo_state_id_mapping[xy]] = True

    C_dqn = np.full((S_dqn, A_dqn), 1.0, dtype=float)
    C_ppo = np.full((S_ppo, A_ppo), 1.0, dtype=float)
    invalid = (P_dqn.sum(axis=2) == 0.0)
    C_dqn[invalid] = np.inf
    invalid = (P_ppo.sum(axis=2) == 0.0)
    C_ppo[invalid] = np.inf

    J_dqn, pi_dqn = dijkstra_policy(P_dqn, C_dqn, dqn_goal_mask)
    J_ppo, pi_ppo = dijkstra_policy(P_ppo, C_ppo, ppo_goal_mask)

    dqn_state_to_opt_policy = {dqn_id_state_mapping[i]: pi_dqn[i] for i in range(len(dqn_id_state_mapping)) if pi_dqn[i] >= 0}
    ppo_state_to_opt_policy = {ppo_id_state_mapping[i]: pi_ppo[i] for i in range(len(ppo_id_state_mapping)) if pi_ppo[i] >= 0}

    cmap = plt.get_cmap('viridis')
    dqn_env.plot_heatmap_over_env(dqn_state_to_opt_policy, save_name=get_save_path('optimal_policy.pdf', save_dir), colormap=cmap)
    dqn_env.plot_heatmap_over_env(dqn_state_to_opt_policy, save_name=get_save_path('optimal_policy.png', save_dir), colormap=cmap)

    # -------------------------------------------------------------------------
    # Create and load DQN models
    # -------------------------------------------------------------------------
    print("Loading DQN models...")
    dqn_results_path = "../sac/results/2025-06-10-21-13"

    dqn_model = dqn.DQN(
        sample_state=dqn_sample_state,
        num_actions=dqn_num_actions,
        learning_rate=0.0,
        discount_factor=0.99,
        exploration_rate=0.01,
        exploration_decay=1.0,
        batch_size=64,
        target_update_frequency=1000,
        replay_buffer_size=10000,
        burnin=1000,
        convolutional=True,
        optimistic_init=False
    )
    dqn_model_init = deepcopy(dqn_model)

    # Load trained DQN
    model_dict = torch.load(f"{dqn_results_path}/dqn_model_9950.pth", weights_only=False)
    dqn_model._net.load_state_dict(model_dict["model_state_dict"])
    dqn_model._net.eval()

    # -------------------------------------------------------------------------
    # Create and load PPO models
    # -------------------------------------------------------------------------
    print("Loading PPO models...")
    ppo_results_path = "../sac/seb_runs/best_ppo"

    ppo_model = ppo.PPO(
        sample_state=ppo_sample_state,
        num_actions=ppo_num_actions,
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
    ppo_init_model = deepcopy(ppo_model)

    # Load trained PPO
    model_dict = torch.load(f"{ppo_results_path}/ppo_model_4000000.pth", weights_only=False)
    ppo_model._net.load_state_dict(model_dict["model_state_dict"])
    ppo_model._net.eval()

    # -------------------------------------------------------------------------
    # Create and load SARSA-λ models
    # -------------------------------------------------------------------------
    print("Loading SARSA-λ models...")
    sarsa_results_path = "../sac/seb_runs/sarsa-run"

    sarsa_model = deep_sarsa.DeepSARSALambda(
        sample_state=ppo_sample_state,
        num_actions=ppo_num_actions,
        learning_rate=0.0001,
        discount_factor=0.99,
        exploration_rate=0.01,
        exploration_decay=1.0,
        lambda_=0.8,
        convolutional=True,
        optimistic_init=0.0,
    )
    sarsa_init_model = deepcopy(sarsa_model)

    # Load trained SARSA-λ
    model_dict = torch.load(f"{sarsa_results_path}/deep_sarsa_lambda_model_500.pth", weights_only=False)
    sarsa_model._net.load_state_dict(model_dict["model_state_dict"])
    sarsa_model._net.eval()

    # -------------------------------------------------------------------------
    # Extract representations and compute RSAs
    # -------------------------------------------------------------------------
    print("\nExtracting DQN init representations...")
    dqn_init_rsas = accumulate_dqn_representations(dqn_model_init, dqn_env, dqn_state_shape,
                                                    save_dir=save_dir, prefix='dqn_init', show=show_plots)

    print("\nExtracting DQN trained representations...")
    dqn_rsas = accumulate_dqn_representations(dqn_model, dqn_env, dqn_state_shape,
                                               save_dir=save_dir, prefix='dqn_trained', show=show_plots)

    print("\nExtracting PPO init representations...")
    ppo_init_rsas = accumulate_ppo_representations(ppo_init_model, ppo_env, ppo_state_shape,
                                                    save_dir=save_dir, prefix='ppo_init', show=show_plots)

    print("\nExtracting PPO trained representations...")
    ppo_rsas = accumulate_ppo_representations(ppo_model, ppo_env, ppo_state_shape,
                                               save_dir=save_dir, prefix='ppo_trained', show=show_plots)

    print("\nExtracting SARSA-λ init representations...")
    sarsa_init_rsas = accumulate_sarsa_representations(sarsa_init_model, ppo_env, ppo_state_shape,
                                                       save_dir=save_dir, prefix='sarsa_init', show=show_plots)

    print("\nExtracting SARSA-λ trained representations...")
    sarsa_rsas = accumulate_sarsa_representations(sarsa_model, ppo_env, ppo_state_shape,
                                                  save_dir=save_dir, prefix='sarsa_trained', show=show_plots)

    # -------------------------------------------------------------------------
    # Compute similarity metrics
    # -------------------------------------------------------------------------
    print("\nComputing similarity metrics...")

    def compute_sims_for_rsas(rsas, reprs_dict, state_to_label, state_id_mapping):
        sims = {}
        for layer in rsas:
            if layer in ['reprs', 'state_to_value', 'state_to_policy']:
                continue
            reprs = reprs_dict[layer]
            mean_off_diag = get_mean_off_diagonal(rsas[layer])
            within_label = get_within_label_similarity(reprs, state_to_label, state_id_mapping)
            sims[layer] = (mean_off_diag, within_label)
        return sims

    # MDP abstraction similarities
    dqn_hom_sims = compute_sims_for_rsas(dqn_rsas, dqn_rsas['reprs'], dqn_state_to_label, dqn_state_id_mapping)
    dqn_init_hom_sims = compute_sims_for_rsas(dqn_init_rsas, dqn_init_rsas['reprs'], dqn_state_to_label, dqn_state_id_mapping)
    ppo_hom_sims = compute_sims_for_rsas(ppo_rsas, ppo_rsas['reprs'], ppo_state_to_label, ppo_state_id_mapping)
    ppo_init_hom_sims = compute_sims_for_rsas(ppo_init_rsas, ppo_init_rsas['reprs'], ppo_state_to_label, ppo_state_id_mapping)

    # Policy abstraction similarities
    dqn_policy_sims = compute_sims_for_rsas(dqn_rsas, dqn_rsas['reprs'], dqn_state_to_opt_policy, dqn_state_id_mapping)
    dqn_init_policy_sims = compute_sims_for_rsas(dqn_init_rsas, dqn_init_rsas['reprs'], dqn_state_to_opt_policy, dqn_state_id_mapping)
    ppo_policy_sims = compute_sims_for_rsas(ppo_rsas, ppo_rsas['reprs'], ppo_state_to_opt_policy, ppo_state_id_mapping)
    ppo_init_policy_sims = compute_sims_for_rsas(ppo_init_rsas, ppo_init_rsas['reprs'], ppo_state_to_opt_policy, ppo_state_id_mapping)
    sarsa_hom_sims = compute_sims_for_rsas(sarsa_rsas, sarsa_rsas['reprs'], ppo_state_to_label, ppo_state_id_mapping)
    sarsa_init_hom_sims = compute_sims_for_rsas(sarsa_init_rsas, sarsa_init_rsas['reprs'], ppo_state_to_label, ppo_state_id_mapping)
    sarsa_policy_sims = compute_sims_for_rsas(sarsa_rsas, sarsa_rsas['reprs'], ppo_state_to_opt_policy, ppo_state_id_mapping)
    sarsa_init_policy_sims = compute_sims_for_rsas(sarsa_init_rsas, sarsa_init_rsas['reprs'], ppo_state_to_opt_policy, ppo_state_id_mapping)

    # -------------------------------------------------------------------------
    # Generate comparison plots
    # -------------------------------------------------------------------------
    print("\nGenerating comparison plots...")

    # Combined DQN/PPO/SARSA comparison
    plot_dqn_ppo_comparison(
        dqn_hom_sims, dqn_init_hom_sims, dqn_policy_sims, dqn_init_policy_sims,
        ppo_hom_sims, ppo_init_hom_sims, ppo_policy_sims, ppo_init_policy_sims,
        sarsa_hom_sims=sarsa_hom_sims, sarsa_init_hom_sims=sarsa_init_hom_sims,
        sarsa_policy_sims=sarsa_policy_sims, sarsa_init_policy_sims=sarsa_init_policy_sims,
        save_dir=save_dir
    )

    # Individual model plots
    plot_single_model_comparison(
        dqn_hom_sims, dqn_init_hom_sims, dqn_policy_sims, dqn_init_policy_sims,
        layer_labels=['State', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='DQN',
        save_dir=save_dir
    )

    plot_single_model_comparison(
        ppo_hom_sims, ppo_init_hom_sims, ppo_policy_sims, ppo_init_policy_sims,
        layer_labels=['State', 'Conv', 'FC1', 'Actor', 'Critic'],
        layer_keys=['state', 'conv', 'shared', 'actor', 'critic'],
        model_name='PPO',
        save_dir=save_dir
    )

    plot_single_model_comparison(
        sarsa_hom_sims, sarsa_init_hom_sims, sarsa_policy_sims, sarsa_init_policy_sims,
        layer_labels=['State', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='SARSA',
        save_dir=save_dir
    )

    # -------------------------------------------------------------------------
    # Depth-wise similarity analysis
    # -------------------------------------------------------------------------
    print("\nGenerating depth-wise similarity plots...")

    # DQN depth-wise analysis (trained model)
    plot_similarity_vs_depth(
        reprs_dict=dqn_rsas['reprs'],
        state_to_depth=dqn_state_to_depth,
        state_to_mdp_label=dqn_state_to_label,
        state_to_policy_label=dqn_state_to_opt_policy,
        state_id_mapping=dqn_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='DQN_trained',
        save_dir=save_dir,
        show=show_plots
    )

    # DQN combined (trained vs init)
    plot_similarity_vs_depth_combined(
        trained_reprs_dict=dqn_rsas['reprs'],
        init_reprs_dict=dqn_init_rsas['reprs'],
        state_to_depth=dqn_state_to_depth,
        state_to_mdp_label=dqn_state_to_label,
        state_to_policy_label=dqn_state_to_opt_policy,
        state_id_mapping=dqn_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='DQN',
        save_dir=save_dir,
        show=show_plots
    )

    # PPO depth-wise analysis (trained model)
    plot_similarity_vs_depth(
        reprs_dict=ppo_rsas['reprs'],
        state_to_depth=ppo_state_to_depth,
        state_to_mdp_label=ppo_state_to_label,
        state_to_policy_label=ppo_state_to_opt_policy,
        state_id_mapping=ppo_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'Actor', 'Critic'],
        layer_keys=['state', 'conv', 'shared', 'actor', 'critic'],
        model_name='PPO_trained',
        save_dir=save_dir,
        show=show_plots
    )

    # PPO combined (trained vs init)
    plot_similarity_vs_depth_combined(
        trained_reprs_dict=ppo_rsas['reprs'],
        init_reprs_dict=ppo_init_rsas['reprs'],
        state_to_depth=ppo_state_to_depth,
        state_to_mdp_label=ppo_state_to_label,
        state_to_policy_label=ppo_state_to_opt_policy,
        state_id_mapping=ppo_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'Actor', 'Critic'],
        layer_keys=['state', 'conv', 'shared', 'actor', 'critic'],
        model_name='PPO',
        save_dir=save_dir,
        show=show_plots
    )

    # SARSA-λ depth-wise analysis (trained model)
    plot_similarity_vs_depth(
        reprs_dict=sarsa_rsas['reprs'],
        state_to_depth=ppo_state_to_depth,
        state_to_mdp_label=ppo_state_to_label,
        state_to_policy_label=ppo_state_to_opt_policy,
        state_id_mapping=ppo_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='SARSA_trained',
        save_dir=save_dir,
        show=show_plots
    )

    # SARSA-λ combined (trained vs init)
    plot_similarity_vs_depth_combined(
        trained_reprs_dict=sarsa_rsas['reprs'],
        init_reprs_dict=sarsa_init_rsas['reprs'],
        state_to_depth=ppo_state_to_depth,
        state_to_mdp_label=ppo_state_to_label,
        state_to_policy_label=ppo_state_to_opt_policy,
        state_id_mapping=ppo_state_id_mapping,
        layer_labels=['Pixels', 'Conv', 'FC1', 'FC2'],
        layer_keys=['state', 'conv', 'shared_1', 'shared_2'],
        model_name='SARSA',
        save_dir=save_dir,
        show=show_plots
    )

    # -------------------------------------------------------------------------
    # Plot reference RSA matrices
    # -------------------------------------------------------------------------
    print("\nPlotting reference RSA matrices...")

    # Binary RSA for policy
    dqn_rsa_policy = compute_binary_rsa_policy(pi_dqn, len(dqn_state_to_label))
    dqn_rsa_mdp = compute_binary_rsa_labels(dqn_state_label, len(dqn_state_to_label))

    plot_reference_rsa_matrix(dqn_rsa_policy, "RSA Policy", save_name="rsa_policy.pdf",
                               save_dir=save_dir, show=show_plots)
    plot_reference_rsa_matrix(dqn_rsa_mdp, "RSA MDP", save_name="rsa_mdp.pdf",
                               save_dir=save_dir, show=show_plots)

    # RSA overlap
    product_rsa = dqn_rsa_mdp * dqn_rsa_policy
    plot_reference_rsa_matrix(product_rsa, "RSA Overlap", save_name="rsa_overlap.pdf",
                               save_dir=save_dir, show=show_plots)

    overlap_percentage = product_rsa.sum() / (len(product_rsa)**2) * 100
    print(f"\nRSA overlap percentage: {overlap_percentage:.2f}%")

    print(f"\nAll plots saved to: {save_dir}")

    if show_plots:
        plt.show()


if __name__ == '__main__':
    run_analysis(save_dir=DEFAULT_SAVE_DIR, show_plots=False)
