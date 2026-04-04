"""
Simple test script for compute_abstract_mdp_depth function.
Modify compute_abstract_mdp_depth and run this script to visualize the result.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple
from collections import deque

from key_door import key_door_env, visualisation_env
from sac import utils


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




def main():
    # Setup environment
    print("Setting up environment...")
    map_path = '../sac/maps/meister_maze.txt'
    map_yaml_path = '../sac/maps/test_meister_maze.yaml'

    env = key_door_env.KeyDoorEnv(
        map_ascii_path=map_path,
        map_yaml_path=map_yaml_path,
        representation="pixel",
        episode_timeout=200
    )
    env = visualisation_env.VisualisationEnv(env)

    pos_env = key_door_env.KeyDoorEnv(
        map_ascii_path=map_path,
        map_yaml_path=map_yaml_path,
        representation="agent_position",
        episode_timeout=200
    )

    # Build state mappings
    state_id_mapping = {state: i for i, state in enumerate(pos_env.positional_state_space)}
    id_state_mapping = {i: state for i, state in enumerate(pos_env.positional_state_space)}

    S = len(pos_env.positional_state_space)
    A = len(pos_env.action_space)

    # Build transition matrix
    print("Building transition matrix...")
    P = np.zeros((S, A, S))
    R = np.zeros((S, A))
    reward_positions = list(pos_env._rewards.keys())

    for state in pos_env.positional_state_space:
        state_index = state_id_mapping[state]
        if state not in reward_positions:
            for action in pos_env.action_space:
                pos_env.reset_environment(train=True)
                pos_env.move_agent_to(state)
                reward, new_state = pos_env.step(action)
                new_state_index = state_id_mapping[new_state[:2]]
                P[state_index][action][new_state_index] = 1
                R[state_index][action] = reward

    # Compute state abstraction
    print("Computing state abstraction...")
    state_blocks, sa_blocks, state_label, sa_label, P_tilde, R_tilde, _ = utils.joint_state_action_abstraction(P, R)

    state_to_label = {id_state_mapping[i]: state_label[i] for i in range(len(state_label))}

    # Compute tree depth by number of direction changes (turns)
    print("Computing tree depth by turns...")
    start_pos = tuple(pos_env._starting_xy)  # Get start position from environment
    print(f"Start position: {start_pos}")
    state_to_depth = compute_state_depth_by_turns(id_state_mapping, P, state_id_mapping, start_pos)

    # Print summary
    K = P_tilde.shape[0]
    max_depth = max(state_to_depth.values()) if state_to_depth else 0
    print(f"\nAbstract states: {K}")
    print(f"Max depth value: {max_depth}")
    print(f"\nDepth distribution:")
    for d in sorted(set(state_to_depth.values())):
        count = sum(1 for v in state_to_depth.values() if v == d)
        print(f"  Depth {d}: {count} states")

    # Visualize
    print("\nSaving visualizations...")
    env.plot_heatmap_over_env(state_to_depth, save_name='test_tree_depth.png')
    env.plot_heatmap_over_env(state_to_label, save_name='test_abstract_labels.png')
    env.plot_heatmap_over_env(state_id_mapping, save_name='test_original_index.png')

    # Save colorbar for depth
    fig_cb, ax_cb = plt.subplots(figsize=(2.0, 0.2))
    values = list(state_to_depth.values())
    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(values), vmax=max(values)))
    cbar = fig_cb.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cbar.ax.tick_params(labelsize=8)
    plt.savefig('test_tree_depth_colorbar.png', dpi=300, bbox_inches='tight')
    plt.close(fig_cb)

    print("\nSaved:")
    print("  - test_tree_depth.png (tree level for each state)")
    print("  - test_tree_depth_colorbar.png")
    print("  - test_abstract_labels.png (abstract state labels)")
    print("  - test_original_index.png (original state indices)")


if __name__ == '__main__':
    main()
