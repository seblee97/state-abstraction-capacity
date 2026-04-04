"""
Plot results from inverted pendulum symmetry analysis.

Usage:
    python plot_inverted_pendulum.py --results results/ppo_results.pkl
    python plot_inverted_pendulum.py --results results/dqn_results.pkl results/ppo_results.pkl
"""

import argparse
import pickle
import os
import re

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

# Plot style settings (matching llm_tree/plot_rsa_results.py)
COLOR_SIMILAR = '#EE7733'  # TealEE7733
COLOR_DISSIMILAR = '#009988'  # Orange


def render_cartpole_state(ax, state, title=None):
    """
    Render a CartPole state visually.

    Args:
        ax: Matplotlib axis to draw on
        state: [x, x_dot, theta, theta_dot] - CartPole state vector
        title: Optional title for the subplot
    """
    x, x_dot, theta, theta_dot = state

    # CartPole dimensions (matching gymnasium CartPole-v1)
    cart_width = 0.5
    cart_height = 0.3
    pole_length = 1.0

    # Scale x position for display (cart position is in [-2.4, 2.4])
    display_x = x

    # Cart body
    cart = mpatches.FancyBboxPatch(
        (display_x - cart_width / 2, 0),
        cart_width, cart_height,
        boxstyle="round,pad=0.02",
        facecolor='#2E86AB',
        edgecolor='black',
        linewidth=1.5
    )
    ax.add_patch(cart)

    # Cart wheels
    wheel_radius = 0.08
    left_wheel = mpatches.Circle(
        (display_x - cart_width / 4, 0),
        wheel_radius,
        facecolor='#1a1a1a',
        edgecolor='black'
    )
    right_wheel = mpatches.Circle(
        (display_x + cart_width / 4, 0),
        wheel_radius,
        facecolor='#1a1a1a',
        edgecolor='black'
    )
    ax.add_patch(left_wheel)
    ax.add_patch(right_wheel)

    # Pole (angle theta is from vertical, positive = clockwise)
    pole_base_x = display_x
    pole_base_y = cart_height
    pole_end_x = pole_base_x + pole_length * np.sin(theta)
    pole_end_y = pole_base_y + pole_length * np.cos(theta)

    # Draw pole as a thick line
    ax.plot([pole_base_x, pole_end_x], [pole_base_y, pole_end_y],
            color='#E94F37', linewidth=8, solid_capstyle='round', zorder=5)

    # Pole pivot point
    pivot = mpatches.Circle(
        (pole_base_x, pole_base_y),
        0.05,
        facecolor='#1a1a1a',
        edgecolor='black',
        zorder=6
    )
    ax.add_patch(pivot)

    # Velocity arrows (with high z-order to appear on top)
    arrow_scale = 0.8

    # Cart velocity arrow (x_dot)
    if abs(x_dot) > 0.1:
        ax.annotate('', xy=(display_x + x_dot * arrow_scale, cart_height / 2),
                   xytext=(display_x, cart_height / 2),
                   arrowprops=dict(arrowstyle='->', color='#44AA44', lw=2,
                                   mutation_scale=12), zorder=10)

    # Angular velocity arrow (theta_dot) - curved arc at pivot with arrowhead
    if abs(theta_dot) > 0.1:
        arc_radius = 0.7
        arc_center = (pole_base_x, pole_base_y)
        # Draw a small arc to indicate rotation direction
        theta_start = np.degrees(-theta)
        theta_end = theta_start + np.sign(theta_dot) * 50
        arc = mpatches.Arc(arc_center, arc_radius * 2, arc_radius * 2,
                          angle=90, theta1=min(theta_start, theta_end),
                          theta2=max(theta_start, theta_end),
                          color='#AA44AA', lw=2, zorder=10)
        ax.add_patch(arc)

        # Add arrowhead at the end of the arc
        # Calculate position at the end of the arc
        arc_end_angle = np.radians(90 + theta_end)
        arrow_x = arc_center[0] + arc_radius * np.cos(arc_end_angle)
        arrow_y = arc_center[1] + arc_radius * np.sin(arc_end_angle)

        # Calculate tangent direction for the arrowhead
        tangent_angle = arc_end_angle + np.sign(theta_dot) * np.pi / 2
        arrow_dx = 0.15 * np.cos(tangent_angle)
        arrow_dy = 0.15 * np.sin(tangent_angle)

        ax.annotate('', xy=(arrow_x + arrow_dx, arrow_y + arrow_dy),
                   xytext=(arrow_x, arrow_y),
                   arrowprops=dict(arrowstyle='->', color='#AA44AA', lw=2,
                                   mutation_scale=10), zorder=10)

    # Ground line
    ax.axhline(y=-0.05, color='#666666', linewidth=2, linestyle='-')
    ax.axhline(y=-0.1, color='#888888', linewidth=8, alpha=0.3)

    # Set axis properties
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')

    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')


def render_symmetric_state_pairs(states, output_dir, n_pairs=5, symmetry_type='policy', seed=42):
    """
    Render pairs of symmetric states and save them to a subdirectory.

    Args:
        states: Array of states [N, 4] where each state is [x, x_dot, theta, theta_dot]
        output_dir: Base output directory
        n_pairs: Number of symmetric pairs to render
        symmetry_type: 'policy' for policy symmetry (s, -s), 'mdp' for MDP homomorphism
        seed: Random seed for selecting pairs
    """
    np.random.seed(seed)

    # Create subdirectory for symmetric state visualizations
    subdir = os.path.join(output_dir, f'symmetric_states_{symmetry_type}')
    os.makedirs(subdir, exist_ok=True)

    print(f"\nRendering {n_pairs} symmetric state pairs ({symmetry_type} symmetry)...")
    print(f"Output directory: {subdir}")

    # For policy/MDP symmetry in CartPole, s and -s have the same optimal action
    # because the dynamics and reward are symmetric: f(-s, a) = -f(s, flip(a))

    # Select random states (avoiding near-zero states which are trivially symmetric)
    valid_indices = []
    for i, s in enumerate(states):
        # Skip states that are too close to origin (trivially symmetric)
        if np.linalg.norm(s) > 0.3:
            valid_indices.append(i)

    if len(valid_indices) < n_pairs:
        print(f"Warning: Only {len(valid_indices)} valid states found, using all of them")
        n_pairs = len(valid_indices)

    selected_indices = np.random.choice(valid_indices, size=n_pairs, replace=False)

    for pair_idx, state_idx in enumerate(selected_indices):
        s1 = states[state_idx]
        s2 = -s1  # Symmetric state under sign flip

        # Create figure with two subplots side by side (total width 3 inches)
        fig, axes = plt.subplots(1, 2, figsize=(3, 1.5))

        # Render original state
        render_cartpole_state(axes[0], s1, title='State s')

        # Render symmetric state
        render_cartpole_state(axes[1], s2, title='State -s')

        plt.tight_layout()

        # Save figure
        output_path = os.path.join(subdir, f'pair_{pair_idx + 1:02d}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {output_path}")

    # Also create a combined figure showing all pairs
    # Each pair takes 3 inches width, arrange in rows
    n_cols = min(n_pairs, 2)  # 2 pairs per row max
    n_rows = (n_pairs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols * 2, figsize=(3 * n_cols, 1.5 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = axes.reshape(1, -1)
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for pair_idx, state_idx in enumerate(selected_indices):
        row = pair_idx // n_cols
        col = (pair_idx % n_cols) * 2

        s1 = states[state_idx]
        s2 = -s1

        render_cartpole_state(axes[row, col], s1, title='s')
        render_cartpole_state(axes[row, col + 1], s2, title='-s')

    # Hide unused subplots
    total_slots = n_rows * n_cols
    for pair_idx in range(n_pairs, total_slots):
        row = pair_idx // n_cols
        col = (pair_idx % n_cols) * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    plt.tight_layout()

    combined_path = os.path.join(subdir, 'all_pairs_combined.png')
    plt.savefig(combined_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved combined figure: {combined_path}")

    return subdir


def split_actor_critic(layer_names, layer_scores):
    """Split layers and scores into actor and critic components.

    Actor: policy_net, action_net, q_net (for DQN)
    Critic: value_net (mlp_extractor.value_net and final value_net)

    Returns:
        (actor_names, actor_scores, critic_names, critic_scores)
    """
    actor_indices = []
    critic_indices = []

    for i, name in enumerate(layer_names):
        # Critic: value network layers
        if "value_net" in name:
            critic_indices.append(i)
        # Actor: policy network, action head, or Q-network (DQN)
        elif "policy_net" in name or "action_net" in name or "q_net" in name:
            actor_indices.append(i)
        else:
            # Default to actor for unknown layers
            actor_indices.append(i)

    actor_names = [layer_names[i] for i in actor_indices]
    critic_names = [layer_names[i] for i in critic_indices]

    actor_scores = {key: [scores[i] for i in actor_indices] for key, scores in layer_scores.items()}
    critic_scores = {key: [scores[i] for i in critic_indices] for key, scores in layer_scores.items()}

    return actor_names, actor_scores, critic_names, critic_scores


def filter_post_activation_layers(layer_names, layer_scores, skip_value_head=False):
    """Filter to only post-activation (ReLU) layers and output heads.

    Skips Linear layers when a corresponding ReLU follows.
    Keeps output heads (action_net, value_net unless skip_value_head=True).

    Args:
        layer_names: List of layer names
        layer_scores: Dict of score lists
        skip_value_head: If True, skip the final value_net output head (for critic plots)

    Returns:
        (filtered_names, filtered_scores)
    """
    # Identify which layers to keep:
    # - ReLU layers (odd indices in standard MLPs)
    # - Output heads (action_net, value_net)
    keep_indices = []

    for i, name in enumerate(layer_names):
        # Skip value head if requested
        if skip_value_head and "value_net" in name and "mlp_extractor" not in name:
            continue

        # Always keep action output head
        if "action_net" in name or ("value_net" in name and "mlp_extractor" not in name):
            keep_indices.append(i)
            continue

        # Extract layer index to determine if Linear (even) or ReLU (odd)
        match = re.search(r'\.(\d+)$', name)
        if match:
            layer_idx = int(match.group(1))
            # Keep ReLU layers (odd indices) - these are post-activation
            if layer_idx % 2 == 1:
                keep_indices.append(i)
        else:
            # No index found - keep it (likely an output head)
            keep_indices.append(i)

    filtered_names = [layer_names[i] for i in keep_indices]
    filtered_scores = {key: [scores[i] for i in keep_indices] for key, scores in layer_scores.items()}

    return filtered_names, filtered_scores


def plot_panel(scores, ax, layers, title, mean_reward, show_ylabel=True, show_xlabel=True, show_legend=True, show_title=True, rotate_labels=False):
    """Plot a single panel showing neural similarity across layers.

    Args:
        scores: Array where each entry is (similar_mean, dissimilar_mean, ...).
                May optionally include (similar_std, dissimilar_std, similar_count, dissimilar_count)
                for error bars.
    """
    x = np.arange(len(layers))
    similar = [s[0] for s in scores]
    dissimilar = [s[1] for s in scores]

    # Check if SEM data is available (6-tuple format)
    has_sem = len(scores) > 0 and len(scores[0]) >= 6
    if has_sem:
        # Compute SEM from std and count
        similar_sem = [s[2] / np.sqrt(s[4]) if s[4] > 0 else 0 for s in scores]
        dissimilar_sem = [s[3] / np.sqrt(s[5]) if s[5] > 0 else 0 for s in scores]

        ax.errorbar(x, similar, yerr=similar_sem, fmt='s-', color=COLOR_SIMILAR,
                    label="Similar States" if show_legend else None, markersize=4, capsize=2, capthick=1)
        ax.errorbar(x, dissimilar, yerr=dissimilar_sem, fmt='s--', color=COLOR_DISSIMILAR,
                    label="Dissimilar States" if show_legend else None, markersize=4, capsize=2, capthick=1)
    else:
        ax.plot(x, similar, 's-', color=COLOR_SIMILAR, label="Similar States" if show_legend else None, markersize=4)
        ax.plot(x, dissimilar, 's--', color=COLOR_DISSIMILAR, label="Dissimilar States" if show_legend else None, markersize=4)

    if show_ylabel:
        ax.set_ylabel("Cosine Sim.", fontsize=8)
    ax.set_xticks(x)
    ax.tick_params(axis='both', labelsize=8)

    # Generate short descriptive layer labels (simplified: L1, L2, Act, Val)
    labels = []
    layer_count = 0
    for l in layers:
        if "action_net" in l:
            labels.append("Act")
        elif "value_net" in l and "mlp_extractor" not in l:
            labels.append("Val")
        else:
            layer_count += 1
            labels.append(f"L{layer_count}")

    ax.set_ylim(-1.1, 1.1)
    if show_xlabel:
        if rotate_labels:
            ax.set_xticklabels(labels, fontsize=8, rotation=45, ha='right')
        else:
            ax.set_xticklabels(labels, fontsize=8)
    else:
        ax.set_xticklabels([])
    if show_legend:
        ax.legend(fontsize=6, frameon=False)
    if show_title:
        ax.set_title(title, fontsize=8)


def load_results(filepath):
    """Load results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def plot_single_algorithm(results, output_path):
    """Plot results for a single algorithm."""
    algorithm = results['algorithm'].upper()
    layer_names = results['layer_names']
    layer_scores = results['layer_scores']
    mean_reward = results['mean_reward']

    print(f"\nPlotting results for {algorithm}...")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # State Symmetry
    m_state = np.array(layer_scores["state"])
    plot_panel(m_state, axes[0], layer_names, f"{algorithm} State Symmetry", mean_reward)

    # LQR Action Symmetry
    m_lqr = np.array(layer_scores["lqr"])
    plot_panel(m_lqr, axes[1], layer_names, f"{algorithm} LQR", mean_reward)

    # Model Action Symmetry
    m_model = np.array(layer_scores["model"])
    plot_panel(m_model, axes[2], layer_names, f"{algorithm} Model Policy", mean_reward)

    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")


def save_individual_panel(scores, layers, title, output_path,
                          show_ylabel=True, show_xlabel=True, show_legend=True, show_title=True, rotate_labels=False):
    """Save a single panel as an individual file."""
    fig, ax = plt.subplots(figsize=(2, 1.5))
    plot_panel(scores, ax, layers, title, mean_reward=0,
               show_ylabel=show_ylabel, show_xlabel=show_xlabel, show_legend=show_legend, show_title=show_title,
               rotate_labels=rotate_labels)
    sns.despine()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Individual panel saved to: {output_path}")


def plot_comparison(results_list, output_path, save_panels=True):
    """Plot comparison of multiple algorithms.

    Layout: 2 rows (model policy, state symmetry) x (2 * n_algorithms) columns.
    For each algorithm: Actor column | Critic column (if available).
    DQN only has actor (Q-network), PPO has both actor and critic.

    Saves both full and simplified combined comparison plots (3.4 x 3 inches).
    """
    n_algorithms = len(results_list)

    print(f"\nPlotting comparison of {n_algorithms} algorithms...")

    # Check which algorithms have critic networks
    has_critic = []
    for results in results_list:
        layer_names = results['layer_names']
        has_critic.append(any("value_net" in name for name in layer_names))

    # Calculate total columns: 2 per algorithm with critic, 1 per algorithm without
    n_cols = sum(2 if hc else 1 for hc in has_critic)

    # Get output directory for individual panels
    output_dir = os.path.dirname(output_path) or '.'

    # Prepare data for all algorithms (full and simplified)
    plot_data = []
    for alg_idx, results in enumerate(results_list):
        algorithm = results['algorithm'].upper()
        layer_names = results['layer_names']
        layer_scores = results['layer_scores']
        mean_reward = results['mean_reward']

        # Split into actor and critic
        actor_names, actor_scores, critic_names, critic_scores = split_actor_critic(layer_names, layer_scores)

        # Title: just algorithm name for DQN (no actor/critic), "Actor" suffix for PPO
        actor_title = algorithm if not has_critic[alg_idx] else f"{algorithm} Actor"

        # Get simplified versions
        simp_actor_names, simp_actor_scores = filter_post_activation_layers(actor_names, actor_scores)
        simp_critic_names, simp_critic_scores = filter_post_activation_layers(critic_names, critic_scores, skip_value_head=True)

        plot_data.append({
            'algorithm': algorithm,
            'has_critic': has_critic[alg_idx],
            'actor_title': actor_title,
            'actor_names': actor_names,
            'actor_scores': actor_scores,
            'critic_names': critic_names,
            'critic_scores': critic_scores,
            'simp_actor_names': simp_actor_names,
            'simp_actor_scores': simp_actor_scores,
            'simp_critic_names': simp_critic_names,
            'simp_critic_scores': simp_critic_scores,
            'mean_reward': mean_reward,
        })

    # Helper function to create combined plot
    def create_combined_plot(use_simplified):
        fig, axes = plt.subplots(2, n_cols, figsize=(3.4, 3), sharey=True)

        # Handle edge cases
        if n_cols == 1:
            axes = axes.reshape(2, 1)

        col_idx = 0
        for alg_idx, data in enumerate(plot_data):
            is_left_col = (col_idx == 0)
            is_last_col = (col_idx + (2 if data['has_critic'] else 1) >= n_cols)

            # Select full or simplified data
            if use_simplified:
                actor_names = data['simp_actor_names']
                actor_scores = data['simp_actor_scores']
                critic_names = data['simp_critic_names']
                critic_scores = data['simp_critic_scores']
            else:
                actor_names = data['actor_names']
                actor_scores = data['actor_scores']
                critic_names = data['critic_names']
                critic_scores = data['critic_scores']

            # Actor column
            if actor_names:
                m_model_actor = np.array(actor_scores["model"])
                m_state_actor = np.array(actor_scores["state"])

                # Row 0: Model Action Symmetry (Actor) - no x labels
                plot_panel(m_model_actor, axes[0, col_idx], actor_names,
                           data['actor_title'], data['mean_reward'],
                           show_ylabel=is_left_col, show_xlabel=False,
                           show_legend=False, show_title=True, rotate_labels=True)

                # Row 1: State Symmetry (Actor) - with rotated x labels
                plot_panel(m_state_actor, axes[1, col_idx], actor_names,
                           "", data['mean_reward'],
                           show_ylabel=is_left_col, show_xlabel=True,
                           show_legend=(is_last_col and not data['has_critic']), show_title=False,
                           rotate_labels=True)

                col_idx += 1

            # Critic column (only if available)
            if data['has_critic'] and critic_names:
                m_model_critic = np.array(critic_scores["model"])
                m_state_critic = np.array(critic_scores["state"])

                # Row 0: Model Action Symmetry (Critic) - no x labels
                plot_panel(m_model_critic, axes[0, col_idx], critic_names,
                           f"{data['algorithm']} Critic", data['mean_reward'],
                           show_ylabel=False, show_xlabel=False,
                           show_legend=False, show_title=True, rotate_labels=True)

                # Row 1: State Symmetry (Critic) - with rotated x labels
                plot_panel(m_state_critic, axes[1, col_idx], critic_names,
                           "", data['mean_reward'],
                           show_ylabel=False, show_xlabel=True,
                           show_legend=is_last_col, show_title=False,
                           rotate_labels=True)

                col_idx += 1

        sns.despine()
        plt.tight_layout()
        return fig

    # Save full combined plot as PDF
    fig_full = create_combined_plot(use_simplified=False)
    base_path = output_path.rsplit('.', 1)[0]
    full_path = f"{base_path}_combined.pdf"
    fig_full.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close(fig_full)
    print(f"Combined plot (full) saved to: {full_path}")

    # Save simplified combined plot as PDF
    fig_simp = create_combined_plot(use_simplified=True)
    simp_path = f"{base_path}_combined_simplified.pdf"
    fig_simp.savefig(simp_path, dpi=300, bbox_inches='tight')
    plt.close(fig_simp)
    print(f"Combined plot (simplified) saved to: {simp_path}")

    # Save individual panels
    if save_panels:
        for data in plot_data:
            algorithm = data['algorithm']
            actor_title = data['actor_title']

            # Full actor panels
            if data['actor_names']:
                m_model = np.array(data['actor_scores']["model"])
                m_state = np.array(data['actor_scores']["state"])
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_actor_model_policy.pdf")
                save_individual_panel(m_model, data['actor_names'], actor_title, panel_path,
                                      show_ylabel=True, show_xlabel=False, show_legend=False, show_title=True)
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_actor_state_symmetry.pdf")
                save_individual_panel(m_state, data['actor_names'], "", panel_path,
                                      show_ylabel=True, show_xlabel=True, show_legend=False, show_title=False, rotate_labels=True)

            # Simplified actor panels
            if data['simp_actor_names']:
                m_model = np.array(data['simp_actor_scores']["model"])
                m_state = np.array(data['simp_actor_scores']["state"])
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_actor_model_policy_simplified.pdf")
                save_individual_panel(m_model, data['simp_actor_names'], actor_title, panel_path,
                                      show_ylabel=True, show_xlabel=False, show_legend=False, show_title=True)
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_actor_state_symmetry_simplified.pdf")
                save_individual_panel(m_state, data['simp_actor_names'], "", panel_path,
                                      show_ylabel=True, show_xlabel=True, show_legend=False, show_title=False, rotate_labels=True)

            # Full critic panels
            if data['has_critic'] and data['critic_names']:
                m_model = np.array(data['critic_scores']["model"])
                m_state = np.array(data['critic_scores']["state"])
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_critic_model_policy.pdf")
                save_individual_panel(m_model, data['critic_names'], f"{algorithm} Critic", panel_path,
                                      show_ylabel=True, show_xlabel=False, show_legend=False, show_title=True)
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_critic_state_symmetry.pdf")
                save_individual_panel(m_state, data['critic_names'], "", panel_path,
                                      show_ylabel=True, show_xlabel=True, show_legend=False, show_title=False, rotate_labels=True)

            # Simplified critic panels
            if data['has_critic'] and data['simp_critic_names']:
                m_model = np.array(data['simp_critic_scores']["model"])
                m_state = np.array(data['simp_critic_scores']["state"])
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_critic_model_policy_simplified.pdf")
                save_individual_panel(m_model, data['simp_critic_names'], f"{algorithm} Critic", panel_path,
                                      show_ylabel=True, show_xlabel=False, show_legend=False, show_title=True)
                panel_path = os.path.join(output_dir, f"{algorithm.lower()}_critic_state_symmetry_simplified.pdf")
                save_individual_panel(m_state, data['simp_critic_names'], "", panel_path,
                                      show_ylabel=True, show_xlabel=True, show_legend=False, show_title=False, rotate_labels=True)


def generate_sample_states_for_visualization(n_states=50, seed=42):
    """
    Generate a set of sample CartPole states for visualization.

    Creates states with varying positions, velocities, and pole angles
    that clearly demonstrate the symmetry.

    Returns:
        Array of states [N, 4] where each state is [x, x_dot, theta, theta_dot]
    """
    np.random.seed(seed)

    states = []

    # Generate states with different characteristics
    # 1. States with pole tilted right (positive theta)
    for _ in range(n_states // 4):
        x = np.random.uniform(-1.5, 1.5)
        x_dot = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(0.05, 0.3)  # 3-17 degrees right
        theta_dot = np.random.uniform(-1.0, 1.0)
        states.append([x, x_dot, theta, theta_dot])

    # 2. States with pole tilted left (negative theta)
    for _ in range(n_states // 4):
        x = np.random.uniform(-1.5, 1.5)
        x_dot = np.random.uniform(-1.0, 1.0)
        theta = np.random.uniform(-0.3, -0.05)  # 3-17 degrees left
        theta_dot = np.random.uniform(-1.0, 1.0)
        states.append([x, x_dot, theta, theta_dot])

    # 3. States with cart moving right
    for _ in range(n_states // 4):
        x = np.random.uniform(0.3, 1.5)
        x_dot = np.random.uniform(0.5, 2.0)
        theta = np.random.uniform(-0.2, 0.2)
        theta_dot = np.random.uniform(-1.0, 1.0)
        states.append([x, x_dot, theta, theta_dot])

    # 4. States with cart moving left
    for _ in range(n_states // 4):
        x = np.random.uniform(-1.5, -0.3)
        x_dot = np.random.uniform(-2.0, -0.5)
        theta = np.random.uniform(-0.2, 0.2)
        theta_dot = np.random.uniform(-1.0, 1.0)
        states.append([x, x_dot, theta, theta_dot])

    return np.array(states)


def main():
    parser = argparse.ArgumentParser(
        description='Plot symmetry analysis results for CartPole')
    parser.add_argument('--results', type=str, nargs='+', required=True,
                        help='Path(s) to results pickle file(s)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for plot (default: auto-generated)')
    parser.add_argument('--render-states', action='store_true',
                        help='Render symmetric state pairs visually')
    parser.add_argument('--n-pairs', type=int, default=5,
                        help='Number of symmetric pairs to render (default: 5)')
    parser.add_argument('--symmetry-type', type=str, default='policy',
                        choices=['policy', 'mdp'],
                        help='Type of symmetry to visualize (default: policy)')

    args = parser.parse_args()

    # Load all results
    results_list = []
    for results_path in args.results:
        if not os.path.exists(results_path):
            print(f"Error: Results file not found: {results_path}")
            continue
        results_list.append(load_results(results_path))

    if not results_list:
        print("Error: No valid results files found")
        return

    # Generate output path if not provided
    if args.output is None:
        if len(results_list) == 1:
            algorithm = results_list[0]['algorithm']
            output_path = f'inverted_pendulum_{algorithm}_symmetry.png'
        else:
            algorithms = '_'.join([r['algorithm'] for r in results_list])
            output_path = f'inverted_pendulum_{algorithms}_comparison.png'
    else:
        output_path = args.output

    # Plot
    if len(results_list) == 1:
        plot_single_algorithm(results_list[0], output_path)
    else:
        plot_comparison(results_list, output_path)

    # Render symmetric state pairs if requested
    if args.render_states:
        # Generate sample states for visualization
        # Use a range of interesting states that show the symmetry clearly
        sample_states = generate_sample_states_for_visualization()
        output_dir = os.path.dirname(output_path) or '.'
        render_symmetric_state_pairs(
            sample_states,
            output_dir,
            n_pairs=args.n_pairs,
            symmetry_type=args.symmetry_type
        )

    print("\nPlotting complete!")


if __name__ == '__main__':
    main()
