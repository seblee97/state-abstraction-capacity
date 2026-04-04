"""
Plot RSA analysis results from LLM evaluation.

This script loads results from evaluate_models.py (with --analyze-representations)
and generates publication-quality plots comparing:
1. Global vs within-label similarities across layers
2. MDP homomorphism vs optimal policy abstractions
3. Comparisons across different graph representations
4. Comparisons across different models

Usage:
    python plot_rsa_results.py                          # Plot small model results
    python plot_rsa_results.py --size large             # Plot large model results
    python plot_rsa_results.py --results-dir my_results # Custom results directory
    python plot_rsa_results.py --models qwen deepseek   # Plot only specific models
"""

import json
import argparse
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

# Plot style settings
COLOR_GLOBAL = '#009988'  # Teal
COLOR_WITHIN_MDP = '#EE7733'  # Orange
COLOR_WITHIN_POLICY = '#CC3311'  # Red
COLOR_WITHIN_DEPTH = '#0077BB'  # Blue
MARKER_SIZE = 40
FIGURE_DPI = 150


def load_results(results_dir: str, model_size: str = "small"):
    """Load results from the summary file for the specified model size."""
    summary_file = Path(results_dir) / f"summary_{model_size}.json"
    if not summary_file.exists():
        raise FileNotFoundError(
            f"Results file not found: {summary_file}\n"
            f"Please run evaluate_models.py with --analyze-representations --size {model_size} first."
        )

    with open(summary_file, 'r') as f:
        return json.load(f)


def extract_layer_metrics(rsa_results: dict, rep_name: str):
    """
    Extract metrics for each layer from RSA results.

    Args:
        rsa_results: Dict[rep_name, Dict[layer_name, metrics]]
        rep_name: Name of the representation to extract

    Returns:
        layer_indices: List of layer indices in order
        metrics: Dict with lists of metrics per layer (including standard errors if available)
    """
    if rep_name not in rsa_results:
        return None, None

    layer_data = rsa_results[rep_name]

    # Sort layers by their index
    def layer_sort_key(name):
        # Extract number from layer name like "layer_0", "layer_10", etc.
        try:
            return int(name.split('_')[1])
        except (IndexError, ValueError):
            return 0

    layer_names = sorted(layer_data.keys(), key=layer_sort_key)
    layer_indices = [layer_sort_key(name) for name in layer_names]

    metrics = {
        'global_similarity': [],
        'within_mdp_label_similarity': [],
        'within_policy_label_similarity': [],
        'within_depth_similarity': [],
        'participation_ratio': [],
        # Standard errors (if available)
        'global_similarity_se': [],
        'within_mdp_label_similarity_se': [],
        'within_policy_label_similarity_se': [],
        'within_depth_similarity_se': [],
    }

    for layer in layer_names:
        layer_metrics = layer_data[layer]
        metrics['global_similarity'].append(layer_metrics.get('global_similarity', 0))
        metrics['within_mdp_label_similarity'].append(layer_metrics.get('within_mdp_label_similarity', 0))
        metrics['within_policy_label_similarity'].append(layer_metrics.get('within_policy_label_similarity', 0))
        metrics['within_depth_similarity'].append(layer_metrics.get('within_depth_similarity', 0))
        metrics['participation_ratio'].append(layer_metrics.get('participation_ratio', 0))
        # Standard errors default to 0 if not available
        metrics['global_similarity_se'].append(layer_metrics.get('global_similarity_se', 0))
        metrics['within_mdp_label_similarity_se'].append(layer_metrics.get('within_mdp_label_similarity_se', 0))
        metrics['within_policy_label_similarity_se'].append(layer_metrics.get('within_policy_label_similarity_se', 0))
        metrics['within_depth_similarity_se'].append(layer_metrics.get('within_depth_similarity_se', 0))

    return layer_indices, metrics


def plot_single_model_representation(
    layer_indices: list,
    metrics: dict,
    title: str,
    save_path: str = None
):
    """
    Plot RSA metrics for a single model and representation.

    Creates a single plot with four curves as a function of layer:
    - Global similarity
    - Within-MDP similarity
    - Within-policy similarity
    - Within-depth similarity

    Includes shaded standard error regions if available.
    """
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    x = np.array(layer_indices)

    global_sim = np.array(metrics['global_similarity'])
    mdp_sim = np.array(metrics['within_mdp_label_similarity'])
    policy_sim = np.array(metrics['within_policy_label_similarity'])
    depth_sim = np.array(metrics['within_depth_similarity'])

    global_se = np.array(metrics['global_similarity_se'])
    mdp_se = np.array(metrics['within_mdp_label_similarity_se'])
    policy_se = np.array(metrics['within_policy_label_similarity_se'])
    depth_se = np.array(metrics['within_depth_similarity_se'])

    # Plot lines
    ax.plot(x, global_sim, '-', color=COLOR_GLOBAL, linewidth=2, label='Global')
    ax.plot(x, mdp_sim, '-', color=COLOR_WITHIN_MDP, linewidth=2, label='MDP Symmetry')
    ax.plot(x, policy_sim, '-', color=COLOR_WITHIN_POLICY, linewidth=2, label=r'$\pi^*$ Symmetry')
    ax.plot(x, depth_sim, '-', color=COLOR_WITHIN_DEPTH, linewidth=2, label='Depth Symmetry')

    # Add shaded standard error regions if available
    if np.any(global_se > 0):
        ax.fill_between(x, global_sim - global_se, global_sim + global_se,
                        color=COLOR_GLOBAL, alpha=0.2)
    if np.any(mdp_se > 0):
        ax.fill_between(x, mdp_sim - mdp_se, mdp_sim + mdp_se,
                        color=COLOR_WITHIN_MDP, alpha=0.2)
    if np.any(policy_se > 0):
        ax.fill_between(x, policy_sim - policy_se, policy_sim + policy_se,
                        color=COLOR_WITHIN_POLICY, alpha=0.2)
    if np.any(depth_se > 0):
        ax.fill_between(x, depth_sim - depth_se, depth_sim + depth_se,
                        color=COLOR_WITHIN_DEPTH, alpha=0.2)

    ax.set_xlabel('Layer', fontsize=8)
    ax.set_ylabel('Cosine Similarity', fontsize=8)
    ax.set_ylim(-0.1, 1.0)
    ax.set_xlim(x[0], x[-1])
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(frameon=False, fontsize=6, loc='best')
    ax.set_title(title, fontsize=8)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def plot_model_comparison(
    summary: dict,
    rep_name: str,
    save_path: str = None
):
    """
    Compare RSA metrics across different models for a single representation.

    Creates a grid of subplots, one row per model, each showing four curves
    with standard error shading.
    """
    rsa_analysis = summary.get('rsa_analysis', {})
    models = list(rsa_analysis.keys())

    if not models:
        print("No RSA analysis results found.")
        return None

    n_models = len(models)
    fig, axs = plt.subplots(n_models, 1, figsize=(3.3, 2.5 * n_models), sharey=True)

    if n_models == 1:
        axs = [axs]

    for row, model_key in enumerate(models):
        model_rsa = rsa_analysis[model_key]
        layer_indices, metrics = extract_layer_metrics(model_rsa, rep_name)

        if layer_indices is None:
            continue

        x = np.array(layer_indices)

        global_sim = np.array(metrics['global_similarity'])
        mdp_sim = np.array(metrics['within_mdp_label_similarity'])
        policy_sim = np.array(metrics['within_policy_label_similarity'])
        depth_sim = np.array(metrics['within_depth_similarity'])

        global_se = np.array(metrics['global_similarity_se'])
        mdp_se = np.array(metrics['within_mdp_label_similarity_se'])
        policy_se = np.array(metrics['within_policy_label_similarity_se'])
        depth_se = np.array(metrics['within_depth_similarity_se'])

        # Plot lines
        axs[row].plot(x, global_sim, '-', color=COLOR_GLOBAL, linewidth=2, label='Global')
        axs[row].plot(x, mdp_sim, '-', color=COLOR_WITHIN_MDP, linewidth=2, label='MDP Symmetry')
        axs[row].plot(x, policy_sim, '-', color=COLOR_WITHIN_POLICY, linewidth=2, label=r'$\pi^*$ Symmetry')
        axs[row].plot(x, depth_sim, '-', color=COLOR_WITHIN_DEPTH, linewidth=2, label='Depth Symmetry')

        # Add shaded standard error regions
        if np.any(global_se > 0):
            axs[row].fill_between(x, global_sim - global_se, global_sim + global_se,
                                  color=COLOR_GLOBAL, alpha=0.2)
        if np.any(mdp_se > 0):
            axs[row].fill_between(x, mdp_sim - mdp_se, mdp_sim + mdp_se,
                                  color=COLOR_WITHIN_MDP, alpha=0.2)
        if np.any(policy_se > 0):
            axs[row].fill_between(x, policy_sim - policy_se, policy_sim + policy_se,
                                  color=COLOR_WITHIN_POLICY, alpha=0.2)
        if np.any(depth_se > 0):
            axs[row].fill_between(x, depth_sim - depth_se, depth_sim + depth_se,
                                  color=COLOR_WITHIN_DEPTH, alpha=0.2)

        axs[row].set_ylabel(f'{model_key}\nCosine Sim.', fontsize=8)
        axs[row].set_ylim(-0.1, 1.0)
        axs[row].set_xlim(x[0], x[-1])
        axs[row].tick_params(axis='both', labelsize=8)

    # Add legend to first row
    axs[0].legend(frameon=False, fontsize=6, loc='best')

    # Set x-axis label on bottom row
    axs[-1].set_xlabel('Layer', fontsize=8)

    fig.suptitle(f'Model Comparison: {rep_name}', fontsize=8, y=1.02)
    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def plot_representation_comparison(
    summary: dict,
    model_key: str,
    save_path: str = None
):
    """
    Compare RSA metrics across different representations for a single model.

    Creates a 2x3 grid (total width 6.8 in) showing how different graph
    representations affect the model's internal representations.
    Includes standard error shading. Labels follow the ascii_vs_edges style.
    """
    REP_ORDER = [
        'edges_directed', 'edges_random', 'ascii_maze',
        'adjacency_list', 'relation_triples', 'path_mapping',
    ]
    REP_LABELS = {
        'edges_directed': 'Edges list repr.\n(ordered)',
        'edges_random': 'Edges list repr.\n(randomized)',
        'ascii_maze': 'ASCII repr.',
        'adjacency_list': 'Adjacency list repr.',
        'relation_triples': 'Relation Triples',
        'path_mapping': 'Root Path Repr.',
    }

    rsa_analysis = summary.get('rsa_analysis', {})

    if model_key not in rsa_analysis:
        print(f"Model {model_key} not found in results.")
        return None

    model_rsa = rsa_analysis[model_key]

    # Filter to representations that exist in data, preserving order
    available_reps = [r for r in REP_ORDER if r in model_rsa]

    if not available_reps:
        print(f"No matching representations found for {model_key}.")
        return None

    ncols = 3
    nrows = -(-len(available_reps) // ncols)  # ceil division
    fig, axs = plt.subplots(nrows, ncols, figsize=(6.8, 1.5 * nrows), sharex='col', sharey='row')

    # Ensure axs is always 2D
    if nrows == 1 and ncols == 1:
        axs = np.array([[axs]])
    elif nrows == 1:
        axs = axs[np.newaxis, :]
    elif ncols == 1:
        axs = axs[:, np.newaxis]

    for idx, rep_name in enumerate(available_reps):
        row = idx // ncols
        col = idx % ncols
        ax = axs[row, col]

        layer_indices, metrics = extract_layer_metrics(model_rsa, rep_name)

        if layer_indices is None:
            ax.set_visible(False)
            continue

        x = np.array(layer_indices)

        global_sim = np.array(metrics['global_similarity'])
        mdp_sim = np.array(metrics['within_mdp_label_similarity'])
        policy_sim = np.array(metrics['within_policy_label_similarity'])
        depth_sim = np.array(metrics['within_depth_similarity'])

        global_se = np.array(metrics['global_similarity_se'])
        mdp_se = np.array(metrics['within_mdp_label_similarity_se'])
        policy_se = np.array(metrics['within_policy_label_similarity_se'])
        depth_se = np.array(metrics['within_depth_similarity_se'])

        # Plot lines
        ax.plot(x, global_sim, '-', color=COLOR_GLOBAL, linewidth=2, label='Global')
        ax.plot(x, mdp_sim, '-', color=COLOR_WITHIN_MDP, linewidth=2, label='MDP Symmetry')
        ax.plot(x, policy_sim, '-', color=COLOR_WITHIN_POLICY, linewidth=2, label=r'$\pi^*$ Symmetry')
        ax.plot(x, depth_sim, '-', color=COLOR_WITHIN_DEPTH, linewidth=2, label='Depth Symmetry')

        # Add shaded standard error regions
        if np.any(global_se > 0):
            ax.fill_between(x, global_sim - global_se, global_sim + global_se,
                            color=COLOR_GLOBAL, alpha=0.2)
        if np.any(mdp_se > 0):
            ax.fill_between(x, mdp_sim - mdp_se, mdp_sim + mdp_se,
                            color=COLOR_WITHIN_MDP, alpha=0.2)
        if np.any(policy_se > 0):
            ax.fill_between(x, policy_sim - policy_se, policy_sim + policy_se,
                            color=COLOR_WITHIN_POLICY, alpha=0.2)
        if np.any(depth_se > 0):
            ax.fill_between(x, depth_sim - depth_se, depth_sim + depth_se,
                            color=COLOR_WITHIN_DEPTH, alpha=0.2)

        display_name = REP_LABELS.get(rep_name, rep_name)
        ax.set_ylabel(f'{display_name}\nCosine Sim.', fontsize=8)
        ax.set_ylim(-0.1, 1.0)
        ax.set_xlim(x[0], x[-1])
        ax.tick_params(axis='both', labelsize=8)

        # Only add x-label on bottom row
        if row == nrows - 1:
            ax.set_xlabel('Layer', fontsize=8)

    # Hide unused axes
    for idx in range(len(available_reps), nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axs[row, col].set_visible(False)

    # Add legend to first subplot
    axs[0, 0].legend(frameon=False, fontsize=6, loc='best')

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def plot_final_layer_summary(
    summary: dict,
    save_path: str = None
):
    """
    Create a summary plot showing final layer metrics for all models and representations.

    This is a bar chart comparing global, within-MDP, within-policy, and within-depth
    similarities at the final layer across all configurations.
    """
    rsa_analysis = summary.get('rsa_analysis', {})

    if not rsa_analysis:
        print("No RSA analysis results found.")
        return None

    # Collect data for bar chart
    data = []
    for model_key, model_rsa in rsa_analysis.items():
        for rep_name, layer_data in model_rsa.items():
            # Get final layer (highest index)
            layer_names = sorted(layer_data.keys(),
                               key=lambda x: int(x.split('_')[1]) if '_' in x else 0)
            if not layer_names:
                continue

            final_layer = layer_names[-1]
            metrics = layer_data[final_layer]

            data.append({
                'model': model_key,
                'representation': rep_name[:10],  # Truncate
                'label': f"{model_key}\n{rep_name[:8]}",
                'global': metrics.get('global_similarity', 0),
                'within_mdp': metrics.get('within_mdp_label_similarity', 0),
                'within_policy': metrics.get('within_policy_label_similarity', 0),
                'within_depth': metrics.get('within_depth_similarity', 0),
                'participation_ratio': metrics.get('participation_ratio', 0)
            })

    if not data:
        print("No data to plot.")
        return None

    # Create grouped bar chart
    n_configs = len(data)
    x = np.arange(n_configs)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(3.3, n_configs * 0.5), 2.5))

    ax.bar(x - 1.5*width, [d['global'] for d in data], width,
           label='Global', color=COLOR_GLOBAL, alpha=0.8)
    ax.bar(x - 0.5*width, [d['within_mdp'] for d in data], width,
           label='MDP Symmetry', color=COLOR_WITHIN_MDP, alpha=0.8)
    ax.bar(x + 0.5*width, [d['within_policy'] for d in data], width,
           label=r'$\pi^*$ Symmetry', color=COLOR_WITHIN_POLICY, alpha=0.8)
    ax.bar(x + 1.5*width, [d['within_depth'] for d in data], width,
           label='Depth Symmetry', color=COLOR_WITHIN_DEPTH, alpha=0.8)
    ax.set_ylabel('Cosine Similarity', fontsize=8)
    ax.set_title('Final Layer Similarity Metrics', fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([d['label'] for d in data], fontsize=8, rotation=45, ha='right')
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(frameon=False, fontsize=6)
    ax.set_ylim(0, 1.0)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def plot_participation_ratio(
    summary: dict,
    save_path: str = None
):
    """
    Plot participation ratio (effective dimensionality) across layers.
    """
    rsa_analysis = summary.get('rsa_analysis', {})

    if not rsa_analysis:
        print("No RSA analysis results found.")
        return None

    # Collect all model/representation combinations
    configs = []
    for model_key, model_rsa in rsa_analysis.items():
        for rep_name in model_rsa.keys():
            configs.append((model_key, rep_name))

    if not configs:
        return None

    n_configs = len(configs)
    fig, ax = plt.subplots(figsize=(3.3, 2.5))

    colors = plt.cm.tab10(np.linspace(0, 1, n_configs))

    for idx, (model_key, rep_name) in enumerate(configs):
        layer_indices, metrics = extract_layer_metrics(rsa_analysis[model_key], rep_name)
        if layer_indices is None:
            continue

        x = np.array(layer_indices)

        label = f"{model_key} - {rep_name[:10]}"
        ax.plot(x, metrics['participation_ratio'], '-', color=colors[idx],
                label=label, linewidth=2, alpha=0.8)

    ax.set_xlabel('Layer', fontsize=8)
    ax.set_ylabel('Participation Ratio', fontsize=8)
    ax.set_title('Effective Dimensionality Across Layers', fontsize=8)
    ax.tick_params(axis='both', labelsize=8)
    ax.legend(frameon=False, fontsize=6, loc='best', ncol=2)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def plot_ascii_vs_edges_comparison(
    summary: dict,
    model_key: str,
    save_path: str = None
):
    """
    Compare RSA metrics between ascii_maze and edges_random representations.

    Creates two subplots with shared x-axis comparing:
    - ASCII repr. (ascii_maze)
    - Edge list repr. (edges_random)
    """
    rsa_analysis = summary.get('rsa_analysis', {})

    if model_key not in rsa_analysis:
        print(f"Model {model_key} not found in results.")
        return None

    model_rsa = rsa_analysis[model_key]

    # Check that both representations exist
    if 'ascii_maze' not in model_rsa or 'edges_random' not in model_rsa:
        print(f"Both ascii_maze and edges_random required. Found: {list(model_rsa.keys())}")
        return None

    fig, axs = plt.subplots(2, 1, figsize=(3.3, 4), sharex=True)

    rep_configs = [
        ('ascii_maze', 'ASCII repr.'),
        ('edges_random', 'Edge list repr.')
    ]

    for row, (rep_name, display_name) in enumerate(rep_configs):
        layer_indices, metrics = extract_layer_metrics(model_rsa, rep_name)

        if layer_indices is None:
            continue

        x = np.array(layer_indices)

        global_sim = np.array(metrics['global_similarity'])
        mdp_sim = np.array(metrics['within_mdp_label_similarity'])
        policy_sim = np.array(metrics['within_policy_label_similarity'])
        depth_sim = np.array(metrics['within_depth_similarity'])

        global_se = np.array(metrics['global_similarity_se'])
        mdp_se = np.array(metrics['within_mdp_label_similarity_se'])
        policy_se = np.array(metrics['within_policy_label_similarity_se'])
        depth_se = np.array(metrics['within_depth_similarity_se'])

        # Plot lines
        axs[row].plot(x, global_sim, '-', color=COLOR_GLOBAL, linewidth=2, label='Global')
        axs[row].plot(x, mdp_sim, '-', color=COLOR_WITHIN_MDP, linewidth=2, label='MDP Symmetry')
        axs[row].plot(x, policy_sim, '-', color=COLOR_WITHIN_POLICY, linewidth=2, label=r'$\pi^*$ Symmetry')
        axs[row].plot(x, depth_sim, '-', color=COLOR_WITHIN_DEPTH, linewidth=2, label='Depth Symmetry')

        # Add shaded standard error regions
        if np.any(global_se > 0):
            axs[row].fill_between(x, global_sim - global_se, global_sim + global_se,
                                  color=COLOR_GLOBAL, alpha=0.2)
        if np.any(mdp_se > 0):
            axs[row].fill_between(x, mdp_sim - mdp_se, mdp_sim + mdp_se,
                                  color=COLOR_WITHIN_MDP, alpha=0.2)
        if np.any(policy_se > 0):
            axs[row].fill_between(x, policy_sim - policy_se, policy_sim + policy_se,
                                  color=COLOR_WITHIN_POLICY, alpha=0.2)
        if np.any(depth_se > 0):
            axs[row].fill_between(x, depth_sim - depth_se, depth_sim + depth_se,
                                  color=COLOR_WITHIN_DEPTH, alpha=0.2)

        axs[row].set_ylabel(f'{display_name}\nCosine Sim.', fontsize=8)
        axs[row].set_ylim(-0.1, 1.0)
        axs[row].set_xlim(x[0], x[-1])
        axs[row].tick_params(axis='both', labelsize=8)

    # Add legend to first row
    axs[0].legend(frameon=False, fontsize=6, loc='best')

    # Set x-axis label on bottom row
    axs[-1].set_xlabel('Layer', fontsize=8)

    sns.despine()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=FIGURE_DPI, transparent=True)
        print(f"Saved: {save_path}")
        plt.close()
    else:
        plt.show()

    return fig


def generate_all_plots(results_dir: str, output_dir: str = None, models: list = None, model_size: str = "small"):
    """
    Generate all plots from RSA analysis results.

    Args:
        results_dir: Directory containing summary_{model_size}.json
        output_dir: Directory to save plots (default: results_dir/plots_{model_size})
        models: List of model keys to plot (default: all)
        model_size: Model size tier ("small" or "large")
    """
    # Load results
    summary = load_results(results_dir, model_size)

    if 'rsa_analysis' not in summary:
        print(f"No RSA analysis found in results. Run evaluate_models.py with --analyze-representations --size {model_size}.")
        return

    rsa_analysis = summary['rsa_analysis']

    # Filter models if specified
    if models:
        rsa_analysis = {k: v for k, v in rsa_analysis.items() if k in models}
        if not rsa_analysis:
            print(f"None of the specified models found. Available: {list(summary['rsa_analysis'].keys())}")
            return

    # Set up output directory
    if output_dir is None:
        output_dir = Path(results_dir) / f"plots_{model_size}"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {output_dir} (model_size={model_size})")

    # Get all representations
    all_reps = set()
    for model_rsa in rsa_analysis.values():
        all_reps.update(model_rsa.keys())

    # 1. Individual plots for each model/representation
    for model_key, model_rsa in rsa_analysis.items():
        model_dir = output_dir / model_key
        model_dir.mkdir(exist_ok=True)

        for rep_name in model_rsa.keys():
            layer_indices, metrics = extract_layer_metrics(model_rsa, rep_name)
            if layer_indices:
                save_path = model_dir / f"{rep_name}_rsa.pdf"
                plot_single_model_representation(
                    layer_indices, metrics,
                    title=f"{model_key} - {rep_name}",
                    save_path=str(save_path)
                )

    # 2. Model comparison plots (one per representation)
    for rep_name in all_reps:
        save_path = output_dir / f"model_comparison_{rep_name}.pdf"
        plot_model_comparison(summary, rep_name, save_path=str(save_path))

    # 3. Representation comparison plots (one per model)
    for model_key in rsa_analysis.keys():
        save_path = output_dir / f"rep_comparison_{model_key}.pdf"
        plot_representation_comparison(summary, model_key, save_path=str(save_path))

    # 4. Summary plots
    plot_final_layer_summary(summary, save_path=str(output_dir / "final_layer_summary.pdf"))
    plot_participation_ratio(summary, save_path=str(output_dir / "participation_ratio.pdf"))

    # 5. ASCII vs Edge list comparison plots (one per model)
    for model_key in rsa_analysis.keys():
        save_path = output_dir / f"ascii_vs_edges_{model_key}.pdf"
        plot_ascii_vs_edges_comparison(summary, model_key, save_path=str(save_path))

    print(f"\nAll plots saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot RSA analysis results from LLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_rsa_results.py                              # Plot all small model results
  python plot_rsa_results.py --size large                 # Plot all large model results
  python plot_rsa_results.py --results-dir my_results     # Custom results dir
  python plot_rsa_results.py --models qwen deepseek       # Plot specific models
  python plot_rsa_results.py --output-dir figures         # Custom output dir
"""
    )
    parser.add_argument(
        "--results-dir",
        default="evaluation_results",
        help="Directory containing evaluation results (default: evaluation_results)"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save plots (default: results_dir/plots_{size})"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Specific models to plot (default: all)"
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Model size tier to plot: 'small' or 'large'. Default: small"
    )

    args = parser.parse_args()

    generate_all_plots(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        models=args.models,
        model_size=args.size
    )


if __name__ == "__main__":
    main()
