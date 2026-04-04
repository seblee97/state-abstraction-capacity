import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

RESULTS_DIR = "evaluation_results"

def load_summary(model_size="small"):
    """Load the summary results for the specified model size."""
    summary_file = Path(RESULTS_DIR) / f"summary_{model_size}.json"
    if not summary_file.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_file}")

    with open(summary_file, 'r') as f:
        return json.load(f)

def create_heatmap(summary, model_size="small"):
    """Create a heatmap of success rates."""
    models = summary["models"]
    representations = summary["representations"]

    # Create data matrix
    data = []
    for model in models:
        row = []
        for rep in representations:
            if model in summary["results"] and rep in summary["results"][model]:
                row.append(summary["results"][model][rep]["success_rate"])
            else:
                row.append(0.0)
        data.append(row)

    df = pd.DataFrame(data, index=models, columns=representations)

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(df.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)

    # Set ticks
    ax.set_xticks(range(len(representations)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(representations, rotation=45, ha='right')
    ax.set_yticklabels(models)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Success Rate (%)', rotation=270, labelpad=20)

    # Add text annotations
    for i in range(len(models)):
        for j in range(len(representations)):
            text = ax.text(j, i, f'{df.values[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)

    ax.set_title(f'Model Performance by Representation Type ({model_size})')
    plt.tight_layout()

    output_file = Path(RESULTS_DIR) / f"heatmap_{model_size}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved to {output_file}")
    plt.close()

def create_bar_chart(summary, model_size="small"):
    """Create bar chart comparing models."""
    models = summary["models"]
    representations = summary["representations"]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = range(len(representations))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        rates = []
        for rep in representations:
            if model in summary["results"] and rep in summary["results"][model]:
                rates.append(summary["results"][model][rep]["success_rate"])
            else:
                rates.append(0.0)

        offset = width * (i - len(models)/2 + 0.5)
        ax.bar([xi + offset for xi in x], rates, width, label=model)

    ax.set_xlabel('Representation Type')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title(f'Model Performance Comparison ({model_size})')
    ax.set_xticks(x)
    ax.set_xticklabels(representations, rotation=45, ha='right')
    ax.legend()
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_file = Path(RESULTS_DIR) / f"bar_chart_{model_size}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Bar chart saved to {output_file}")
    plt.close()

def analyze_step_distribution(summary, model_name, model_size="small"):
    """Analyze distribution of steps taken for a specific model."""
    model_file = Path(RESULTS_DIR) / f"{model_name}_{model_size}_results.json"
    if not model_file.exists():
        print(f"Results file not found for {model_name} ({model_size})")
        return

    with open(model_file, 'r') as f:
        results = json.load(f)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (rep_name, rep_data) in enumerate(results.items()):
        if idx >= len(axes):
            break

        steps_success = [r["steps"] for r in rep_data["results"] if r["success"]]
        steps_fail = [r["steps"] for r in rep_data["results"] if not r["success"]]

        ax = axes[idx]
        if steps_success:
            ax.hist(steps_success, bins=20, alpha=0.7, label='Success', color='green')
        if steps_fail:
            ax.hist(steps_fail, bins=20, alpha=0.7, label='Failed', color='red')

        ax.set_xlabel('Number of Steps')
        ax.set_ylabel('Count')
        ax.set_title(f'{rep_name}\nSuccess: {rep_data["success_rate"]:.1f}%')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Step Distribution for {model_name} ({model_size})')
    plt.tight_layout()

    output_file = Path(RESULTS_DIR) / f"{model_name}_{model_size}_step_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Step distribution saved to {output_file}")
    plt.close()

def print_detailed_stats(summary):
    """Print detailed statistics."""
    print("\n" + "="*80)
    print("DETAILED STATISTICS")
    print("="*80)

    for model in summary["models"]:
        if model not in summary["results"]:
            continue

        print(f"\n{model.upper()}")
        print("-" * 80)

        for rep in summary["representations"]:
            if rep not in summary["results"][model]:
                continue

            data = summary["results"][model][rep]
            print(f"  {rep:20s}: {data['success_rate']:6.2f}% "
                  f"({data['success_count']}/{data['total_count']})")

def main():
    """Main visualization function."""
    parser = argparse.ArgumentParser(
        description="Visualize LLM evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python visualize_results.py                # Visualize small model results
  python visualize_results.py --size large   # Visualize large model results
"""
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Model size tier to visualize: 'small' or 'large'. Default: small"
    )

    args = parser.parse_args()
    model_size = args.size

    print(f"Loading results for {model_size} models...")
    summary = load_summary(model_size)

    print("\nCreating visualizations...")
    create_heatmap(summary, model_size)
    create_bar_chart(summary, model_size)

    # Create step distributions for each model
    for model in summary["models"]:
        analyze_step_distribution(summary, model, model_size)

    print_detailed_stats(summary)

    print(f"\nAll visualizations saved to {RESULTS_DIR}/")

if __name__ == "__main__":
    main()
