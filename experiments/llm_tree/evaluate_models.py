import json
import os
from pathlib import Path
import argparse
import numpy as np

# Add imports for LLM inference
# You'll need to install: pip install transformers torch accelerate
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError:
    print("Please install required packages: pip install transformers torch accelerate")
    exit(1)

# Import utility functions from utils.py
from utils import (
    load_graph_structure,
    get_representation_files,
    get_all_start_nodes,
    get_neighbors,
    get_neighbors_with_directions,
    parse_node_id,
    build_graph_transition_matrices,
    compute_mdp_homomorphism_labels,
    compute_optimal_policy_labels,
    compute_depth_labels,
    analyze_llm_representations,
    compute_similarity_metrics,
    plot_similarity_comparison
)

# --- Configuration ---
TREE_REPRESENTATIONS_DIR = "tree_representations"
RESULTS_DIR = "evaluation_results"
MAX_STEPS = 40  # Maximum navigation steps allowed

# Model configurations by size tier
SMALL_MODELS = {
    "ministral": "mistralai/Ministral-8B-Instruct-2410",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
}

LARGE_MODELS = {
    "mistral": "mistralai/Mistral-Small-24B-Instruct-2501",
    "qwen": "Qwen/Qwen2.5-32B-Instruct",
    "deepseek": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
}

# Default to small models for backwards compatibility
ALL_MODELS = SMALL_MODELS

# Global graph structure (loaded from adjacency list)
adjacency_graph = {}
direction_graph = None  # For spatial maze format: {node: {direction: neighbor}}
target_node = None
start_node = None

# --- Model management ---
class ModelEvaluator:
    def __init__(self, model_name, model_path, layer_interval=1):
        self.model_name = model_name
        self.model_path = model_path
        self.layer_interval = layer_interval
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_layers = None
        self.layer_indices = None

    def load_model(self):
        """Load the model and tokenizer."""
        print(f"Loading {self.model_name} from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        print(f"Model loaded on {self.device}")

        # Get number of layers and set up extraction indices
        self._setup_layer_indices()

    def _setup_layer_indices(self):
        """Determine which transformer layers to extract from."""
        config = self.model.config

        # Try different attribute names for number of layers
        if hasattr(config, 'num_hidden_layers'):
            self.num_layers = config.num_hidden_layers
        elif hasattr(config, 'n_layer'):
            self.num_layers = config.n_layer
        elif hasattr(config, 'num_layers'):
            self.num_layers = config.num_layers
        else:
            # Fallback: try to infer from model structure
            self.num_layers = 32  # Common default

        # Extract from layers at specified interval (always include first and last)
        layer_indices_list = list(range(0, self.num_layers, self.layer_interval))
        # Ensure last layer is included
        if (self.num_layers - 1) not in layer_indices_list:
            layer_indices_list.append(self.num_layers - 1)
        self.layer_indices = {f'layer_{i}': i for i in layer_indices_list}

        print(f"Model has {self.num_layers} layers. Extracting from {len(self.layer_indices)} layers (interval={self.layer_interval}).")

    def unload_model(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.num_layers = None
            self.layer_indices = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def generate_response(self, prompt, max_new_tokens=100):
        """Generate response from the model."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from response
        response = response[len(prompt):].strip()
        return response

    def extract_representations(self, prompt):
        """
        Extract hidden states from the model BEFORE generating any response.

        This captures the model's internal representation of the prompt,
        where the only difference between states is the last token (node ID).

        Args:
            prompt: Full context including graph representation and current node

        Returns:
            representations: Dict[layer_name, np.ndarray] - hidden states for each layer
                Each array has shape (hidden_dim,) representing the last token's hidden state
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )

        # hidden_states is a tuple of (num_layers + 1) tensors
        # Each tensor has shape (batch_size, sequence_length, hidden_dim)
        # Index 0 is the embedding layer output, indices 1-N are transformer layer outputs
        hidden_states = outputs.hidden_states

        representations = {}
        for layer_name, layer_idx in self.layer_indices.items():
            # Get the hidden state for this layer
            # We add 1 because hidden_states[0] is the embedding layer
            layer_hidden = hidden_states[layer_idx + 1]

            # Extract the last token's representation
            # Shape: (batch_size, seq_len, hidden_dim) -> (hidden_dim,)
            last_token_repr = layer_hidden[0, -1, :].cpu().numpy()
            representations[layer_name] = last_token_repr

        return representations

# --- Navigation task ---
def run_navigation_task(evaluator, representation_content, nav_start_node, goal_node, graph, dir_graph=None):
    """Run a single navigation task.

    Args:
        evaluator: ModelEvaluator instance
        representation_content: Text representation of the graph
        nav_start_node: Starting node ID
        goal_node: Target node ID
        graph: Adjacency graph dict mapping node -> neighbors
        dir_graph: Optional direction graph for spatial maze format
    """
    current_node = nav_start_node
    history = []
    conversation = representation_content + f"\n\nYou are currently at Node {current_node}. Your goal is to reach Node {goal_node}."

    for step in range(MAX_STEPS):
        # Get valid neighbors for current node
        valid_neighbors = get_neighbors(graph, current_node)

        # Build prompt with direction info if available
        if dir_graph is not None:
            directions = get_neighbors_with_directions(dir_graph, current_node)
            # Format as "Direction: Node" pairs
            neighbor_list = ", ".join(f"{d}: Node {n}" for d, n in directions.items())
            prompt = (f"{conversation}\n\n"
                     f"Which node do you want to move to? "
                     f"Available moves from your current position: {neighbor_list}\n"
                     f"Respond with ONLY the node number you want to move to.\n"
                     f"Next node:")
        else:
            neighbor_list = ", ".join(str(n) for n in valid_neighbors)
            prompt = (f"{conversation}\n\n"
                     f"Which node do you want to move to? "
                     f"You can move to any of these connected nodes: {neighbor_list}\n"
                     f"Respond with ONLY the node number you want to move to.\n"
                     f"Next node:")

        # Get model response
        try:
            response = evaluator.generate_response(prompt, max_new_tokens=20)
        except Exception as e:
            history.append({
                "step": step + 1,
                "current_node": current_node,
                "target_node_id": None,
                "error": str(e)
            })
            return {
                "success": False,
                "steps": step + 1,
                "history": history,
                "reason": f"Error: {str(e)}"
            }

        # Parse target node ID
        target_node_id = parse_node_id(response)

        if target_node_id is None:
            history.append({
                "step": step + 1,
                "current_node": current_node,
                "response": response,
                "target_node_id": None,
                "valid_neighbors": valid_neighbors
            })
            return {
                "success": False,
                "steps": step + 1,
                "history": history,
                "reason": "Could not parse node ID from response"
            }

        # Check if target node is a valid neighbor
        if target_node_id not in valid_neighbors:
            # Invalid move - stay at current node and give feedback
            history.append({
                "step": step + 1,
                "current_node": current_node,
                "response": response,
                "target_node_id": target_node_id,
                "valid_neighbors": valid_neighbors,
                "valid": False
            })
            conversation += f"\n\nInvalid move: Node {target_node_id} is not connected to Node {current_node}. You remain at Node {current_node}."
            # Continue to next iteration without changing current_node
            continue

        # Execute valid move - include direction if available
        move_info = {
            "step": step + 1,
            "current_node": current_node,
            "response": response,
            "target_node_id": target_node_id,
            "valid_neighbors": valid_neighbors,
            "valid": True
        }

        # Add direction info if available
        if dir_graph is not None:
            directions = get_neighbors_with_directions(dir_graph, current_node)
            for d, n in directions.items():
                if n == target_node_id:
                    move_info["direction"] = d
                    break

        history.append(move_info)

        # Update conversation with direction if available
        if dir_graph is not None and "direction" in move_info:
            conversation += f"\n\nYou moved {move_info['direction']} to Node {target_node_id}."
        else:
            conversation += f"\n\nYou moved to Node {target_node_id}."

        current_node = target_node_id

        # Check if target reached
        if current_node == goal_node:
            return {
                "success": True,
                "steps": step + 1,
                "history": history,
                "reason": "Target reached"
            }

    # Max steps exceeded
    return {
        "success": False,
        "steps": MAX_STEPS,
        "history": history,
        "reason": "Max steps exceeded"
    }


# --- RSA Analysis ---
def collect_representations(evaluator, representation_content, start_nodes, goal_node):
    """
    Collect hidden state representations for all start nodes.

    Extracts representations BEFORE any navigation action, so the only
    difference between states is the start node ID in the prompt.

    Args:
        evaluator: ModelEvaluator instance with loaded model
        representation_content: Text representation of the graph
        start_nodes: List of start node IDs
        goal_node: Target node ID

    Returns:
        layer_representations: Dict[layer_name, Dict[node_id, np.ndarray]]
    """
    from collections import defaultdict

    layer_representations = defaultdict(dict)

    print(f"    Collecting representations for {len(start_nodes)} nodes...")

    for i, start_node in enumerate(start_nodes):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"      Node {i+1}/{len(start_nodes)}", end="\r")

        # Build initial prompt (before any navigation)
        initial_prompt = (
            f"{representation_content}\n\n"
            f"You are currently at Node {start_node}. "
            f"Your goal is to reach Node {goal_node}."
        )

        # Extract representations
        reprs = evaluator.extract_representations(initial_prompt)

        for layer_name, repr_vec in reprs.items():
            layer_representations[layer_name][start_node] = repr_vec

    print(f"      Collected representations for {len(start_nodes)} nodes")

    return dict(layer_representations)


def run_rsa_analysis(
    evaluator,
    representations,
    start_nodes,
    goal_node,
    graph,
    node_to_mdp_label,
    node_to_policy_label,
    node_to_idx,
    node_to_depth,
    model_key,
    results_dir,
    model_size="small"
):
    """
    Run RSA analysis for all graph representations.

    Args:
        evaluator: ModelEvaluator instance
        representations: Dict[rep_name, rep_content]
        start_nodes: List of start node IDs
        goal_node: Target node ID
        graph: Adjacency graph
        node_to_mdp_label: MDP homomorphism labels
        node_to_policy_label: Optimal policy labels
        node_to_idx: Node to index mapping
        node_to_depth: Depth labels for each node
        model_key: Name of the model
        results_dir: Directory to save results
        model_size: Size tier of the model ("small" or "large")

    Returns:
        rsa_results: Dict[rep_name, Dict[layer_name, metrics]]
    """
    rsa_results = {}

    for rep_name, rep_content in representations.items():
        print(f"\n  RSA Analysis for representation: {rep_name}")

        # Collect representations for all start nodes
        layer_reprs = collect_representations(
            evaluator, rep_content, start_nodes, goal_node
        )

        # Run analysis
        save_dir = Path(results_dir) / f"{model_key}_{model_size}" / "rsa_analysis" / rep_name

        analysis_results = analyze_llm_representations(
            layer_reprs,
            node_to_mdp_label,
            node_to_policy_label,
            node_to_idx,
            node_to_depth=node_to_depth,
            save_dir=str(save_dir),
            representation_name=rep_name
        )

        rsa_results[rep_name] = analysis_results

        # Print summary
        for layer_name, metrics in analysis_results.items():
            print(f"    {layer_name}:")
            print(f"      Global sim: {metrics['global_similarity']:.4f}")
            print(f"      Within-MDP sim: {metrics['within_mdp_label_similarity']:.4f}")
            print(f"      Within-policy sim: {metrics['within_policy_label_similarity']:.4f}")
            print(f"      Within-depth sim: {metrics['within_depth_similarity']:.4f}")
            print(f"      Participation ratio: {metrics['participation_ratio']:.2f}")

    return rsa_results


# --- Main evaluation ---
def evaluate_all(models_to_evaluate=None, analyze_representations=False, model_size="small", layer_interval=1):
    """Run evaluation for all models and representations.

    Args:
        models_to_evaluate: List of model keys to evaluate. If None, evaluate all models.
        analyze_representations: If True, collect hidden states and run RSA analysis.
        model_size: "small" for ~7B models, "large" for ~24-32B models.
        layer_interval: Interval for layer extraction (1=every layer, 2=every other, etc.).
    """
    global adjacency_graph, direction_graph, target_node, start_node

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Select model set based on size
    if model_size == "large":
        MODEL_SET = LARGE_MODELS
        print(f"Using LARGE models (~24-32B parameters)")
    else:
        MODEL_SET = SMALL_MODELS
        print(f"Using SMALL models (~7-8B parameters)")

    # Determine which models to evaluate
    if models_to_evaluate is None:
        MODELS = MODEL_SET
    else:
        MODELS = {k: v for k, v in MODEL_SET.items() if k in models_to_evaluate}

        # Check for invalid model names
        invalid_models = set(models_to_evaluate) - set(MODEL_SET.keys())
        if invalid_models:
            print(f"Warning: Invalid model names: {', '.join(invalid_models)}")
            print(f"Available models for {model_size} size: {', '.join(MODEL_SET.keys())}")

        if not MODELS:
            print("Error: No valid models specified")
            print(f"Available models for {model_size} size: {', '.join(MODEL_SET.keys())}")
            exit(1)

    # Load graph structure from adjacency list
    print("Loading graph structure...")
    adjacency_graph, target_node, start_node, direction_graph = load_graph_structure(TREE_REPRESENTATIONS_DIR)
    print(f"Loaded graph with {len(adjacency_graph)} nodes")
    if start_node is not None:
        print(f"Start node: {start_node}")
    print(f"Target node: {target_node}")
    if direction_graph is not None:
        print("Spatial maze format detected (with cardinal directions)")

    # Get representations
    rep_files = get_representation_files(TREE_REPRESENTATIONS_DIR)
    print(f"Found {len(rep_files)} representation files")

    # Get start nodes
    start_nodes = get_all_start_nodes(adjacency_graph, target_node)
    print(f"Testing {len(start_nodes)} start nodes")

    # Load representations
    representations = {}
    for rep_name, rep_file in rep_files.items():
        with open(rep_file, 'r') as f:
            representations[rep_name] = f.read()

    # If analyzing representations, compute graph labels first
    node_to_mdp_label = None
    node_to_policy_label = None
    node_to_idx = None
    node_to_depth = None

    if analyze_representations:
        print("\n" + "=" * 60)
        print("Computing graph structure labels for RSA analysis...")
        print("=" * 60)

        # Build transition matrices
        P, C, R, node_to_idx, idx_to_node, action_meanings = build_graph_transition_matrices(
            adjacency_graph, target_node
        )

        # Compute MDP homomorphism labels
        state_label, state_blocks = compute_mdp_homomorphism_labels(P, R)
        node_to_mdp_label = {idx_to_node[i]: int(state_label[i]) for i in range(len(state_label))}
        print(f"Found {len(state_blocks)} MDP homomorphism classes")

        # Compute optimal policy labels
        node_to_policy_label, node_to_cost, pi = compute_optimal_policy_labels(
            P, C, node_to_idx[target_node], idx_to_node
        )
        num_policy_classes = len(set(node_to_policy_label.values()))
        print(f"Found {num_policy_classes} optimal policy classes")

        # Compute depth labels (distance from target node)
        node_to_depth = compute_depth_labels(adjacency_graph, start_node=None, target_node=target_node)
        num_depth_classes = len(set(node_to_depth.values()))
        print(f"Found {num_depth_classes} depth classes (max depth: {max(node_to_depth.values())})")

    # Evaluate each model
    all_results = {}
    all_rsa_results = {}

    for model_key, model_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Evaluating {model_key}")
        print(f"{'='*60}")

        evaluator = ModelEvaluator(model_key, model_path, layer_interval=layer_interval)

        try:
            evaluator.load_model()
        except Exception as e:
            print(f"Failed to load {model_key}: {e}")
            continue

        model_results = {}

        # Run RSA analysis if requested (before navigation to avoid state changes)
        if analyze_representations:
            print("\n  Running RSA analysis...")
            rsa_results = run_rsa_analysis(
                evaluator,
                representations,
                start_nodes,
                target_node,
                adjacency_graph,
                node_to_mdp_label,
                node_to_policy_label,
                node_to_idx,
                node_to_depth,
                model_key,
                RESULTS_DIR,
                model_size
            )
            all_rsa_results[model_key] = rsa_results

        # Run navigation evaluation
        for rep_name, rep_content in representations.items():
            print(f"\n  Testing representation: {rep_name}")
            rep_results = []

            for i, nav_start in enumerate(start_nodes):
                print(f"    Start node {i+1}/{len(start_nodes)}: {nav_start}", end=" ")

                result = run_navigation_task(
                    evaluator,
                    rep_content,
                    nav_start,
                    target_node,
                    adjacency_graph,
                    direction_graph
                )

                result["start_node"] = nav_start
                result["target_node"] = target_node
                rep_results.append(result)

                print(f"-> {'SUCCESS' if result['success'] else 'FAILED'} ({result['steps']} steps)")

            # Calculate success rate
            success_count = sum(1 for r in rep_results if r["success"])
            success_rate = success_count / len(rep_results) * 100

            model_results[rep_name] = {
                "success_rate": success_rate,
                "success_count": success_count,
                "total_count": len(rep_results),
                "results": rep_results
            }

            # Add RSA results if available
            if analyze_representations and rep_name in all_rsa_results.get(model_key, {}):
                model_results[rep_name]["rsa_analysis"] = all_rsa_results[model_key][rep_name]

            print(f"  {rep_name}: {success_rate:.1f}% success ({success_count}/{len(rep_results)})")

        all_results[model_key] = model_results

        # Save results for this model
        output_file = Path(RESULTS_DIR) / f"{model_key}_{model_size}_results.json"
        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        print(f"\nSaved results to {output_file}")

        # Unload model
        evaluator.unload_model()

    # Save summary
    summary = {
        "model_size": model_size,
        "layer_interval": layer_interval,
        "models": list(all_results.keys()),
        "representations": list(rep_files.keys()),
        "num_start_nodes": len(start_nodes),
        "start_node": start_node,
        "target_node": target_node,
        "max_steps": MAX_STEPS,
        "spatial_maze_format": direction_graph is not None,
        "analyze_representations": analyze_representations,
        "results": {}
    }

    for model_key, model_results in all_results.items():
        summary["results"][model_key] = {
            rep_name: {
                "success_rate": data["success_rate"],
                "success_count": data["success_count"],
                "total_count": data["total_count"]
            }
            for rep_name, data in model_results.items()
        }

    # Add RSA summary if available
    if analyze_representations:
        summary["rsa_analysis"] = all_rsa_results
        summary["graph_labels"] = {
            "mdp_homomorphism": node_to_mdp_label,
            "optimal_policy": node_to_policy_label
        }

    summary_file = Path(RESULTS_DIR) / f"summary_{model_size}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Summary saved to {summary_file}")

    # Print final summary table
    print("\nSummary Table:")
    print(f"{'Model':<15} {'Representation':<20} {'Success Rate':<15}")
    print("-" * 50)
    for model_key in all_results:
        for rep_name, data in all_results[model_key].items():
            print(f"{model_key:<15} {rep_name:<20} {data['success_rate']:>6.1f}%")

    return all_results, all_rsa_results if analyze_representations else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate LLM models on graph navigation tasks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Model sizes:
  small: {', '.join(SMALL_MODELS.keys())} (~7-8B parameters)
  large: {', '.join(LARGE_MODELS.keys())} (~24-32B parameters)

Examples:
  python evaluate_models.py                              # Evaluate all small models
  python evaluate_models.py --size large                 # Evaluate all large models
  python evaluate_models.py qwen deepseek                # Evaluate specific small models
  python evaluate_models.py --size large qwen            # Evaluate specific large models
  python evaluate_models.py --analyze-representations   # Run with RSA analysis
  python evaluate_models.py --layer-interval 4          # Extract every 4th layer
"""
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="Models to evaluate (space-separated). If not specified, all models in the selected size tier will be evaluated."
    )
    parser.add_argument(
        "--size",
        choices=["small", "large"],
        default="small",
        help="Model size tier: 'small' (~7-8B) or 'large' (~24-32B). Default: small"
    )
    parser.add_argument(
        "--analyze-representations",
        action="store_true",
        help="Run RSA analysis on LLM hidden representations. "
             "Extracts hidden states before navigation and computes similarity metrics "
             "based on MDP homomorphism and optimal policy labels."
    )
    parser.add_argument(
        "--layer-interval",
        type=int,
        default=1,
        help="Interval for layer extraction (1=every layer, 2=every other layer, etc.). "
             "First and last layers are always included. Default: 1"
    )

    args = parser.parse_args()

    # If no models specified, evaluate all
    models_to_evaluate = args.models if args.models else None

    evaluate_all(
        models_to_evaluate,
        analyze_representations=args.analyze_representations,
        model_size=args.size,
        layer_interval=args.layer_interval
    )
