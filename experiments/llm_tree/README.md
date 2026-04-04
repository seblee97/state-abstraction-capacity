# LLM Tree Navigation Evaluation

This project evaluates small language models (7B parameter range) on tree navigation tasks using different graph representations.

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Generate tree representations:
```bash
python generate_tree.py
```

This creates 6 different representations in `tree_representations/`:

**Important:** The evaluation script reads the graph structure from these files (specifically from `4_adjacency_list.txt`). You must run `generate_tree.py` before running the evaluation.
- `1_edges_ordered.txt`: Edge list in sorted order
- `2_edges_random.txt`: Edge list in random order
- `3_ascii_tree.txt`: ASCII tree visualization
- `4_adjacency_list.txt`: JSON adjacency list (randomized)
- `5_relation_triples.txt`: Parent/child relation triples (randomized)
- `6_path_mapping.txt`: Node to path string mapping (randomized)

## Running Evaluation

### Step 1: Verify Setup (Optional but Recommended)

Test that the graph loading works correctly:
```bash
python test_graph_loading.py
```

You should see:
```
✅ Found adjacency list file
✅ Loaded graph with 31 nodes
✅ Target node: 21
✅ All tests passed!
```

### Step 2: Run Evaluation

Run the evaluation script:
```bash
python evaluate_models.py
```

This will:
1. Load graph structure from `tree_representations/4_adjacency_list.txt`
2. Load each model sequentially (Ministral, Qwen, DeepSeek)
3. Test each model on all 6 representations
4. For each representation, test navigation from all possible start nodes to the target node
5. Save detailed results to `evaluation_results/`

**Note:** This process can take several hours depending on your hardware. Models are loaded one at a time to manage memory.

## Model Configuration

The script evaluates these models by default:
- **Ministral**: `mistralai/Ministral-8B-Instruct-2410`
- **Qwen**: `Qwen/Qwen2.5-7B-Instruct`
- **DeepSeek**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

To modify models, edit the `MODELS` dictionary in `evaluate_models.py`.

## Visualizing Results

After evaluation completes, generate visualizations:
```bash
python visualize_results.py
```

This creates:
- `heatmap.png`: Success rate heatmap (models × representations)
- `bar_chart.png`: Bar chart comparing models across representations
- `{model}_step_distribution.png`: Step distribution histograms for each model

## Results Format

Results are saved in JSON format:

### Individual Model Results (`{model}_results.json`)
```json
{
  "edges_ordered": {
    "success_rate": 85.5,
    "success_count": 25,
    "total_count": 30,
    "results": [
      {
        "start_node": 5,
        "target_node": 31,
        "success": true,
        "steps": 8,
        "history": [...],
        "reason": "Target reached"
      },
      ...
    ]
  },
  ...
}
```

### Summary (`summary.json`)
```json
{
  "models": ["ministral", "qwen", "deepseek"],
  "representations": ["edges_ordered", ...],
  "num_start_nodes": 30,
  "target_node": 31,
  "max_steps": 20,
  "results": {
    "ministral": {
      "edges_ordered": {
        "success_rate": 85.5,
        "success_count": 25,
        "total_count": 30
      },
      ...
    },
    ...
  }
}
```

## Navigation Task

The task follows these rules:
1. Agent starts at a given node in the graph
2. Target is always the last leaf node (explicitly shown in representations)
3. At each step, the agent is shown which nodes are connected to the current node
4. The agent must respond with a node ID to move to
5. **Invalid moves:** Attempting to move to a non-connected node results in staying at the current node with feedback, but the agent can continue
6. Maximum 40 steps allowed
7. Success = reaching target node within step limit

**Example interaction (successful move):**
```
Current: Node 15. Goal: Node 31.
You can move to: 7, 30, 31
Response: 31
→ You moved to Node 31. Success!
```

**Example interaction (invalid move with recovery):**
```
Current: Node 15. Goal: Node 31.
You can move to: 7, 30, 31
Response: 25
→ Invalid move: Node 25 is not connected to Node 15. You remain at Node 15.

Current: Node 15. Goal: Node 31.
You can move to: 7, 30, 31
Response: 31
→ You moved to Node 31. Success!
```

**Key advantage:** This tests whether models truly understand the graph structure rather than just following procedural tree navigation rules. Models can recover from mistakes, testing their ability to learn from feedback.

## Configuration

Edit these parameters in the scripts:

### `generate_tree.py`
- `SEED`: Random seed for tree generation
- `DEPTH`: Tree depth (default: 5, giving 31 nodes)

**Note:** The evaluation script automatically reads the graph structure from the generated files, so you don't need to manually sync any parameters between scripts.

### `evaluate_models.py`
- `MAX_STEPS`: Maximum navigation steps (default: 40)
- `MODELS`: Dictionary of models to evaluate

**Important:** The evaluation script loads the graph structure from `tree_representations/4_adjacency_list.txt`, so any changes to the tree (SEED, DEPTH) require re-running `generate_tree.py`.

## Memory Requirements

- Each 7B model requires ~14GB GPU VRAM (fp16) or ~28GB RAM (fp32)
- Models are loaded sequentially to reduce peak memory
- Use `device_map="auto"` for automatic device placement

## Troubleshooting

**Out of memory:**
- Reduce batch size or use smaller models
- Enable CPU offloading with `device_map="auto"`
- Run one model at a time manually

**Model download issues:**
- Ensure you have HuggingFace access tokens for gated models
- Use `huggingface-cli login` to authenticate

**Slow inference:**
- Use GPU if available
- Reduce `MAX_STEPS` to limit evaluation time
- Test on subset of start nodes first

## Example Workflow

```bash
# 1. Generate tree representations
python generate_tree.py

# 2. Run evaluation (may take hours)
python evaluate_models.py

# 3. Generate visualizations
python visualize_results.py

# 4. View results
cat evaluation_results/summary.json
```

## Extending the Evaluation

To add a new model:
1. Add entry to `MODELS` dict in `evaluate_models.py`
2. Ensure model is compatible with `AutoModelForCausalLM`

To add a new representation:
1. Add generation function to `generate_tree.py`
2. Add file to `representations` dict in `main()`
3. Re-run evaluation

To modify the task:
1. Edit `get_valid_actions()` for different action spaces
2. Edit `parse_action()` for different response formats
3. Edit navigation prompt in `run_navigation_task()`
