# Refactoring Summary: Separation of Concerns

## Problem

The original `evaluate_models.py` script duplicated the tree generation logic from `generate_tree.py`:
- Manually reconstructed the tree structure using the same random seed
- Recalculated node mappings and neighbor relationships
- Risk of inconsistency if generation parameters changed

## Solution

Refactored `evaluate_models.py` to read the graph structure directly from generated files:

### Key Changes

#### 1. **Removed Duplicated Code**

**Before:**
```python
SEED = 42
DEPTH = 5
NUM_NODES = 2**DEPTH - 1

def load_tree_map():
    nodes = list(range(1, NUM_NODES + 1))
    random.seed(SEED)
    random.shuffle(nodes)
    tree_map = {i: nodes[i] for i in range(NUM_NODES)}
    return tree_map

tree_map = load_tree_map()

def get_neighbors(current_node_id):
    idx = find_node_index(current_node_id)
    # Calculate parent, left child, right child...
    # Manually compute tree structure
```

**After:**
```python
adjacency_graph = {}
target_node = None

def load_graph_structure():
    # Read adjacency list file
    # Parse JSON graph structure
    # Extract target node
    return adjacency_graph, target_node

def get_neighbors(current_node_id):
    return adjacency_graph.get(current_node_id, [])
```

#### 2. **Single Source of Truth**

- `generate_tree.py` creates the graph structure and saves it
- `evaluate_models.py` reads the graph structure from files
- No need to synchronize seeds or parameters

#### 3. **Simplified Dependencies**

**Removed:**
- `random` module (no longer needed)
- `SEED`, `DEPTH`, `NUM_NODES` constants
- `load_tree_map()`, `find_node_index()` functions
- Complex tree traversal logic

**Added:**
- `load_graph_structure()` - parses adjacency list file
- Global `adjacency_graph` dict - stores graph structure
- Global `target_node` - stores target from file

## Benefits

### 1. **Maintainability**
- Change graph generation → just re-run `generate_tree.py`
- No need to update evaluation script
- Single place to modify graph structure

### 2. **Correctness**
- Evaluation uses exact same graph as in representation files
- No risk of seed mismatch or calculation errors
- Target node read from file (not recalculated)

### 3. **Flexibility**
- Can evaluate any graph structure (not just binary trees)
- Can modify generation parameters without touching evaluation code
- Can manually create graph files for testing

### 4. **Simplicity**
- Evaluation script is now ~50 lines shorter
- Clearer separation of responsibilities
- Easier to understand and debug

## File Structure

```
llm_tree/
├── generate_tree.py          # Creates graph & representations
│   └── Outputs to: tree_representations/
│       ├── 1_edges_ordered.txt
│       ├── 2_edges_random.txt
│       ├── 3_ascii_tree.txt
│       ├── 4_adjacency_list.txt  ← Source of truth
│       ├── 5_relation_triples.txt
│       └── 6_path_mapping.txt
│
├── evaluate_models.py         # Reads graph from files
│   └── Reads: tree_representations/4_adjacency_list.txt
│   └── Outputs to: evaluation_results/
│
└── test_graph_loading.py      # Verify graph loading works
```

## Workflow

### Before (Duplicated Logic)
```
1. Run generate_tree.py
   → Creates representations with SEED=42, DEPTH=5

2. Run evaluate_models.py
   → Must use same SEED=42, DEPTH=5
   → Recreates same tree structure
   → Risk: If parameters don't match → wrong graph!
```

### After (Single Source)
```
1. Run generate_tree.py
   → Creates representations
   → Saves graph structure in adjacency_list.txt

2. Run evaluate_models.py
   → Loads graph from adjacency_list.txt
   → Always correct, always matches representations
```

## Testing

Run the test script to verify graph loading:

```bash
conda run -n sac python test_graph_loading.py
```

Expected output:
```
✅ Found adjacency list file
✅ Loaded graph with 31 nodes
✅ Target node: 21
✅ All neighbor references are valid
✅ Number of start nodes: 30
✅ All tests passed!
```

## Migration Notes

If you have existing evaluation code:

1. **No changes needed to `generate_tree.py`**
   - Already outputs adjacency list file
   - Already includes target node

2. **Update `evaluate_models.py`**
   - Remove: `SEED`, `DEPTH`, `NUM_NODES`, `load_tree_map()`
   - Add: `load_graph_structure()` call at start
   - Replace: `get_neighbors()` with dict lookup

3. **Verify with test**
   - Run `test_graph_loading.py`
   - Ensure all tests pass

## Future Improvements

This refactoring enables:

1. **Different graph types**
   - Create non-tree graphs
   - Test on regular graphs, grids, etc.

2. **Dynamic generation**
   - Generate graphs on-the-fly
   - Test different sizes/depths

3. **Custom test cases**
   - Manually create challenging graphs
   - Add adversarial examples

4. **Batch evaluation**
   - Evaluate multiple graph configurations
   - Compare performance across graph types
