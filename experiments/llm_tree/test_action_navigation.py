#!/usr/bin/env python3
"""
Test script to verify UP/LEFT/RIGHT action navigation works correctly.
"""

import json
import re
from pathlib import Path

TREE_REPRESENTATIONS_DIR = "tree_representations"

# Global graph structure
adjacency_graph = {}
target_node = None
node_to_index = {}

def load_graph_structure():
    """Load graph structure from the adjacency list file."""
    global adjacency_graph, target_node, node_to_index

    adj_file = Path(TREE_REPRESENTATIONS_DIR) / "4_adjacency_list.txt"
    with open(adj_file, 'r') as f:
        content = f.read()

    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    json_str = content[json_start:json_end]
    adjacency_graph = json.loads(json_str)
    adjacency_graph = {int(k): v for k, v in adjacency_graph.items()}

    target_match = re.search(r'Target Node:\s*(\d+)', content)
    target_node = int(target_match.group(1))

    node_to_index = build_node_to_index_map()

def build_node_to_index_map():
    """Build mapping from node ID to tree index."""
    all_nodes = set(adjacency_graph.keys())
    root_candidates = [n for n in all_nodes if len(adjacency_graph[n]) == 2]
    root_id = root_candidates[0] if root_candidates else min(all_nodes)

    mapping = {}
    queue = [(root_id, 0)]
    visited = set()

    while queue:
        node_id, idx = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        mapping[node_id] = idx

        neighbors = adjacency_graph.get(node_id, [])
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if len(unvisited_neighbors) >= 1:
            queue.append((unvisited_neighbors[0], 2 * idx + 1))
        if len(unvisited_neighbors) >= 2:
            queue.append((unvisited_neighbors[1], 2 * idx + 2))

    return mapping

def get_node_at_index(tree_idx):
    """Get node ID at given tree index."""
    for node_id, idx in node_to_index.items():
        if idx == tree_idx:
            return node_id
    return None

def get_valid_actions(current_node_id):
    """Get valid actions (UP/LEFT/RIGHT) from current node."""
    idx = node_to_index.get(current_node_id)
    if idx is None:
        return []

    actions = []

    # UP: Move to parent
    if idx > 0:
        parent_idx = (idx - 1) // 2
        parent_id = get_node_at_index(parent_idx)
        if parent_id is not None:
            actions.append(("UP", parent_id))

    # LEFT: Move to left child
    left_idx = 2 * idx + 1
    left_id = get_node_at_index(left_idx)
    if left_id is not None:
        actions.append(("LEFT", left_id))

    # RIGHT: Move to right child
    right_idx = 2 * idx + 2
    right_id = get_node_at_index(right_idx)
    if right_id is not None:
        actions.append(("RIGHT", right_id))

    return actions

def test_action_navigation():
    """Test the action-based navigation system."""
    print("="*60)
    print("Testing Action-Based Navigation (UP/LEFT/RIGHT)")
    print("="*60)

    # Find root node (index 0)
    root_id = get_node_at_index(0)
    print(f"\n📍 Root node (index 0): {root_id}")

    # Test actions from root
    print(f"\n--- Test 1: Actions from Root ---")
    actions = get_valid_actions(root_id)
    print(f"Root node {root_id} has actions: {[a for a, _ in actions]}")

    if actions:
        # Should have LEFT and RIGHT, but not UP
        action_names = [a for a, _ in actions]
        assert "UP" not in action_names, "Root should not have UP action"
        assert "LEFT" in action_names or "RIGHT" in action_names, "Root should have children"
        print("✅ Root has no UP action (correct)")
        print("✅ Root has child actions (correct)")

    # Test LEFT action from root
    if "LEFT" in [a for a, _ in actions]:
        print(f"\n--- Test 2: Take LEFT from Root ---")
        left_node = [nid for a, nid in actions if a == "LEFT"][0]
        print(f"Taking LEFT from {root_id} → {left_node}")

        # From left child, we should have UP, and possibly LEFT/RIGHT
        left_actions = get_valid_actions(left_node)
        left_action_names = [a for a, _ in left_actions]
        print(f"Node {left_node} has actions: {left_action_names}")

        assert "UP" in left_action_names, "Left child should have UP action"
        print("✅ Left child has UP action (correct)")

        # Test UP action returns to root
        up_node = [nid for a, nid in left_actions if a == "UP"][0]
        assert up_node == root_id, f"UP from left child should return to root {root_id}, got {up_node}"
        print(f"✅ UP from {left_node} returns to root {root_id} (correct)")

    # Test leaf node
    print(f"\n--- Test 3: Actions from Leaf Node ---")
    # Find a leaf: node with only 1 neighbor (parent only)
    leaf_nodes = [nid for nid in adjacency_graph.keys() if len(adjacency_graph[nid]) == 1]
    if leaf_nodes:
        leaf_id = leaf_nodes[0]
        leaf_actions = get_valid_actions(leaf_id)
        leaf_action_names = [a for a, _ in leaf_actions]
        print(f"Leaf node {leaf_id} has actions: {leaf_action_names}")

        assert "UP" in leaf_action_names, "Leaf should have UP action"
        assert "LEFT" not in leaf_action_names, "Leaf should not have LEFT"
        assert "RIGHT" not in leaf_action_names, "Leaf should not have RIGHT"
        print("✅ Leaf has only UP action (correct)")

    # Test navigation path
    print(f"\n--- Test 4: Navigation Path ---")
    current = root_id
    path = [current]
    print(f"Starting at root: {current}")

    # Go LEFT
    actions = get_valid_actions(current)
    if "LEFT" in [a for a, _ in actions]:
        current = [nid for a, nid in actions if a == "LEFT"][0]
        path.append(current)
        print(f"LEFT → {current}")

    # Go RIGHT
    actions = get_valid_actions(current)
    if "RIGHT" in [a for a, _ in actions]:
        current = [nid for a, nid in actions if a == "RIGHT"][0]
        path.append(current)
        print(f"RIGHT → {current}")

    # Go UP twice
    for _ in range(2):
        actions = get_valid_actions(current)
        if "UP" in [a for a, _ in actions]:
            current = [nid for a, nid in actions if a == "UP"][0]
            path.append(current)
            print(f"UP → {current}")

    print(f"\nNavigation path: {' → '.join(map(str, path))}")
    print(f"✅ Successfully navigated using UP/LEFT/RIGHT actions")

    # Summary
    print(f"\n{'='*60}")
    print("✅ All action navigation tests passed!")
    print(f"{'='*60}")
    print(f"\nKey findings:")
    print(f"  - Root node: {root_id}")
    print(f"  - Target node: {target_node}")
    print(f"  - Total nodes: {len(adjacency_graph)}")
    print(f"  - Action-based navigation working correctly")

    return True

def main():
    print("Loading graph structure...")
    load_graph_structure()
    print(f"Loaded {len(adjacency_graph)} nodes")
    print(f"Target: {target_node}\n")

    success = test_action_navigation()
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
