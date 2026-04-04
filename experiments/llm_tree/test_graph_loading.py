#!/usr/bin/env python3
"""
Test script to verify graph loading works correctly.
Run this before running the full evaluation.
"""

import json
import re
from pathlib import Path

TREE_REPRESENTATIONS_DIR = "tree_representations"

def test_graph_loading():
    """Test that we can load the graph structure from adjacency list."""

    adj_file = Path(TREE_REPRESENTATIONS_DIR) / "4_adjacency_list.txt"

    if not adj_file.exists():
        print(f"❌ Adjacency list file not found: {adj_file}")
        print("Please run: python generate_tree.py")
        return False

    print(f"✅ Found adjacency list file: {adj_file}")

    with open(adj_file, 'r') as f:
        content = f.read()

    # Extract JSON
    json_start = content.find('{')
    json_end = content.rfind('}') + 1
    json_str = content[json_start:json_end]

    # Parse adjacency list
    adjacency_graph = json.loads(json_str)
    adjacency_graph = {int(k): v for k, v in adjacency_graph.items()}

    print(f"✅ Loaded graph with {len(adjacency_graph)} nodes")

    # Extract target node
    target_match = re.search(r'Target Node:\s*(\d+)', content)
    if target_match:
        target_node = int(target_match.group(1))
        print(f"✅ Target node: {target_node}")
    else:
        print("❌ Could not find target node")
        return False

    # Verify graph structure
    print("\n📊 Graph Statistics:")
    print(f"   Total nodes: {len(adjacency_graph)}")
    print(f"   Target node: {target_node}")

    # Check a few nodes
    sample_nodes = sorted(adjacency_graph.keys())[:3]
    print("\n🔍 Sample nodes and their neighbors:")
    for node in sample_nodes:
        neighbors = adjacency_graph[node]
        print(f"   Node {node}: neighbors = {neighbors}")

    # Verify all nodes have valid neighbors
    all_node_ids = set(adjacency_graph.keys())
    for node_id, neighbors in adjacency_graph.items():
        for neighbor in neighbors:
            if neighbor not in all_node_ids:
                print(f"❌ Node {node_id} has invalid neighbor: {neighbor}")
                return False

    print("\n✅ All neighbor references are valid")

    # Test get_neighbors function
    def get_neighbors(node_id):
        return adjacency_graph.get(node_id, [])

    test_node = sample_nodes[0]
    neighbors = get_neighbors(test_node)
    print(f"\n✅ get_neighbors({test_node}) = {neighbors}")

    # Test getting all start nodes
    start_nodes = [n for n in adjacency_graph.keys() if n != target_node]
    print(f"\n✅ Number of start nodes (excluding target): {len(start_nodes)}")

    print("\n" + "="*60)
    print("✅ All tests passed! Graph loading works correctly.")
    print("="*60)

    return True

if __name__ == "__main__":
    success = test_graph_loading()
    exit(0 if success else 1)
