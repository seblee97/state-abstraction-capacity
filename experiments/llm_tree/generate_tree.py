import random
import json
import os

# --- Configuration ---
SEED = 42          # Change this to generate a different random numbering
DEPTH = 4          # Depth of the fractal maze (number of splits)
OUTPUT_DIR = "tree_representations"

# --- Constants & Setup ---
random.seed(SEED)
NUM_NODES = 2**DEPTH - 1  # For depth 4, this is 15 nodes
nodes = list(range(1, NUM_NODES + 1))
random.shuffle(nodes)  # Randomize node IDs

# Maps logical tree index (0-based) to the randomized Node ID
# Index 0 is root (start). Children of i are 2i+1 and 2i+2.
tree_map = {i: nodes[i] for i in range(NUM_NODES)}

# --- Spatial Direction Mapping ---
# The fractal maze alternates between horizontal and vertical splits:
# - Level 0 (root): Start position
# - Level 1: First split goes East, then splits North/South
# - Level 2: Vertical corridors split into East/West
# - Level 3: Horizontal corridors split into North/South
# - And so on...
#
# At each junction:
# - Even depth (0, 2, 4...): horizontal movement, children are North/South
# - Odd depth (1, 3, 5...): vertical movement, children are East/West

def get_depth(index):
    """Get the depth of a node in the tree (0-indexed)."""
    if index == 0:
        return 0
    return get_depth((index - 1) // 2) + 1

def get_child_directions(parent_index):
    """
    Get the directions to children based on parent's depth.
    Even depth: children are North (left) and South (right)
    Odd depth: children are West (left) and East (right)
    """
    depth = get_depth(parent_index)
    if depth % 2 == 0:  # Horizontal corridor, splits vertically
        return ("North", "South")
    else:  # Vertical corridor, splits horizontally
        return ("West", "East")

def get_direction_to_parent(child_index):
    """Get the direction from child back to parent."""
    if child_index == 0:
        return None
    parent_index = (child_index - 1) // 2
    is_left_child = (child_index == 2 * parent_index + 1)

    depth = get_depth(parent_index)
    if depth % 2 == 0:  # Parent splits vertically
        return "South" if is_left_child else "North"  # Opposite direction
    else:  # Parent splits horizontally
        return "East" if is_left_child else "West"  # Opposite direction

def get_spatial_path(index):
    """
    Get the spatial path from start to this node.
    Returns a list of directions taken.
    """
    if index == 0:
        return []

    parent_index = (index - 1) // 2
    is_left_child = (index == 2 * parent_index + 1)

    parent_path = get_spatial_path(parent_index)

    # First move from root is always East
    if parent_index == 0:
        parent_path.append("East")

    # Add direction based on which child we are
    left_dir, right_dir = get_child_directions(parent_index)
    if is_left_child:
        parent_path.append(left_dir)
    else:
        parent_path.append(right_dir)

    return parent_path

# --- The Standard Navigation Text ---
NAV_TASK_BLURB = (
    "TASK DESCRIPTION:\n"
    "You are an agent navigating a spatial maze with corridors and junctions. "
    "Your position is defined by the current node ID. At each step, you will be given "
    "a list of connected nodes that you can move to. You must respond with the node ID "
    "you want to move to next.\n"
    "Your goal is to traverse the maze efficiently to reach a target node ID.\n"
    "The maze has a fractal structure: corridors alternate between horizontal (East/West) "
    "and vertical (North/South) directions at each junction.\n"
    "Below is the representation of the maze.\n"
    "---------------------------------------------------\n\n"
)

# --- Graph Representation Descriptions ---
REPRESENTATION_DESCRIPTIONS = {
    "1_edges_directed.txt": (
        "GRAPH FORMAT: Directed Edge List (Ordered)\n"
        "Each line shows a connection between two nodes with the cardinal direction.\n"
        "Format: 'NodeA --Direction--> NodeB' means moving Direction from A reaches B.\n"
        "Edges are listed in a structured order following the maze hierarchy.\n\n"
    ),
    "2_edges_random.txt": (
        "GRAPH FORMAT: Directed Edge List (Randomized)\n"
        "Each line shows a connection between two nodes with the cardinal direction.\n"
        "Format: 'NodeA --Direction--> NodeB' means moving Direction from A reaches B.\n"
        "Edges are listed in random order with no structural hints.\n\n"
    ),
    "3_ascii_maze.txt": (
        "GRAPH FORMAT: ASCII Maze Diagram\n"
        "A visual representation of the fractal maze structure.\n"
        "Shows junctions and corridors with cardinal directions labeled.\n\n"
    ),
    "4_adjacency_list.txt": (
        "GRAPH FORMAT: Adjacency List with Directions (JSON)\n"
        "A dictionary mapping each node ID to its neighbors with direction labels.\n"
        "Format: {node: {direction: neighbor_node, ...}, ...}\n"
        "Node order and direction order are randomized.\n\n"
    ),
    "5_relation_triples.txt": (
        "GRAPH FORMAT: Spatial Relation Triples\n"
        "Explicit directional relations: North(from, to), South(from, to), East(from, to), West(from, to).\n"
        "Triples are listed in random order.\n\n"
    ),
    "6_path_mapping.txt": (
        "GRAPH FORMAT: Spatial Path Mapping\n"
        "Each node is identified by its path from start using cardinal directions.\n"
        "Example: 'East-North-West' means go East, then North, then West from start.\n"
        "Mappings are listed in random order.\n\n"
    ),
}

# --- Helper Functions ---

def get_edges_directed(sort_order=True):
    """Generates a list of directed edges with cardinal directions."""
    edges = []

    for i in range(NUM_NODES):
        current_id = tree_map[i]
        left = 2 * i + 1
        right = 2 * i + 2

        if left < NUM_NODES:
            left_dir, right_dir = get_child_directions(i)
            left_id = tree_map[left]
            right_id = tree_map[right]

            # Forward edges (parent to children)
            edges.append(f"{current_id} --{left_dir}--> {left_id}")
            edges.append(f"{current_id} --{right_dir}--> {right_id}")

            # Backward edges (children to parent)
            opposite_left = {"North": "South", "South": "North", "East": "West", "West": "East"}[left_dir]
            opposite_right = {"North": "South", "South": "North", "East": "West", "West": "East"}[right_dir]
            edges.append(f"{left_id} --{opposite_left}--> {current_id}")
            edges.append(f"{right_id} --{opposite_right}--> {current_id}")

    if not sort_order:
        random.shuffle(edges)

    # Add target node information (last leaf node)
    last_leaf_index = NUM_NODES - 1
    target_node_id = tree_map[last_leaf_index]
    edges_str = "\n".join(edges)
    edges_str += f"\n\nStart Node: {tree_map[0]}"
    edges_str += f"\nTarget Node: {target_node_id}"

    return edges_str

def get_ascii_maze():
    """Generates an ASCII representation of the fractal maze."""
    lines = []

    # Build a simple text representation showing the structure
    lines.append("FRACTAL MAZE STRUCTURE")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Start: Node {tree_map[0]}")
    lines.append("")

    def _recurse(index, indent, came_from):
        node_id = tree_map[index]
        depth = get_depth(index)

        # Show current node with its incoming direction
        if index == 0:
            lines.append(f"{indent}[{node_id}] (Start)")
        else:
            lines.append(f"{indent}[{node_id}] (came from {came_from})")

        left = 2 * index + 1
        right = 2 * index + 2

        if left < NUM_NODES:
            left_dir, right_dir = get_child_directions(index)
            new_indent = indent + "    "

            lines.append(f"{indent}  |")
            lines.append(f"{indent}  +--{left_dir}-->")
            _recurse(left, new_indent, left_dir)

            lines.append(f"{indent}  |")
            lines.append(f"{indent}  +--{right_dir}-->")
            _recurse(right, new_indent, right_dir)

    _recurse(0, "", "")

    # Add target node information
    last_leaf_index = NUM_NODES - 1
    target_node_id = tree_map[last_leaf_index]
    lines.append("")
    lines.append("=" * 40)
    lines.append(f"Target Node: {target_node_id}")

    return "\n".join(lines)

def get_adjacency_list():
    """Generates a JSON adjacency list with direction labels."""
    adj = {tree_map[i]: {} for i in range(NUM_NODES)}

    for i in range(NUM_NODES):
        current_id = tree_map[i]

        # Add parent connection
        if i > 0:
            parent_idx = (i - 1) // 2
            parent_id = tree_map[parent_idx]
            direction_to_parent = get_direction_to_parent(i)
            adj[current_id][direction_to_parent] = parent_id

        # Add children connections
        left = 2 * i + 1
        right = 2 * i + 2

        if left < NUM_NODES:
            left_dir, right_dir = get_child_directions(i)
            adj[current_id][left_dir] = tree_map[left]
            adj[current_id][right_dir] = tree_map[right]

    # Randomize the order of nodes and directions
    node_keys = list(adj.keys())
    random.shuffle(node_keys)
    adj_randomized = {}
    for k in node_keys:
        dirs = list(adj[k].items())
        random.shuffle(dirs)
        adj_randomized[k] = dict(dirs)

    # Convert to string (pretty printed)
    result = json.dumps(adj_randomized, indent=2, sort_keys=False)

    # Add start and target node information
    last_leaf_index = NUM_NODES - 1
    target_node_id = tree_map[last_leaf_index]
    result += f"\n\nStart Node: {tree_map[0]}"
    result += f"\nTarget Node: {target_node_id}"

    return result

def get_triples():
    """Generates explicit spatial relation triples."""
    triples = []

    for i in range(NUM_NODES):
        current_id = tree_map[i]
        left = 2 * i + 1
        right = 2 * i + 2

        if left < NUM_NODES:
            left_dir, right_dir = get_child_directions(i)
            left_id = tree_map[left]
            right_id = tree_map[right]

            # Forward relations
            triples.append(f"{left_dir}({current_id}, {left_id})")
            triples.append(f"{right_dir}({current_id}, {right_id})")

            # Backward relations
            opposite = {"North": "South", "South": "North", "East": "West", "West": "East"}
            triples.append(f"{opposite[left_dir]}({left_id}, {current_id})")
            triples.append(f"{opposite[right_dir]}({right_id}, {current_id})")

    # Randomize the order of triples
    random.shuffle(triples)

    # Add start and target node information
    last_leaf_index = NUM_NODES - 1
    target_node_id = tree_map[last_leaf_index]
    result = "\n".join(triples)
    result += f"\n\nStart Node: {tree_map[0]}"
    result += f"\nTarget Node: {target_node_id}"

    return result

def get_paths():
    """Generates spatial path strings mapped to node IDs."""
    paths = []

    for i in range(NUM_NODES):
        node_id = tree_map[i]
        spatial_path = get_spatial_path(i)

        if len(spatial_path) == 0:
            path_str = "Start"
        else:
            path_str = "-".join(spatial_path)

        paths.append(f"Node {node_id}: {path_str}")

    # Randomize the order of path mappings
    random.shuffle(paths)

    # Add start and target node information
    last_leaf_index = NUM_NODES - 1
    target_node_id = tree_map[last_leaf_index]
    result = "\n".join(paths)
    result += f"\n\nStart Node: {tree_map[0]}"
    result += f"\nTarget Node: {target_node_id}"

    return result

# --- Main Execution ---

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    representations = {
        "1_edges_directed.txt": get_edges_directed(sort_order=True),
        "2_edges_random.txt": get_edges_directed(sort_order=False),
        "3_ascii_maze.txt": get_ascii_maze(),
        "4_adjacency_list.txt": get_adjacency_list(),
        "5_relation_triples.txt": get_triples(),
        "6_path_mapping.txt": get_paths()
    }

    print(f"Generating fractal maze with Seed {SEED}, Depth {DEPTH}...")
    print(f"Number of nodes: {NUM_NODES}")
    print(f"Start Node ID: {tree_map[0]}")
    print(f"Target Node ID: {tree_map[NUM_NODES - 1]}")

    for filename, content in representations.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(NAV_TASK_BLURB)
            f.write(REPRESENTATION_DESCRIPTIONS[filename])
            f.write(content)
        print(f"Created: {filepath}")

if __name__ == "__main__":
    main()
