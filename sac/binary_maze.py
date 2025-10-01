import numpy as np

def generate_binary_tree_maze(tree_depth=5):
    """
    Generate a maze.txt with actual corridors forming a binary tree.
    
    The maze has T-junctions at each node, with corridors connecting them.
    - Action 0 (left) goes to left child
    - Action 1 (right) goes to right child  
    - Action 2 (back) goes to parent
    
    Returns:
    --------
    maze_str : str
        The maze as a string
    start_pos : tuple
        (y, x) coordinates of the start position 'S'
    reward_pos : tuple
        (y, x) coordinates of the reward position 'R'
    """
    
    # Corridor width (must be odd for centered junctions)
    corridor_width = 3
    
    # Calculate positions for each node in the tree
    # Each leaf needs horizontal space
    num_leaves = 2**(tree_depth - 1)
    leaf_spacing = 8  # Horizontal space per leaf
    
    # Calculate natural dimensions
    natural_width = num_leaves * leaf_spacing + 2
    level_height = 6
    natural_height = tree_depth * level_height + 4
    
    # Make it square by using the maximum dimension
    maze_size = max(natural_width, natural_height)
    maze_width = maze_size
    maze_height = maze_size
    
    # Calculate offsets to center the tree
    x_offset = (maze_width - natural_width) // 2
    y_offset = (maze_height - natural_height) // 2
    
    # Initialize with walls
    maze = [['#' for _ in range(maze_width)] for _ in range(maze_height)]
    
    # Store start and reward positions
    start_pos = None
    reward_pos = None
    
    # Helper function to carve corridor
    def carve_horizontal(y, x1, x2):
        """Carve a horizontal corridor"""
        y_start = y - corridor_width // 2
        y_end = y + corridor_width // 2 + 1
        for row in range(y_start, y_end):
            if 0 <= row < maze_height:
                for col in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= col < maze_width:
                        maze[row][col] = ' '
    
    def carve_vertical(x, y1, y2):
        """Carve a vertical corridor"""
        x_start = x - corridor_width // 2
        x_end = x + corridor_width // 2 + 1
        for col in range(x_start, x_end):
            if 0 <= col < maze_width:
                for row in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= row < maze_height:
                        maze[row][col] = ' '
    
    # Build tree level by level
    node_positions = {}  # Maps (level, node_idx) -> (y, x)
    
    for level in range(tree_depth):
        num_nodes = 2**level
        y = y_offset + 2 + level * level_height
        
        for node_idx in range(num_nodes):
            # Calculate x position - distribute evenly
            total_span = num_leaves * leaf_spacing
            node_span = total_span / num_nodes
            x = x_offset + int(1 + (node_idx + 0.5) * node_span)
            
            node_positions[(level, node_idx)] = (y, x)
            
            # Carve the T-junction area
            carve_vertical(x, y - 2, y + 2)
            carve_horizontal(y, x - 2, x + 2)
            
            # Place markers and store positions
            if level == 0:
                maze[y][x] = 'S'
                start_pos = (y, x)
            elif level == tree_depth - 1 and node_idx == 0:
                maze[y][x] = 'R'
                reward_pos = (y, x)
            
            # Connect to parent
            if level > 0:
                parent_idx = node_idx // 2
                parent_y, parent_x = node_positions[(level - 1, parent_idx)]
                
                # Vertical corridor from parent down
                carve_vertical(x, parent_y + 2, y - 2)
                
                # Horizontal corridor from x to parent_x at intermediate height
                mid_y = (parent_y + y) // 2
                carve_horizontal(mid_y, parent_x, x)
                
                # Connect the horizontal corridor to both vertical corridors
                carve_vertical(parent_x, parent_y + 2, mid_y)
                carve_vertical(x, mid_y, y - 2)
    
    maze_str = '\n'.join(''.join(row) for row in maze)
    return maze_str, start_pos, reward_pos

def generate_fractal_maze(depth=4, maze_size=63):
    """
    Generate a fractal maze with binary tree structure.
    
    Starting from the left, a horizontal corridor runs to the middle,
    then splits into vertical corridors (up/down). Each vertical corridor
    splits into horizontal corridors at half the distance, and so on.
    
    Parameters:
    -----------
    depth : int
        Number of recursive splits (fractal depth)
    maze_size : int
        Size of the square maze
    
    Returns:
    --------
    maze_str : str
        The maze as a string
    start_pos : tuple
        (y, x) coordinates of start position 'S'
    reward_pos : tuple
        (y, x) coordinates of reward position 'R'
    """
    
    # Initialize maze with walls
    maze = [['#' for _ in range(maze_size)] for _ in range(maze_size)]
    
    # Corridor width
    corridor_width = 3
    half_width = corridor_width // 2
    
    def carve_horizontal(y, x1, x2):
        """Carve a horizontal corridor from x1 to x2 at height y"""
        for dy in range(-half_width, half_width + 1):
            row = y + dy
            if 0 <= row < maze_size:
                for x in range(min(x1, x2), max(x1, x2) + 1):
                    if 0 <= x < maze_size:
                        maze[row][x] = ' '
    
    def carve_vertical(x, y1, y2):
        """Carve a vertical corridor from y1 to y2 at position x"""
        for dx in range(-half_width, half_width + 1):
            col = x + dx
            if 0 <= col < maze_size:
                for y in range(min(y1, y2), max(y1, y2) + 1):
                    if 0 <= y < maze_size:
                        maze[y][col] = ' '
    
    def fractal_split(x, y, direction, width, height, level):
        """
        Recursively create fractal corridors.
        
        Parameters:
        -----------
        x, y : int
            Current position
        width, height : int
            Available space in each direction
        level : int
            Current recursion depth
        is_horizontal : bool
            True if current corridor is horizontal, False if vertical
        """
        if level < 0 or (width < 10 and height < 10):
            return
        
        if direction == 'right':
            # Carve horizontal corridor for half the width
            x_end = x + width // 2
        else:
            # Carve horizontal corridor for half the width
            x_end = x - width // 2
        carve_horizontal(y, x, x_end)
        
        level -= 1
        if level < 0:
            return
        # At the end, split into two vertical corridors (up and down)
        split_x = x_end
        
        # Vertical corridor - goes up for half the available space above and down
        # for half the available space below
        y_up = y - height // 4
        y_down = y + height // 4
        carve_vertical(split_x, y_down, y_up)
        
        # Recursively continue - vertical corridors split horizontally in BOTH directions
        # Upper branch splits left and right
        new_width = width // 2
        new_height = height // 2
        
        # Upper-left
        fractal_split(split_x, y_up, 'left', new_width, new_height, level - 1)
        
        # Upper-right
        fractal_split(split_x, y_up, 'right', new_width, new_height, level - 1)
        
        # Lower branch splits left and right
        # Lower-left
        fractal_split(split_x, y_down, 'left', new_width, new_height, level - 1)
        
        # Lower-right
        fractal_split(split_x, y_down, 'right', new_width, new_height, level - 1)

    # Start position: left side, middle height
    start_y = maze_size // 2
    start_x = 2
    
    # Available space
    available_width = maze_size - 4
    available_height = maze_size - 4
    
    # Begin fractal generation starting with horizontal corridor
    fractal_split(start_x, start_y, 'right', available_width, available_height, depth)
    
    # Place start marker
    maze[start_y][start_x] = 'S'
    start_pos = (start_y, start_x)
    
    # Find reward position - look for rightmost open space
    reward_pos = None
    for x in range(maze_size - 1, -1, -1):
        for y in range(maze_size):
            if maze[y][x] == ' ':
                maze[y][x] = 'R'
                reward_pos = (y, x)
                break
        if reward_pos:
            break
    
    # Fallback if no open space found
    if not reward_pos:
        reward_pos = (start_y, start_x + 10)
        if maze[reward_pos[0]][reward_pos[1]] == ' ':
            maze[reward_pos[0]][reward_pos[1]] = 'R'
    
    # Convert maze to string
    maze_str = '\n'.join(''.join(row) for row in maze)
    
    return maze_str, start_pos, reward_pos

def generate_maze_from_mdp(tree_depth, P, R):
    """
    Generate maze from MDP definition.
    
    Parameters:
    -----------
    tree_depth : int
        Depth of binary tree
    P : np.ndarray  
        Transition matrix
    R : np.ndarray
        Reward matrix
    
    Returns:
    --------
    maze_str : str
        The maze as a string
    start_pos : tuple
        (y, x) coordinates of the start position 'S'
    reward_pos : tuple
        (y, x) coordinates of the reward position 'R'
    """
    return generate_fractal_maze(tree_depth)


# Example usage
if __name__ == "__main__":
    tree_depth = 6
    S, A = np.sum([2**x for x in range(tree_depth)]), 3
    
    # Build MDP as in paper
    P = np.zeros((S, A, S))
    for r in range(tree_depth - 1):
        for s in range(2**r):
            P[2**r - 1 + s, 0, 2**(r + 1) - 1 + 2 * s] = 1.0
            P[2**r - 1 + s, 1, 2**(r + 1) - 1 + 2 * s + 1] = 1.0
            if r > 0:
                P[2**r - 1 + s, 2, 2**(r - 1) - 1 + s // 2] = 1.0
    
    for s in range(2**(tree_depth - 1)):
        P[2**(tree_depth - 1) - 1 + s, 2, 2**(tree_depth - 2) - 1 + s // 2] = 1.0
    
    R = np.zeros((S, A))
    R[2**(tree_depth - 1) - 1, 0] = 1.0
    
    # Generate maze
    maze_str, start_pos, reward_pos = generate_maze_from_mdp(tree_depth, P, R)
    print(maze_str)
    print(f"\nStart position: {start_pos}")
    print(f"Reward position: {reward_pos}")
    print(f"\nMaze dimensions: {len(maze_str.split(chr(10)))} rows x {len(maze_str.split(chr(10))[0])} columns")
    
    with open('maze.txt', 'w') as f:
        f.write(maze_str)