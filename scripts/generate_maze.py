# generate_maze.py
"""
Utility to generate valid mazes and convert to bitmask format.

Uses mazelib library for high-quality maze generation.
Install: pip install mazelib
"""

import numpy as np
import json
import argparse
import matplotlib.pyplot as plt

try:
    from mazelib import Maze
    from mazelib.generate.Prims import Prims
    from mazelib.generate.BacktrackingGenerator import BacktrackingGenerator
    from mazelib.generate.Wilsons import Wilsons
    MAZELIB_AVAILABLE = True
except ImportError:
    MAZELIB_AVAILABLE = False
    print("Warning: mazelib not installed. Install with: pip install mazelib")


def generate_maze_with_mazelib(width, height, start=(0, 0), seed=None, algorithm='prims'):
    """
    Generate a maze using mazelib library.

    Args:
        width: Maze width (desired final grid size)
        height: Maze height (desired final grid size)
        start: Starting cell for generation (not used directly by mazelib)
        seed: Random seed for reproducibility
        algorithm: 'prims', 'backtracker', or 'wilsons'

    Returns:
        A 2D numpy array of shape (height, width) where True = corridor, False = wall
    """
    if not MAZELIB_AVAILABLE:
        raise ImportError("mazelib is required. Install with: pip install mazelib")

    if seed is not None:
        np.random.seed(seed)

    # mazelib generates grids with dimensions (2*h + 1) x (2*w + 1)
    # because it places walls between cells.
    # To get a final size of (height, width), we need to calculate the input dimensions.
    # For exact size, we use: input_dim = (target_dim - 1) // 2
    # This gives us a slightly smaller maze that we can pad if needed.

    input_height = max(1, (height - 1) // 2)
    input_width = max(1, (width - 1) // 2)

    # Create maze object
    m = Maze()
    m.generator = {
        'prims': Prims(input_height, input_width),
        'backtracker': BacktrackingGenerator(input_height, input_width),
        'wilsons': Wilsons(input_height, input_width),
    }[algorithm]

    # Generate the maze
    m.generate()

    # mazelib returns a grid where:
    # - 1 = wall
    # - 0 = corridor
    # We want True = corridor, False = wall
    raw_maze = (m.grid == 0)

    # The generated maze might not be exactly the requested size
    # Pad or crop to get exact dimensions
    current_h, current_w = raw_maze.shape

    if current_h == height and current_w == width:
        maze_grid = raw_maze
    elif current_h < height or current_w < width:
        # Pad with walls if too small
        maze_grid = np.zeros((height, width), dtype=bool)
        maze_grid[:current_h, :current_w] = raw_maze
    else:
        # Crop if too large (shouldn't happen with our calculation, but just in case)
        maze_grid = raw_maze[:height, :width]

    return maze_grid


def maze_to_bitmask(maze_grid):
    """
    Convert a boolean maze grid to bitmask format.

    Args:
        maze_grid: 2D numpy array where True = corridor, False = wall

    Returns:
        2D list of integers with bitmask encoding:
        - Bit 0 (value 1): up available
        - Bit 1 (value 2): down available
        - Bit 2 (value 4): left available
        - Bit 3 (value 8): right available
    """
    height, width = maze_grid.shape
    bitmask = np.zeros((height, width), dtype=int)

    for y in range(height):
        for x in range(width):
            if not maze_grid[y, x]:
                # Wall cell, no available actions
                bitmask[y, x] = 0
                continue

            mask = 0

            # Check up (y+1)
            if y + 1 < height and maze_grid[y + 1, x]:
                mask |= 1  # Bit 0

            # Check down (y-1)
            if y - 1 >= 0 and maze_grid[y - 1, x]:
                mask |= 2  # Bit 1

            # Check left (x-1)
            if x - 1 >= 0 and maze_grid[y, x - 1]:
                mask |= 4  # Bit 2

            # Check right (x+1)
            if x + 1 < width and maze_grid[y, x + 1]:
                mask |= 8  # Bit 3

            bitmask[y, x] = mask

    return bitmask.tolist()


def ensure_path_exists(maze_grid, start, goal):
    """
    Check if a path exists from start to goal using BFS.

    Args:
        maze_grid: 2D boolean array
        start: (x, y) tuple
        goal: (x, y) tuple

    Returns:
        True if path exists, False otherwise
    """
    from collections import deque

    height, width = maze_grid.shape
    visited = np.zeros((height, width), dtype=bool)
    queue = deque([start])
    visited[start[1], start[0]] = True

    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

    while queue:
        x, y = queue.popleft()

        if (x, y) == goal:
            return True

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if (0 <= nx < width and 0 <= ny < height and
                maze_grid[ny, nx] and not visited[ny, nx]):
                visited[ny, nx] = True
                queue.append((nx, ny))

    return False


def carve_path_if_needed(maze_grid, start, goal):
    """
    If no path exists, carve a simple path from start to goal.
    This ensures the maze is always solvable.
    """
    from collections import deque

    if ensure_path_exists(maze_grid, start, goal):
        return maze_grid

    print(f"No path found, carving path from {start} to {goal}")

    # Simple A* pathfinding to carve a path
    height, width = maze_grid.shape

    # Use BFS but carve cells as we go
    visited = np.zeros((height, width), dtype=bool)
    parent = {}
    queue = deque([start])
    visited[start[1], start[0]] = True
    maze_grid[start[1], start[0]] = True

    directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
    found = False

    while queue and not found:
        x, y = queue.popleft()

        if (x, y) == goal:
            found = True
            break

        for dx, dy in directions:
            nx, ny = x + dx, y + dy

            if 0 <= nx < width and 0 <= ny < height and not visited[ny, nx]:
                visited[ny, nx] = True
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))

    # Backtrack and carve path
    if found:
        current = goal
        while current in parent:
            maze_grid[current[1], current[0]] = True
            current = parent[current]

    return maze_grid


def visualize_maze(maze_grid, bitmask, start, goal, title="Generated Maze"):
    """Visualize the maze with start and goal points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    height, width = maze_grid.shape

    # Left: Boolean maze - walls are black, corridors are white
    # Invert the maze so True (corridor) = white, False (wall) = black
    ax1.imshow(~maze_grid, cmap='binary', interpolation='nearest')
    ax1.scatter(start[0], start[1], marker='o', s=200, color='green', label='Start', zorder=5)
    ax1.scatter(goal[0], goal[1], marker='*', s=300, color='red', label='Goal', zorder=5)
    ax1.set_title('Maze Grid (Black=Wall, White=Corridor)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: Bitmask visualization with corridors
    ax2.set_xlim(-0.5, width - 0.5)
    ax2.set_ylim(-0.5, height - 0.5)
    ax2.set_aspect('equal')

    for y in range(height):
        for x in range(width):
            if bitmask[y][x] == 0:
                # Wall - black
                ax2.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                # Corridor - white
                ax2.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray', linewidth=0.5))

                # Show bitmask value as text
                ax2.text(x, y, str(bitmask[y][x]), ha='center', va='center', fontsize=8, color='blue')

    ax2.scatter(start[0], start[1], marker='o', s=200, color='green', label='Start', zorder=5)
    ax2.scatter(goal[0], goal[1], marker='*', s=300, color='red', label='Goal', zorder=5)
    ax2.set_title('Bitmask Encoding (Black=Wall, White=Corridor)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.invert_yaxis()  # Match array indexing

    plt.suptitle(title)
    plt.tight_layout()

    return fig


def main():
    parser = argparse.ArgumentParser(description="Generate maze and convert to bitmask format")
    parser.add_argument("--width", type=int, default=10, help="Maze width")
    parser.add_argument("--height", type=int, default=10, help="Maze height")
    parser.add_argument("--start-x", type=int, default=0, help="Start X coordinate")
    parser.add_argument("--start-y", type=int, default=0, help="Start Y coordinate")
    parser.add_argument("--goal-x", type=int, default=9, help="Goal X coordinate")
    parser.add_argument("--goal-y", type=int, default=5, help="Goal Y coordinate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--algorithm", type=str, default="prims",
                       choices=["prims", "backtracker", "wilsons"],
                       help="Maze generation algorithm (prims is recommended)")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--name", type=str, default="maze", help="Maze name for output")
    args = parser.parse_args()

    start = (args.start_x, args.start_y)
    goal = (args.goal_x, args.goal_y)

    print(f"Generating {args.width}x{args.height} maze using {args.algorithm}...")
    print(f"Start: {start}, Goal: {goal}")
    print(f"Seed: {args.seed}")

    # Generate maze using mazelib
    maze_grid = generate_maze_with_mazelib(
        args.width, args.height, start=start, seed=args.seed, algorithm=args.algorithm
    )

    # Ensure goal is reachable
    maze_grid = carve_path_if_needed(maze_grid, start, goal)

    # Verify path exists
    if ensure_path_exists(maze_grid, start, goal):
        print("✓ Path from start to goal exists!")
    else:
        print("✗ No path found (this shouldn't happen)")

    # Convert to bitmask
    bitmask = maze_to_bitmask(maze_grid)

    # Print bitmask in JSON format
    print("\nBitmask encoding:")
    print(json.dumps(bitmask, indent=2))

    # Save to file if requested
    if args.output:
        output_data = {
            "name": args.name,
            "width": args.width,
            "height": args.height,
            "start_pos": list(start),
            "goal_pos": list(goal),
            "maze_layout": bitmask,
            "seed": args.seed,
            "algorithm": args.algorithm,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved maze to {args.output}")

    # Visualize if requested
    if args.visualize:
        fig = visualize_maze(maze_grid, bitmask, start, goal,
                           title=f"{args.name} ({args.algorithm}, seed={args.seed})")
        plt.show()

    return bitmask


if __name__ == "__main__":
    main()
