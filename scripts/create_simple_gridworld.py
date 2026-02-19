# create_simple_gridworld.py
"""
Create simple gridworld layouts for testing.

Layout A: 3x3 block at bottom edge (rows 7-9, cols 3-5)
Layout B: 4x4 block at bottom edge (rows 6-9, cols 3-6)

Start: (0, 0)
Goal: (9, 9)
"""

import json
import numpy as np
import os


def create_layout_with_obstacle(obstacle_rows, obstacle_cols):
    """
    Create a 10x10 gridworld with all cells open except for one rectangular obstacle.

    Args:
        obstacle_rows: tuple (row_start, row_end) for obstacle
        obstacle_cols: tuple (col_start, col_end) for obstacle

    Returns:
        10x10 bitmask layout
    """
    # Initialize all cells as fully open (all 4 directions available = 15)
    layout = np.full((10, 10), 15, dtype=int)

    # Set obstacle cells to 0 (walls)
    row_start, row_end = obstacle_rows
    col_start, col_end = obstacle_cols
    layout[row_start:row_end, col_start:col_end] = 0

    # Now update adjacent cells to block directions toward the obstacle
    # For each navigable cell, check if each direction leads to obstacle
    for y in range(10):
        for x in range(10):
            if layout[y, x] == 0:  # Skip obstacle cells
                continue

            bitmask = 0

            # Check up (y+1)
            if y + 1 < 10 and layout[y + 1, x] != 0:
                bitmask |= 1

            # Check down (y-1)
            if y - 1 >= 0 and layout[y - 1, x] != 0:
                bitmask |= 2

            # Check left (x-1)
            if x - 1 >= 0 and layout[y, x - 1] != 0:
                bitmask |= 4

            # Check right (x+1)
            if x + 1 < 10 and layout[y, x + 1] != 0:
                bitmask |= 8

            layout[y, x] = bitmask

    return layout.tolist()


def main():
    """Generate simple gridworld layouts and update config."""

    # Layout A: 3x3 obstacle at bottom edge (rows 7-9, cols 3-5)
    print("Creating Layout A (3x3 obstacle at bottom edge)...")
    layout_a = create_layout_with_obstacle(
        obstacle_rows=(7, 10),  # rows 7,8,9
        obstacle_cols=(3, 6)    # cols 3,4,5
    )

    # Layout B: 4x4 obstacle at bottom edge (rows 6-9, cols 3-6)
    print("Creating Layout B (4x4 obstacle at bottom edge)...")
    layout_b = create_layout_with_obstacle(
        obstacle_rows=(6, 10),  # rows 6,7,8,9
        obstacle_cols=(3, 7)    # cols 3,4,5,6
    )

    # Count navigable cells
    navigable_a = sum(1 for row in layout_a for cell in row if cell > 0)
    navigable_b = sum(1 for row in layout_b for cell in row if cell > 0)

    print(f"\nLayout A: {navigable_a} navigable cells (out of 100)")
    print(f"  - 3×3 obstacle at bottom edge (rows 7-9, cols 3-5)")
    print(f"Layout B: {navigable_b} navigable cells (out of 100)")
    print(f"  - 4×4 obstacle at bottom edge (rows 6-9, cols 3-6)")

    # Load existing config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "gridworld.json"
    )

    with open(config_path, "r") as f:
        config = json.load(f)

    # Update layouts
    config["data_collection"]["env_params"]["maze_layout"] = layout_a
    config["finetuning"]["env_params"]["maze_layout"] = layout_b

    # Update goal to (9, 9)
    config["data_collection"]["cost_fn_params"]["goal_state"] = [9, 9]
    config["data_collection"]["domain_randomization"]["goal_state"]["min"] = [8, 8]
    config["data_collection"]["domain_randomization"]["goal_state"]["max"] = [9, 9]

    config["finetuning"]["cost_fn_params"]["goal_state"] = [9, 9]
    config["finetuning"]["evaluator_params"]["goal_state"] = [9, 9]
    config["finetuning"]["evaluator_params"]["cost_fn_params"]["goal_state"] = [9, 9]

    # Save updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nUpdated {config_path} with simple layouts")
    print(f"  - Layout A: 3×3 obstacle at rows 7-9, cols 3-5 (bottom edge)")
    print(f"  - Layout B: 4×4 obstacle at rows 6-9, cols 3-6 (bottom edge)")
    print(f"  - Start: (0, 0)")
    print(f"  - Goal: (9, 9)")

    # Visualize
    print("\nLayout A (3×3 bottom obstacle):")
    print_layout(layout_a)

    print("\nLayout B (4×4 bottom obstacle):")
    print_layout(layout_b)


def print_layout(layout):
    """Print a simple ASCII visualization of the layout."""
    print("  ", end="")
    for x in range(10):
        print(f"{x}", end=" ")
    print()

    for y in range(10):
        print(f"{y} ", end="")
        for x in range(10):
            if layout[y][x] == 0:
                print("#", end=" ")  # Obstacle
            elif (x, y) == (0, 0):
                print("S", end=" ")  # Start
            elif (x, y) == (9, 9):
                print("G", end=" ")  # Goal
            else:
                print(".", end=" ")  # Open
        print()


if __name__ == "__main__":
    main()
