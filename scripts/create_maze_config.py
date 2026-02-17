# create_maze_config.py
"""
Generate two different mazes (A and B) and update gridworld.json config.

Usage:
    python scripts/create_maze_config.py --config gridworld.json --seed-a 42 --seed-b 123 --visualize
"""

import json
import argparse
import os
import matplotlib.pyplot as plt
from generate_maze import (
    generate_maze_with_mazelib,
    maze_to_bitmask,
    ensure_path_exists,
    carve_path_if_needed,
    visualize_maze,
)


def update_config_with_mazes(config_path, maze_a_bitmask, maze_b_bitmask, start_pos, goal_a, goal_b):
    """
    Update the gridworld config file with new maze layouts.

    Args:
        config_path: Path to gridworld.json
        maze_a_bitmask: Bitmask for maze A (data collection/pretraining)
        maze_b_bitmask: Bitmask for maze B (finetuning)
        start_pos: [x, y] starting position
        goal_a: [x, y] goal position for maze A
        goal_b: [x, y] goal position for maze B
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update data_collection section (maze A)
    config["data_collection"]["env_params"]["maze_layout"] = maze_a_bitmask
    config["data_collection"]["env_params"]["start_pos"] = start_pos
    config["data_collection"]["cost_fn_params"]["goal_state"] = goal_a
    config["data_collection"]["domain_randomization"]["goal_state"]["min"] = [goal_a[0] - 1, goal_a[1] - 1]
    config["data_collection"]["domain_randomization"]["goal_state"]["max"] = [goal_a[0], goal_a[1]]

    # Update finetuning section (maze B)
    config["finetuning"]["env_params"]["maze_layout"] = maze_b_bitmask
    config["finetuning"]["env_params"]["start_pos"] = start_pos
    config["finetuning"]["cost_fn_params"]["goal_state"] = goal_b
    config["finetuning"]["evaluator_params"]["goal_state"] = goal_b

    # Write back to file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✓ Updated {config_path} with new mazes")


def main():
    parser = argparse.ArgumentParser(description="Generate maze pair for gridworld experiment")
    parser.add_argument("--config", type=str, default="gridworld.json", help="Config file to update")
    parser.add_argument("--width", type=int, default=10, help="Maze width")
    parser.add_argument("--height", type=int, default=10, help="Maze height")
    parser.add_argument("--start-x", type=int, default=0, help="Start X coordinate")
    parser.add_argument("--start-y", type=int, default=0, help="Start Y coordinate")
    parser.add_argument("--goal-a-x", type=int, default=9, help="Maze A goal X")
    parser.add_argument("--goal-a-y", type=int, default=5, help="Maze A goal Y")
    parser.add_argument("--goal-b-x", type=int, default=9, help="Maze B goal X")
    parser.add_argument("--goal-b-y", type=int, default=5, help="Maze B goal Y")
    parser.add_argument("--seed-a", type=int, default=42, help="Seed for maze A")
    parser.add_argument("--seed-b", type=int, default=123, help="Seed for maze B")
    parser.add_argument("--algorithm", type=str, default="prims",
                       choices=["prims", "backtracker", "wilsons"],
                       help="Maze generation algorithm (prims is recommended)")
    parser.add_argument("--visualize", action="store_true", help="Show visualization")
    parser.add_argument("--save-images", action="store_true", help="Save visualizations to files")
    args = parser.parse_args()

    start = (args.start_x, args.start_y)
    goal_a = (args.goal_a_x, args.goal_a_y)
    goal_b = (args.goal_b_x, args.goal_b_y)

    print("="*60)
    print("Generating Maze A (Data Collection / Pretraining)")
    print("="*60)
    print(f"Size: {args.width}x{args.height}")
    print(f"Start: {start}, Goal: {goal_a}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Seed: {args.seed_a}\n")

    # Generate maze A using mazelib
    maze_a_grid = generate_maze_with_mazelib(
        args.width, args.height, start=start, seed=args.seed_a, algorithm=args.algorithm
    )
    maze_a_grid = carve_path_if_needed(maze_a_grid, start, goal_a)

    if ensure_path_exists(maze_a_grid, start, goal_a):
        print("✓ Maze A: Path from start to goal exists!")
    else:
        print("✗ Maze A: No path found!")
        return

    maze_a_bitmask = maze_to_bitmask(maze_a_grid)

    print("\n" + "="*60)
    print("Generating Maze B (Finetuning)")
    print("="*60)
    print(f"Size: {args.width}x{args.height}")
    print(f"Start: {start}, Goal: {goal_b}")
    print(f"Algorithm: {args.algorithm}")
    print(f"Seed: {args.seed_b}\n")

    # Generate maze B using mazelib
    maze_b_grid = generate_maze_with_mazelib(
        args.width, args.height, start=start, seed=args.seed_b, algorithm=args.algorithm
    )
    maze_b_grid = carve_path_if_needed(maze_b_grid, start, goal_b)

    if ensure_path_exists(maze_b_grid, start, goal_b):
        print("✓ Maze B: Path from start to goal exists!")
    else:
        print("✗ Maze B: No path found!")
        return

    maze_b_bitmask = maze_to_bitmask(maze_b_grid)

    # Update config file
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    update_config_with_mazes(
        config_path,
        maze_a_bitmask,
        maze_b_bitmask,
        [start[0], start[1]],
        [goal_a[0], goal_a[1]],
        [goal_b[0], goal_b[1]]
    )

    # Visualize and/or save
    if args.visualize or args.save_images:
        fig_a = visualize_maze(
            maze_a_grid, maze_a_bitmask, start, goal_a,
            title=f"Maze A (Pretraining) - {args.algorithm}, seed {args.seed_a}"
        )
        fig_b = visualize_maze(
            maze_b_grid, maze_b_bitmask, start, goal_b,
            title=f"Maze B (Finetuning) - {args.algorithm}, seed {args.seed_b}"
        )

        if args.save_images:
            os.makedirs("figures", exist_ok=True)
            fig_a.savefig(f"figures/maze_a_{args.algorithm}_seed{args.seed_a}.png", dpi=150, bbox_inches='tight')
            fig_b.savefig(f"figures/maze_b_{args.algorithm}_seed{args.seed_b}.png", dpi=150, bbox_inches='tight')
            print(f"\n✓ Saved visualizations to figures/")

        if args.visualize:
            plt.show()
        else:
            plt.close(fig_a)
            plt.close(fig_b)

    print("\n" + "="*60)
    print("Complete! You can now run:")
    print("="*60)
    print(f"1. python scripts/collect_gridworld_data.py --config {args.config}")
    print(f"2. python scripts/pretrain.py --config {args.config}")
    print(f"3. python scripts/run_gridworld.py --config {args.config} --lambdas 0.0 1.0 10.0")
    print("="*60)


if __name__ == "__main__":
    main()
