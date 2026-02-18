# analyze_data_coverage.py
"""
Analyze the coverage of state-action pairs in collected gridworld data.

Usage:
    python scripts/analyze_data_coverage.py --data-path ./data/pretraining_data/gridworld_buffer.pkl
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import json


def analyze_coverage(data_path, config_path):
    """Analyze and visualize state-action coverage."""

    # Load data
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states = data["states"][0]  # Shape: (num_transitions, 2)
    actions = data["actions"][0]  # Shape: (num_transitions, 1)

    # Load maze layout
    with open(config_path, "r") as f:
        config = json.load(f)
    maze_layout = np.array(config["data_collection"]["env_params"]["maze_layout"])

    print(f"Data loaded from {data_path}")
    print(f"Total transitions: {len(states)}")
    print(f"State shape: {states.shape}")
    print(f"Action shape: {actions.shape}")

    # Round states to grid cells
    states_rounded = np.round(states).astype(int)
    actions_int = np.round(actions).astype(int).flatten()

    # Count unique (state, action) pairs
    state_action_pairs = set()
    for s, a in zip(states_rounded, actions_int):
        state_action_pairs.add((s[0], s[1], a))

    # Count visits per cell (ignoring action)
    cell_visits = {}
    for s in states_rounded:
        cell = (s[0], s[1])
        cell_visits[cell] = cell_visits.get(cell, 0) + 1

    # Count visits per (cell, action) pair
    state_action_visits = {}
    for s, a in zip(states_rounded, actions_int):
        sa = (s[0], s[1], a)
        state_action_visits[sa] = state_action_visits.get(sa, 0) + 1

    # Determine navigable cells from maze
    navigable_cells = set()
    for y in range(10):
        for x in range(10):
            if maze_layout[y, x] > 0:  # Non-zero bitmask = navigable
                navigable_cells.add((x, y))

    # Calculate theoretical number of state-action pairs
    # For each navigable cell, count available actions from bitmask
    theoretical_pairs = 0
    for x, y in navigable_cells:
        bitmask = maze_layout[y, x]
        for action in range(4):
            if (bitmask >> action) & 1:  # Action is available
                theoretical_pairs += 1

    print(f"\n{'='*60}")
    print("Coverage Analysis")
    print(f"{'='*60}")
    print(f"Navigable cells in maze: {len(navigable_cells)}")
    print(f"Theoretical state-action pairs (based on maze): {theoretical_pairs}")
    print(f"Unique state-action pairs observed: {len(state_action_pairs)}")
    print(f"Coverage: {len(state_action_pairs)/theoretical_pairs*100:.1f}%")
    print(f"\nUnique cells visited: {len(cell_visits)}")
    print(f"Cell coverage: {len(cell_visits)/len(navigable_cells)*100:.1f}%")

    # Find unvisited cells
    unvisited_cells = navigable_cells - set((x, y) for x, y, _ in state_action_pairs)
    if unvisited_cells:
        print(f"\nUnvisited navigable cells: {len(unvisited_cells)}")
        for cell in sorted(unvisited_cells):
            print(f"  {cell}")
    else:
        print("\n✓ All navigable cells were visited!")

    # Visualize coverage
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot 1: Maze layout
    ax = axes[0, 0]
    for y in range(10):
        for x in range(10):
            if maze_layout[y, x] == 0:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray'))
    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_title('Maze Layout')
    ax.grid(True, alpha=0.3)

    # Plot 2: Cell visit heatmap
    ax = axes[0, 1]
    visit_grid = np.zeros((10, 10))
    for (x, y), count in cell_visits.items():
        if 0 <= x < 10 and 0 <= y < 10:
            visit_grid[y, x] = count
    im = ax.imshow(visit_grid, cmap='hot', interpolation='nearest', origin='lower')
    ax.set_title(f'Cell Visits (total: {len(cell_visits)}/{len(navigable_cells)})')
    plt.colorbar(im, ax=ax, label='Visit count')

    # Plot 3-6: Action-specific coverage (one per action)
    action_names = ['Up', 'Down', 'Left', 'Right']
    for action_idx in range(4):
        ax = axes[(action_idx+2)//3, (action_idx+2)%3]

        action_grid = np.zeros((10, 10))
        for (x, y, a), count in state_action_visits.items():
            if a == action_idx and 0 <= x < 10 and 0 <= y < 10:
                action_grid[y, x] = count

        # Overlay maze walls
        for y in range(10):
            for x in range(10):
                if maze_layout[y, x] == 0:
                    action_grid[y, x] = np.nan

        im = ax.imshow(action_grid, cmap='viridis', interpolation='nearest', origin='lower')
        ax.set_title(f'Action {action_idx} ({action_names[action_idx]}) Coverage')
        plt.colorbar(im, ax=ax, label='Visit count')

    plt.tight_layout()

    # Save figure
    save_dir = os.path.dirname(data_path)
    fig_path = os.path.join(save_dir, "coverage_analysis.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Coverage visualization saved to {fig_path}")

    plt.show()

    # Print statistics per action
    print(f"\n{'='*60}")
    print("Action Usage Statistics")
    print(f"{'='*60}")
    for action_idx in range(4):
        count = sum(1 for _, _, a in state_action_pairs if a == action_idx)
        visits = sum(count for (_, _, a), count in state_action_visits.items() if a == action_idx)
        print(f"Action {action_idx} ({action_names[action_idx]:5s}): {count:3d} unique pairs, {visits:5d} total uses")


def main():
    parser = argparse.ArgumentParser(description="Analyze gridworld data coverage")
    parser.add_argument("--data-path", type=str,
                       default="./data/pretraining_data/gridworld_buffer.pkl",
                       help="Path to the data pickle file")
    parser.add_argument("--config", type=str, default="gridworld.json",
                       help="Config file name")
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)

    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        print("Please run data collection first:")
        print(f"  python scripts/collect_gridworld_data.py --config {args.config}")
        return

    analyze_coverage(args.data_path, config_path)


if __name__ == "__main__":
    main()
