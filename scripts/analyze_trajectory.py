#!/usr/bin/env python
"""Analyze saved trajectory data from unicycle comparison runs."""

import numpy as np
import argparse
import sys


def analyze(npz_path, num_steps=30, case_filter=None):
    """Print trajectory analysis for unicycle dynamics."""
    data = np.load(npz_path)

    print(f"Loaded: {npz_path}")
    print(f"Keys: {list(data.keys())}")
    print()

    # Find unique case names
    case_names = [k.replace('_states', '') for k in data.keys() if k.endswith('_states')]

    if case_filter:
        case_names = [c for c in case_names if case_filter.lower() in c.lower()]

    for name in case_names:
        states = data[f'{name}_states']
        actions = data[f'{name}_actions'] if f'{name}_actions' in data else None

        print(f"=== {name.upper().replace('_', ' ')} ===")
        print(f"Shape: states={states.shape}, actions={actions.shape if actions is not None else 'N/A'}")
        print()

        # State layout: [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
        print(f"{'Step':>4} | {'Evader pos':^13} | {'Evader vel':^13} | {'Unicycle pos':^13} | {'heading':>8} | {'speed':>5} | {'dist':>5}")
        print("-" * 95)

        # Sample evenly across the trajectory
        total_steps = len(states)
        if total_steps <= num_steps:
            indices = list(range(total_steps))
        else:
            # Always include first and last, sample the rest evenly
            indices = [0] + list(np.linspace(1, total_steps - 2, num_steps - 2, dtype=int)) + [total_steps - 1]
            indices = sorted(set(indices))  # Remove duplicates

        prev_idx = -1
        for i in indices:
            # Show gap indicator if we skipped steps
            if prev_idx >= 0 and i - prev_idx > 1:
                print(f"  ...")
            prev_idx = i

            s = states[i]
            p1x, p1y, v1x, v1y = s[0], s[1], s[2], s[3]
            p2x, p2y, alpha2, v2 = s[4], s[5], s[6], s[7]

            dist = np.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)
            heading_deg = np.degrees(alpha2)

            print(f"{i:4d} | ({p1x:+5.2f}, {p1y:+5.2f}) | ({v1x:+5.2f}, {v1y:+5.2f}) | ({p2x:+5.2f}, {p2y:+5.2f}) | {heading_deg:+7.1f}° | {v2:5.2f} | {dist:5.2f}")

        # Summary stats
        p1_pos = states[:, 0:2]
        p2_pos = states[:, 4:6]
        distances = np.sqrt(np.sum((p1_pos - p2_pos)**2, axis=1))
        headings = np.degrees(states[:, 6])
        speeds = states[:, 7]

        print()
        print(f"Summary:")
        print(f"  Distance: {distances[0]:.2f} -> {distances[-1]:.2f} (min={distances.min():.2f}, max={distances.max():.2f})")
        print(f"  Unicycle heading: {headings[0]:.1f}° -> {headings[-1]:.1f}° (range: {headings.min():.1f}° to {headings.max():.1f}°, delta={headings.max()-headings.min():.1f}°)")
        print(f"  Unicycle speed: {speeds[0]:.2f} -> {speeds[-1]:.2f} (range: {speeds.min():.2f} to {speeds.max():.2f})")
        print(f"  Evader total distance traveled: {np.sum(np.sqrt(np.sum(np.diff(p1_pos, axis=0)**2, axis=1))):.2f}")
        print(f"  Unicycle total distance traveled: {np.sum(np.sqrt(np.sum(np.diff(p2_pos, axis=0)**2, axis=1))):.2f}")
        print()
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze unicycle trajectory data")
    parser.add_argument("--path", type=str, default="./comparison_results/unicycle_trajectories_seed_42.npz",
                        help="Path to .npz file")
    parser.add_argument("--steps", type=int, default=30, help="Number of steps to print")
    parser.add_argument("--case", type=str, default=None, help="Filter to specific case (e.g. 'active')")
    args = parser.parse_args()

    analyze(args.path, args.steps, args.case)
