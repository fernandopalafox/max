#!/usr/bin/env python
"""Create animated GIF from saved trajectory data."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse
import os


def animate_trajectory(npz_path, case="perfect_info", output_path=None, fps=20, trail_length=20):
    """Create an animated GIF of the trajectory."""
    data = np.load(npz_path)

    states_key = f"{case}_states"
    if states_key not in data:
        print(f"Case '{case}' not found. Available: {[k.replace('_states', '') for k in data.keys() if '_states' in k]}")
        return

    states = data[states_key]
    print(f"Animating {case}: {len(states)} frames")

    # State layout: [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
    evader_pos = states[:, 0:2]
    evader_vel = states[:, 2:4]
    unicycle_pos = states[:, 4:6]
    unicycle_heading = states[:, 6]
    unicycle_speed = states[:, 7]

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Compute bounds with padding
    all_x = np.concatenate([evader_pos[:, 0], unicycle_pos[:, 0]])
    all_y = np.concatenate([evader_pos[:, 1], unicycle_pos[:, 1]])
    margin = 1.0
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    # Initialize plot elements
    evader_trail, = ax.plot([], [], 'b-', alpha=0.5, linewidth=1.5, label='Evader trail')
    unicycle_trail, = ax.plot([], [], 'r-', alpha=0.5, linewidth=1.5, label='Unicycle trail')
    evader_dot, = ax.plot([], [], 'bo', markersize=10)
    # Unicycle triangle (will update vertices each frame)
    unicycle_tri, = ax.fill([], [], fc='red', ec='darkred', linewidth=2, zorder=10, label='Unicycle')

    # Text annotations
    time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, fontsize=12, verticalalignment='top')
    dist_text = ax.text(0.02, 0.93, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')
    speed_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, fontsize=10, verticalalignment='top')

    ax.legend(loc='upper right')
    ax.set_title(f'{case.replace("_", " ").title()}')

    def init():
        evader_trail.set_data([], [])
        unicycle_trail.set_data([], [])
        evader_dot.set_data([], [])
        unicycle_tri.set_xy(np.empty((0, 2)))
        time_text.set_text('')
        dist_text.set_text('')
        speed_text.set_text('')
        return evader_trail, unicycle_trail, evader_dot, unicycle_tri, time_text, dist_text, speed_text

    def update(frame):
        # Trail (last N points)
        start = max(0, frame - trail_length)
        evader_trail.set_data(evader_pos[start:frame+1, 0], evader_pos[start:frame+1, 1])
        unicycle_trail.set_data(unicycle_pos[start:frame+1, 0], unicycle_pos[start:frame+1, 1])

        # Current positions
        evader_dot.set_data([evader_pos[frame, 0]], [evader_pos[frame, 1]])

        # Draw unicycle as a triangle pointing in heading direction
        heading = unicycle_heading[frame]
        px, py = unicycle_pos[frame]

        # Triangle size - scale relative to axis extent so it looks consistent
        x_extent = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_extent = ax.get_ylim()[1] - ax.get_ylim()[0]
        size = 0.01 * max(x_extent, y_extent)  # 1% of the larger axis dimension

        # Triangle vertices: front point + two back corners
        front_x = px + size * np.cos(heading)
        front_y = py + size * np.sin(heading)

        # Back corners (perpendicular to heading, offset back)
        back_offset = size * 0.6
        side_offset = size * 0.4
        back_x = px - back_offset * np.cos(heading)
        back_y = py - back_offset * np.sin(heading)

        # Perpendicular direction
        perp_x = -np.sin(heading)
        perp_y = np.cos(heading)

        left_x = back_x + side_offset * perp_x
        left_y = back_y + side_offset * perp_y
        right_x = back_x - side_offset * perp_x
        right_y = back_y - side_offset * perp_y

        # Update triangle vertices
        unicycle_tri.set_xy(np.array([
            [front_x, front_y],
            [left_x, left_y],
            [right_x, right_y]
        ]))

        # Text updates
        dist = np.sqrt(np.sum((evader_pos[frame] - unicycle_pos[frame])**2))
        time_text.set_text(f'Step: {frame}/{len(states)-1}')
        dist_text.set_text(f'Distance: {dist:.2f}')
        speed_text.set_text(f'Unicycle speed: {unicycle_speed[frame]:.2f}')

        return evader_trail, unicycle_trail, evader_dot, unicycle_tri, time_text, dist_text, speed_text

    anim = animation.FuncAnimation(fig, update, init_func=init, frames=len(states),
                                   interval=1000/fps, blit=False)

    if output_path is None:
        output_path = npz_path.replace('.npz', f'_{case}.gif')

    print(f"Saving to {output_path}...")
    anim.save(output_path, writer='pillow', fps=fps)
    print(f"Done! Saved {output_path}")

    plt.close()
    return output_path


def animate_all_cases(npz_path, output_dir=None, fps=20, trail_length=20):
    """Create animated GIFs for all cases in the file."""
    data = np.load(npz_path)
    cases = [k.replace('_states', '') for k in data.keys() if '_states' in k]

    if output_dir is None:
        output_dir = os.path.dirname(npz_path)

    for case in cases:
        output_path = os.path.join(output_dir, f"unicycle_{case}.gif")
        animate_trajectory(npz_path, case=case, output_path=output_path, fps=fps, trail_length=trail_length)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Animate unicycle trajectory")
    parser.add_argument("--path", type=str, default="./comparison_results/unicycle_trajectories_seed_42.npz",
                        help="Path to .npz file")
    parser.add_argument("--case", type=str, default=None,
                        help="Case to animate (e.g. 'perfect_info'). If not specified, animates all.")
    parser.add_argument("--output", type=str, default=None, help="Output path for GIF")
    parser.add_argument("--fps", type=int, default=20, help="Frames per second")
    parser.add_argument("--trail", type=int, default=30, help="Trail length (number of past positions to show)")
    args = parser.parse_args()

    if args.case:
        animate_trajectory(args.path, case=args.case, output_path=args.output, fps=args.fps, trail_length=args.trail)
    else:
        animate_all_cases(args.path, fps=args.fps, trail_length=args.trail)
