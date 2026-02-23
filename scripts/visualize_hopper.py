# visualize_hopper.py
#
# Standalone script to visualize hopper trajectories from saved buffer data.
# Can also be imported and called from finetuning scripts.
#
# Usage:
#   python scripts/visualize_hopper.py --buffer ./data/pretraining_data/hopper_buffer.pkl --config hopper.json
#   python scripts/visualize_hopper.py --buffer ./data/pretraining_data/hopper_buffer.pkl --config hopper.json --output hopper.gif

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle
import tempfile
import argparse
import os
import pickle
import json


def make_hopper_animation(states, config, fps=30):
    """
    Create a GIF animation showing the hopper's articulated leg chain.

    The hopper is drawn as a series of connected links:
        torso (hip) -> thigh -> leg (shin) -> foot

    Args:
        states: Array of shape (N, 11) with hopper states.
                [x, y, theta_thigh, theta_leg, theta_foot,
                 xdot, ydot, omega_thigh, omega_leg, omega_foot, ground_contact]
        config: Config dict containing env_params.
        fps: Frames per second for the animation.

    Returns:
        Path to the saved GIF file.
    """
    states = np.array(states)
    dt = config.get("env_params", config).get("dt", 0.01)

    # Link lengths
    env_p = config.get("env_params", config)
    thigh_len = env_p.get("thigh_length", 0.45)
    leg_len = env_p.get("leg_length", 0.5)
    foot_len = env_p.get("foot_length", 0.39)

    # Subsample to ~400 frames max
    skip = max(1, len(states) // 400)
    frames = states[::skip]

    # Compute axis limits from trajectory
    all_x = frames[:, 0]
    all_y = frames[:, 1]
    x_margin = max(2.0, (all_x.max() - all_x.min()) * 0.3)
    x_min = all_x.min() - x_margin
    x_max = all_x.max() + x_margin
    y_min = -0.5
    y_max = max(3.0, all_y.max() + 1.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Hopper Locomotion")
    ax.grid(True, alpha=0.3)

    # Ground line
    ax.axhline(y=0, color='saddlebrown', linewidth=3, zorder=1)
    ax.fill_between([x_min - 10, x_max + 10], -0.5, 0,
                    color='saddlebrown', alpha=0.15, zorder=1)

    # Trajectory trace (hip path)
    traj_line, = ax.plot([], [], color="blue", linewidth=1, alpha=0.4, zorder=2)

    # Hopper links
    torso_dot = Circle((0, 0), 0.06, fc="darkblue", ec="black", lw=1.5, zorder=10)
    ax.add_patch(torso_dot)
    thigh_line, = ax.plot([], [], color="navy", linewidth=4, solid_capstyle="round", zorder=8)
    knee_dot = Circle((0, 0), 0.04, fc="steelblue", ec="black", lw=1, zorder=9)
    ax.add_patch(knee_dot)
    leg_line, = ax.plot([], [], color="royalblue", linewidth=3.5, solid_capstyle="round", zorder=7)
    ankle_dot = Circle((0, 0), 0.04, fc="steelblue", ec="black", lw=1, zorder=9)
    ax.add_patch(ankle_dot)
    foot_line, = ax.plot([], [], color="cornflowerblue", linewidth=3,
                         solid_capstyle="round", zorder=6)
    foot_tip_dot = Circle((0, 0), 0.03, fc="orange", ec="black", lw=1, zorder=9)
    ax.add_patch(foot_tip_dot)

    # Info text
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10,
                        fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3),
                        zorder=20)
    vel_text = ax.text(0.02, 0.88, "", transform=ax.transAxes, fontsize=9,
                       bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8, pad=0.3),
                       zorder=20)

    def _compute_joint_positions(s):
        """Compute hip, knee, ankle, foot tip positions from state."""
        x, y = s[0], s[1]
        th_t, th_l, th_f = s[2], s[3], s[4]

        # Cumulative angles (from vertical)
        cum_t = th_t
        cum_l = th_t + th_l
        cum_f = th_t + th_l + th_f

        # Hip (torso) position
        hip_x, hip_y = x, y

        # Knee
        knee_x = hip_x + thigh_len * np.sin(cum_t)
        knee_y = hip_y - thigh_len * np.cos(cum_t)

        # Ankle
        ankle_x = knee_x + leg_len * np.sin(cum_l)
        ankle_y = knee_y - leg_len * np.cos(cum_l)

        # Foot tip
        foot_x = ankle_x + foot_len * np.sin(cum_f)
        foot_y = ankle_y - foot_len * np.cos(cum_f)

        return (hip_x, hip_y), (knee_x, knee_y), (ankle_x, ankle_y), (foot_x, foot_y)

    def update(i):
        t = i * skip * dt
        s = frames[i]

        hip, knee, ankle, foot_tip = _compute_joint_positions(s)

        # Trajectory trace
        traj_line.set_data(frames[:i + 1, 0], frames[:i + 1, 1])

        # Hip (torso)
        torso_dot.set_center(hip)

        # Thigh
        thigh_line.set_data([hip[0], knee[0]], [hip[1], knee[1]])
        knee_dot.set_center(knee)

        # Leg (shin)
        leg_line.set_data([knee[0], ankle[0]], [knee[1], ankle[1]])
        ankle_dot.set_center(ankle)

        # Foot
        foot_line.set_data([ankle[0], foot_tip[0]], [ankle[1], foot_tip[1]])
        foot_tip_dot.set_center(foot_tip)

        # Info
        time_text.set_text(f"t = {t:.2f}s")
        fwd_vel = s[5]
        height = s[1]
        contact = s[10]
        vel_text.set_text(f"v_x={fwd_vel:.2f}  h={height:.2f}  contact={'Y' if contact > 0.5 else 'N'}")

        return [traj_line, torso_dot, thigh_line, knee_dot,
                leg_line, ankle_dot, foot_line, foot_tip_dot,
                time_text, vel_text]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // fps, blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    anim.save(tmp.name, writer="pillow", fps=fps)
    plt.close(fig)
    return tmp.name


def plot_hopper_states(states, config):
    """
    Static plot of hopper state components over time.

    Returns a matplotlib figure with 4 subplots:
      1. Height (y) over time
      2. Forward velocity (x_dot) over time
      3. Joint angles over time
      4. Joint angular velocities over time
    """
    states = np.array(states)
    dt = config.get("env_params", config).get("dt", 0.01)
    time = np.arange(len(states)) * dt

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=True)

    # Height
    ax = axes[0, 0]
    ax.plot(time, states[:, 1], label="height (y)", color="blue")
    ax.axhline(0.3, color='red', linestyle='--', alpha=0.5, label="min_height")
    ax.set_ylabel("Height (m)")
    ax.set_title("Torso Height")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Forward velocity
    ax = axes[0, 1]
    ax.plot(time, states[:, 5], label="x_dot", color="green")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Forward Velocity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Joint angles
    ax = axes[1, 0]
    ax.plot(time, states[:, 2], label="thigh")
    ax.plot(time, states[:, 3], label="leg")
    ax.plot(time, states[:, 4], label="foot")
    ax.set_ylabel("Angle (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Joint Angles")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Joint angular velocities
    ax = axes[1, 1]
    ax.plot(time, states[:, 7], label="thigh")
    ax.plot(time, states[:, 8], label="leg")
    ax.plot(time, states[:, 9], label="foot")
    ax.set_ylabel("Angular Vel (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Joint Angular Velocities")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def make_hopper_animation_from_buffer(buffers, buffer_idx, config, fps=30):
    """
    Convenience wrapper that extracts states from a buffer dict
    (same format as run_drone.py's make_drone_animation).
    """
    states = np.array(buffers["states"][0, :buffer_idx, :])
    return make_hopper_animation(states, config, fps=fps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize hopper trajectory")
    parser.add_argument(
        "--buffer", type=str, required=True,
        help="Path to buffer pickle file (e.g. ./data/pretraining_data/hopper_buffer.pkl)",
    )
    parser.add_argument(
        "--config", type=str, default="hopper.json",
        help="Config filename in configs folder",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output GIF path. If not specified, opens in viewer.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Max number of steps to visualize (defaults to all).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Animation FPS")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Use data_collection config for env_params (matches the buffer)
    config = full_config.get("data_collection", full_config.get("finetuning", {}))

    # Load buffer
    with open(args.buffer, "rb") as f:
        data = pickle.load(f)

    states = data["states"][0]  # (N, 11) - remove agent dim
    dones = data.get("dones", np.zeros(len(states)))

    # Find first episode
    done_indices = np.where(dones > 0)[0]
    if len(done_indices) > 0:
        ep_end = done_indices[0] + 1
    else:
        ep_end = len(states)

    if args.max_steps:
        ep_end = min(ep_end, args.max_steps)

    episode_states = states[:ep_end]
    print(f"Visualizing {len(episode_states)} steps from buffer")

    # Create animation
    gif_path = make_hopper_animation(episode_states, config, fps=args.fps)

    if args.output:
        import shutil
        shutil.move(gif_path, args.output)
        print(f"Animation saved to {args.output}")
    else:
        # Also create static plot
        fig = plot_hopper_states(episode_states, config)
        plt.show()
        print(f"Animation saved to {gif_path}")

    # Static state plot
    fig = plot_hopper_states(episode_states, config)
    fig.savefig(gif_path.replace(".gif", "_states.png"), dpi=150)
    print(f"State plot saved to {gif_path.replace('.gif', '_states.png')}")
