# collect_data_hopper.py
"""
Data collection script for Hopper environment using iCEM planning
with ground truth dynamics for standing task.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.planners import init_planner
from max.costs import init_cost
import argparse
import os
import pickle
import json
import time


def make_collect_episode_fn(
    reset_fn,
    step_fn,
    get_obs_fn,
    planner,
    max_episode_length,
):
    """Create a JIT-compiled episode collection function using lax.scan."""

    def collect_episode(
        key,
        planner_state,
        train_state,
    ):
        """Collect a single episode of transitions using mjx.Data state."""
        key, reset_key = jax.random.split(key)
        data = reset_fn(reset_key)
        current_obs = get_obs_fn(data)

        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }

        def step_body(carry, step_idx):
            data, current_obs, planner_state, key = carry

            key, planner_key = jax.random.split(key)
            planner_state = planner_state.replace(key=planner_key)

            actions, planner_state = planner.solve(planner_state, data, cost_params)
            action = actions[0][None, :]  # Add agent dim

            # Extract state and obs before stepping
            state_array = jnp.concatenate([data.qpos, data.qvel])
            obs = current_obs[0]  # Remove agent dim

            # Step environment
            data, next_obs, rewards, terminated, truncated, info = step_fn(
                data, step_idx, action
            )

            carry = (data, next_obs, planner_state, key)
            outputs = (state_array, obs, action[0], rewards[0])
            return carry, outputs

        init_carry = (data, current_obs, planner_state, key)
        step_indices = jnp.arange(max_episode_length)

        final_carry, outputs = jax.lax.scan(step_body, init_carry, step_indices)
        states, obs, actions, rewards = outputs

        _, _, _, key = final_carry

        return states, obs, actions, rewards, key

    return jax.jit(collect_episode)


def plot_hopper_trajectory(states, config):
    """
    Plot hopper-specific visualization:
    - Left: Height over time (torso z position)
    - Middle: Orientation (rooty angle) over time
    - Right: Joint angles over time
    """
    states = np.array(states)  # 14D states [qpos(7), qvel(7)]
    dt = 0.02  # hopper control dt
    time_axis = np.arange(len(states)) * dt

    # Height is qpos[1] (rootz)
    rootz = states[:, 1]

    # Orientation is qpos[2] (rooty pitch angle)
    rooty = states[:, 2]

    # Velocities
    vel_x = states[:, 7]  # vel_rootx
    vel_z = states[:, 8]  # vel_rootz

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Height and target
    ax = axes[0]
    ax.plot(time_axis, rootz, label="Root Z Height", color="blue", linewidth=2)
    ax.axhline(0.6, color='red', linestyle='--', linewidth=2,
               label="Target Height (0.6)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Hopper Height")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Middle: Orientation
    ax = axes[1]
    ax.plot(time_axis, np.degrees(rooty), label="Pitch Angle",
            color="green", linewidth=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2,
               label="Upright (0 deg)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Angle (degrees)")
    ax.set_title("Hopper Orientation")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Joint positions
    ax = axes[2]
    joint_labels = ["waist", "hip", "knee", "ankle"]
    for i, label in enumerate(joint_labels):
        ax.plot(time_axis, np.degrees(states[:, 3 + i]), label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (degrees)")
    ax.set_title("Joint Positions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def create_hopper_animation(states, max_frames=100, save_path=None):
    """
    Creates an animated GIF showing the hopper as a side-view stick figure.

    The hopper is a planar system (x-z plane), so we show a side view.
    The camera follows the hopper's forward progress.

    Args:
        states: Array of 14D states [qpos (7), qvel (7)].
        max_frames: Maximum number of frames (subsampled if needed)
        save_path: Optional path to save GIF. If None, uses temp file.

    Returns:
        Path to the saved GIF file.
    """
    import tempfile

    states = np.array(states)
    dt = 0.02

    # Subsample if too many frames
    orig_len = len(states)
    if len(states) > max_frames:
        indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        states = states[indices]
        effective_dt = dt * (orig_len / max_frames)
    else:
        effective_dt = dt

    # Link lengths based on hopper.xml geom definitions
    TORSO_HEIGHT = 0.25  # fromto="0 0 -.05 0 0 .2"
    PELVIS_LENGTH = 0.15  # fromto="0 0 0 0 0 -.15"
    THIGH_LENGTH = 0.33  # fromto="0 0 0 0 0 -.33"
    CALF_LENGTH = 0.32   # fromto="0 0 0 0 0 -.32"
    FOOT_LENGTH = 0.25   # fromto="-.08 0 0 .17 0 0"

    # Initial torso position (from XML: pos="0 0 1")
    INITIAL_Z = 1.0

    def get_hopper_points(state):
        """Compute joint positions from state using forward kinematics."""
        rootx = state[0]
        rootz = state[1] + INITIAL_Z  # rootz is displacement from initial
        rooty = state[2]  # Pitch angle

        # Joint angles (relative)
        waist_angle = state[3]
        hip_angle = state[4]
        knee_angle = state[5]
        ankle_angle = state[6]

        # Torso center
        torso_center = np.array([rootx, rootz])

        # Torso top and bottom (oriented by rooty)
        torso_top = torso_center + 0.2 * np.array([np.sin(rooty), np.cos(rooty)])
        torso_bottom = torso_center - 0.05 * np.array(
            [np.sin(rooty), np.cos(rooty)])

        # Pelvis attaches at torso bottom, hangs down with waist rotation
        pelvis_top = torso_bottom
        cumulative_angle = rooty + waist_angle
        pelvis_bottom = pelvis_top - PELVIS_LENGTH * np.array(
            [np.sin(cumulative_angle), np.cos(cumulative_angle)])

        # Thigh attaches at pelvis bottom, with hip rotation
        thigh_top = pelvis_bottom
        cumulative_angle += hip_angle
        thigh_bottom = thigh_top - THIGH_LENGTH * np.array(
            [np.sin(cumulative_angle), np.cos(cumulative_angle)])

        # Calf attaches at thigh bottom, with knee rotation
        calf_top = thigh_bottom
        cumulative_angle += knee_angle
        calf_bottom = calf_top - CALF_LENGTH * np.array(
            [np.sin(cumulative_angle), np.cos(cumulative_angle)])

        # Foot attaches at calf bottom, with ankle rotation
        foot_center = calf_bottom
        cumulative_angle += ankle_angle
        # Foot is horizontal, extends back and forward
        foot_back = foot_center - 0.08 * np.array(
            [np.cos(cumulative_angle), -np.sin(cumulative_angle)])
        foot_front = foot_center + 0.17 * np.array(
            [np.cos(cumulative_angle), -np.sin(cumulative_angle)])

        return {
            'torso': (torso_bottom, torso_top),
            'pelvis': (pelvis_top, pelvis_bottom),
            'thigh': (thigh_top, thigh_bottom),
            'calf': (calf_top, calf_bottom),
            'foot': (foot_back, foot_front),
            'rootx': rootx,
            'rootz': rootz,
        }

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lines
    torso_line, = ax.plot([], [], 'b-', linewidth=6, solid_capstyle='round')
    pelvis_line, = ax.plot([], [], 'b-', linewidth=5, solid_capstyle='round')
    thigh_line, = ax.plot([], [], 'r-', linewidth=4, solid_capstyle='round')
    calf_line, = ax.plot([], [], 'r-', linewidth=3, solid_capstyle='round')
    foot_line, = ax.plot([], [], 'g-', linewidth=4, solid_capstyle='round')

    # Joint markers
    joints, = ax.plot([], [], 'ko', markersize=4, zorder=5)

    # Ground line
    ground_line, = ax.plot([], [], 'k-', linewidth=2)

    # Trail showing path
    trail_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.3)

    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.grid(True, alpha=0.3)

    trail_x = []
    trail_z = []

    def init():
        torso_line.set_data([], [])
        pelvis_line.set_data([], [])
        thigh_line.set_data([], [])
        calf_line.set_data([], [])
        foot_line.set_data([], [])
        joints.set_data([], [])
        ground_line.set_data([], [])
        trail_line.set_data([], [])
        return (torso_line, pelvis_line, thigh_line, calf_line, foot_line,
                joints, ground_line, trail_line)

    def animate(frame):
        points = get_hopper_points(states[frame])
        rootx = points['rootx']
        rootz = points['rootz']

        # Update trail
        trail_x.append(rootx)
        trail_z.append(rootz)
        trail_line.set_data(trail_x, trail_z)

        # Update body segments
        t = points['torso']
        torso_line.set_data([t[0][0], t[1][0]], [t[0][1], t[1][1]])

        p = points['pelvis']
        pelvis_line.set_data([p[0][0], p[1][0]], [p[0][1], p[1][1]])

        th = points['thigh']
        thigh_line.set_data([th[0][0], th[1][0]], [th[0][1], th[1][1]])

        c = points['calf']
        calf_line.set_data([c[0][0], c[1][0]], [c[0][1], c[1][1]])

        f = points['foot']
        foot_line.set_data([f[0][0], f[1][0]], [f[0][1], f[1][1]])

        # Joint markers
        all_x = [t[0][0], t[1][0], p[1][0], th[1][0], c[1][0]]
        all_z = [t[0][1], t[1][1], p[1][1], th[1][1], c[1][1]]
        joints.set_data(all_x, all_z)

        # Update view to follow hopper
        view_width = 3.0
        ax.set_xlim(rootx - view_width / 2, rootx + view_width / 2)
        ax.set_ylim(-0.2, 1.8)

        # Ground line
        ground_line.set_data([rootx - view_width, rootx + view_width], [0, 0])

        # Update title
        time_val = frame * effective_dt
        ax.set_title(f'Hopper Animation | t={time_val:.2f}s | height={rootz:.2f}m')

        return (torso_line, pelvis_line, thigh_line, calf_line, foot_line,
                joints, ground_line, trail_line)

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(states), interval=50, blit=False
    )

    if save_path is None:
        save_path = tempfile.mktemp(suffix='.gif')

    anim.save(save_path, writer='pillow', fps=20)
    plt.close(fig)

    return save_path


def plot_histograms(buffers, buffer_idx, config, max_samples=50000):
    """Plot histograms of observations and actions."""
    dim_obs = config.get("dim_obs", 15)
    observations = np.array(buffers["states"][0, :buffer_idx, :dim_obs])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    if buffer_idx > max_samples:
        indices = np.random.choice(buffer_idx, size=max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]

    dim_action = actions.shape[1]

    fig, axes = plt.subplots(2, max(7, dim_action), figsize=(24, 8))

    # Observation histograms - show key states
    obs_indices = [0, 1, 2, 3, 6, 7, 8]
    obs_labels = ["rootz", "rooty", "waist", "hip", "ankle", "vel_x", "vel_z"]
    for i, (idx, label) in enumerate(zip(obs_indices, obs_labels)):
        if i >= axes.shape[1]:
            break
        ax = axes[0, i]
        ax.hist(observations[:, idx], bins=50, alpha=0.7, color='blue')
        ax.set_title(f"Obs: {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Action histograms
    action_labels = ["waist", "hip", "knee", "ankle"]
    for i in range(dim_action):
        ax = axes[1, i]
        ax.hist(actions[:, i], bins=50, alpha=0.7, color='green')
        ax.set_title(f"Action: {action_labels[i]}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for i in range(len(obs_indices), axes.shape[1]):
        axes[0, i].axis('off')
    for i in range(dim_action, axes.shape[1]):
        axes[1, i].axis('off')

    plt.tight_layout()
    return fig


def save_buffer(buffers, buffer_idx, save_path, env_name, episode_info=None):
    """Save buffer data to disk as pickle file."""
    os.makedirs(save_path, exist_ok=True)

    data = {
        "states": np.array(buffers["states"][:, :buffer_idx, :]),
        "actions": np.array(buffers["actions"][:, :buffer_idx, :]),
        "rewards": np.array(buffers["rewards"][:, :buffer_idx]),
        "dones": np.array(buffers["dones"][:buffer_idx]),
        "num_transitions": buffer_idx,
    }
    if episode_info is not None:
        data["episode_info"] = episode_info

    file_path = os.path.join(save_path, f"{env_name}_buffer.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Buffer saved to {file_path}")


def main(config, save_dir):
    """Main data collection loop for Hopper."""
    wandb.init(
        project=config.get("wandb_project", "hopper_data_collection"),
        config=config,
        name=f"hopper_collect_{config['num_episodes']}ep",
        reinit=True,
    )

    key = jax.random.key(config["seed"])
    plot_freq = config.get("plot_freq", 10)

    # Initialize components
    print("Initializing components...")
    reset_fn, step_fn, get_obs_fn = init_env(config)
    normalizer, norm_params = init_normalizer(config)

    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        config, dynamics_model, init_params, trainer_key
    )

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    # Create JIT-compiled episode collection function
    dim_obs = config.get("dim_obs", 15)
    collect_episode = make_collect_episode_fn(
        reset_fn,
        step_fn,
        get_obs_fn,
        planner,
        config["max_episode_length"],
    )
    print("Initialization complete.")

    # Initialize buffer
    buffer_size = config["num_episodes"] * config["max_episode_length"]
    buffers = init_jax_buffers(
        config["num_agents"],
        buffer_size,
        dim_obs,
        config["dim_action"],
    )
    buffer_idx = 0
    episode_info = []

    print(f"Starting hopper data collection: {config['num_episodes']} episodes")

    # Collection loop
    for ep in range(config["num_episodes"]):
        episode_info.append({"episode": ep})

        # Collect episode
        ep_start = time.perf_counter()
        states, obs, actions, rewards, key = collect_episode(
            key,
            planner_state,
            train_state,
        )
        jax.block_until_ready(rewards)
        ep_time = time.perf_counter() - ep_start

        ep_len = config["max_episode_length"]

        # Add observations to buffer
        for t in range(ep_len):
            if buffer_idx >= buffer_size:
                print(f"Warning: Buffer full at episode {ep}, truncating.")
                break
            buffers = update_buffer_dynamic(
                buffers,
                buffer_idx,
                obs[t: t + 1],
                actions[t: t + 1],
                rewards[t: t + 1],
                jnp.zeros(1),
                jnp.zeros(1),
                float(t == ep_len - 1),
            )
            buffer_idx += 1

        # Log metrics
        avg_height = float(jnp.mean(states[:, 1]))  # rootz
        avg_orientation = float(jnp.mean(jnp.abs(states[:, 2])))  # |rooty|

        print(
            f"Episode {ep + 1}/{config['num_episodes']}: "
            f"time={ep_time:.2f}s, avg_height={avg_height:.3f}, "
            f"avg_orientation={np.degrees(avg_orientation):.1f}deg"
        )
        wandb.log({
            "episode/time": ep_time,
            "episode/avg_height": avg_height,
            "episode/avg_orientation_deg": np.degrees(avg_orientation),
        }, step=ep)

        # Plot at plot_freq
        if ep == 0 or (ep + 1) % plot_freq == 0:
            fig = plot_hopper_trajectory(states, config)
            wandb.log({
                "episode/trajectory": wandb.Image(fig, caption=f"Episode {ep+1}")
            }, step=ep)
            plt.close(fig)

            gif_path = create_hopper_animation(states)
            wandb.log({
                "episode/animation": wandb.Video(gif_path, fps=20, format="gif")
            }, step=ep)

    # Final histograms
    fig = plot_histograms(buffers, buffer_idx, config)
    wandb.log({"data/histograms": wandb.Image(fig)})
    plt.close(fig)

    wandb.summary["total_transitions"] = buffer_idx

    # Save buffer to disk
    save_buffer(buffers, buffer_idx, save_dir, config["env_name"], episode_info)

    wandb.finish()
    print(f"\nData collection complete. Total transitions: {buffer_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data for Hopper offline training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hopper.json",
        help="Config filename in configs folder",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save collected data (overrides config)",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="Wandb project name (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    args = parser.parse_args()

    # Load config from JSON file
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Extract data_collection section as the config
    config = full_config["data_collection"]

    # Apply CLI overrides
    if args.save_dir is not None:
        config["save_path"] = args.save_dir
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.seed is not None:
        config["seed"] = args.seed

    save_dir = config["save_path"]

    main(config, save_dir)
