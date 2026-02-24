# collect_data_swimmer.py
"""
Data collection script for Swimmer environment using iCEM planning
with ground truth dynamics.
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

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


def sample_goal_position(key, dr_config):
    """Sample a goal position from domain randomization config."""
    bounds = dr_config["goal_position"]
    min_val = bounds["min"]
    max_val = bounds["max"]
    key, kx, ky = jax.random.split(key, 3)
    goal_x = jax.random.uniform(kx, minval=min_val, maxval=max_val)
    goal_y = jax.random.uniform(ky, minval=min_val, maxval=max_val)
    return jnp.array([float(goal_x), float(goal_y)]), key


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
        goal_position,
    ):
        """Collect a single episode of transitions using mjx.Data state."""
        key, reset_key = jax.random.split(key)
        data = reset_fn(reset_key)
        current_obs = get_obs_fn(data)

        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "goal_position": goal_position,
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


def plot_swimmer_trajectory(states, goal_position, config):
    """
    Plot swimmer-specific visualization for goal-reaching:
    - Left: Top-down XY trajectory with goal marker
    - Middle: Distance to goal over time
    - Right: Joint angles over time
    """
    states = np.array(states)  # 16D states for 6-link swimmer
    goal_position = np.array(goal_position)
    dt = 0.03  # swimmer control dt
    time_axis = np.arange(len(states)) * dt

    # X position is qpos[0] = states[:, 0]
    x_position = states[:, 0]
    # Y position is qpos[1] = states[:, 1]
    y_position = states[:, 1]
    # Distance to goal
    dist_to_goal = np.sqrt(
        (x_position - goal_position[0])**2 +
        (y_position - goal_position[1])**2
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: Top-down XY trajectory with goal
    ax = axes[0]
    ax.plot(x_position, y_position, label="Trajectory", color="blue", linewidth=2)
    ax.scatter(x_position[0], y_position[0], color="green", s=100,
               marker="o", label="Start", zorder=5)
    ax.scatter(x_position[-1], y_position[-1], color="blue", s=100,
               marker="x", label="End", zorder=5)
    ax.scatter(goal_position[0], goal_position[1], color="red", s=200,
               marker="*", label="Goal", zorder=6)
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Swimmer Goal-Reaching (Top-Down)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect('equal')

    # Middle: Distance to goal over time
    ax = axes[1]
    ax.plot(time_axis, dist_to_goal, label="Distance to Goal",
            color="purple", linewidth=2)
    ax.axhline(0, color='red', linestyle='--', linewidth=2, label="Goal")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Distance to Goal Over Time")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Joint positions (qpos[3:8] for 5 joints)
    ax = axes[2]
    joint_labels = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4"]
    for i, label in enumerate(joint_labels):
        ax.plot(time_axis, states[:, 3 + i], label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title("Joint Positions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def create_swimmer_xy_animation(states, max_frames=100, save_path=None):
    """
    Creates an animated GIF showing the swimmer from a top-down view.

    The swimmer is visualized as connected line segments (links).
    The camera follows the swimmer's centroid.

    Args:
        states: Array of 16D states [qpos (8), qvel (8)].
        max_frames: Maximum number of frames (subsampled if needed)
        save_path: Optional path to save GIF.

    Returns:
        Path to the saved GIF file.
    """
    import tempfile

    states = np.array(states)
    dt = 0.03

    # Subsample if too many frames
    if len(states) > max_frames:
        indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        states = states[indices]
        effective_dt = dt * (len(states) / max_frames)
    else:
        effective_dt = dt

    # Link length (from MuJoCo XML: bodies at "0 .1 0" offset)
    link_length = 0.1

    def get_swimmer_points(state):
        """Compute link positions from state using forward kinematics."""
        rootx = state[0]
        rooty = state[1]
        rootz_angle = state[2]  # Rotation around z-axis (heading)

        # Joint angles (5 joints for 6-link swimmer)
        joint_angles = state[3:8]

        # Head position and orientation
        head_pos = np.array([rootx, rooty])
        head_angle = rootz_angle

        # Compute each link position using forward kinematics
        positions = [head_pos.copy()]
        current_pos = head_pos.copy()
        current_angle = head_angle

        for i, joint_angle in enumerate(joint_angles):
            # Each link extends in the direction of current heading
            # and joint adds rotation
            current_angle += joint_angle
            # Link extends along the body axis
            direction = np.array([
                np.cos(current_angle),
                np.sin(current_angle)
            ])
            current_pos = current_pos + link_length * direction
            positions.append(current_pos.copy())

        return np.array(positions)

    # Set up figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Initialize lines for swimmer body
    body_line, = ax.plot([], [], 'b-', linewidth=4, solid_capstyle='round')
    joints, = ax.plot([], [], 'ko', markersize=6, zorder=5)
    head_marker, = ax.plot([], [], 'go', markersize=10, zorder=6)

    # Trail showing path
    trail_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.3)

    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True, alpha=0.3)

    # Store trail positions
    trail_x = []
    trail_y = []

    def init():
        body_line.set_data([], [])
        joints.set_data([], [])
        head_marker.set_data([], [])
        trail_line.set_data([], [])
        return body_line, joints, head_marker, trail_line

    def animate(frame):
        positions = get_swimmer_points(states[frame])

        # Update trail (track head position)
        trail_x.append(positions[0, 0])
        trail_y.append(positions[0, 1])
        trail_line.set_data(trail_x, trail_y)

        # Update body line
        body_line.set_data(positions[:, 0], positions[:, 1])

        # Update joint markers
        joints.set_data(positions[:, 0], positions[:, 1])

        # Update head marker
        head_marker.set_data([positions[0, 0]], [positions[0, 1]])

        # Update view to follow swimmer
        centroid = np.mean(positions, axis=0)
        view_size = 0.8
        ax.set_xlim(centroid[0] - view_size, centroid[0] + view_size)
        ax.set_ylim(centroid[1] - view_size, centroid[1] + view_size)

        # Update title with time and velocity
        forward_vel = states[frame, 8]  # qvel[0]
        time = frame * effective_dt
        ax.set_title(f'Swimmer Animation | t={time:.2f}s | vel={forward_vel:.3f} m/s')

        return body_line, joints, head_marker, trail_line

    anim = FuncAnimation(
        fig, animate, init_func=init,
        frames=len(states), interval=50, blit=False
    )

    # Save as GIF
    if save_path is None:
        save_path = tempfile.mktemp(suffix='.gif')

    anim.save(save_path, writer='pillow', fps=20)
    plt.close(fig)

    return save_path


def plot_histograms(buffers, buffer_idx, config, max_samples=50000):
    """Plot histograms of observations and actions."""
    dim_obs = config.get("dim_obs", 14)
    observations = np.array(buffers["states"][0, :buffer_idx, :dim_obs])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    # Subsample if buffer is too large
    if buffer_idx > max_samples:
        indices = np.random.choice(buffer_idx, size=max_samples, replace=False)
        observations = observations[indices]
        actions = actions[indices]

    norm_params = config["normalization_params"]
    dim_action = actions.shape[1]

    # Create figure with observations (top) and actions (bottom)
    fig, axes = plt.subplots(2, max(dim_obs, dim_action), figsize=(24, 8))

    # Observation histograms (top row)
    # 16D observation: rootx, rooty, rootz, joint_0-4, velocities
    obs_labels = [
        "rootx", "rooty", "rootz", "j0", "j1", "j2", "j3", "j4",
        "v_rx", "v_ry", "v_rz", "v_j0", "v_j1", "v_j2", "v_j3", "v_j4"
    ]
    for i in range(min(dim_obs, axes.shape[1])):
        ax = axes[0, i]
        ax.hist(observations[:, i], bins=50, alpha=0.7, color='blue')
        label = obs_labels[i] if i < len(obs_labels) else f"obs_{i}"
        ax.set_title(f"Obs: {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Action histograms (bottom row)
    action_labels = ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4"]
    for i in range(dim_action):
        ax = axes[1, i]
        ax.hist(actions[:, i], bins=50, alpha=0.7, color='green')
        ax.axvline(norm_params["action"]["min"][i], color='red',
                   linestyle='--', linewidth=2)
        ax.axvline(norm_params["action"]["max"][i], color='red',
                   linestyle='--', linewidth=2)
        ax.set_title(f"Action: {action_labels[i]}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Hide unused subplots
    for i in range(dim_obs, axes.shape[1]):
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
    """Main data collection loop for Swimmer."""
    wandb.init(
        project=config.get("wandb_project", "swimmer_data_collection"),
        config=config,
        name=f"swimmer_collect_{config['num_episodes']}ep",
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
    dim_obs = config.get("dim_obs", 14)
    collect_episode = make_collect_episode_fn(
        reset_fn,
        step_fn,
        get_obs_fn,
        planner,
        config["max_episode_length"],
    )
    print("Initialization complete.")

    # Initialize buffer - store 14D observations
    buffer_size = config["num_episodes"] * config["max_episode_length"]
    buffers = init_jax_buffers(
        config["num_agents"],
        buffer_size,
        dim_obs,
        config["dim_action"],
    )
    buffer_idx = 0
    episode_info = []

    print(f"Starting swimmer data collection: {config['num_episodes']} episodes")

    # Collection loop
    for ep in range(config["num_episodes"]):
        # Sample goal position from domain randomization
        if "domain_randomization" in config:
            goal_position, key = sample_goal_position(
                key, config["domain_randomization"]
            )
        else:
            goal_position = jnp.array(config["cost_fn_params"]["goal_position"])

        episode_info.append({
            "episode": ep,
            "goal_position": goal_position.tolist()
        })

        # Collect episode
        ep_start = time.perf_counter()
        states, obs, actions, rewards, key = collect_episode(
            key,
            planner_state,
            train_state,
            goal_position,
        )
        # Block until computation is complete for accurate timing
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
                obs[t:t + 1],
                actions[t:t + 1],
                rewards[t:t + 1],
                jnp.zeros(1),
                jnp.zeros(1),
                float(t == ep_len - 1),
            )
            buffer_idx += 1

        # Log metrics
        final_x = float(states[-1, 0])
        final_y = float(states[-1, 1])
        final_dist = float(jnp.sqrt(
            (final_x - goal_position[0])**2 +
            (final_y - goal_position[1])**2
        ))

        print(
            f"Episode {ep + 1}/{config['num_episodes']}: "
            f"time={ep_time:.2f}s, final_dist={final_dist:.3f}, "
            f"goal=({goal_position[0]:.2f}, {goal_position[1]:.2f})"
        )
        wandb.log({
            "episode/time": ep_time,
            "episode/final_distance": final_dist,
            "episode/final_x": final_x,
            "episode/final_y": final_y,
            "episode/goal_x": float(goal_position[0]),
            "episode/goal_y": float(goal_position[1]),
        }, step=ep)

        # Plot episode trajectory at plot_freq
        if ep == 0 or (ep + 1) % plot_freq == 0:
            fig = plot_swimmer_trajectory(states, goal_position, config)
            wandb.log({
                "episode/trajectory": wandb.Image(fig, caption=f"Episode {ep+1}")
            }, step=ep)
            plt.close(fig)

            gif_path = create_swimmer_xy_animation(states)
            wandb.log({
                "episode/animation": wandb.Video(gif_path, fps=20, format="gif")
            }, step=ep)

    # Final histograms
    fig = plot_histograms(buffers, buffer_idx, config)
    wandb.log({"data/histograms": wandb.Image(fig)})
    plt.close(fig)

    # Log normalization bounds to summary
    norm_params_config = config["normalization_params"]
    wandb.summary["norm_bounds/action_min"] = norm_params_config["action"]["min"]
    wandb.summary["norm_bounds/action_max"] = norm_params_config["action"]["max"]
    wandb.summary["total_transitions"] = buffer_idx

    # Save buffer to disk
    save_buffer(buffers, buffer_idx, save_dir, config["env_name"], episode_info)

    wandb.finish()
    print(f"\nData collection complete. Total transitions: {buffer_idx}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect data for Swimmer offline training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="swimmer.json",
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
