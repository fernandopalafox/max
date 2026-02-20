# collect_data_cheetah.py
"""
Data collection script for HalfCheetah environment using iCEM planning
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


def sample_target_velocity(key, dr_config):
    """Sample a target velocity from domain randomization config."""
    vel_bounds = dr_config["target_velocity"]
    min_val = vel_bounds["min"]
    max_val = vel_bounds["max"]
    key, subkey = jax.random.split(key)
    target_vel = jax.random.uniform(subkey, minval=min_val, maxval=max_val)
    return float(target_vel), key


def make_collect_episode_fn(
    reset_fn,
    step_fn,
    get_obs_fn,
    get_state_array_fn,
    planner,
    max_episode_length,
):
    """Create a JIT-compiled episode collection function using lax.scan."""

    def collect_episode(
        key,
        planner_state,
        train_state,
        target_velocity,
    ):
        """Collect a single episode of transitions using mjx.Data state."""
        key, reset_key = jax.random.split(key)
        data = reset_fn(reset_key)
        current_obs = get_obs_fn(data)

        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "target_velocity": target_velocity,
        }

        def step_body(carry, step_idx):
            data, current_obs, planner_state, key = carry

            key, planner_key = jax.random.split(key)
            planner_state = planner_state.replace(key=planner_key)

            actions, planner_state = planner.solve(planner_state, data, cost_params)
            action = actions[0][None, :]  # Add agent dim

            # Extract state and obs before stepping
            state_array = get_state_array_fn(data)
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


def plot_cheetah_trajectory(states, target_velocity, config):
    """
    Plot cheetah-specific visualization:
    - Left: X position over time (forward progress)
    - Middle: Forward velocity over time with target line
    - Right: Joint angles over time
    """
    states = np.array(states)  # 18D states
    dt = config["env_params"]["dt"]
    time = np.arange(len(states)) * dt

    # X position is qpos[0] = states[:, 0]
    x_position = states[:, 0]

    # Forward velocity is qvel[0] = states[:, 9]
    x_velocity = states[:, 9]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Left: X position (forward progress)
    ax = axes[0]
    ax.plot(time, x_position, label="X Position", color="blue", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X Position (m)")
    ax.set_title("Forward Progress")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Middle: Forward velocity
    ax = axes[1]
    ax.plot(time, x_velocity, label="Forward Velocity", color="green", linewidth=2)
    ax.axhline(
        target_velocity, color='red', linestyle='--', linewidth=2,
        label=f"Target: {target_velocity:.1f}"
    )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Forward Velocity Tracking")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Right: Joint positions (qpos[3:9] for leg joints)
    ax = axes[2]
    joint_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
    for i, label in enumerate(joint_labels):
        ax.plot(time, states[:, 3 + i], label=label, alpha=0.7)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint Angle (rad)")
    ax.set_title("Leg Joint Positions")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    return fig


def create_cheetah_xy_animation(states, dt, max_frames=100, save_path=None):
    """
    Creates an animated GIF showing the cheetah as a stick-figure mesh.

    The camera follows the cheetah's forward progress.

    Args:
        states: Array of 18D states [qpos (9), qvel (9)]
        dt: Timestep in seconds
        max_frames: Maximum number of frames (subsampled if needed)
        save_path: Optional path to save GIF. If None, uses temp file.

    Returns:
        Path to the saved GIF file.
    """
    import tempfile

    states = np.array(states)

    # Subsample if too many frames
    if len(states) > max_frames:
        indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        states = states[indices]
        effective_dt = dt * (len(states) / max_frames)
    else:
        effective_dt = dt

    # Link lengths - scaled to fit within torso height of 0.7m
    # The cheetah's legs bend, so total extended length can exceed standing height
    # Using slightly shorter lengths for visualization clarity
    torso_length = 1.0
    thigh_length = 0.23
    shin_length = 0.23
    foot_length = 0.14

    # Initial torso height from MuJoCo XML (torso starts at z=0.7)
    TORSO_INITIAL_Z = 0.7

    def get_cheetah_points(state):
        """Compute joint positions from state using forward kinematics."""
        rootx = state[0]
        # rootz (qpos[1]) is displacement from initial z position
        rootz = TORSO_INITIAL_Z + state[1]
        rooty = state[2]  # Pitch angle

        # Joint angles
        bthigh_angle = state[3]
        bshin_angle = state[4]
        bfoot_angle = state[5]
        fthigh_angle = state[6]
        fshin_angle = state[7]
        ffoot_angle = state[8]

        # Torso center and endpoints
        torso_front = np.array([
            rootx + torso_length / 2 * np.cos(rooty),
            rootz + torso_length / 2 * np.sin(rooty)
        ])
        torso_back = np.array([
            rootx - torso_length / 2 * np.cos(rooty),
            rootz - torso_length / 2 * np.sin(rooty)
        ])

        # Back leg (attached at torso_back)
        # Thigh hangs down from hip; joint angle 0 means pointing straight down
        back_hip = torso_back.copy()
        angle = rooty - np.pi / 2 + bthigh_angle
        back_knee = back_hip + thigh_length * np.array([np.cos(angle), np.sin(angle)])
        angle += bshin_angle
        back_ankle = back_knee + shin_length * np.array([np.cos(angle), np.sin(angle)])
        angle += bfoot_angle
        back_toe = back_ankle + foot_length * np.array([np.cos(angle), np.sin(angle)])

        # Front leg (attached at torso_front)
        front_hip = torso_front.copy()
        angle = rooty - np.pi / 2 + fthigh_angle
        front_knee = front_hip + thigh_length * np.array([np.cos(angle), np.sin(angle)])
        angle += fshin_angle
        front_ankle = front_knee + shin_length * np.array([np.cos(angle), np.sin(angle)])
        angle += ffoot_angle
        front_toe = front_ankle + foot_length * np.array([np.cos(angle), np.sin(angle)])

        return {
            'torso': (torso_back, torso_front),
            'back_leg': [back_hip, back_knee, back_ankle, back_toe],
            'front_leg': [front_hip, front_knee, front_ankle, front_toe],
            'rootx': rootx,
            'rootz': rootz,
        }

    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Initialize lines
    torso_line, = ax.plot([], [], 'b-', linewidth=6, solid_capstyle='round')
    back_thigh, = ax.plot([], [], 'r-', linewidth=4, solid_capstyle='round')
    back_shin, = ax.plot([], [], 'r-', linewidth=3, solid_capstyle='round')
    back_foot, = ax.plot([], [], 'r-', linewidth=2, solid_capstyle='round')
    front_thigh, = ax.plot([], [], 'g-', linewidth=4, solid_capstyle='round')
    front_shin, = ax.plot([], [], 'g-', linewidth=3, solid_capstyle='round')
    front_foot, = ax.plot([], [], 'g-', linewidth=2, solid_capstyle='round')

    # Joint markers
    joints, = ax.plot([], [], 'ko', markersize=4, zorder=5)

    # Ground line and fill
    ground_line, = ax.plot([], [], 'k-', linewidth=2)
    ground_fill = ax.axhspan(-0.5, 0, color='#8B4513', alpha=0.3)  # Brown ground fill

    # Trail showing path
    trail_line, = ax.plot([], [], 'b--', linewidth=1, alpha=0.3)

    ax.set_aspect('equal')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Z Position (m)')
    ax.grid(True, alpha=0.3)

    # Store trail positions
    trail_x = []
    trail_z = []

    def init():
        torso_line.set_data([], [])
        back_thigh.set_data([], [])
        back_shin.set_data([], [])
        back_foot.set_data([], [])
        front_thigh.set_data([], [])
        front_shin.set_data([], [])
        front_foot.set_data([], [])
        joints.set_data([], [])
        ground_line.set_data([], [])
        trail_line.set_data([], [])
        return (torso_line, back_thigh, back_shin, back_foot,
                front_thigh, front_shin, front_foot, joints, ground_line, trail_line)

    def animate(frame):
        points = get_cheetah_points(states[frame])
        rootx = points['rootx']
        rootz = points['rootz']

        # Update trail
        trail_x.append(rootx)
        trail_z.append(rootz)
        trail_line.set_data(trail_x, trail_z)

        # Update torso
        torso = points['torso']
        torso_line.set_data([torso[0][0], torso[1][0]], [torso[0][1], torso[1][1]])

        # Update back leg segments
        bl = points['back_leg']
        back_thigh.set_data([bl[0][0], bl[1][0]], [bl[0][1], bl[1][1]])
        back_shin.set_data([bl[1][0], bl[2][0]], [bl[1][1], bl[2][1]])
        back_foot.set_data([bl[2][0], bl[3][0]], [bl[2][1], bl[3][1]])

        # Update front leg segments
        fl = points['front_leg']
        front_thigh.set_data([fl[0][0], fl[1][0]], [fl[0][1], fl[1][1]])
        front_shin.set_data([fl[1][0], fl[2][0]], [fl[1][1], fl[2][1]])
        front_foot.set_data([fl[2][0], fl[3][0]], [fl[2][1], fl[3][1]])

        # Joint markers (all joints)
        all_joints_x = [bl[0][0], bl[1][0], bl[2][0], bl[3][0],
                        fl[0][0], fl[1][0], fl[2][0], fl[3][0]]
        all_joints_z = [bl[0][1], bl[1][1], bl[2][1], bl[3][1],
                        fl[0][1], fl[1][1], fl[2][1], fl[3][1]]
        joints.set_data(all_joints_x, all_joints_z)

        # Update view to follow cheetah (camera moves with it)
        view_width = 4.0
        ax.set_xlim(rootx - view_width / 2, rootx + view_width / 2)
        # Ground is at z=0, torso center at ~0.7, show some margin above and below
        ax.set_ylim(-0.3, 1.5)

        # Ground line
        ground_line.set_data([rootx - view_width, rootx + view_width], [0, 0])

        # Update title with time and velocity
        forward_vel = states[frame, 9]  # qvel[0]
        time = frame * effective_dt
        ax.set_title(f'Cheetah Animation | t={time:.2f}s | vel={forward_vel:.2f} m/s')

        return (torso_line, back_thigh, back_shin, back_foot,
                front_thigh, front_shin, front_foot, joints, ground_line, trail_line)

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


def plot_histograms(buffers, buffer_idx, config):
    """Plot histograms of observations and actions."""
    # Buffer stores 17D observations
    dim_obs = config.get("dim_obs", 17)
    observations = np.array(buffers["states"][0, :buffer_idx, :dim_obs])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    norm_params = config["normalization_params"]

    dim_action = actions.shape[1]

    # Create figure with observations and actions
    fig, axes = plt.subplots(2, max(dim_obs, dim_action), figsize=(24, 8))

    # Observation histograms (top row) - show a subset
    obs_indices = [0, 1, 2, 8, 9, 10, 11]  # rootz, rooty, bthigh, ffoot, vel_x...
    obs_labels = ["rootz", "rooty", "bthigh", "ffoot",
                  "vel_x", "vel_z", "vel_y"]
    for i, (idx, label) in enumerate(zip(obs_indices, obs_labels)):
        if i >= axes.shape[1]:
            break
        ax = axes[0, i]
        ax.hist(observations[:, idx], bins=50, alpha=0.7, color='blue')
        ax.set_title(f"Obs: {label}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")

    # Action histograms (bottom row)
    action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
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
    """Main data collection loop for HalfCheetah."""
    wandb.init(
        project=config.get("wandb_project", "cheetah_data_collection"),
        config=config,
        name=f"cheetah_collect_{config['num_episodes']}ep",
        reinit=True,
    )

    key = jax.random.key(config["seed"])
    plot_freq = config.get("plot_freq", 10)

    # Initialize components
    print("Initializing components...")
    reset_fn, step_fn, get_obs_fn, get_state_array_fn = init_env(config)
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
    dim_obs = config.get("dim_obs", 17)
    collect_episode = make_collect_episode_fn(
        reset_fn,
        step_fn,
        get_obs_fn,
        get_state_array_fn,
        planner,
        config["max_episode_length"],
    )
    print("Initialization complete.")

    # Initialize buffer - store 17D observations
    dim_obs = config.get("dim_obs", 17)
    buffer_size = config["num_episodes"] * config["max_episode_length"]
    buffers = init_jax_buffers(
        config["num_agents"],
        buffer_size,
        dim_obs,  # Store 17D observations
        config["dim_action"],
    )
    buffer_idx = 0
    episode_info = []

    print(f"Starting cheetah data collection: {config['num_episodes']} episodes")

    # Collection loop
    for ep in range(config["num_episodes"]):
        # Sample target velocity from domain randomization
        if "domain_randomization" in config:
            target_velocity, key = sample_target_velocity(
                key, config["domain_randomization"]
            )
        else:
            target_velocity = config["cost_fn_params"]["target_velocity"]

        episode_info.append({
            "episode": ep,
            "target_velocity": target_velocity
        })

        # Collect episode
        ep_start = time.perf_counter()
        states, obs, actions, rewards, key = collect_episode(
            key,
            planner_state,
            train_state,
            target_velocity,
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
                obs[t : t + 1],
                actions[t : t + 1],
                rewards[t : t + 1],
                jnp.zeros(1),
                jnp.zeros(1),
                float(t == ep_len - 1),
            )
            buffer_idx += 1

        # Log metrics
        avg_velocity = float(jnp.mean(states[:, 9]))  # Forward velocity
        final_x = float(states[-1, 0])  # Final x position

        print(
            f"Episode {ep + 1}/{config['num_episodes']}: "
            f"time={ep_time:.2f}s, avg_vel={avg_velocity:.2f}, "
            f"final_x={final_x:.2f}, target={target_velocity:.2f}"
        )
        wandb.log({
            "episode/length": ep_len,
            "episode/time": ep_time,
            "episode/avg_velocity": avg_velocity,
            "episode/final_x_position": final_x,
            "episode/target_velocity": target_velocity,
        }, step=ep)

        # Plot episode trajectory at plot_freq
        if ep == 0 or (ep + 1) % plot_freq == 0:
            fig = plot_cheetah_trajectory(states, target_velocity, config)
            wandb.log({f"episode/trajectory_ep_{ep+1}": wandb.Image(fig)}, step=ep)
            plt.close(fig)

            gif_path = create_cheetah_xy_animation(
                states, config["env_params"]["dt"]
            )
            wandb.log({
                f"episode/cheetah_anim_ep_{ep+1}": wandb.Video(gif_path, fps=20, format="gif")
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
        description="Collect data for HalfCheetah offline training"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cheetah.json",
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
