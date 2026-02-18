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


def sample_target_velocity(key, dr_config):
    """Sample a target velocity from domain randomization config."""
    vel_bounds = dr_config["target_velocity"]
    min_val = vel_bounds["min"]
    max_val = vel_bounds["max"]
    key, subkey = jax.random.split(key)
    target_vel = jax.random.uniform(subkey, minval=min_val, maxval=max_val)
    return float(target_vel), key


def collect_episode(
    key,
    reset_fn,
    step_fn,
    get_obs_fn,
    planner,
    planner_state,
    train_state,
    target_velocity,
    max_episode_length,
):
    """Collect a single episode of transitions."""
    episode_states = []  # 18D states
    episode_obs = []  # 17D observations for buffer
    episode_actions = []
    episode_rewards = []

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)  # 18D state
    current_obs = get_obs_fn(state)  # 17D observation

    for step_idx in range(max_episode_length):
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "target_velocity": target_velocity,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        # Plan using 18D state
        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # Add agent dim

        episode_states.append(state)
        episode_obs.append(current_obs[0])  # Remove agent dim, store 17D
        episode_actions.append(action[0])

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, step_idx, action
        )
        episode_rewards.append(rewards[0])
        current_obs = next_obs

        if terminated or truncated:
            break

    return (
        jnp.stack(episode_states),  # 18D states for visualization
        jnp.stack(episode_obs),  # 17D observations for buffer
        jnp.stack(episode_actions),
        jnp.array(episode_rewards),
        len(episode_states),
        key,
    )


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


def create_cheetah_xy_animation(states, dt, max_frames=200):
    """
    Creates a simple animation showing the cheetah's forward progress.
    Returns a figure with the final trajectory plotted.
    """
    states = np.array(states)

    # X position over time
    x_position = states[:, 0]
    time = np.arange(len(states)) * dt

    # Subsample if too many frames
    if len(states) > max_frames:
        indices = np.linspace(0, len(states) - 1, max_frames, dtype=int)
        x_position = x_position[indices]
        time = time[indices]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(time, x_position, 'b-', linewidth=2, label='X Position')
    ax.scatter(time[0], x_position[0], color='green', s=100,
               marker='o', label='Start', zorder=5)
    ax.scatter(time[-1], x_position[-1], color='red', s=100,
               marker='x', label='End', zorder=5)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("X Position (m)")
    ax.set_title("Cheetah Forward Progress")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


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
        states, obs, actions, rewards, ep_len, key = collect_episode(
            key,
            reset_fn,
            step_fn,
            get_obs_fn,
            planner,
            planner_state,
            train_state,
            target_velocity,
            config["max_episode_length"],
        )

        # Add 17D observations to buffer
        for t in range(ep_len):
            if buffer_idx >= buffer_size:
                print(f"Warning: Buffer full at episode {ep}, truncating.")
                break
            buffers = update_buffer_dynamic(
                buffers,
                buffer_idx,
                obs[t : t + 1],  # 17D observation
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
            f"len={ep_len}, avg_vel={avg_velocity:.2f}, "
            f"final_x={final_x:.2f}, target={target_velocity:.2f}"
        )
        wandb.log({
            "episode/length": ep_len,
            "episode/avg_velocity": avg_velocity,
            "episode/final_x_position": final_x,
            "episode/target_velocity": target_velocity,
        }, step=ep)

        # Plot episode trajectory at plot_freq
        if ep == 0 or (ep + 1) % plot_freq == 0:
            # Trajectory plot
            fig = plot_cheetah_trajectory(states, target_velocity, config)
            wandb.log({f"episode/trajectory_ep_{ep+1}": wandb.Image(fig)}, step=ep)
            plt.close(fig)

            # XY progress plot
            fig = create_cheetah_xy_animation(
                states, config["env_params"]["dt"]
            )
            wandb.log({f"episode/xy_progress_ep_{ep+1}": wandb.Image(fig)}, step=ep)
            plt.close(fig)

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
    print(f"Data collection complete. Total transitions: {buffer_idx}")


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
