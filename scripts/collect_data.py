# collect_data.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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


def sample_goal_state(key, dr_config):
    """Sample a goal state from domain randomization config."""
    goal_bounds = dr_config["goal_state"]
    min_val = jnp.array(goal_bounds["min"])
    max_val = jnp.array(goal_bounds["max"])
    key, subkey = jax.random.split(key)
    goal = jax.random.uniform(subkey, shape=min_val.shape, minval=min_val, maxval=max_val)
    return goal, key


def collect_episode(
    key,
    reset_fn,
    step_fn,
    get_obs_fn,
    planner,
    planner_state,
    train_state,
    goal_state,
    max_episode_length,
):
    """Collect a single episode of transitions."""
    episode_states = []
    episode_actions = []
    episode_rewards = []

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step_idx in range(max_episode_length):
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "goal_state": goal_state,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # Add agent dim

        episode_states.append(current_obs[0])  # Remove agent dim
        episode_actions.append(action[0])

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, step_idx, action
        )
        episode_rewards.append(rewards[0])
        current_obs = next_obs

        if terminated or truncated:
            break

    return (
        jnp.stack(episode_states),
        jnp.stack(episode_actions),
        jnp.array(episode_rewards),
        len(episode_states),
        key,
    )


def plot_episode_trajectory(states, goal_state, config):
    """Plot XY trajectory (left) and state components over time (right)."""
    states = np.array(states)
    pos_x, pos_y = states[:, 0], states[:, 1]
    dt = config["env_params"]["dt"]
    time = np.arange(len(states)) * dt

    norm_params = config["normalization_params"]["state"]
    state_min = norm_params["min"]
    state_max = norm_params["max"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Left: XY trajectory
    ax = axes[0]
    ax.plot(pos_x, pos_y, label="Path", color="blue", linewidth=2, alpha=0.8)
    ax.scatter(pos_x[0], pos_y[0], marker="o", s=100, color="blue", label="Start", zorder=5)
    ax.scatter(pos_x[-1], pos_y[-1], marker="x", s=100, color="blue", label="End", zorder=5)
    ax.scatter(goal_state[0], goal_state[1], marker="*", s=200, color="red", label="Goal", zorder=5)
    ax.set_xlim(state_min[0], state_max[0])
    ax.set_ylim(state_min[1], state_max[1])
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("XY Trajectory")
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')

    # Middle: Position over time
    ax = axes[1]
    ax.plot(time, states[:, 0], label="pos_x")
    ax.plot(time, states[:, 1], label="pos_y")
    ax.axhline(goal_state[0], color='g', linestyle='--', alpha=0.7, label="goal_x")
    ax.axhline(goal_state[1], color='orange', linestyle='--', alpha=0.7, label="goal_y")
    ax.set_ylabel("Position")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(state_min[0], state_max[0])
    ax.set_title("Position vs Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Right: Velocity over time
    ax = axes[2]
    ax.plot(time, states[:, 2], label="vel_x")
    ax.plot(time, states[:, 3], label="vel_y")
    ax.axhline(goal_state[2], color='g', linestyle='--', alpha=0.7, label="goal_vx")
    ax.axhline(goal_state[3], color='orange', linestyle='--', alpha=0.7, label="goal_vy")
    ax.set_ylabel("Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(state_min[2], state_max[2])
    ax.set_title("Velocity vs Time")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_histograms(buffers, buffer_idx, config):
    """Plot histograms of states and actions with normalization bounds as red lines."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    norm_params = config["normalization_params"]
    state_labels = config.get("state_labels", [f"s{i}" for i in range(states.shape[1])])

    dim_state = states.shape[1]
    dim_action = actions.shape[1]

    fig, axes = plt.subplots(2, max(dim_state, dim_action), figsize=(4 * max(dim_state, dim_action), 8))

    # State histograms (top row)
    for i in range(dim_state):
        ax = axes[0, i]
        ax.hist(states[:, i], bins=50, alpha=0.7, color='blue')
        ax.axvline(norm_params["state"]["min"][i], color='red', linestyle='--', linewidth=2, label='bounds')
        ax.axvline(norm_params["state"]["max"][i], color='red', linestyle='--', linewidth=2)
        ax.set_title(f"State: {state_labels[i]}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend()

    # Action histograms (bottom row)
    for i in range(dim_action):
        ax = axes[1, i]
        ax.hist(actions[:, i], bins=50, alpha=0.7, color='green')
        ax.axvline(norm_params["action"]["min"][i], color='red', linestyle='--', linewidth=2, label='bounds')
        ax.axvline(norm_params["action"]["max"][i], color='red', linestyle='--', linewidth=2)
        ax.set_title(f"Action: a{i}")
        ax.set_xlabel("Value")
        ax.set_ylabel("Count")
        if i == 0:
            ax.legend()

    # Hide unused subplots
    for i in range(dim_state, max(dim_state, dim_action)):
        axes[0, i].axis('off')
    for i in range(dim_action, max(dim_state, dim_action)):
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
    """Main data collection loop."""
    wandb.init(
        project=config.get("wandb_project", "data_collection"),
        config=config,
        name=f"collect_{config['num_episodes']}ep",
        reinit=True,
    )

    key = jax.random.key(config["seed"])
    plot_freq = config.get("plot_freq", 10)

    # Initialize components
    reset_fn, step_fn, get_obs_fn = init_env(config)
    normalizer, norm_params = init_normalizer(config)

    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    # Initialize buffer (size = num_episodes * max_episode_length)
    buffer_size = config["num_episodes"] * config["max_episode_length"]
    buffers = init_jax_buffers(
        config["num_agents"],
        buffer_size,
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0
    episode_info = []

    print(f"Starting data collection: {config['num_episodes']} episodes")

    # Collection loop
    for ep in range(config["num_episodes"]):
        # Sample goal from domain randomization
        if "domain_randomization" in config:
            goal_state, key = sample_goal_state(key, config["domain_randomization"])
            goal_state = np.array(goal_state)
        else:
            goal_state = np.array(config["cost_fn_params"]["goal_state"])

        episode_info.append({"episode": ep, "goal_state": goal_state.tolist()})

        # Collect episode
        states, actions, rewards, ep_len, key = collect_episode(
            key,
            reset_fn,
            step_fn,
            get_obs_fn,
            planner,
            planner_state,
            train_state,
            goal_state,
            config["max_episode_length"],
        )

        # Add to buffer
        for t in range(ep_len):
            if buffer_idx >= buffer_size:
                print(f"Warning: Buffer full at episode {ep}, truncating.")
                break
            buffers = update_buffer_dynamic(
                buffers,
                buffer_idx,
                states[t : t + 1],
                actions[t : t + 1],
                rewards[t : t + 1],
                jnp.zeros(1),
                jnp.zeros(1),
                float(t == ep_len - 1),
            )
            buffer_idx += 1

        print(f"Episode {ep + 1}/{config['num_episodes']}: length={ep_len}, buffer_idx={buffer_idx}")
        wandb.log({"episode/length": ep_len}, step=ep)

        # Plot episode trajectory at plot_freq
        if (ep + 1) % plot_freq == 0:
            fig = plot_episode_trajectory(states, goal_state, config)
            wandb.log({f"episode/ep_{ep+1}": wandb.Image(fig)}, step=ep)
            plt.close(fig)

    # Final histograms - using matplotlib and logging as image
    fig = plot_histograms(buffers, buffer_idx, config)
    wandb.log({"data/histograms": wandb.Image(fig)})
    plt.close(fig)

    # Log normalization bounds to summary for reference
    norm_params_config = config["normalization_params"]
    wandb.summary["norm_bounds/state_min"] = norm_params_config["state"]["min"]
    wandb.summary["norm_bounds/state_max"] = norm_params_config["state"]["max"]
    wandb.summary["norm_bounds/action_min"] = norm_params_config["action"]["min"]
    wandb.summary["norm_bounds/action_max"] = norm_params_config["action"]["max"]

    # Save buffer to disk
    save_buffer(buffers, buffer_idx, save_dir, config["env_name"], episode_info)

    wandb.finish()
    print("Data collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect data for offline training")
    parser.add_argument(
        "--config",
        type=str,
        default="linear_with_nn.json",
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
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
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
