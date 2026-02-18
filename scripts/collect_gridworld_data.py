# collect_gridworld_data.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
import argparse
import os
import pickle
import json


def collect_episode_random_from_start(
    key,
    start_state,
    step_fn,
    get_obs_fn,
    max_episode_length,
):
    """Collect a single episode using random action exploration from a specific start state."""
    episode_states = []
    episode_actions = []
    episode_rewards = []

    # Start from specified state instead of reset
    state = start_state
    current_obs = get_obs_fn(state)

    for step_idx in range(max_episode_length):
        # Random action from {0, 1, 2, 3}
        key, action_key = jax.random.split(key)
        action_int = jax.random.randint(action_key, shape=(1,), minval=0, maxval=4)
        action = action_int.astype(jnp.float32)

        episode_states.append(current_obs[0])  # Remove agent dim
        episode_actions.append(action)

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


def collect_episode_random(
    key,
    reset_fn,
    step_fn,
    get_obs_fn,
    max_episode_length,
):
    """Collect a single episode using random action exploration."""
    episode_states = []
    episode_actions = []
    episode_rewards = []

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step_idx in range(max_episode_length):
        # Random action from {0, 1, 2, 3}
        key, action_key = jax.random.split(key)
        action_int = jax.random.randint(action_key, shape=(1,), minval=0, maxval=4)
        action = action_int.astype(jnp.float32)

        episode_states.append(current_obs[0])  # Remove agent dim
        episode_actions.append(action)

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


def plot_maze_trajectory(states, maze_layout, episode_num, save_dir):
    """Plot the agent's path through the maze."""
    states_np = np.array(states)
    maze_arr = np.array(maze_layout)

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw maze walls and corridors (black = wall, white = corridor)
    for y in range(10):
        for x in range(10):
            bitmask = maze_arr[y, x]
            if bitmask == 0:
                # Wall cell - black
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                # Corridor cell - white
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray', linewidth=0.5))

    # Plot trajectory
    ax.plot(states_np[:, 0], states_np[:, 1], 'b-', alpha=0.6, linewidth=2, label='Path')
    ax.scatter(states_np[0, 0], states_np[0, 1], marker='o', s=150, color='green', label='Start', zorder=5)
    ax.scatter(states_np[-1, 0], states_np[-1, 1], marker='X', s=150, color='red', label='End', zorder=5)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Episode {episode_num} Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig


def save_buffer(buffers, buffer_idx, save_path, env_name):
    """Save buffer data to pickle file."""
    os.makedirs(save_path, exist_ok=True)

    data = {
        "states": np.array(buffers["states"][:, :buffer_idx, :]),
        "actions": np.array(buffers["actions"][:, :buffer_idx, :]),
        "rewards": np.array(buffers["rewards"][:, :buffer_idx]),
        "dones": np.array(buffers["dones"][:buffer_idx]),
        "num_transitions": buffer_idx,
    }

    filename = f"{env_name}_buffer.pkl"
    filepath = os.path.join(save_path, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"\nData saved to {filepath}")
    print(f"Total transitions: {buffer_idx}")
    print(f"States shape: {data['states'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    return filepath


def get_navigable_cells(maze_layout):
    """Extract all navigable (x, y) positions from the maze."""
    navigable = []
    for y in range(len(maze_layout)):
        for x in range(len(maze_layout[0])):
            if maze_layout[y][x] > 0:  # Non-zero bitmask = navigable
                navigable.append((x, y))
    return navigable


def main(config):
    wandb.init(
        project=config.get("wandb_project", "gridworld_data_collection"),
        config=config,
        name=f"collect_{config['env_name']}",
        reinit=True,
    )

    key = jax.random.key(config["seed"])

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Get all navigable cells from maze
    maze_layout = config["env_params"]["maze_layout"]
    navigable_cells = get_navigable_cells(maze_layout)

    print(f"Found {len(navigable_cells)} navigable cells in the maze")
    print(f"Will collect {config['episodes_per_cell']} episodes from each cell")

    # Initialize buffers
    num_agents = config["num_agents"]
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]

    # Buffer size based on cells × episodes_per_cell × max_length
    episodes_per_cell = config.get("episodes_per_cell", 5)
    max_episode_length = config["max_episode_length"]
    total_episodes = len(navigable_cells) * episodes_per_cell
    buffer_size = total_episodes * max_episode_length

    buffers = init_jax_buffers(num_agents, buffer_size, dim_state, dim_action)
    buffer_idx = 0

    plot_freq = config.get("plot_freq", 50)
    episode_counter = 0

    print(f"Collecting data from all {len(navigable_cells)} navigable cells...")
    print(f"Episodes per cell: {episodes_per_cell}")
    print(f"Total episodes: {total_episodes}")
    print(f"Max episode length: {max_episode_length}")

    # Iterate over each navigable cell
    for cell_idx, (x, y) in enumerate(navigable_cells):
        # Create start state for this cell
        start_state = jnp.array([float(x), float(y)])

        # Collect multiple episodes from this cell
        for ep_in_cell in range(episodes_per_cell):
            states, actions, rewards, ep_len, key = collect_episode_random_from_start(
                key, start_state, step_fn, get_obs_fn, max_episode_length
            )

            # Add to buffer
            for t in range(ep_len):
                buffers = update_buffer_dynamic(
                    buffers,
                    buffer_idx,
                    states[t : t + 1],      # Shape: (1, 2) = (num_agents, dim_state)
                    actions[t : t + 1],     # Shape: (1, 1) = (num_agents, dim_action)
                    rewards[t : t + 1],     # Shape: (1,) = (num_agents,)
                    jnp.zeros(1),           # log_pis
                    jnp.zeros(1),           # values
                    float(t == ep_len - 1),  # done flag
                )
                buffer_idx += 1

            # Logging
            episode_counter += 1
            wandb.log({
                "episode": episode_counter,
                "cell_index": cell_idx,
                "start_x": x,
                "start_y": y,
                "episode_length": ep_len,
                "episode_reward": float(jnp.sum(rewards)),
            })

            # Plot trajectory
            if episode_counter % plot_freq == 0:
                print(f"Episode {episode_counter}/{total_episodes}: Cell ({x},{y}), length={ep_len}")
                fig = plot_maze_trajectory(
                    states,
                    maze_layout,
                    episode_counter,
                    config["save_path"]
                )
                wandb.log({f"trajectory/ep_{episode_counter}": wandb.Image(fig)})
                plt.close(fig)

    print(f"\nCompleted {episode_counter} episodes from {len(navigable_cells)} cells")

    # Save buffer
    save_path = save_buffer(
        buffers,
        buffer_idx,
        config["save_path"],
        config["env_name"]
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect gridworld data with random exploration")
    parser.add_argument("--config", type=str, default="gridworld.json")
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    config = full_config["data_collection"]

    # Override with command line args
    if args.num_episodes:
        config["num_episodes"] = args.num_episodes
    if args.seed:
        config["seed"] = args.seed
    if args.save_dir:
        config["save_path"] = args.save_dir
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project

    main(config)
