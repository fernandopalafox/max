# collect_hopper_data.py

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
import argparse
import os
import pickle
import json


def collect_random_episode(
    key,
    reset_fn,
    step_fn,
    get_obs_fn,
    max_episode_length,
    max_torque,
    dim_action,
):
    """Collect a single episode using random actions."""
    episode_states = []
    episode_actions = []
    episode_rewards = []

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step_idx in range(max_episode_length):
        # Sample random action
        key, action_key = jax.random.split(key)
        raw_action = jax.random.uniform(
            action_key,
            shape=(dim_action,),
            minval=-max_torque,
            maxval=max_torque,
        )
        action = raw_action[None, :]  # Add agent dim

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


def save_buffer(buffers, buffer_idx, save_path, env_name):
    """Save buffer data to disk as pickle file."""
    os.makedirs(save_path, exist_ok=True)

    data = {
        "states": np.array(buffers["states"][:, :buffer_idx, :]),
        "actions": np.array(buffers["actions"][:, :buffer_idx, :]),
        "rewards": np.array(buffers["rewards"][:, :buffer_idx]),
        "dones": np.array(buffers["dones"][:buffer_idx]),
        "num_transitions": buffer_idx,
    }

    file_path = os.path.join(save_path, f"{env_name}_buffer.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Buffer saved to {file_path}")


def main(config, save_dir):
    """Main data collection loop using random actions."""
    wandb.init(
        project=config.get("wandb_project", "hopper_data_collection"),
        config=config,
        name=f"collect_hopper_{config['num_episodes']}ep",
        reinit=True,
    )

    key = jax.random.key(config["seed"])

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    max_torque = config["env_params"].get("max_torque", 100.0)
    dim_action = config["dim_action"]

    # Allocate buffer
    buffer_size = config["num_episodes"] * config["max_episode_length"]
    buffers = init_jax_buffers(
        config["num_agents"],
        buffer_size,
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    print(f"Starting hopper data collection: {config['num_episodes']} episodes with random actions")

    for ep in range(config["num_episodes"]):
        states, actions, rewards, ep_len, key = collect_random_episode(
            key,
            reset_fn,
            step_fn,
            get_obs_fn,
            config["max_episode_length"],
            max_torque,
            dim_action,
        )

        # Add transitions to buffer
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

    # Save buffer
    save_buffer(buffers, buffer_idx, save_dir, config["env_name"])

    wandb.finish()
    print("Hopper data collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect hopper data via random actions")
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
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    config = full_config["data_collection"]

    # Apply CLI overrides
    if args.save_dir is not None:
        config["save_path"] = args.save_dir
    if args.seed is not None:
        config["seed"] = args.seed

    save_dir = config["save_path"]

    main(config, save_dir)
