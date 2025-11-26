# ippo_tracking.py

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from max.normalizers import (
    init_normalizer,
    init_rolling_return_normalizer,
    update_rolling_return_stats,
)
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.policies import init_policy
from max.policy_trainers import init_policy_trainer
from max.policy_evaluators import evaluate_policy
from max.environments import init_env
import argparse
import copy
import time
import functools
import os
import pickle
import json

def main(config, save_dir):
    wandb.init(
        project="lqr",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    print("Setting up environment")
    # TODO: create a single-agent environment where first player is controlled and others are scripted
    # first player is the learning agent. Second agent just has a fixed policy. Their policy is a one-step LQR to the first agent's position.
    # This is the same way the pursuit-evasion environment is structured. 
    # Make the reward be the LQR cost of the first agent only.
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize policy 
    # TODO: initialize icem policy

    # Initialize dynamics
    # TODO: import from dynamics module. Use pursuit-evader dynamics where the params are the pursuer's LQR weights
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    # Init dynamics evaluator
    # TODO : import from dynamics evaluator module

    # Initialize dynamics trainer
    # TODO : import from dynamics traines module

    # Initialize buffer
    num_agents = config["num_agents"]
    buffers = init_jax_buffers(
        config["num_agents"],
        config["train_policy_freq"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    print(
        f"Starting online training for {config['total_steps']} steps "
        f"with train_policy_freq = {config['train_policy_freq']}..."
    )

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["reward"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    rollout_start_time = time.time()
    time_policy_inference = 0.0
    time_env_step = 0.0
    time_buffering = 0.0
    time_reward_norm = 0.0
    time_env_reset = 0.0
    time_wandb_log = 0.0
    time_train_prep = 0.0

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state) 

    for step in range(1, config["total_steps"] + 1):

        key, action_key, reset_key = jax.random.split(key, 3)

        # Select action
        t_start_policy = time.time()
        dyn_params = None

        agent_action_keys = jax.random.split(action_key, num_agents)

        actions, values, log_pis = policy_step(
            policy_train_state.params,
            current_obs,
            dyn_params,
            agent_action_keys,
        ) # TODO: modify icem output so that it uses dummy log_pis and values
        time_policy_inference += time.time() - t_start_policy

        # Execute action in environment
        t_start_env = time.time()
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, actions
        )
        time_env_step += time.time() - t_start_env

        # Accumulate reward components
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]

        done = terminated or truncated

        episode_length += 1

        # Add data to the buffer
        t_start_buffer = time.time()
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs,
            actions,
            rewards,
            log_pis,
            values,
            float(done),
        )
        buffer_idx += 1
        time_buffering += time.time() - t_start_buffer

        current_obs = next_obs

        # Reset environment if done
        if done:
            t_start_env_reset = time.time()

            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)
            time_env_reset += time.time() - t_start_env_reset

            t_start_wandb_log = time.time()
            episode_log = {"episode/length": episode_length}

            if episode_length > 0:
                for info_key in reward_component_keys_to_avg:
                    avg_val = (
                        episode_reward_components[info_key] / episode_length
                    )
                    episode_log[f"rewards/{info_key}"] = float(avg_val)

            wandb.log(episode_log, step=step)
            time_wandb_log += time.time() - t_start_wandb_log

            # Reset accumulators
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train policy
        if step % config["train_policy_freq"] == 0:
            # Unused for policies like iCEM 


        # Train model
        if step % config["train_model_freq"] == 0:
            # Call the dynamics trainer here

            # Reset buffers
            buffers = init_jax_buffers(
                config["num_agents"],
                config["train_policy_freq"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    if save_dir:
        run_name = config.get(
            "wandb_run_name", f"lqr_model_{config['seed']}"
        )
        save_path = os.path.join(save_dir, run_name)

        os.makedirs(save_path, exist_ok=True)

        print(f"\nSaving final model parameters to {save_path}...")

        # TODO: save dynamics model parameters instead
        target_params_np = jax.device_get(policy_train_state.params)

        file_path = os.path.join(save_path, "policy_params.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(target_params_np, f)

        print(f"Policy parameters saved to {file_path}")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LQR control experiments."
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the W&B run.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--meta-seed",
        type=int,
        default=42,
        help="A seed to generate the run seeds.",
    )
    # TODO: change to save dynamics model parameters
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_policies",
        help="Directory to save the trained policy parameters.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    # TODO: create a lqr.json config file. Use run_dogfight as inspiration
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lqr.json")
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

    if args.meta_seed is not None:
        rng = np.random.default_rng(args.meta_seed)
        seeds = rng.integers(low=0, high=2**32 - 1, size=args.num_seeds)
    else:
        seeds = range(args.num_seeds)

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Starting run for seed #{seed} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = int(seed)
        run_config["wandb_group"] = "lqr"

        if args.run_name:
            run_name_base = args.run_name
        else:
            run_name_base = "lqr"

        if args.num_seeds > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_seed_{seed_idx}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")