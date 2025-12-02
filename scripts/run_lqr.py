# run_lqr.py

import jax
import numpy as np
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.dynamics_evaluators import DynamicsEvaluator
from max.planners import init_planner
from max.costs import init_cost
import argparse
import copy
import time
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

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    # Initialize dynamics trainer
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        config, dynamics_model, init_params, trainer_key
    )

    # Initialize dynamics evaluator
    evaluator = DynamicsEvaluator(dynamics_model.pred_one_step)

    # Initialize cost function
    cost_fn = init_cost(config, dynamics_model)

    # Initialize planner
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

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

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state) 

    for step in range(1, config["total_steps"] + 1):

        key, action_key, reset_key = jax.random.split(key, 3)

        # Select action
        dyn_params = None

        agent_action_keys = jax.random.split(action_key, num_agents)

        # Execute policy
        actions, values, log_pis = policy_step(
            None,
            current_obs,
            dyn_params,
            agent_action_keys,
        )

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, actions
        )
        done = terminated or truncated
        episode_length += 1
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]

        # Update buffer
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
            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)
            
            # Log and reset episode stats
            episode_log = {"episode/length": episode_length}
            if episode_length > 0:
                for info_key in reward_component_keys_to_avg:
                    avg_val = (
                        episode_reward_components[info_key] / episode_length
                    )
                    episode_log[f"rewards/{info_key}"] = float(avg_val)
            wandb.log(episode_log, step=step)
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train policy
        if step % config["train_policy_freq"] == 0:
            # Unused for policies like iCEM
            pass

        # Train model
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            # Get the most recent complete transition from buffer
            # We need buffer_idx >= 2 because we need (s_t, a_t, s_{t+1})
            # buffer[i] contains state at time i, action at time i
            # buffer[i+1] contains state at time i+1 (which is next_state for transition i)

            # Train on the global state (same for all agents) and evader's action (agent 1)
            prev_idx = buffer_idx - 2
            curr_idx = buffer_idx - 1

            # Note: We use the global state (agent 0's obs = agent 1's obs = full state)
            # and the evader's action (agent 1)
            train_data = {
                "states": buffers["state"][0:1, prev_idx:prev_idx+1, :],  # Global state at t
                "actions": buffers["action"][1:2, prev_idx:prev_idx+1, :],  # Evader's action at t
                "next_states": buffers["state"][0:1, curr_idx:curr_idx+1, :],  # Global state at t+1
            }

            # Reshape to (1, dim) for batch_size=1
            train_data = {
                "states": train_data["states"].reshape(1, -1),
                "actions": train_data["actions"].reshape(1, -1),
                "next_states": train_data["next_states"].reshape(1, -1),
            }

            # Update dynamics parameters using EKF
            train_state, loss = trainer.train(train_state, train_data, step=step)
            wandb.log({"train/model_loss": float(loss)}, step=step)

            # Reset buffers
            buffers = init_jax_buffers(
                config["num_agents"],
                config["train_policy_freq"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # Save model parameters
    if save_dir:
        run_name = config.get(
            "wandb_run_name", f"lqr_model_{config['seed']}"
        )
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving final model parameters to {save_path}...")
        dynamics_params_np = jax.device_get(train_state.params)
        file_path = os.path.join(save_path, "dynamics_params.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(dynamics_params_np, f)
        print(f"Dynamics parameters saved to {file_path}")
        if train_state.covariance is not None:
            cov_path = os.path.join(save_path, "param_covariance.pkl")
            cov_np = jax.device_get(train_state.covariance)
            with open(cov_path, "wb") as f:
                pickle.dump(cov_np, f)
            print(f"Parameter covariance saved to {cov_path}")

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
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models",
        help="Directory to save the learned dynamics model parameters.",
    )
    args = parser.parse_args()

    # Load config from JSON file
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