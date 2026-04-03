# run_hopper.py

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
from max.dynamics_evaluators import init_evaluator
from max.planners import init_planner
from max.costs import init_cost
from visualize_hopper import make_hopper_animation, plot_hopper_states
import argparse
import copy
import os
import pickle
import json


def main(config, save_dir):
    wandb.init(
        project=config.get("wandb_project", "hopper_finetuning"),
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

    # Count trainable parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
    wandb.config.update({"num_params": num_params})

    # Initialize evaluator
    evaluator = init_evaluator(config)

    # Initial evaluation before training
    eval_results = evaluator.evaluate(train_state.params)
    initial_cov_trace_per_param = None
    if train_state.covariance is not None:
        cov_trace = jnp.trace(train_state.covariance)
        initial_cov_trace_per_param = cov_trace / train_state.covariance.shape[0]
    wandb.log(
        {**eval_results, "eval/cov_trace_delta": 0.0},
        step=0,
    )

    # Initialize cost function
    cost_fn = init_cost(config, dynamics_model)

    # Initialize planner
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    # Initialize buffer
    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    print(f"Starting hopper simulation for {config['total_steps']} steps")

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["reward", "dist_to_target"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)
    goal_state = np.array(config["cost_fn_params"]["goal_state"])

    for step in range(1, config["total_steps"] + 1):

        # Compute actions
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "goal_state": goal_state,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # add agent dim

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]

        # Update buffer
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs,
            action,
            rewards,
            jnp.zeros_like(rewards),  # dummy value
            jnp.zeros_like(rewards),  # dummy log_pi
            float(done),
        )
        buffer_idx += 1

        current_obs = next_obs

        # Reset environment if done
        if done:
            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)
            print(f"Episode finished at step {step}.")

            # Log and reset episode stats
            episode_log = {"episode/length": episode_length}
            if episode_length > 0:
                for info_key in reward_component_keys_to_avg:
                    avg_val = episode_reward_components[info_key] / episode_length
                    episode_log[f"rewards/{info_key}"] = float(avg_val)
            wandb.log(episode_log, step=step)
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train policy (unused for iCEM)
        if step % config["train_policy_freq"] == 0:
            pass

        # Evaluate model
        if step % config["eval_freq"] == 0:
            eval_results = evaluator.evaluate(train_state.params)

            # Track covariance trace if available
            if train_state.covariance is not None:
                cov_trace = jnp.trace(train_state.covariance)
                cov_trace_per_param = cov_trace / train_state.covariance.shape[0]
                cov_trace_delta = cov_trace_per_param - initial_cov_trace_per_param
                wandb.log(
                    {
                        **eval_results,
                        "eval/cov_trace_delta": float(cov_trace_delta),
                    },
                    step=step,
                )
            else:
                wandb.log(
                    {**eval_results, "eval/cov_trace_delta": 0.0},
                    step=step,
                )

        # Train model (need at least one full transition)
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)
            wandb.log({"train/model_loss": float(loss)}, step=step)

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"],
                config["buffer_size"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # Save model parameters
    if save_dir:
        run_name = config.get("wandb_run_name", f"hopper_model_{config['seed']}")
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

    # Generate and log hopper visualization
    if buffer_idx > 0:
        states = np.array(buffers["states"][0, :buffer_idx, :])

        print("\nGenerating hopper animation...")
        gif_path = make_hopper_animation(states, config, fps=30)
        wandb.log(
            {"trajectory/animation": wandb.Video(gif_path, format="gif")},
            step=config["total_steps"],
        )
        print("Animation logged to wandb.")

        print("Generating state components plot...")
        fig = plot_hopper_states(states, config)
        wandb.log(
            {"trajectory/state_components": wandb.Image(fig)},
            step=config["total_steps"],
        )
        plt.close(fig)
        print("State components plot logged to wandb.")

    wandb.finish()
    print("Run complete.")
    return eval_results.get("eval/terminal_goal_cost")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hopper experiments.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the W&B run.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=None,
        help="List of weight_info values to sweep.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to run for each lambda value.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="hopper.json",
        help="Config filename in configs folder. Defaults to hopper.json.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models",
        help="Directory to save the learned dynamics model parameters.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["finetuning"]

    run_name_base = args.run_name or "hopper"

    if args.lambdas is None:
        lambdas = [CONFIG["cost_fn_params"]["weight_info"]]
    else:
        lambdas = args.lambdas

    # Generate seeds using JAX RNG from config seed
    base_key = jax.random.key(CONFIG["seed"])
    seed_keys = jax.random.split(base_key, args.num_seeds)
    seeds = [int(jax.random.bits(k)) for k in seed_keys]

    for lam_idx, lam in enumerate(lambdas, start=1):
        for seed_idx, seed in enumerate(seeds, start=1):
            print(f"--- Starting run for lam{lam_idx} (lambda={lam}), seed {seed_idx}/{args.num_seeds} ---")
            run_config = copy.deepcopy(CONFIG)
            run_config["seed"] = seed
            run_config["cost_fn_params"]["weight_info"] = lam

            # Build run name: base_lam{val}_seed{idx}
            run_name = run_name_base
            if args.lambdas is not None:
                run_name = f"{run_name}_lam{lam}"
            if args.num_seeds > 1:
                run_name = f"{run_name}_{seed_idx}"
            run_config["wandb_run_name"] = run_name

            main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
