# run_linear.py

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
import argparse
import copy
import os
import pickle
import json


def plot_linear_trajectory(buffers, buffer_idx, config):
    """Plot agent trajectory and target point."""
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    # Extract positions
    pos_x, pos_y = states[:, 0], states[:, 1]
    goal = config["cost_fn_params"]["goal_state"][:2]

    # Get normalization bounds for positions
    norm_params = config["normalization_params"]["state"]
    x_min, x_max = norm_params["min"][0], norm_params["max"][0]
    y_min, y_max = norm_params["min"][1], norm_params["max"][1]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(pos_x, pos_y, label="Path", color="blue", linewidth=2, alpha=0.8)

    # Mark start (circle) and end (x) points
    ax.scatter(pos_x[0], pos_y[0], marker="o", s=100, color="blue",
               label="Start", zorder=5)
    ax.scatter(pos_x[-1], pos_y[-1], marker="x", s=100, color="blue",
               label="End", zorder=5)

    # Mark goal point
    ax.scatter(goal[0], goal[1], marker="*", s=200, color="red",
               label="Goal", zorder=5)

    # Use normalization bounds for axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Formatting
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Linear Tracking Trajectory")
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')
    plt.tight_layout()

    return fig


def plot_state_components(buffers, buffer_idx, config):
    """Plot positions and velocities over time with normalization bounds."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    time = np.arange(buffer_idx) * dt

    norm_params = config["normalization_params"]["state"]
    state_min = norm_params["min"]
    state_max = norm_params["max"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # Positions subplot (pos_x, pos_y)
    ax = axes[0]
    ax.plot(time, states[:, 0], label="pos_x")
    ax.plot(time, states[:, 1], label="pos_y")
    ax.axhline(state_min[0], color='r', linestyle='--', alpha=0.5, label="bounds")
    ax.axhline(state_max[0], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Position")
    ax.set_ylim(state_min[0], state_max[0])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Velocities subplot (vel_x, vel_y)
    ax = axes[1]
    ax.plot(time, states[:, 2], label="vel_x")
    ax.plot(time, states[:, 3], label="vel_y")
    ax.axhline(state_min[2], color='r', linestyle='--', alpha=0.5, label="bounds")
    ax.axhline(state_max[2], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Velocity")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(state_min[2], state_max[2])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_eval_trajectory(eval_results, config):
    """Plot trajectory from rollout_with_trajectory evaluator results."""
    trajectory = eval_results["trajectory"]
    goal = eval_results["goal_state"][:2]

    pos_x, pos_y = trajectory[:, 0], trajectory[:, 1]

    norm_params = config["normalization_params"]["state"]
    x_min, x_max = norm_params["min"][0], norm_params["max"][0]
    y_min, y_max = norm_params["min"][1], norm_params["max"][1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(pos_x, pos_y, label="Path", color="blue", linewidth=2, alpha=0.8)
    ax.scatter(pos_x[0], pos_y[0], marker="o", s=100, color="blue", label="Start", zorder=5)
    ax.scatter(pos_x[-1], pos_y[-1], marker="x", s=100, color="blue", label="End", zorder=5)
    ax.scatter(goal[0], goal[1], marker="*", s=200, color="red", label="Goal", zorder=5)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(f"Eval Rollout Trajectory ({len(trajectory)} steps)")
    ax.legend(loc="best")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_aspect('equal')
    plt.tight_layout()

    return fig


def main(config, save_dir, plot_eval=False):
    wandb.init(
        project=config.get("wandb_project", "linear"),
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
    init_cov_trace = (
        jnp.trace(train_state.covariance) / train_state.covariance.shape[0]
        if train_state.covariance is not None
        else 0.0
    )
    wandb.log({**eval_results, "eval/cov_trace": float(init_cov_trace)}, step=0)

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

    print(
        f"Starting simulation for {config['total_steps']} steps "
    )

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
        action = actions[0][None, :]  # hacky add agent dim

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

        # Train policy
        if step % config["train_policy_freq"] == 0:
            # Unused for policies like iCEM
            pass

        # Evaluate model
        if step % config["eval_freq"] == 0:
            # Run rollout evaluation
            eval_results = evaluator.evaluate(train_state.params)

            # Track covariance trace if available
            cov_trace = (
                jnp.trace(train_state.covariance)
                if train_state.covariance is not None
                else 0.0
            )
            cov_trace_per_param = cov_trace / train_state.covariance.shape[0] if train_state.covariance is not None else 0.0

            wandb.log(
                {
                    **eval_results,
                    "eval/cov_trace": float(cov_trace_per_param),
                },
                step=step,
            )

        # Train model
        # buffer_idx >= 2 to ensure we have at least one full transition
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)
            wandb.log({"train/model_loss": float(loss)}, step=step)

        # Handle Buffer Overflow
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
        run_name = config.get("wandb_run_name", f"linear_model_{config['seed']}")
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

    # Plot and log trajectory
    if save_dir:
        print("\nGenerating trajectory plot...")
        fig = plot_linear_trajectory(buffers, buffer_idx, config)
        wandb.log({"trajectory/linear_plot": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)
        print("Trajectory plot logged to wandb.")

        print("Generating state components plot...")
        fig = plot_state_components(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/state_components": wandb.Image(fig)}, step=config["total_steps"]
        )
        plt.close(fig)
        print("State components plot logged to wandb.")

    # Final evaluation with trajectory plot using rollout_with_plot evaluator
    if plot_eval:
        print("\nRunning final evaluation with trajectory plot...")
        plot_eval_config = copy.deepcopy(config)
        plot_eval_config["evaluator_type"] = "rollout_with_trajectory"
        final_evaluator = init_evaluator(plot_eval_config)
        final_eval_results = final_evaluator.evaluate(train_state.params)

        # Log metrics (excluding trajectory/actions/goal_state which are arrays)
        metrics_to_log = {k: v for k, v in final_eval_results.items()
                         if not isinstance(v, np.ndarray)}
        wandb.log(metrics_to_log, step=config["total_steps"])

        # Plot and log trajectory
        fig = plot_eval_trajectory(final_eval_results, config)
        wandb.log({"final_eval_traj": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)
        print("Final eval trajectory logged to wandb as 'final_eval_traj'")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run linear tracking experiments.")
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
        "--seed",
        type=int,
        default=42,
        help="Starting random seed.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to run for each lambda value.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models",
        help="Directory to save the learned dynamics model parameters.",
    )
    parser.add_argument(
        "--plot-eval",
        action="store_true",
        help="Run final evaluation with trajectory plot logged to wandb.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="linear.json",
        help="Config filename in configs folder. Defaults to linear.json.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["finetuning"]

    run_name_base = args.run_name or "linear"

    if args.lambdas is None:
        lambdas = [
            CONFIG["cost_fn_params"]["weight_info"]
        ]
    else:
        lambdas = args.lambdas

    for lam_idx, lam in enumerate(lambdas, start=1):
        for seed_idx in range(1, args.num_seeds + 1):
            seed = args.seed + seed_idx - 1
            print(f"--- Starting run for lam{lam_idx} (lambda={lam}), run {seed_idx}/{args.num_seeds} ---")
            run_config = copy.deepcopy(CONFIG)
            run_config["seed"] = seed
            run_config["wandb_group"] = "linear_tracking"
            run_config["cost_fn_params"]["weight_info"] = lam

            # Build run name: base_lam{idx}_seed{idx}
            run_name = run_name_base
            if args.lambdas is not None:
                run_name = f"{run_name}_lam{lam}"
            if args.num_seeds > 1:
                run_name = f"{run_name}_{seed_idx}"
            run_config["wandb_run_name"] = run_name

            main(run_config, save_dir=args.save_dir, plot_eval=args.plot_eval)

    print("All experiments complete.")
