# run_drone.py

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
from max.dynamics_evaluators import DynamicsEvaluator
from max.planners import init_planner
from max.costs import init_cost
import argparse
import copy
import os
import pickle
import json


def plot_drone_trajectory(buffers, buffer_idx, config):
    """Plot drone trajectory and state evolution."""
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    # Extract position, angle, and velocities
    p_x, p_y = states[:, 0], states[:, 1]
    phi = states[:, 2]
    v_x, v_y = states[:, 3], states[:, 4]
    phi_dot = states[:, 5]
    time = np.arange(len(p_x)) * config["env_params"]["dt"]

    # Get goal position
    goal_pos = config["cost_fn_params"]["goal_position"]

    # Create plot with four subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: XY trajectory
    axes[0, 0].plot(p_x, p_y, color="blue", linewidth=2, alpha=0.7, label="Trajectory")
    axes[0, 0].scatter(p_x[0], p_y[0], marker="o", s=100, color="green",
                       label="Start", zorder=5)
    axes[0, 0].scatter(p_x[-1], p_y[-1], marker="x", s=100, color="red",
                       label="End", zorder=5)
    axes[0, 0].scatter([goal_pos[0]], [goal_pos[1]], marker="*", s=200, color="gold",
                       label="Goal", zorder=5)
    axes[0, 0].set_xlabel("X Position (m)")
    axes[0, 0].set_ylabel("Y Position (m)")
    axes[0, 0].set_title("Drone XY Trajectory")
    axes[0, 0].legend(loc="best")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_aspect('equal')

    # Top-right: Position over time
    axes[0, 1].plot(time, p_x, label="p_x", color="blue", linewidth=2)
    axes[0, 1].plot(time, p_y, label="p_y", color="orange", linewidth=2)
    axes[0, 1].axhline(y=goal_pos[0], color="blue", linestyle="--", alpha=0.5)
    axes[0, 1].axhline(y=goal_pos[1], color="orange", linestyle="--", alpha=0.5, label="Goal y")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Position (m)")
    axes[0, 1].set_title("Position Over Time")
    axes[0, 1].legend(loc="best")
    axes[0, 1].grid(True, alpha=0.3)

    # Bottom-left: Pitch angle over time
    axes[1, 0].plot(time, np.degrees(phi), color="purple", linewidth=2)
    axes[1, 0].axhline(y=0, color="red", linestyle="--", alpha=0.5, label="Level")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Pitch Angle (deg)")
    axes[1, 0].set_title("Pitch Angle Over Time")
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.3)

    # Bottom-right: Velocities over time
    axes[1, 1].plot(time, v_x, label="v_x", color="blue", linewidth=2)
    axes[1, 1].plot(time, v_y, label="v_y", color="orange", linewidth=2)
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Velocity (m/s)")
    axes[1, 1].set_title("Velocities Over Time")
    axes[1, 1].legend(loc="best")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main(config, save_dir):
    wandb.init(
        project="planar_drone",
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

    key, action_key = jax.random.split(key)
    eval_actions = jax.random.uniform(
        action_key,
        shape=(config["eval_traj_horizon"], config["dim_action"]),
        minval=jnp.array(config["normalization_params"]["action"]["min"]),
        maxval=jnp.array(config["normalization_params"]["action"]["max"]),
    )

    key, reset_key = jax.random.split(key)
    eval_trajectory = [reset_fn(reset_key)]
    current_state = eval_trajectory[0]
    for action in eval_actions:
        next_state, _, _, _, _, _ = step_fn(current_state, 0, action)
        eval_trajectory.append(next_state)
        current_state = next_state

    eval_trajectory_data = {
        "trajectory": jnp.array(eval_trajectory),
        "actions": eval_actions,
    }

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

    initial_multi_step_loss = evaluator.compute_multi_step_loss(
        train_state.params, eval_trajectory_data
    )
    initial_one_step_loss = evaluator.compute_one_step_loss(
        train_state.params, eval_trajectory_data
    )
    wandb.log(
        {
            "eval/multi_step_loss": initial_multi_step_loss,
            "eval/one_step_loss": initial_one_step_loss,
        },
        step=0,
    )

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["reward", "position_error"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step in range(1, config["total_steps"] + 1):

        # Compute actions
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
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

            if step % config["eval_freq"] == 0:
                multi_step_loss = evaluator.compute_multi_step_loss(
                    train_state.params, eval_trajectory_data
                )
                one_step_loss = evaluator.compute_one_step_loss(
                    train_state.params, eval_trajectory_data
                )

                # Track covariance trace if available
                cov_trace = (
                    jnp.trace(train_state.covariance)
                    if train_state.covariance is not None
                    else 0.0
                )

                wandb.log(
                    {
                        "eval/multi_step_loss": multi_step_loss,
                        "eval/one_step_loss": one_step_loss,
                        "eval/cov_trace": cov_trace,
                    },
                    step=step,
                )

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
        run_name = config.get("wandb_run_name", f"drone_model_{config['seed']}")
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
    if buffer_idx > 0:
        print("\nGenerating trajectory plot...")
        fig = plot_drone_trajectory(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/drone_plot": wandb.Image(fig)}, step=config["total_steps"]
        )
        plt.close(fig)
        print("Trajectory plot logged to wandb.")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planar drone experiments.")
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
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "drone.json"
    )
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
        run_config["wandb_group"] = "planar_drone_wind"

        if args.run_name:
            run_name_base = args.run_name
        else:
            run_name_base = "drone"

        if args.num_seeds > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_seed_{seed_idx}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
