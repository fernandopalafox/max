# finetune_cheetah.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import make_cheetah_env, EnvParams
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

# Import animation function from data collection script
from collect_data_cheetah import create_cheetah_xy_animation


def plot_cheetah_velocity(buffers, buffer_idx, config, target_velocity):
    """Plot forward velocity and target velocity over time."""
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    dt = config["env_params"]["dt"]
    time = np.arange(buffer_idx) * dt

    # Forward velocity is at index 8 in 17D state
    forward_vel = states[:, 8]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, forward_vel, label="Forward Velocity", color="blue", linewidth=2)
    ax.axhline(target_velocity, color="red", linestyle="--", label=f"Target ({target_velocity:.2f} m/s)")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Cheetah Forward Velocity Tracking")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_state_components(buffers, buffer_idx, config):
    """Plot joint angles and velocities over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    time = np.arange(buffer_idx) * dt

    state_labels = config.get("state_labels", [f"s{i}" for i in range(17)])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Joint angles (indices 0-7)
    ax = axes[0]
    for i in range(8):
        ax.plot(time, states[:, i], label=state_labels[i], alpha=0.8)
    ax.set_ylabel("Joint Angle (rad)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Joint Positions")

    # Velocities (indices 8-16)
    ax = axes[1]
    for i in range(8, 17):
        ax.plot(time, states[:, i], label=state_labels[i], alpha=0.8)
    ax.set_ylabel("Velocity")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocities")

    plt.tight_layout()
    return fig


def main(config):
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    # Read settings from config
    save_dir = config.get("save_dir", None)
    plot_run = config.get("plot_run", True)

    # Initialize cheetah environment
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)

    # Initialize learned dynamics model
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

    # Initialize cost function (uses learned model for rollouts)
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

    # Fixed target velocity from config
    target_velocity = config["cost_fn_params"]["target_velocity"]

    print(
        f"Starting cheetah finetuning for {config['total_steps']} steps "
        f"(target velocity: {target_velocity} m/s)"
    )

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["forward_vel"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Initial covariance tracking
    initial_cov_trace_per_param = None
    if train_state.covariance is not None:
        cov_trace = jnp.trace(train_state.covariance)
        initial_cov_trace_per_param = cov_trace / train_state.covariance.shape[0]

    # Initial evaluation before training
    eval_results = evaluator.evaluate(train_state.params)
    wandb.log(
        {
            **eval_results,
            "eval/cov_trace_delta": 0.0,
        },
        step=0,
    )

    # Track 18D states for animation (includes rootx)
    full_states_for_animation = []

    # Main training loop
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    current_obs = get_obs_fn(mjx_data).squeeze()  # 17D observation

    for step in range(1, config["total_steps"] + 1):
        # Store full 18D state for animation before stepping
        full_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        full_states_for_animation.append(np.array(full_state))

        # Compute actions using planner with learned model
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "target_velocity": target_velocity,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, current_obs, cost_params)
        action = actions[0][None, :]  # Add agent dimension

        # Step environment with MJX ground truth
        mjx_data, next_obs, rewards, terminated, truncated, info = step_fn(
            mjx_data, episode_length, action
        )
        next_obs = next_obs.squeeze()  # 17D
        done = terminated or truncated
        episode_length += 1

        # Track rewards
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += float(info[info_key])

        # Update buffer with 17D observations
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs[None, :],  # Add agent dim
            action,
            rewards,
            jnp.zeros_like(rewards),  # dummy value
            jnp.zeros_like(rewards),  # dummy log_pi
            float(done),
        )
        buffer_idx += 1

        current_obs = next_obs

        # Log step metrics
        velocity_error = (info["forward_vel"] - target_velocity) ** 2
        wandb.log({
            "step/forward_vel": float(info["forward_vel"]),
            "step/target_velocity": target_velocity,
            "step/velocity_error": float(velocity_error),
        }, step=step)

        # Reset environment if done
        if done:
            key, reset_key = jax.random.split(key)
            mjx_data = reset_fn(reset_key)
            current_obs = get_obs_fn(mjx_data).squeeze()

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

        # Evaluate model
        if step % config["eval_freq"] == 0:
            # Run rollout evaluation
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
                    {
                        **eval_results,
                        "eval/cov_trace_delta": 0.0,
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
        run_name = config.get("wandb_run_name", f"cheetah_model_{config['seed']}")
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
    if plot_run and buffer_idx > 0:
        print("\nGenerating velocity plot...")
        fig = plot_cheetah_velocity(buffers, buffer_idx, config, target_velocity)
        wandb.log({"trajectory/velocity_plot": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)
        print("Velocity plot logged to wandb.")

        print("Generating state components plot...")
        fig = plot_state_components(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/state_components": wandb.Image(fig)}, step=config["total_steps"]
        )
        plt.close(fig)
        print("State components plot logged to wandb.")

        # Generate animation from full 18D states
        if len(full_states_for_animation) > 0:
            print("Generating cheetah animation...")
            full_states_array = np.array(full_states_for_animation)
            gif_path = create_cheetah_xy_animation(
                full_states_array, config["env_params"]["dt"]
            )
            wandb.log({
                "trajectory/animation": wandb.Video(gif_path, fps=20, format="gif")
            }, step=config["total_steps"])
            print("Animation logged to wandb.")

    # Final evaluation with trajectory plot
    print("\nRunning final evaluation with trajectory plot...")
    plot_eval_config = copy.deepcopy(config)
    plot_eval_config["evaluator_type"] = "rollout_with_trajectory"
    final_evaluator = init_evaluator(plot_eval_config)
    final_eval_results = final_evaluator.evaluate(train_state.params)

    # Log metrics (excluding trajectory/actions/goal_state which are arrays)
    metrics_to_log = {k: v for k, v in final_eval_results.items()
                     if not isinstance(v, np.ndarray)}
    wandb.log(metrics_to_log, step=config["total_steps"])

    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cheetah finetuning experiments.")
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
        help="Number of seeds to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cheetah.json",
        help="Config filename in configs folder. Defaults to cheetah.json.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["finetuning"]

    run_name_base = args.run_name or "cheetah_finetune"

    # Generate seeds using JAX RNG from config seed
    base_key = jax.random.key(CONFIG["seed"])
    seed_keys = jax.random.split(base_key, args.num_seeds)
    seeds = [int(jax.random.bits(k)) for k in seed_keys]

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"--- Starting run for seed {seed_idx}/{args.num_seeds} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = seed

        # Build run name
        run_name = run_name_base
        if args.num_seeds > 1:
            run_name = f"{run_name}_{seed_idx}"
        run_config["wandb_run_name"] = run_name

        wandb.init(
            project=run_config.get("wandb_project", "cheetah_finetuning"),
            config=run_config,
            name=run_config.get("wandb_run_name"),
            reinit=True,
        )
        main(run_config)
        wandb.finish()

    print("All experiments complete.")
