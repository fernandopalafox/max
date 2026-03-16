# finetune_cheetah.py

# Enable deterministic GPU operations for debugging (set before importing JAX)
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import time

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import make_cheetah_env, EnvParams
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.samplers import init_sampler
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


def plot_cheetah_velocity(buffers, buffer_idx, config):
    """Plot forward velocity over time."""
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    dt = 0.01
    time = np.arange(buffer_idx) * dt

    # Forward velocity is at index 8 in 17D state
    forward_vel = states[:, 8]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, forward_vel, label="Forward Velocity", color="blue", linewidth=2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Cheetah Forward Velocity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_action_trajectory(buffers, buffer_idx, config):
    """Plot applied actions over time."""
    actions = np.array(buffers["actions"][0, :buffer_idx, :])  # (T, dim_action)
    dt = 0.01
    time_axis = np.arange(buffer_idx) * dt
    dim_action = actions.shape[1]

    fig, axes = plt.subplots(dim_action, 1, figsize=(12, 2 * dim_action), sharex=True)
    if dim_action == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(time_axis, actions[:, i], lw=1.5)
        ax.set_ylabel(f"a{i}")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Applied Actions", fontsize=10)
    plt.tight_layout()
    return fig


def plot_state_components(buffers, buffer_idx, config):
    """Plot joint angles and velocities over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = 0.01
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
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    # Read settings from config
    save_dir = config.get("save_dir", None)
    plot_run = config.get("plot_run", True)
    plot_eval = config.get("plot_eval", False)

    # Initialize cheetah environment
    print(f"[{time.time()-t0:.2f}s] Initializing environment...")
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)
    print(f"[{time.time()-t0:.2f}s] Environment initialized")

    # Initialize learned dynamics model
    print(f"[{time.time()-t0:.2f}s] Initializing dynamics model...")
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )
    print(f"[{time.time()-t0:.2f}s] Dynamics model initialized")

    # Initialize dynamics trainer
    print(f"[{time.time()-t0:.2f}s] Initializing trainer...")
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        config, dynamics_model, init_params, trainer_key
    )
    print(f"[{time.time()-t0:.2f}s] Trainer initialized")

    # Initialize sampler
    print(f"[{time.time()-t0:.2f}s] Initializing sampler...")
    sampler = init_sampler(config["sampler"])
    print(f"[{time.time()-t0:.2f}s] Sampler initialized")

    # Count trainable parameters
    num_params = sum(x.size for x in jax.tree_util.tree_leaves(train_state.params))
    wandb.config.update({"num_params": num_params})

    # Initialize evaluator
    print(f"[{time.time()-t0:.2f}s] Initializing evaluator...")
    evaluator = init_evaluator(config)
    print(f"[{time.time()-t0:.2f}s] Evaluator initialized")

    # Initialize cost function (uses learned model for rollouts)
    print(f"[{time.time()-t0:.2f}s] Initializing cost function...")
    cost_fn = init_cost(config, dynamics_model)
    print(f"[{time.time()-t0:.2f}s] Cost function initialized")

    # Initialize planner
    print(f"[{time.time()-t0:.2f}s] Initializing planner...")
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)
    print(f"[{time.time()-t0:.2f}s] Planner initialized")

    # Initialize buffer
    print(f"[{time.time()-t0:.2f}s] Initializing buffers...")
    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0
    print(f"[{time.time()-t0:.2f}s] Buffers initialized")

    print(f"Starting cheetah finetuning for {config['total_steps']} steps")

    episode_length = 0
    episode_total_reward = 0.0

    # Initial covariance tracking
    initial_cov_trace_per_param = None
    if train_state.covariance is not None:
        cov_trace = jnp.trace(train_state.covariance)
        initial_cov_trace_per_param = cov_trace / train_state.covariance.shape[0]

    # Initial evaluation before training
    print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    eval_results = evaluator.evaluate(train_state.params)
    print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")
    # Log only scalar metrics (ignore trajectory/actions/goal_state)
    initial_metrics = {
        k: v for k, v in eval_results.items()
        if isinstance(v, (int, float))
    }
    initial_metrics["eval/cov_trace_delta"] = 0.0
    wandb.log(initial_metrics, step=0)

    if plot_eval and "trajectory" in eval_results:
        print(f"[{time.time()-t0:.2f}s] Generating initial eval animation...")
        traj = eval_results["trajectory"]
        full_states = np.concatenate([traj.qpos, traj.qvel], axis=-1)
        gif_path = create_cheetah_xy_animation(full_states)
        wandb.log({"eval/animation": wandb.Video(gif_path, fps=20, format="gif")}, step=0)
        print(f"[{time.time()-t0:.2f}s] Initial eval animation logged.")

    # Track 18D states for animation (includes rootx)
    full_states_for_animation = []

    # Main training loop
    print(f"[{time.time()-t0:.2f}s] Starting main loop...")
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    current_obs = get_obs_fn(mjx_data).squeeze()  # 17D observation
    print(f"[{time.time()-t0:.2f}s] First reset done, entering loop")

    # Timing accumulators for loop
    t_planner = 0.0
    t_step = 0.0
    t_train = 0.0
    t_eval = 0.0
    last_report = time.time()

    total_accumulated_loss = 0.0
    for step in range(1, config["total_steps"] + 1):
        step_start = time.time()

        # Store full 18D state for animation before stepping
        full_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        full_states_for_animation.append(np.array(full_state))

        # Compute actions using planner with learned model
        _t0 = time.time()
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, current_obs, cost_params)
        action = actions[0][None, :]  # Add agent dimension
        dt_planner = time.time() - _t0
        t_planner += dt_planner

        # Step environment with MJX ground truth
        _t0 = time.time()
        mjx_data, next_obs, rewards, terminated, truncated, _ = step_fn(
            mjx_data, episode_length, action
        )
        dt_step = time.time() - _t0
        t_step += dt_step
        next_obs = next_obs.squeeze()  # 17D
        done = terminated or truncated
        episode_length += 1
        episode_total_reward += float(rewards[0])

        # Update buffer with 17D observations
        _t0 = time.time()
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
        dt_buffer = time.time() - _t0
        buffer_idx += 1

        current_obs = next_obs

        dt_wandb = 0.0

        # Reset environment if done
        dt_reset = 0.0
        if done:
            _t0 = time.time()
            key, reset_key = jax.random.split(key)
            mjx_data = reset_fn(reset_key)
            current_obs = get_obs_fn(mjx_data).squeeze()
            dt_reset = time.time() - _t0

            print(f"Episode finished at step {step}.")

            # Log and reset episode stats
            wandb.log({
                "episode/length": episode_length,
                "rewards/episode_reward": episode_total_reward,
            }, step=step)
            episode_length = 0
            episode_total_reward = 0.0

        # Train model
        dt_train = 0.0
        if step % config["train_model_freq"] == 0:
            _t0 = time.time()
            key, sample_key = jax.random.split(key)
            train_data = sampler.sample(sample_key, buffers, buffer_idx)
            if train_data is not None:
                train_state, loss = trainer.train(train_state, train_data, step=step)
                total_accumulated_loss += float(loss)
                wandb.log({
                    "train/model_loss": float(loss),
                    "train/total_accumulated_loss": total_accumulated_loss
                }, step=step)
            dt_train = time.time() - _t0
            t_train += dt_train

        # Evaluate model
        dt_eval = 0.0
        if step % config["eval_freq"] == 0:
            _t0 = time.time()
            # Run rollout evaluation (returns metrics + trajectory)
            eval_results = evaluator.evaluate(train_state.params)
            dt_eval = time.time() - _t0
            t_eval += dt_eval

            # Log only scalar metrics (ignore trajectory/actions/goal_state)
            metrics_to_log = {
                k: v for k, v in eval_results.items()
                if isinstance(v, (int, float))
            }

            # Track covariance trace if available
            if train_state.covariance is not None:
                cov_trace = jnp.trace(train_state.covariance)
                cov_trace_per_param = cov_trace / train_state.covariance.shape[0]
                cov_trace_delta = cov_trace_per_param - initial_cov_trace_per_param
                metrics_to_log["eval/cov_trace_delta"] = float(cov_trace_delta)
            else:
                metrics_to_log["eval/cov_trace_delta"] = 0.0

            wandb.log(metrics_to_log, step=step)

        step_total = time.time() - step_start
        print(f"[Step {step}] total={step_total:.3f}s | planner={dt_planner:.3f}s, env_step={dt_step:.3f}s, buffer={dt_buffer:.3f}s, wandb={dt_wandb:.3f}s, reset={dt_reset:.3f}s, train={dt_train:.3f}s, eval={dt_eval:.3f}s")

        # Handle Buffer Overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"],
                config["buffer_size"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # Final timing summary
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total planner time: {t_planner:.2f}s")
    print(f"Total step time: {t_step:.2f}s")
    print(f"Total train time: {t_train:.2f}s")
    print(f"Total eval time: {t_eval:.2f}s")
    print(f"======================\n")

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
        fig = plot_cheetah_velocity(buffers, buffer_idx, config)
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

        print("Generating action trajectory plot...")
        fig = plot_action_trajectory(buffers, buffer_idx, config)
        wandb.log({"trajectory/actions": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)
        print("Action trajectory plot logged to wandb.")

        # Generate animation from full 18D states
        if len(full_states_for_animation) > 0:
            print("Generating cheetah animation...")
            full_states_array = np.array(full_states_for_animation)
            gif_path = create_cheetah_xy_animation(full_states_array)
            wandb.log({
                "trajectory/animation": wandb.Video(gif_path, fps=20, format="gif")
            }, step=config["total_steps"])
            print("Animation logged to wandb.")

    if plot_eval:
        print(f"\n[{time.time()-t0:.2f}s] Running final evaluation...")
        final_eval_results = evaluator.evaluate(train_state.params)
        print(f"[{time.time()-t0:.2f}s] Final evaluation complete.")
        if "trajectory" in final_eval_results:
            traj = final_eval_results["trajectory"]
            full_states = np.concatenate([traj.qpos, traj.qvel], axis=-1)
            gif_path = create_cheetah_xy_animation(full_states)
            wandb.log({"eval/animation": wandb.Video(gif_path, fps=20, format="gif")}, step=config["total_steps"])
            print("Final eval animation logged to wandb.")

    print("Run complete.")
    return total_accumulated_loss


def run_sweep():
    """Entry point for wandb sweep agents. Simple single-seed run."""
    wandb.init()

    # Load base config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "cheetah.json"
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)

    run_config = copy.deepcopy(full_config["finetuning"])

    # Apply wandb.config overrides (handles dotted keys)
    for key, value in wandb.config.items():
        keys = key.split(".")
        target = run_config
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value

    main(run_config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cheetah finetuning experiments.")
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
        help="Number of seeds to run.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cheetah.json",
        help="Config filename in configs folder. Defaults to cheetah.json.",
    )
    args = parser.parse_args()

    # Check if running as wandb sweep agent
    if os.environ.get("WANDB_SWEEP_ID"):
        run_sweep()
    else:
        # Load config from JSON file
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", args.config
        )
        with open(config_path, "r") as f:
            full_config = json.load(f)
        CONFIG = full_config["finetuning"]

        run_name_base = args.run_name or "cheetah_finetune"

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

                # Build run name: base_lam{idx}_seed{idx}
                run_name = run_name_base
                if args.lambdas is not None:
                    run_name = f"{run_name}_lam{lam}"
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
