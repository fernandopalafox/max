# run_unicycle.py

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
import time  # TIMING

def plot_unicycle_trajectory(buffers, buffer_idx, config):
    """Plot evader and pursuer x-y positions from unicycle buffer.

    State layout for unicycle:
    - evader: pos_x (0), pos_y (1), vel_x (2), vel_y (3)
    - pursuer: pos_x (4), pos_y (5), speed (6), angle (7)
    """
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    # Extract positions
    evader_x, evader_y = states[:, 0], states[:, 1]
    pursuer_x, pursuer_y = states[:, 4], states[:, 5]
    # Extract pursuer heading (unicycle-specific)
    pursuer_angle = states[:, 7]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
    ax.plot(pursuer_x, pursuer_y, label="Pursuer", color="red", linewidth=2, alpha=0.8)

    # Mark start (circle) and end (x) points
    ax.scatter(evader_x[0], evader_y[0], marker="o", s=100, color="blue",
               label="Evader Start", zorder=5)
    ax.scatter(evader_x[-1], evader_y[-1], marker="x", s=100, color="blue",
               label="Evader End", zorder=5)
    ax.scatter(pursuer_x[0], pursuer_y[0], marker="o", s=100, color="red",
               label="Pursuer Start", zorder=5)
    ax.scatter(pursuer_x[-1], pursuer_y[-1], marker="x", s=100, color="red",
               label="Pursuer End", zorder=5)

    # Draw heading arrows for pursuer at intervals (unicycle visualization)
    arrow_interval = max(1, len(pursuer_x) // 15)
    arrow_length = 0.3
    for i in range(0, len(pursuer_x), arrow_interval):
        dx = arrow_length * np.cos(pursuer_angle[i])
        dy = arrow_length * np.sin(pursuer_angle[i])
        ax.arrow(pursuer_x[i], pursuer_y[i], dx, dy,
                 head_width=0.1, head_length=0.05, fc="red", ec="red", alpha=0.5)

    # Formatting
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Unicycle Pursuit-Evasion Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    return fig


def main(config, save_dir):
    wandb.init(
        project="unicycle",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    # Ground truth parameters for evaluation (extract before init_env modifies config)
    true_tracking_weight = config["env_params"]["true_tracking_weight"]
    true_params = {
        "model": {
            "tracking_weight": true_tracking_weight,
        },
        "normalizer": None,
    }

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
    reward_component_keys_to_avg = ["reward"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    # TIMING
    t_loop_start = time.perf_counter()

    for step in range(1, config["total_steps"] + 1):


        # Compute actions
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        _t0 = time.perf_counter()  # TIMING
        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # hacky add agent dim

        print(state[0:4], action)

        # TODO: REMOVE THIS
        _, mpc_info = dynamics_model.pred_one_step_with_info(
            train_state.params, state, action
        )
        mpc_grad_norm = float(mpc_info["mpc_grad_norm"])
        mpc_cost = float(mpc_info["mpc_cost"])
        _t_plan = time.perf_counter() - _t0  # TIMING

        # Step environment
        _t0 = time.perf_counter()  # TIMING
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        _t_step = time.perf_counter() - _t0  # TIMING
        done = terminated or truncated
        episode_length += 1
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]

        # Update buffer
        _t0 = time.perf_counter()  # TIMING
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
        _t_buf = time.perf_counter() - _t0  # TIMING
        buffer_idx += 1

        _t_train, _t_eval = 0., 0.  # TIMING

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
            _t0 = time.perf_counter()  # TIMING
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)
            _t_train = time.perf_counter() - _t0  # TIMING
            wandb.log({"train/model_loss": float(loss)}, step=step)

            if step % config["eval_freq"] == 0:
                _t0 = time.perf_counter()  # TIMING
                multi_step_loss = evaluator.compute_multi_step_loss(
                    train_state.params, eval_trajectory_data
                )
                one_step_loss = evaluator.compute_one_step_loss(
                    train_state.params, eval_trajectory_data
                )

                # Compute parameter difference from true dynamics
                diff_tree = jax.tree.map(lambda x, y: x - y, train_state.params, true_params)
                param_diff = sum(jnp.linalg.norm(leaf) for leaf in jax.tree_util.tree_leaves(diff_tree))
                cov_trace = jnp.trace(train_state.covariance) if train_state.covariance is not None else 0.0

                # Compute MPC gradient norm (monitors MPC convergence)
                eval_log = {
                    "eval/multi_step_loss": multi_step_loss,
                    "eval/one_step_loss": one_step_loss,
                    "eval/param_diff": param_diff,
                    "eval/cov_trace": cov_trace,
                    "eval/mpc_grad_norm": mpc_grad_norm,
                    "eval/mpc_cost": mpc_cost,
                }
                _t_eval = time.perf_counter() - _t0  # TIMING

                wandb.log(eval_log, step=step)

        print(f"[{step}] plan={_t_plan:.3f} step={_t_step:.3f} buf={_t_buf:.3f} train={_t_train:.3f} eval={_t_eval:.3f}")  # TIMING

        # Handle Buffer Overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"],
                config["buffer_size"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # TIMING - print summary
    print(f"\n[TIMING] total={time.perf_counter() - t_loop_start:.1f}s")

    # Save model parameters
    if save_dir:
        run_name = config.get("wandb_run_name", f"lqr_model_{config['seed']}")
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
        fig = plot_unicycle_trajectory(buffers, buffer_idx, config)
        wandb.log({"trajectory/xy_plot": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)
        print("Trajectory plot logged to wandb.")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run unicycle control experiments.")
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
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "unicycle.json")
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
        run_config["wandb_group"] = "unicycle"

        if args.run_name:
            run_name_base = args.run_name
        else:
            run_name_base = "unicycle"

        if args.num_seeds > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_seed_{seed_idx}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
