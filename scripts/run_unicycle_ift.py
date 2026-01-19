# run_unicycle_ift.py
#
# Minimal adaptation of run_lqr.py to test unicycle dynamics with
# implicit gradient-based information gathering.
#
# Key changes from run_lqr.py:
# 1. Uses unicycle_mpc dynamics (already has implicit differentiation)
# 2. Loads unicycle.json config instead of lqr.json
# 3. Adjusted plot function for unicycle state representation

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


def plot_unicycle_trajectory(buffers, buffer_idx, config):
    """Plot evader and unicycle (pursuer) x-y positions."""
    # Extract states (agent 0, valid timesteps only)
    states = np.array(buffers["states"][0, :buffer_idx, :])

    # State layout for unicycle_mpc:
    # [evader_pos_x, evader_pos_y, evader_vel_x, evader_vel_y,
    #  unicycle_pos_x, unicycle_pos_y, unicycle_heading, unicycle_speed]
    evader_x, evader_y = states[:, 0], states[:, 1]
    unicycle_x, unicycle_y = states[:, 4], states[:, 5]

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
    ax.plot(unicycle_x, unicycle_y, label="Unicycle (MPC)", color="red", linewidth=2, alpha=0.8)

    # Mark start (circle) and end (x) points
    ax.scatter(evader_x[0], evader_y[0], marker="o", s=100, color="blue",
               label="Evader Start", zorder=5)
    ax.scatter(evader_x[-1], evader_y[-1], marker="x", s=100, color="blue",
               label="Evader End", zorder=5)
    ax.scatter(unicycle_x[0], unicycle_y[0], marker="o", s=100, color="red",
               label="Unicycle Start", zorder=5)
    ax.scatter(unicycle_x[-1], unicycle_y[-1], marker="x", s=100, color="red",
               label="Unicycle End", zorder=5)

    # Draw heading arrows for unicycle at intervals
    arrow_interval = max(1, buffer_idx // 10)
    headings = states[:, 6]
    for i in range(0, buffer_idx, arrow_interval):
        dx = 0.3 * np.cos(headings[i])
        dy = 0.3 * np.sin(headings[i])
        ax.arrow(unicycle_x[i], unicycle_y[i], dx, dy,
                 head_width=0.1, head_length=0.05, fc='red', ec='red', alpha=0.5)

    # Formatting
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title("Unicycle MPC Pursuit-Evasion Trajectory")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    return fig


def main(config, save_dir):
    wandb.init(
        project="unicycle_ift",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    # Ground truth parameters for evaluation
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    fix_theta2 = config["dynamics_params"].get("fix_theta2", False)

    # Match true_params structure to learnable params
    if fix_theta2:
        true_params = {
            "model": {
                "theta1": jnp.array(true_theta1),
            },
            "normalizer": None,
        }
    else:
        true_params = {
            "model": {
                "theta1": jnp.array(true_theta1),
                "theta2": jnp.array(true_theta2),
            },
            "normalizer": None,
        }

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics (unicycle_mpc with implicit differentiation)
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    # Initialize dynamics trainer (EKF for online parameter learning)
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        config, dynamics_model, init_params, trainer_key
    )

    # Initialize dynamics evaluator
    evaluator = DynamicsEvaluator(dynamics_model.pred_one_step)

    # Generate evaluation trajectory with random actions
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

    # Initialize cost function (info_gathering uses implicit gradients via JAX autodiff)
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

    # Extract config values for logging
    weight_info = config["cost_fn_params"]["weight_info"]
    init_distance = config["env_params"].get("min_init_distance", 5.0)

    print(f"Starting unicycle MPC simulation for {config['total_steps']} steps")
    print(f"True params: theta1={true_theta1}, theta2={true_theta2}")
    if fix_theta2:
        print(f"Init params: theta1={init_params['model']['theta1']} (theta2 fixed at {true_theta2})")
    else:
        print(f"Init params: theta1={init_params['model']['theta1']}, theta2={init_params['model']['theta2']}")
    print(f"Info gathering weight: {weight_info}")
    print(f"Min initial distance: {init_distance}")

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
            "config/weight_info": weight_info,
            "config/init_distance": init_distance,
        },
        step=0,
    )

    episode_length = 0
    # unicycle_mpc env returns "distance" in info, not "reward"
    reward_component_keys_to_avg = ["distance"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step in range(1, config["total_steps"] + 1):
        # Compute actions using planner
        # cost_params includes dyn_params and params_cov_model for info gathering
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # add agent dim

        # Log iCEM convergence info (if available)
        if planner_state.iter_costs is not None and step % config["eval_freq"] == 0:
            iter_costs = planner_state.iter_costs
            wandb.log(
                {
                    "icem/final_cost": float(planner_state.final_cost),
                    "icem/iter0_cost": float(iter_costs[0]),
                    "icem/iter_last_cost": float(iter_costs[-1]),
                    "icem/cost_improvement": float(iter_costs[0] - iter_costs[-1]),
                },
                step=step,
            )

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
            jnp.zeros_like(rewards),
            jnp.zeros_like(rewards),
            float(done),
        )
        buffer_idx += 1

        current_obs = next_obs

        # Reset environment if done
        if done:
            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)
            print(f"Episode finished at step {step}.")

            episode_log = {"episode/length": episode_length}
            if episode_length > 0:
                for info_key in reward_component_keys_to_avg:
                    avg_val = episode_reward_components[info_key] / episode_length
                    episode_log[f"rewards/{info_key}"] = float(avg_val)
            wandb.log(episode_log, step=step)
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train model (EKF update with implicit gradients from dynamics)
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

                # Compute parameter difference from true dynamics
                diff_tree = jax.tree.map(lambda x, y: x - y, train_state.params, true_params)
                param_diff = sum(jnp.linalg.norm(leaf) for leaf in jax.tree_util.tree_leaves(diff_tree))
                cov_trace = jnp.trace(train_state.covariance) if train_state.covariance is not None else 0.0

                # Log current parameter estimates
                current_theta1 = float(train_state.params["model"]["theta1"])
                # theta2 may be fixed (not in params)
                fix_theta2 = config["dynamics_params"].get("fix_theta2", False)
                if fix_theta2:
                    current_theta2 = true_theta2  # fixed at true value
                else:
                    current_theta2 = float(train_state.params["model"]["theta2"])

                wandb.log(
                    {
                        "eval/multi_step_loss": multi_step_loss,
                        "eval/one_step_loss": one_step_loss,
                        "eval/param_diff": param_diff,
                        "eval/cov_trace": cov_trace,
                        "params/theta1": current_theta1,
                        "params/theta2": current_theta2,
                        "params/theta1_error": abs(current_theta1 - true_theta1),
                        "params/theta2_error": abs(current_theta2 - true_theta2),
                        "config/weight_info": weight_info,
                        "config/init_distance": init_distance,
                        "config/fix_theta2": fix_theta2,
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
        run_name = config.get("wandb_run_name", f"unicycle_model_{config['seed']}")
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
    parser = argparse.ArgumentParser(description="Run unicycle MPC with IFT experiments.")
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
        help="A seed to generate the run seeds (used with --num-seeds).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Specific random seed to use (overrides --meta-seed and --num-seeds).",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models",
        help="Directory to save the learned dynamics model parameters.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default=None,
        choices=["random", "icem"],
        help="Override planner type (default uses config).",
    )
    parser.add_argument(
        "--weight-info",
        type=float,
        default=None,
        help="Override information gathering weight (default: 5000.0 from config).",
    )
    parser.add_argument(
        "--init-distance",
        type=float,
        default=None,
        help="Override minimum initial distance between agents (default: 5.0 from config).",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=None,
        help="Override total simulation steps (default: 150 from config).",
    )
    parser.add_argument(
        "--pursuer-heading",
        type=float,
        default=None,
        help="Fixed initial pursuer heading in radians (default: random). E.g., 0=right, 1.57=up, 3.14=left",
    )
    parser.add_argument(
        "--fix-theta2",
        action="store_true",
        help="Fix theta2 at true value and only learn theta1.",
    )
    parser.add_argument(
        "--mpc-horizon",
        type=int,
        default=None,
        help="Pursuer MPC planning horizon T (default: 4 from config).",
    )
    args = parser.parse_args()

    # Load unicycle config (not lqr.json)
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "unicycle.json")
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

    # Determine seeds to run
    if args.seed is not None:
        # Single specific seed overrides everything
        seeds = [args.seed]
    elif args.meta_seed is not None:
        rng = np.random.default_rng(args.meta_seed)
        seeds = rng.integers(low=0, high=2**32 - 1, size=args.num_seeds)
    else:
        seeds = range(args.num_seeds)

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Starting run for seed #{seed} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = int(seed)
        run_config["wandb_group"] = "unicycle_ift"

        # Override planner if specified
        if args.planner:
            run_config["planner_type"] = args.planner

        # Override weight_info if specified
        if args.weight_info is not None:
            run_config["cost_fn_params"]["weight_info"] = args.weight_info

        # Override init_distance if specified
        if args.init_distance is not None:
            run_config["env_params"]["min_init_distance"] = args.init_distance

        # Override total_steps if specified
        if args.total_steps is not None:
            run_config["total_steps"] = args.total_steps

        # Override pursuer heading if specified
        if args.pursuer_heading is not None:
            run_config["env_params"]["init_pursuer_heading"] = args.pursuer_heading

        # Fix theta2 if specified
        if args.fix_theta2:
            run_config["dynamics_params"]["fix_theta2"] = True

        # Override MPC horizon if specified
        if args.mpc_horizon is not None:
            run_config["dynamics_params"]["mpc_horizon"] = args.mpc_horizon

        # Build run name with planner type label, weight_info, and init_distance
        planner_type = run_config["planner_type"]
        planner_label = "random_actions" if planner_type == "random" else planner_type
        weight_info = run_config["cost_fn_params"]["weight_info"]
        init_dist = run_config["env_params"].get("min_init_distance", 5.0)

        if args.run_name:
            run_name_base = f"{args.run_name}_{planner_label}_w{weight_info}_d{init_dist}"
        else:
            run_name_base = f"unicycle_ift_{planner_label}_w{weight_info}_d{init_dist}"

        if args.num_seeds > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_seed_{seed_idx}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
