# run_gridworld.py

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


def plot_gridworld_trajectory(buffers, buffer_idx, maze_layout, goal_state, config):
    """Plot agent trajectory on the maze grid."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    maze_arr = np.array(maze_layout)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw maze walls and corridors (black = wall, white = corridor)
    for y in range(10):
        for x in range(10):
            bitmask = maze_arr[y, x]
            if bitmask == 0:
                # Wall cell - black
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                # Corridor cell - white
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray', linewidth=0.5))

    # Plot trajectory
    pos_x, pos_y = states[:, 0], states[:, 1]
    ax.plot(pos_x, pos_y, 'b-', alpha=0.6, linewidth=2, label='Path')
    ax.scatter(pos_x[0], pos_y[0], marker='o', s=150, color='green', label='Start', zorder=5)
    ax.scatter(pos_x[-1], pos_y[-1], marker='x', s=150, color='blue', label='End', zorder=5)
    ax.scatter(goal_state[0], goal_state[1], marker='*', s=250, color='red', label='Goal', zorder=5)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Gridworld Trajectory')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def plot_position_over_time(buffers, buffer_idx, config):
    """Plot x and y positions over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    time = np.arange(buffer_idx)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    # X position
    ax = axes[0]
    ax.plot(time, states[:, 0], label="X position", color='blue')
    ax.axhline(0, color='r', linestyle='--', alpha=0.3)
    ax.axhline(9, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel("X")
    ax.set_ylim(-0.5, 9.5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Y position
    ax = axes[1]
    ax.plot(time, states[:, 1], label="Y position", color='green')
    ax.axhline(0, color='r', linestyle='--', alpha=0.3)
    ax.axhline(9, color='r', linestyle='--', alpha=0.3)
    ax.set_ylabel("Y")
    ax.set_xlabel("Step")
    ax.set_ylim(-0.5, 9.5)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_eval_trajectory(eval_results, maze_layout, config):
    """Plot trajectory from rollout evaluator results."""
    trajectory = eval_results["trajectory"]
    goal = eval_results["goal_state"]
    maze_arr = np.array(maze_layout)

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw maze (black = wall, white = corridor)
    for y in range(10):
        for x in range(10):
            bitmask = maze_arr[y, x]
            if bitmask == 0:
                # Wall - black
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                # Corridor - white
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray', linewidth=0.5))

    # Plot trajectory
    pos_x, pos_y = trajectory[:, 0], trajectory[:, 1]
    ax.plot(pos_x, pos_y, 'b-', alpha=0.6, linewidth=2, label='Path')
    ax.scatter(pos_x[0], pos_y[0], marker='o', s=150, color='green', label='Start', zorder=5)
    ax.scatter(pos_x[-1], pos_y[-1], marker='x', s=150, color='blue', label='End', zorder=5)
    ax.scatter(goal[0], goal[1], marker='*', s=250, color='red', label='Goal', zorder=5)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title(f'Eval Rollout Trajectory ({len(trajectory)} steps)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def main(config, save_dir, plot_eval=False):
    wandb.init(
        project=config.get("wandb_project", "gridworld"),
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
        f"Starting gridworld simulation for {config['total_steps']} steps "
    )

    episode_length = 0

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)
    goal_state = np.array(config["cost_fn_params"]["goal_state"])

    for step in range(1, config["total_steps"] + 1):

        # Compute actions using iCEM planner
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "goal_state": goal_state,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)

        # Round action to discrete {0, 1, 2, 3} for gridworld
        action_continuous = actions[0][None, :]  # Add agent dim
        action_discrete = jnp.round(jnp.clip(action_continuous, 0.0, 3.0))

        # Step environment with discrete action
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action_discrete
        )
        done = terminated or truncated
        episode_length += 1

        # Update buffer (store the discrete action that was actually taken)
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs,
            action_discrete,
            rewards,
            jnp.zeros_like(rewards),
            jnp.zeros_like(rewards),
            float(done),
        )
        buffer_idx += 1

        current_obs = next_obs

        # Reset environment if done
        if done:
            key, reset_key = jax.random.split(key)
            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)

            dist_to_goal = float(jnp.linalg.norm(state - goal_state))
            print(f"Episode finished at step {step}, length={episode_length}, dist_to_goal={dist_to_goal:.2f}")

            wandb.log({
                "episode/length": episode_length,
                "episode/dist_to_goal": dist_to_goal,
            }, step=step)

            episode_length = 0

        # Train policy (unused for iCEM)
        if step % config["train_policy_freq"] == 0:
            pass

        # Evaluate model
        if step % config["eval_freq"] == 0:
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
            print(f"Buffer full at step {step}, resetting buffer")
            buffer_idx = 0

    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval_results = evaluator.evaluate(train_state.params)
    wandb.log({f"final/{k}": v for k, v in final_eval_results.items()})

    # Save model parameters
    os.makedirs(save_dir, exist_ok=True)
    params_np = jax.device_get(train_state.params)
    params_path = os.path.join(save_dir, "dynamics_params.pkl")
    with open(params_path, "wb") as f:
        pickle.dump(params_np, f)
    print(f"Saved dynamics params to {params_path}")

    # Save covariance if exists
    if train_state.covariance is not None:
        cov_np = jax.device_get(train_state.covariance)
        cov_path = os.path.join(save_dir, "param_covariance.pkl")
        with open(cov_path, "wb") as f:
            pickle.dump(cov_np, f)
        print(f"Saved param covariance to {cov_path}")

    # Plot trajectory on maze
    if plot_eval and "trajectory" in final_eval_results:
        fig1 = plot_eval_trajectory(
            final_eval_results,
            config["env_params"]["maze_layout"],
            config
        )
        wandb.log({"final/eval_trajectory": wandb.Image(fig1)})
        plt.close(fig1)

    # Plot training trajectory
    if buffer_idx > 0:
        fig2 = plot_gridworld_trajectory(
            buffers,
            buffer_idx,
            config["env_params"]["maze_layout"],
            goal_state,
            config
        )
        wandb.log({"final/training_trajectory": wandb.Image(fig2)})
        plt.close(fig2)

        fig3 = plot_position_over_time(buffers, buffer_idx, config)
        wandb.log({"final/position_over_time": wandb.Image(fig3)})
        plt.close(fig3)

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gridworld finetuning experiment")
    parser.add_argument("--config", type=str, default="gridworld.json")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--save-dir", type=str, default="./trained_models/gridworld")
    parser.add_argument("--plot-eval", action="store_true")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Use finetuning section
    base_config = full_config["finetuning"]

    # Run experiments for each lambda and seed
    for lambda_val in args.lambdas:
        for seed_idx in range(args.num_seeds):
            config = copy.deepcopy(base_config)

            # Set info weight
            config["cost_fn_params"]["weight_info"] = lambda_val

            # Set seed
            if args.seed is not None:
                config["seed"] = args.seed + seed_idx
            else:
                config["seed"] = config.get("seed", 123) + seed_idx

            # Set wandb config
            if args.wandb_group:
                config["wandb_group"] = args.wandb_group
            if args.wandb_project:
                config["wandb_project"] = args.wandb_project

            run_name = f"lambda{lambda_val}_seed{config['seed']}"
            config["wandb_run_name"] = run_name

            # Save directory per experiment
            exp_save_dir = os.path.join(args.save_dir, run_name)

            print(f"\n{'='*60}")
            print(f"Running: lambda={lambda_val}, seed={config['seed']}")
            print(f"{'='*60}\n")

            main(config, exp_save_dir, plot_eval=args.plot_eval)
