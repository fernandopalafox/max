# run_unicycle_wandb.py
# Runs experiments with wandb logging, sweeping over lambda (weight_info) values
# Produces plots for: covariance trace, parameter estimation error

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.planners import init_planner
from max.costs import init_cost
import argparse
import copy
import os
import json


def plot_trajectory(states, title="Trajectory", arrow_every=10):
    """Plot a single trajectory with heading arrows."""
    fig, ax = plt.subplots(figsize=(8, 8))

    evader_x, evader_y = states[:, 0], states[:, 1]
    evader_vx, evader_vy = states[:, 2], states[:, 3]
    unicycle_x, unicycle_y = states[:, 4], states[:, 5]
    unicycle_alpha = states[:, 6]
    unicycle_v = states[:, 7]

    ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
    ax.plot(unicycle_x, unicycle_y, label="Pursuer", color="red", linewidth=2, alpha=0.8)

    # Velocity arrows for evader
    for i in range(0, len(states), arrow_every):
        ax.quiver(evader_x[i], evader_y[i], evader_vx[i], evader_vy[i],
                  color='darkblue', scale=30, width=0.005)

    # Heading arrows for unicycle
    for i in range(0, len(states), arrow_every):
        dx = unicycle_v[i] * np.cos(unicycle_alpha[i]) * 0.3
        dy = unicycle_v[i] * np.sin(unicycle_alpha[i]) * 0.3
        ax.arrow(unicycle_x[i], unicycle_y[i], dx, dy,
                head_width=0.15, head_length=0.08, fc='darkred', ec='darkred')

    ax.scatter(evader_x[0], evader_y[0], marker="o", s=100, color="blue", zorder=5, label="E Start")
    ax.scatter(evader_x[-1], evader_y[-1], marker="x", s=100, color="blue", zorder=5, label="E End")
    ax.scatter(unicycle_x[0], unicycle_y[0], marker="o", s=100, color="red", zorder=5, label="P Start")
    ax.scatter(unicycle_x[-1], unicycle_y[-1], marker="x", s=100, color="red", zorder=5, label="P End")

    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.axis("equal")
    plt.tight_layout()

    return fig


def run_random(config, key, initial_state):
    """Run with random actions (no planning) - baseline."""
    config = copy.deepcopy(config)
    weight_control = config["cost_fn_params"]["weight_control"]

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize buffer
    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    # Get action bounds from config
    action_low = jnp.array(config["normalization_params"]["action"]["min"])
    action_high = jnp.array(config["normalization_params"]["action"]["max"])

    # Run simulation with random actions
    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  random"):
        # Sample random action
        key, action_key = jax.random.split(key)
        action = jax.random.uniform(action_key, shape=(1, config["dim_action"]),
                                    minval=action_low, maxval=action_high)

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        # Update buffer
        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        # No param learning for random - just log placeholder
        wandb.log({
            "eval/cov_trace": 0.0,
            "eval/param_diff": 0.0,
        }, step=step)

        # Reset if done
        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    # Extract final trajectory
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    return states, actions


def run_with_lambda(config, key, initial_state, weight_info, case_name):
    """
    Run with a specific lambda (weight_info) value.
    """
    config = copy.deepcopy(config)

    # Save true params before init_env pops them
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    true_params = jnp.array([true_theta1, true_theta2])

    # Set the weight_info (lambda)
    config["cost_fn_params"]["weight_info"] = weight_info

    # Set cost type based on lambda
    if weight_info > 0:
        config["cost_type"] = "info_gathering"
    else:
        config["cost_type"] = "evasion_only"

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    # Initialize trainer (EKF) - always learn
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)
    current_params = train_state.params
    current_cov = train_state.covariance

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

    # Run simulation
    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc=f"  {case_name}"):
        # Compute actions
        if weight_info > 0 and current_cov is not None:
            cost_params = {
                "dyn_params": current_params,
                "params_cov_model": current_cov,
            }
        else:
            cost_params = {"dyn_params": current_params}

        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        # Update buffer
        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        # Compute parameter estimation error (L2 norm)
        current_theta = jnp.array([
            current_params['model']['theta1'],
            current_params['model']['theta2']
        ])
        param_diff = float(jnp.linalg.norm(current_theta - true_params))

        # Compute covariance trace
        cov_trace = float(jnp.trace(current_cov)) if current_cov is not None else 0.0

        # Log metrics (matching run_lqr.py style)
        wandb.log({
            "eval/cov_trace": cov_trace,
            "eval/param_diff": param_diff,
        }, step=step)

        # Reset if done
        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        # Train model (EKF update)
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)
            current_params = train_state.params
            current_cov = train_state.covariance

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    # Extract final trajectory
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])

    # Log final trajectory plot
    fig = plot_trajectory(states, title=case_name)
    wandb.log({"trajectory/xy_plot": wandb.Image(fig)}, step=config["total_steps"])
    plt.close(fig)

    return states, actions


def main(config, args):
    """Run experiments with different lambda values."""

    # Define lambda values to test
    if args.lambdas:
        lambda_values = [float(x) for x in args.lambdas.split(",")]
    else:
        lambda_values = [0.0, 10.0, 100.0]

    key = jax.random.key(config["seed"])

    # Get a common initial state for fair comparison
    reset_fn, _, _ = init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)

    print(f"Initial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"pursuer=({initial_state[4]:.2f}, {initial_state[5]:.2f})")

    # Use the same run_key for all cases
    key, run_key = jax.random.split(key)

    # Run random baseline if requested
    if args.include_random:
        run_name = "random"
        if args.num_seeds > 1:
            run_name = f"{run_name}_seed_{config['seed']}"

        wandb.init(
            project=args.project,
            config={**config, "λ": "random"},
            group=args.group or "λ_sweep",
            name=run_name,
            reinit=True,
        )

        print(f"\n=== Running random ===")
        run_random(config, run_key, initial_state)
        wandb.finish()

    # Run each lambda value
    for weight_info in lambda_values:
        # Format lambda value nicely (no decimal for integers)
        if weight_info == int(weight_info):
            val_str = str(int(weight_info))
        else:
            val_str = str(weight_info)

        case_name = f"λ={val_str}"

        run_name = case_name
        if args.num_seeds > 1:
            run_name = f"{run_name}_seed_{config['seed']}"

        wandb.init(
            project=args.project,
            config={**config, "λ": weight_info},
            group=args.group or "λ_sweep",
            name=run_name,
            reinit=True,
        )

        print(f"\n=== Running {case_name} ===")
        run_with_lambda(config, run_key, initial_state, weight_info, case_name)
        wandb.finish()

    print("\nAll experiments complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiments with lambda sweep.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--meta-seed", type=int, default=42, help="Seed to generate run seeds")
    parser.add_argument("--config", type=str, default="unicycle",
                        help="Config name (without .json)")
    parser.add_argument("--project", type=str, default="info-gathering",
                        help="Wandb project name")
    parser.add_argument("--group", type=str, default=None,
                        help="Wandb group name")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override total_steps from config")
    parser.add_argument("--lambdas", type=str, default=None,
                        help="Comma-separated list of lambda values (e.g., '0,1,10,50')")
    parser.add_argument("--include-random", action="store_true",
                        help="Include random baseline (no planning)")
    args = parser.parse_args()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", f"{args.config}.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Using config: {args.config}")

    if args.steps:
        config["total_steps"] = args.steps
        config["buffer_size"] = args.steps + 10

    # Generate seeds
    if args.num_seeds > 1:
        rng = np.random.default_rng(args.meta_seed)
        seeds = rng.integers(low=0, high=2**32 - 1, size=args.num_seeds)
    else:
        seeds = [args.seed]

    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}")
        print(f"SEED {seed_idx + 1}/{len(seeds)}: {seed}")
        print(f"{'='*50}")

        run_config = copy.deepcopy(config)
        run_config["seed"] = int(seed)

        main(run_config, args)

    print("\nDone.")
