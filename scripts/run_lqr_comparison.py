# run_lqr_comparison.py
# Compares learning (EKF + info-gathering) vs perfect info (known pursuer model) trajectories

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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


def plot_metrics(states_dict, actions_dict, weight_control, save_path=None):
    """Plot distance, cumulative distance cost, and control cost over time for all cases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'Perfect Info': 'green', 'No Learning': 'gray',
              'Passive Learning': 'orange', 'Active Learning': 'blue'}

    for title in states_dict.keys():
        states = states_dict[title]
        actions = actions_dict[title]

        evader_pos = states[:, 0:2]
        pursuer_pos = states[:, 4:6]

        # Distance over time
        distances = np.sqrt(np.sum((evader_pos - pursuer_pos)**2, axis=1))

        # Cumulative distance (evader wants to maximize)
        cumulative_dist = np.cumsum(distances)

        # Control cost: weight_control * ||u||^2 at each step
        # actions has shape (T, 2) for evader control
        control_costs = weight_control * np.sum(actions**2, axis=1)
        cumulative_control = np.cumsum(control_costs)

        color = colors.get(title, 'black')
        axes[0].plot(distances, label=title, color=color, linewidth=2)
        axes[1].plot(cumulative_dist, label=title, color=color, linewidth=2)
        axes[2].plot(cumulative_control, label=title, color=color, linewidth=2)

    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Distance")
    axes[0].set_title("Distance Between Evader and Pursuer")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Cumulative Distance")
    axes[1].set_title("Cumulative Distance (higher = better evasion)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("Timestep")
    axes[2].set_ylabel("Cumulative Control Cost")
    axes[2].set_title(f"Cumulative Control Cost (weight={weight_control})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Metrics plot saved to {save_path}")

    return fig


def plot_comparison(results, save_path=None, arrow_every=20):
    """Plot all trajectories for comparison with velocity arrows."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, states) in zip(axes, results.items()):
        evader_x, evader_y = states[:, 0], states[:, 1]
        evader_vx, evader_vy = states[:, 2], states[:, 3]
        pursuer_x, pursuer_y = states[:, 4], states[:, 5]
        pursuer_vx, pursuer_vy = states[:, 6], states[:, 7]

        ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
        ax.plot(pursuer_x, pursuer_y, label="Pursuer", color="red", linewidth=2, alpha=0.8)

        # Velocity arrows every N steps (higher scale = smaller arrows)
        for i in range(0, len(states), arrow_every):
            ax.quiver(evader_x[i], evader_y[i], evader_vx[i], evader_vy[i],
                      color='darkblue', scale=50)
            ax.quiver(pursuer_x[i], pursuer_y[i], pursuer_vx[i], pursuer_vy[i],
                      color='darkred', scale=50)

        ax.scatter(evader_x[0], evader_y[0], marker="o", s=100, color="blue", zorder=5, label="Start")
        ax.scatter(evader_x[-1], evader_y[-1], marker="x", s=100, color="blue", zorder=5, label="End")
        ax.scatter(pursuer_x[0], pursuer_y[0], marker="o", s=100, color="red", zorder=5)
        ax.scatter(pursuer_x[-1], pursuer_y[-1], marker="x", s=100, color="red", zorder=5)

        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        ax.axis("equal")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Comparison plot saved to {save_path}")

    return fig


def run_learning(config, key, reset_key, initial_state):
    """Run the learning case with EKF and info-gathering cost."""
    print("\n=== Running LEARNING case (EKF + info-gathering) ===")

    # Initialize environment (deepcopy because init_env mutates config)
    config = copy.deepcopy(config)
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with initial (wrong) parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Active Learning params: q_cholesky diag = {jnp.diag(init_params['model']['q_cholesky'])}, r_cholesky diag = {jnp.diag(init_params['model']['r_cholesky'])}")

    # Initialize dynamics trainer (EKF)
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Initialize cost function (info_gathering)
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

    for step in range(1, config["total_steps"] + 1):
        # Compute actions
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
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

        # Reset if done
        if done:
            print(f"  Learning: Episode finished at step {step}")
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

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    # Extract trajectory and actions
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  Learning: Completed {buffer_idx} steps")
    return states, actions


def run_perfect_info(config, key, reset_key, initial_state):
    """Run the perfect information case with known pursuer model."""
    print("\n=== Running PERFECT INFO case (known model, evasion only) ===")

    # Modify config for perfect info
    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    # Use true parameters for dynamics
    true_q_diag = config["env_params"]["true_q_diag"]
    true_r_diag = config["env_params"]["true_r_diag"]
    config["dynamics_params"]["init_q_diag"] = true_q_diag
    config["dynamics_params"]["init_r_diag"] = true_r_diag

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with TRUE parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Perfect Info params: q_cholesky diag = {jnp.diag(init_params['model']['q_cholesky'])}, r_cholesky diag = {jnp.diag(init_params['model']['r_cholesky'])}")

    # Initialize cost function (evasion_only - no info term)
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

    for step in range(1, config["total_steps"] + 1):
        # Compute actions (no covariance needed)
        cost_params = {"dyn_params": init_params}
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

        # Reset if done
        if done:
            print(f"  Perfect info: Episode finished at step {step}")
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

    # Extract trajectory and actions
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  Perfect info: Completed {buffer_idx} steps")
    return states, actions


def run_no_learning(config, key, reset_key, initial_state):
    """Run with wrong parameters and no learning (no EKF updates)."""
    print("\n=== Running NO LEARNING case (wrong model, no updates) ===")

    # Modify config - use evasion only cost, keep wrong init params
    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with WRONG parameters (default init values)
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    # Initialize cost function (evasion_only)
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

    for step in range(1, config["total_steps"] + 1):
        # Compute actions with wrong params (no covariance needed)
        cost_params = {"dyn_params": init_params}
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

        # Reset if done
        if done:
            print(f"  No learning: Episode finished at step {step}")
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        # NO EKF updates - just use wrong params throughout

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    # Extract trajectory and actions
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  No learning: Completed {buffer_idx} steps")
    return states, actions


def run_passive_learning(config, key, reset_key, initial_state):
    """Run with EKF learning but no info-gain objective (evasion only cost)."""
    print("\n=== Running PASSIVE LEARNING case (EKF updates, evasion only cost) ===")

    # Modify config - evasion only cost, but still do EKF
    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with WRONG parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    # Initialize dynamics trainer (EKF) - will learn passively
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Initialize cost function (evasion_only - no info term)
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

    for step in range(1, config["total_steps"] + 1):
        # Compute actions using current learned params
        cost_params = {"dyn_params": train_state.params}
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

        # Reset if done
        if done:
            print(f"  Passive learning: Episode finished at step {step}")
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        # Train model (EKF update) - learning passively from whatever data we get
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    # Extract trajectory and actions
    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  Passive learning: Completed {buffer_idx} steps")
    return states, actions


def main(config, save_dir):
    """Run all cases and compare."""
    key = jax.random.key(config["seed"])

    # Get a common initial state for fair comparison
    # Note: init_env pops true_q_diag/true_r_diag from config, so we deepcopy
    reset_fn, _, _ = init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)
    print(f"Initial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"pursuer=({initial_state[4]:.2f}, {initial_state[5]:.2f})")

    results = {}  # {name: (states, actions)}

    # Use the same run_key for all cases so iCEM random sampling is comparable
    key, run_key = jax.random.split(key)

    # 1. Perfect info (oracle baseline)
    results["Perfect Info"] = run_perfect_info(config, run_key, reset_key, initial_state)

    # 2. No learning (wrong params, no updates)
    results["No Learning"] = run_no_learning(config, run_key, reset_key, initial_state)

    # 3. Passive learning (EKF but no info-gain objective)
    results["Passive Learning"] = run_passive_learning(config, run_key, reset_key, initial_state)

    # 4. Active learning (EKF + info-gain objective)
    results["Active Learning"] = run_learning(config, run_key, reset_key, initial_state)

    # Separate states and actions for different uses
    states_only = {k: v[0] for k, v in results.items()}
    actions_only = {k: v[1] for k, v in results.items()}

    # Get control weight for cost computation
    weight_control = config["cost_fn_params"]["weight_control"]

    # Save trajectories and actions
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        traj_path = os.path.join(save_dir, f"trajectories_seed_{config['seed']}.npz")
        # Save both states and actions
        save_dict = {}
        for k in results.keys():
            key = k.replace(" ", "_").lower()
            save_dict[f"{key}_states"] = results[k][0]
            save_dict[f"{key}_actions"] = results[k][1]
        np.savez(traj_path, **save_dict)
        print(f"Trajectories and actions saved to {traj_path}")
        save_path = os.path.join(save_dir, f"comparison_seed_{config['seed']}.png")
    else:
        save_path = None

    fig = plot_comparison(states_only, save_path)

    # Second plot: distance, cumulative distance, and control cost over time
    metrics_path = save_path.replace(".png", "_metrics.png") if save_path else None
    fig2 = plot_metrics(states_only, actions_only, weight_control, metrics_path)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    for title in results.keys():
        states = states_only[title]
        actions = actions_only[title]

        evader_pos = states[:, 0:2]
        pursuer_pos = states[:, 4:6]
        distances = np.sqrt(np.sum((evader_pos - pursuer_pos)**2, axis=1))
        cumulative_dist = np.sum(distances)

        # Compute control cost
        control_costs = weight_control * np.sum(actions**2, axis=1)
        cumulative_control = np.sum(control_costs)

        # Total cost: -distance + control_cost (evader minimizes this)
        total_cost = -cumulative_dist + cumulative_control

        print(f"{title}:")
        print(f"  Initial distance: {distances[0]:.2f}")
        print(f"  Final distance:   {distances[-1]:.2f}")
        print(f"  Min distance:     {distances.min():.2f} (step {distances.argmin()})")
        print(f"  Max distance:     {distances.max():.2f} (step {distances.argmax()})")
        print(f"  Mean distance:    {distances.mean():.2f}")
        print(f"  Cumulative dist:  {cumulative_dist:.2f}")
        print(f"  Control cost:     {cumulative_control:.2f}")
        print(f"  Total cost:       {total_cost:.2f} (lower is better for evader)")
        print()

    plt.show()

    print("\n=== Comparison complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare learning vs perfect info pursuit-evasion.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default="./comparison_results",
        help="Directory to save comparison plots.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "lqr.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["seed"] = args.seed
    main(config, save_dir=args.save_dir)
