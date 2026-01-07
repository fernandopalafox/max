# run_unicycle_comparison.py
# Compares learning (EKF + info-gathering) vs perfect info trajectories for unicycle MPC dynamics

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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


def plot_metrics(states_dict, actions_dict, weight_control, save_path=None):
    """Plot distance, cumulative distance cost, and control cost over time for all cases."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'Perfect Info': 'green', 'No Learning': 'gray',
              'Passive Learning': 'orange', 'Active Learning': 'blue'}

    for title in states_dict.keys():
        states = states_dict[title]
        actions = actions_dict[title]

        evader_pos = states[:, 0:2]
        unicycle_pos = states[:, 4:6]

        # Distance over time
        distances = np.sqrt(np.sum((evader_pos - unicycle_pos)**2, axis=1))

        # Cumulative distance (evader wants to maximize)
        cumulative_dist = np.cumsum(distances)

        # Control cost: weight_control * ||u||^2 at each step
        control_costs = weight_control * np.sum(actions**2, axis=1)
        cumulative_control = np.cumsum(control_costs)

        color = colors.get(title, 'black')
        axes[0].plot(distances, label=title, color=color, linewidth=2)
        axes[1].plot(cumulative_dist, label=title, color=color, linewidth=2)
        axes[2].plot(cumulative_control, label=title, color=color, linewidth=2)

    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Distance")
    axes[0].set_title("Distance Between Evader and Unicycle")
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


def plot_learning_curves(learning_histories, config, save_path=None):
    """Plot parameter estimates and uncertainty over time for learning cases."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    colors = {'Passive Learning': 'orange', 'Active Learning': 'blue'}

    for name, (theta1_hist, theta2_hist, cov_hist) in learning_histories.items():
        color = colors.get(name, 'black')
        steps = np.arange(len(theta1_hist))

        axes[0].plot(steps, theta1_hist, label=name, color=color, linewidth=2)
        axes[1].plot(steps, theta2_hist, label=name, color=color, linewidth=2)
        axes[2].plot(steps, cov_hist, label=name, color=color, linewidth=2)

    # Add true value lines
    axes[0].axhline(true_theta1, color='green', linestyle='--', label=f'True ({true_theta1})')
    axes[1].axhline(true_theta2, color='green', linestyle='--', label=f'True ({true_theta2})')

    axes[0].set_xlabel("EKF Update Step")
    axes[0].set_ylabel("theta1 estimate")
    axes[0].set_title("Theta1 Convergence (position cost weight)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("EKF Update Step")
    axes[1].set_ylabel("theta2 estimate")
    axes[1].set_title("Theta2 Convergence (accel scaling)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_xlabel("EKF Update Step")
    axes[2].set_ylabel("Covariance Trace")
    axes[2].set_title("Total Uncertainty (trace of covariance)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Learning curves saved to {save_path}")

    return fig


def plot_comparison(results, save_path=None, arrow_every=10):
    """Plot all trajectories for comparison with heading arrows for unicycle."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (title, states) in zip(axes, results.items()):
        # Evader: double integrator with position and velocity
        evader_x, evader_y = states[:, 0], states[:, 1]
        evader_vx, evader_vy = states[:, 2], states[:, 3]

        # Unicycle: position, heading, speed
        unicycle_x, unicycle_y = states[:, 4], states[:, 5]
        unicycle_alpha = states[:, 6]  # heading angle
        unicycle_v = states[:, 7]      # speed

        ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
        ax.plot(unicycle_x, unicycle_y, label="Unicycle", color="red", linewidth=2, alpha=0.8)

        # Velocity arrows for evader every N steps
        for i in range(0, len(states), arrow_every):
            ax.quiver(evader_x[i], evader_y[i], evader_vx[i], evader_vy[i],
                      color='darkblue', scale=30, width=0.005)

        # Heading arrows for unicycle every N steps
        for i in range(0, len(states), arrow_every):
            dx = unicycle_v[i] * np.cos(unicycle_alpha[i]) * 0.3
            dy = unicycle_v[i] * np.sin(unicycle_alpha[i]) * 0.3
            ax.arrow(unicycle_x[i], unicycle_y[i], dx, dy,
                    head_width=0.15, head_length=0.08, fc='darkred', ec='darkred')

        ax.scatter(evader_x[0], evader_y[0], marker="o", s=100, color="blue", zorder=5, label="E Start")
        ax.scatter(evader_x[-1], evader_y[-1], marker="x", s=100, color="blue", zorder=5, label="E End")
        ax.scatter(unicycle_x[0], unicycle_y[0], marker="o", s=100, color="red", zorder=5, label="U Start")
        ax.scatter(unicycle_x[-1], unicycle_y[-1], marker="x", s=100, color="red", zorder=5, label="U End")

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

    # Save true params before init_env pops them
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with initial (wrong) parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Active Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    # Track parameter estimates and uncertainty over time
    theta1_history = [float(init_params['model']['theta1'])]
    theta2_history = [float(init_params['model']['theta2'])]

    # Initialize dynamics trainer (EKF)
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Track covariance trace (total uncertainty)
    cov_trace_history = [float(jnp.trace(train_state.covariance))]

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

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Active Learning"):
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

            # Log parameter estimates and uncertainty
            theta1_history.append(float(train_state.params['model']['theta1']))
            theta2_history.append(float(train_state.params['model']['theta2']))
            cov_trace_history.append(float(jnp.trace(train_state.covariance)))

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
    print(f"  Final params: theta1 = {train_state.params['model']['theta1']:.4f}, theta2 = {train_state.params['model']['theta2']:.4f}")

    # Print parameter convergence summary
    print(f"  Theta1 history: {theta1_history[0]:.3f} -> {theta1_history[-1]:.3f} (true: {true_theta1})")
    print(f"  Theta2 history: {theta2_history[0]:.3f} -> {theta2_history[-1]:.3f} (true: {true_theta2})")
    print(f"  Cov trace: {cov_trace_history[0]:.4f} -> {cov_trace_history[-1]:.4f}")

    return states, actions, theta1_history, theta2_history, cov_trace_history


def run_perfect_info(config, key, reset_key, initial_state):
    """Run the perfect information case with known unicycle model."""
    print("\n=== Running PERFECT INFO case (known model, evasion only) ===")

    # Modify config for perfect info
    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    # Use true parameters for dynamics
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    config["dynamics_params"]["init_theta1"] = true_theta1
    config["dynamics_params"]["init_theta2"] = true_theta2

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with TRUE parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Perfect Info params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

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

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Perfect Info"):
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
    print(f"  DEBUG No Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

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

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  No Learning"):
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
    return states, actions


def run_passive_learning(config, key, reset_key, initial_state):
    """Run with EKF learning but no info-gain objective (evasion only cost)."""
    print("\n=== Running PASSIVE LEARNING case (EKF updates, evasion only cost) ===")

    # Modify config - evasion only cost, but still do EKF
    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    # Save true params before init_env pops them
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics with WRONG parameters
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Passive Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    # Track parameter estimates and uncertainty over time
    theta1_history = [float(init_params['model']['theta1'])]
    theta2_history = [float(init_params['model']['theta2'])]

    # Initialize dynamics trainer (EKF) - will learn passively
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Track covariance trace (total uncertainty)
    cov_trace_history = [float(jnp.trace(train_state.covariance))]

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

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Passive Learning"):
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

            # Log parameter estimates and uncertainty
            theta1_history.append(float(train_state.params['model']['theta1']))
            theta2_history.append(float(train_state.params['model']['theta2']))
            cov_trace_history.append(float(jnp.trace(train_state.covariance)))

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
    print(f"  Final params: theta1 = {train_state.params['model']['theta1']:.4f}, theta2 = {train_state.params['model']['theta2']:.4f}")

    # Print parameter convergence summary
    print(f"  Theta1 history: {theta1_history[0]:.3f} -> {theta1_history[-1]:.3f} (true: {true_theta1})")
    print(f"  Theta2 history: {theta2_history[0]:.3f} -> {theta2_history[-1]:.3f} (true: {true_theta2})")
    print(f"  Cov trace: {cov_trace_history[0]:.4f} -> {cov_trace_history[-1]:.4f}")

    return states, actions, theta1_history, theta2_history, cov_trace_history


def main(config, save_dir):
    """Run all cases and compare."""
    key = jax.random.key(config["seed"])

    # Get a common initial state for fair comparison
    # Note: init_env pops true_theta1/true_theta2 from config, so we deepcopy
    reset_fn, _, _ = init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)
    print(f"Initial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"unicycle=({initial_state[4]:.2f}, {initial_state[5]:.2f}), "
          f"heading={np.degrees(initial_state[6]):.1f}deg, speed={initial_state[7]:.2f}")

    results = {}  # {name: (states, actions)}
    learning_histories = {}  # {name: (theta1_hist, theta2_hist, cov_hist)}

    # Use the same run_key for all cases so iCEM random sampling is comparable
    key, run_key = jax.random.split(key)

    # 1. Perfect info (oracle baseline)
    results["Perfect Info"] = run_perfect_info(config, run_key, reset_key, initial_state)

    # 2. No learning (wrong params, no updates)
    results["No Learning"] = run_no_learning(config, run_key, reset_key, initial_state)

    # 3. Passive learning (EKF but no info-gain objective)
    passive_result = run_passive_learning(config, run_key, reset_key, initial_state)
    results["Passive Learning"] = (passive_result[0], passive_result[1])
    learning_histories["Passive Learning"] = (passive_result[2], passive_result[3], passive_result[4])

    # 4. Active learning (EKF + info-gain objective)
    active_result = run_learning(config, run_key, reset_key, initial_state)
    results["Active Learning"] = (active_result[0], active_result[1])
    learning_histories["Active Learning"] = (active_result[2], active_result[3], active_result[4])

    # Separate states and actions for different uses
    states_only = {k: v[0] for k, v in results.items()}
    actions_only = {k: v[1] for k, v in results.items()}

    # Get control weight for cost computation
    weight_control = config["cost_fn_params"]["weight_control"]

    # Save trajectories and actions
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        traj_path = os.path.join(save_dir, f"unicycle_trajectories_seed_{config['seed']}.npz")
        # Save both states and actions
        save_dict = {}
        for k in results.keys():
            key_name = k.replace(" ", "_").lower()
            save_dict[f"{key_name}_states"] = results[k][0]
            save_dict[f"{key_name}_actions"] = results[k][1]
        np.savez(traj_path, **save_dict)
        print(f"Trajectories and actions saved to {traj_path}")
        save_path = os.path.join(save_dir, f"unicycle_comparison_seed_{config['seed']}.png")
    else:
        save_path = None

    fig = plot_comparison(states_only, save_path)

    # Second plot: distance, cumulative distance, and control cost over time
    metrics_path = save_path.replace(".png", "_metrics.png") if save_path else None
    fig2 = plot_metrics(states_only, actions_only, weight_control, metrics_path)

    # Third plot: parameter convergence for learning cases
    learning_path = save_path.replace(".png", "_learning.png") if save_path else None
    fig3 = plot_learning_curves(learning_histories, config, learning_path)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    print(f"True parameters: theta1 = {true_theta1}, theta2 = {true_theta2}")
    print(f"Init parameters: theta1 = {config['dynamics_params']['init_theta1']}, theta2 = {config['dynamics_params']['init_theta2']}")
    print()

    for title in results.keys():
        states = states_only[title]
        actions = actions_only[title]

        evader_pos = states[:, 0:2]
        unicycle_pos = states[:, 4:6]
        distances = np.sqrt(np.sum((evader_pos - unicycle_pos)**2, axis=1))
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


def run_single_case(config, case, save_dir):
    """Run a single case and save results."""
    key = jax.random.key(config["seed"])

    # Get a common initial state
    reset_fn, _, _ = init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)
    print(f"Initial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"unicycle=({initial_state[4]:.2f}, {initial_state[5]:.2f}), "
          f"heading={np.degrees(initial_state[6]):.1f}deg, speed={initial_state[7]:.2f}")

    key, run_key = jax.random.split(key)

    os.makedirs(save_dir, exist_ok=True)

    if case == "perfect_info":
        states, actions = run_perfect_info(config, run_key, reset_key, initial_state)
        np.savez(os.path.join(save_dir, f"case_perfect_info_seed_{config['seed']}.npz"),
                 states=states, actions=actions)
    elif case == "no_learning":
        states, actions = run_no_learning(config, run_key, reset_key, initial_state)
        np.savez(os.path.join(save_dir, f"case_no_learning_seed_{config['seed']}.npz"),
                 states=states, actions=actions)
    elif case == "passive_learning":
        result = run_passive_learning(config, run_key, reset_key, initial_state)
        states, actions = result[0], result[1]
        np.savez(os.path.join(save_dir, f"case_passive_learning_seed_{config['seed']}.npz"),
                 states=states, actions=actions,
                 theta1_history=np.array(result[2]),
                 theta2_history=np.array(result[3]),
                 cov_trace_history=np.array(result[4]))
    elif case == "active_learning":
        result = run_learning(config, run_key, reset_key, initial_state)
        states, actions = result[0], result[1]
        np.savez(os.path.join(save_dir, f"case_active_learning_seed_{config['seed']}.npz"),
                 states=states, actions=actions,
                 theta1_history=np.array(result[2]),
                 theta2_history=np.array(result[3]),
                 cov_trace_history=np.array(result[4]))
    else:
        raise ValueError(f"Unknown case: {case}")

    print(f"Saved {case} results to {save_dir}")


def combine_results(config, save_dir):
    """Load individual case files and generate plots."""
    seed = config["seed"]

    # Load all case files
    cases = ["perfect_info", "no_learning", "passive_learning", "active_learning"]
    results = {}
    learning_histories = {}

    for case in cases:
        path = os.path.join(save_dir, f"case_{case}_seed_{seed}.npz")
        if not os.path.exists(path):
            print(f"Missing {path} - run with --case {case} first")
            return

        data = np.load(path)
        case_name = case.replace("_", " ").title()
        results[case_name] = (data["states"], data["actions"])

        if "theta1_history" in data:
            learning_histories[case_name] = (
                data["theta1_history"],
                data["theta2_history"],
                data["cov_trace_history"]
            )

    # Separate states and actions
    states_only = {k: v[0] for k, v in results.items()}
    actions_only = {k: v[1] for k, v in results.items()}

    weight_control = config["cost_fn_params"]["weight_control"]

    # Save combined trajectories
    traj_path = os.path.join(save_dir, f"unicycle_trajectories_seed_{seed}.npz")
    save_dict = {}
    for k in results.keys():
        key_name = k.replace(" ", "_").lower()
        save_dict[f"{key_name}_states"] = results[k][0]
        save_dict[f"{key_name}_actions"] = results[k][1]
    np.savez(traj_path, **save_dict)
    print(f"Combined trajectories saved to {traj_path}")

    # Generate plots
    save_path = os.path.join(save_dir, f"unicycle_comparison_seed_{seed}.png")
    fig = plot_comparison(states_only, save_path)

    metrics_path = save_path.replace(".png", "_metrics.png")
    fig2 = plot_metrics(states_only, actions_only, weight_control, metrics_path)

    if learning_histories:
        learning_path = save_path.replace(".png", "_learning.png")
        fig3 = plot_learning_curves(learning_histories, config, learning_path)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    print(f"True parameters: theta1 = {true_theta1}, theta2 = {true_theta2}")
    print(f"Init parameters: theta1 = {config['dynamics_params']['init_theta1']}, theta2 = {config['dynamics_params']['init_theta2']}")
    print()

    for title in results.keys():
        states = states_only[title]
        actions = actions_only[title]

        evader_pos = states[:, 0:2]
        unicycle_pos = states[:, 4:6]
        distances = np.sqrt(np.sum((evader_pos - unicycle_pos)**2, axis=1))
        cumulative_dist = np.sum(distances)

        control_costs = weight_control * np.sum(actions**2, axis=1)
        cumulative_control = np.sum(control_costs)

        total_cost = -cumulative_dist + cumulative_control

        print(f"{title}:")
        print(f"  Final distance:   {distances[-1]:.2f}")
        print(f"  Mean distance:    {distances.mean():.2f}")
        print(f"  Cumulative dist:  {cumulative_dist:.2f}")
        print(f"  Control cost:     {cumulative_control:.2f}")
        print(f"  Total cost:       {total_cost:.2f}")
        print()

    plt.show()
    print("\n=== Combination complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare learning vs perfect info for unicycle MPC.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default="./comparison_results",
        help="Directory to save comparison plots.",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        choices=["perfect_info", "no_learning", "passive_learning", "active_learning"],
        help="Run a single case (for parallel execution). If not specified, runs all cases sequentially.",
    )
    parser.add_argument(
        "--combine", action="store_true",
        help="Combine individual case results and generate plots (use after running cases in parallel).",
    )
    parser.add_argument(
        "--steps", type=int, default=None,
        help="Override total_steps (default: use config value, typically 250).",
    )
    parser.add_argument(
        "--config", type=str, default="unicycle",
        help="Config name (without .json). Options: unicycle, unicycle_aggressive, unicycle_slow_turn",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", f"{args.config}.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Using config: {args.config}")

    config["seed"] = args.seed
    if args.steps:
        config["total_steps"] = args.steps
        config["buffer_size"] = args.steps + 10  # Ensure buffer is large enough

    if args.combine:
        combine_results(config, args.save_dir)
    elif args.case:
        run_single_case(config, args.case, args.save_dir)
    else:
        main(config, save_dir=args.save_dir)
