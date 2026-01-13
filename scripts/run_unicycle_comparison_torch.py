# run_unicycle_comparison_torch.py
# Same as run_unicycle_comparison.py but uses PyTorch MPC for the environment

import jax
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env, EnvParams
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.planners import init_planner
from max.costs import init_cost
import argparse
import copy
import os
import json

# Import PyTorch unicycle dynamics
from IFT_torch import unicycle_step as torch_unicycle_step


# ============================================================
# PyTorch MPC Environment
# ============================================================

def make_torch_unicycle_env(config):
    """
    Create environment that uses PyTorch MPC for the unicycle pursuer.
    """
    env_params = config["env_params"]
    dynamics_config = config["dynamics_params"]

    params = EnvParams(
        dt=env_params.get("dt", 0.1),
        num_agents=env_params.get("num_agents", 1),
        box_half_width=env_params.get("box_half_width", 10.0),
        max_episode_steps=env_params.get("max_episode_steps", 500),
        evader_max_accel=env_params.get("evader_max_accel", 2.0),
        evader_max_speed=env_params.get("evader_max_speed", 3.0),
    )

    dt = dynamics_config["dt"]
    true_theta1 = env_params["true_theta1"]
    true_theta2 = env_params["true_theta2"]
    weight_w = dynamics_config.get("weight_w", 0.1)
    weight_a = dynamics_config.get("weight_a", 1.0)
    weight_speed = dynamics_config.get("weight_speed", 5.0)
    target_speed = dynamics_config.get("target_speed", 1.0)

    # Clamp bounds (match JAX implementation)
    max_angular_vel = 10.0
    max_accel = 20.0

    def solve_mpc_torch(x2_0_np, target_pos_np):
        """Solve MPC using PyTorch L-BFGS, matching JAX's 2-step horizon."""
        x0 = torch.tensor(x2_0_np, dtype=torch.float64)
        p_target = torch.tensor(target_pos_np, dtype=torch.float64)
        theta1_t = torch.tensor(true_theta1, dtype=torch.float64)
        theta2_t = torch.tensor(true_theta2, dtype=torch.float64)

        n_u = 2

        def cost_fn(u0):
            """Cost matching JAX mpc_cost with 2-step rollout, u1=0."""
            u1 = torch.zeros(2, dtype=torch.float64)

            # Rollout 2 steps
            x1 = torch_unicycle_step(x0, u0, dt, theta2_t)
            x2 = torch_unicycle_step(x1, u1, dt, theta2_t)

            # Terminal position error
            pos_err = x2[:2] - p_target
            tracking = theta1_t * torch.sum(pos_err ** 2)

            # Control costs (only u0)
            w, a = u0
            turn_cost = weight_w * w ** 2
            accel_cost = weight_a * a ** 2

            # Speed regulation
            v_final = x2[3]
            speed_cost = weight_speed * (v_final - target_speed) ** 2

            return tracking + turn_cost + accel_cost + speed_cost

        # Optimize only u0
        u0_flat = torch.zeros(n_u, requires_grad=True)

        opt = torch.optim.LBFGS(
            [u0_flat],
            lr=1.0,
            max_iter=200,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            J = cost_fn(u0_flat)
            J.backward()
            return J

        opt.step(closure)

        with torch.no_grad():
            u_star = u0_flat.numpy().copy()

        # Clamp (match JAX)
        u_star[0] = np.clip(u_star[0], -max_angular_vel, max_angular_vel)
        u_star[1] = np.clip(u_star[1], -max_accel, max_accel)

        return u_star

    def step_dynamics(state_np, evader_action_np):
        """Step dynamics using PyTorch MPC for unicycle."""
        p1 = state_np[0:2]
        v1 = state_np[2:4]
        p2 = state_np[4:6]
        alpha2 = state_np[6]
        v2 = state_np[7]

        a1 = evader_action_np.squeeze()

        x2_0 = np.array([p2[0], p2[1], alpha2, v2])
        target_pos = p1

        u2_star = solve_mpc_torch(x2_0, target_pos)

        # Update evader (double integrator)
        next_v1 = v1 + a1 * dt
        next_p1 = p1 + v1 * dt + 0.5 * a1 * dt**2

        # Update unicycle
        x2_torch = torch.tensor(x2_0, dtype=torch.float64)
        u2_torch = torch.tensor(u2_star, dtype=torch.float64)
        x2_next = torch_unicycle_step(x2_torch, u2_torch, dt, true_theta2).numpy()

        next_state = np.array([
            next_p1[0], next_p1[1], next_v1[0], next_v1[1],
            x2_next[0], x2_next[1], x2_next[2], x2_next[3]
        ])

        return next_state

    @jax.jit
    def reset_fn(key):
        key1, key2, key3 = jax.random.split(key, 3)

        p1 = jax.random.uniform(
            key1, shape=(2,),
            minval=-0.5 * params.box_half_width,
            maxval=0.5 * params.box_half_width,
        )
        v1 = jnp.zeros(2)

        p2 = jax.random.uniform(
            key2, shape=(2,),
            minval=-0.5 * params.box_half_width,
            maxval=0.5 * params.box_half_width,
        )
        alpha2 = jax.random.uniform(key3, minval=-jnp.pi, maxval=jnp.pi)
        v2 = 0.5

        return jnp.array([p1[0], p1[1], v1[0], v1[1], p2[0], p2[1], alpha2, v2])

    @jax.jit
    def get_obs_fn(state):
        return state[None, :]

    def step_fn(state, step_count, action):
        clipped_action = np.clip(
            np.array(action),
            -params.evader_max_accel,
            params.evader_max_accel
        )

        next_state = step_dynamics(np.array(state), clipped_action)
        next_state = jnp.array(next_state)

        evader_pos = next_state[0:2]
        opponent_pos = next_state[4:6]
        dist_sq = jnp.sum((evader_pos - opponent_pos) ** 2)

        evader_reward = dist_sq
        rewards = jnp.array([evader_reward])

        terminated = False
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {"distance": jnp.sqrt(dist_sq)}

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


# ============================================================
# Plotting functions (same as original)
# ============================================================

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

        distances = np.sqrt(np.sum((evader_pos - unicycle_pos)**2, axis=1))
        cumulative_dist = np.cumsum(distances)

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
        evader_x, evader_y = states[:, 0], states[:, 1]
        evader_vx, evader_vy = states[:, 2], states[:, 3]

        unicycle_x, unicycle_y = states[:, 4], states[:, 5]
        unicycle_alpha = states[:, 6]
        unicycle_v = states[:, 7]

        ax.plot(evader_x, evader_y, label="Evader", color="blue", linewidth=2, alpha=0.8)
        ax.plot(unicycle_x, unicycle_y, label="Unicycle", color="red", linewidth=2, alpha=0.8)

        for i in range(0, len(states), arrow_every):
            ax.quiver(evader_x[i], evader_y[i], evader_vx[i], evader_vy[i],
                      color='darkblue', scale=30, width=0.005)

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


# ============================================================
# Run functions (using PyTorch environment)
# ============================================================

def run_learning(config, key, reset_key, initial_state):
    """Run the learning case with EKF and info-gathering cost (PyTorch env)."""
    print("\n=== Running LEARNING case (EKF + info-gathering) [PyTorch MPC] ===")

    config = copy.deepcopy(config)

    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    # Use PyTorch environment
    reset_fn, step_fn, get_obs_fn = make_torch_unicycle_env(config)

    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Active Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    theta1_history = [float(init_params['model']['theta1'])]
    theta2_history = [float(init_params['model']['theta2'])]

    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    cov_trace_history = [float(jnp.trace(train_state.covariance))]

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Active Learning"):
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)

            theta1_history.append(float(train_state.params['model']['theta1']))
            theta2_history.append(float(train_state.params['model']['theta2']))
            cov_trace_history.append(float(jnp.trace(train_state.covariance)))

        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  Final params: theta1 = {train_state.params['model']['theta1']:.4f}, theta2 = {train_state.params['model']['theta2']:.4f}")

    print(f"  Theta1 history: {theta1_history[0]:.3f} -> {theta1_history[-1]:.3f} (true: {true_theta1})")
    print(f"  Theta2 history: {theta2_history[0]:.3f} -> {theta2_history[-1]:.3f} (true: {true_theta2})")
    print(f"  Cov trace: {cov_trace_history[0]:.4f} -> {cov_trace_history[-1]:.4f}")

    return states, actions, theta1_history, theta2_history, cov_trace_history


def run_perfect_info(config, key, reset_key, initial_state):
    """Run the perfect information case (PyTorch env)."""
    print("\n=== Running PERFECT INFO case (known model, evasion only) [PyTorch MPC] ===")

    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]
    config["dynamics_params"]["init_theta1"] = true_theta1
    config["dynamics_params"]["init_theta2"] = true_theta2

    reset_fn, step_fn, get_obs_fn = make_torch_unicycle_env(config)

    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Perfect Info params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Perfect Info"):
        cost_params = {"dyn_params": init_params}
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    return states, actions


def run_no_learning(config, key, reset_key, initial_state):
    """Run with wrong parameters and no learning (PyTorch env)."""
    print("\n=== Running NO LEARNING case (wrong model, no updates) [PyTorch MPC] ===")

    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    reset_fn, step_fn, get_obs_fn = make_torch_unicycle_env(config)

    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG No Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  No Learning"):
        cost_params = {"dyn_params": init_params}
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    return states, actions


def run_passive_learning(config, key, reset_key, initial_state):
    """Run with EKF learning but no info-gain objective (PyTorch env)."""
    print("\n=== Running PASSIVE LEARNING case (EKF updates, evasion only cost) [PyTorch MPC] ===")

    config = copy.deepcopy(config)
    config["cost_type"] = "evasion_only"

    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    reset_fn, step_fn, get_obs_fn = make_torch_unicycle_env(config)

    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)
    print(f"  DEBUG Passive Learning params: theta1 = {init_params['model']['theta1']}, theta2 = {init_params['model']['theta2']}")

    theta1_history = [float(init_params['model']['theta1'])]
    theta2_history = [float(init_params['model']['theta2'])]

    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    cov_trace_history = [float(jnp.trace(train_state.covariance))]

    cost_fn = init_cost(config, dynamics_model)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    state = initial_state
    current_obs = get_obs_fn(state)
    episode_length = 0

    for step in tqdm(range(1, config["total_steps"] + 1), desc="  Passive Learning"):
        cost_params = {"dyn_params": train_state.params}
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]

        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1

        buffers = update_buffer_dynamic(
            buffers, buffer_idx, current_obs, action, rewards,
            jnp.zeros_like(rewards), jnp.zeros_like(rewards), float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        if done:
            state = initial_state
            current_obs = get_obs_fn(state)
            episode_length = 0

        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)

            theta1_history.append(float(train_state.params['model']['theta1']))
            theta2_history.append(float(train_state.params['model']['theta2']))
            cov_trace_history.append(float(jnp.trace(train_state.covariance)))

        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    states = np.array(buffers["states"][0, :buffer_idx, :])
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    print(f"  Final params: theta1 = {train_state.params['model']['theta1']:.4f}, theta2 = {train_state.params['model']['theta2']:.4f}")

    print(f"  Theta1 history: {theta1_history[0]:.3f} -> {theta1_history[-1]:.3f} (true: {true_theta1})")
    print(f"  Theta2 history: {theta2_history[0]:.3f} -> {theta2_history[-1]:.3f} (true: {true_theta2})")
    print(f"  Cov trace: {cov_trace_history[0]:.4f} -> {cov_trace_history[-1]:.4f}")

    return states, actions, theta1_history, theta2_history, cov_trace_history


# ============================================================
# Main
# ============================================================

def main(config, save_dir):
    """Run all cases and compare."""
    key = jax.random.key(config["seed"])

    # Get initial state using JAX env (for consistency with JAX version)
    from max.environments import init_env as jax_init_env
    reset_fn, _, _ = jax_init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)
    print(f"Initial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"unicycle=({initial_state[4]:.2f}, {initial_state[5]:.2f}), "
          f"heading={np.degrees(initial_state[6]):.1f}deg, speed={initial_state[7]:.2f}")

    results = {}
    learning_histories = {}

    key, run_key = jax.random.split(key)

    # 1. Perfect info
    results["Perfect Info"] = run_perfect_info(config, run_key, reset_key, initial_state)

    # 2. No learning
    results["No Learning"] = run_no_learning(config, run_key, reset_key, initial_state)

    # 3. Passive learning
    passive_result = run_passive_learning(config, run_key, reset_key, initial_state)
    results["Passive Learning"] = (passive_result[0], passive_result[1])
    learning_histories["Passive Learning"] = (passive_result[2], passive_result[3], passive_result[4])

    # 4. Active learning
    active_result = run_learning(config, run_key, reset_key, initial_state)
    results["Active Learning"] = (active_result[0], active_result[1])
    learning_histories["Active Learning"] = (active_result[2], active_result[3], active_result[4])

    states_only = {k: v[0] for k, v in results.items()}
    actions_only = {k: v[1] for k, v in results.items()}

    weight_control = config["cost_fn_params"]["weight_control"]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        traj_path = os.path.join(save_dir, f"unicycle_trajectories_seed_{config['seed']}.npz")
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

    metrics_path = save_path.replace(".png", "_metrics.png") if save_path else None
    fig2 = plot_metrics(states_only, actions_only, weight_control, metrics_path)

    learning_path = save_path.replace(".png", "_learning.png") if save_path else None
    fig3 = plot_learning_curves(learning_histories, config, learning_path)

    # Print summary statistics
    print("\n=== SUMMARY STATISTICS (PyTorch MPC) ===")
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

    print("\n=== Comparison complete (PyTorch MPC) ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare learning vs perfect info for unicycle MPC (PyTorch backend).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--save-dir", type=str, default="./comparison_results_torch",
        help="Directory to save comparison plots.",
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

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", f"{args.config}.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    print(f"Using config: {args.config} (PyTorch MPC backend)")

    config["seed"] = args.seed
    if args.steps:
        config["total_steps"] = args.steps
        config["buffer_size"] = args.steps + 10

    main(config, save_dir=args.save_dir)
