"""
Compare JAX vs PyTorch MPC implementations in the full active learning workflow.

This runs the active learning experiment twice:
1. With JAX MPC (existing implementation)
2. With PyTorch MPC (wrapped to match JAX interface)

Then compares the resulting trajectories.
"""

import jax
import jax.numpy as jnp
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import os
import json
import argparse

from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.planners import init_planner
from max.costs import init_cost

# Import PyTorch unicycle dynamics
from IFT_torch import unicycle_step as torch_unicycle_step


# ============================================================
# PyTorch MPC Environment Wrapper
# ============================================================

def make_torch_unicycle_env(config):
    """
    Create environment that uses PyTorch MPC for the unicycle pursuer.
    Matches the interface of make_unicycle_mpc_env from environments.py.
    """
    from max.environments import EnvParams

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

        horizon = 2
        n_u = 2

        def cost_fn(u_seq):
            """Cost matching JAX mpc_cost with 2-step rollout, u1=0."""
            # Only use first control, second is zero (like JAX)
            u0 = u_seq[0]
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

        # Optimize only u0 (to match JAX which optimizes single control)
        u0_flat = torch.zeros(n_u, requires_grad=True)

        opt = torch.optim.LBFGS(
            [u0_flat],
            lr=1.0,
            max_iter=200,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            u_seq = u0_flat.unsqueeze(0)  # Shape [1, 2]
            J = cost_fn(u_seq)
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
        # Unpack state
        p1 = state_np[0:2]
        v1 = state_np[2:4]
        p2 = state_np[4:6]
        alpha2 = state_np[6]
        v2 = state_np[7]

        # Evader action
        a1 = evader_action_np.squeeze()

        # Unicycle state for MPC
        x2_0 = np.array([p2[0], p2[1], alpha2, v2])
        target_pos = p1  # Track evader's current position

        # Solve MPC
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
        """Reset environment with random initial positions."""
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
        """Step using PyTorch MPC (not JIT-able)."""
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
# Run active learning with specified environment
# ============================================================

def run_active_learning(config, key, initial_state, use_torch_env=False, desc="Active Learning"):
    """Run active learning case, optionally using PyTorch MPC environment."""
    config = copy.deepcopy(config)

    true_theta1 = config["env_params"]["true_theta1"]
    true_theta2 = config["env_params"]["true_theta2"]

    # Initialize environment
    if use_torch_env:
        reset_fn, step_fn, get_obs_fn = make_torch_unicycle_env(config)
    else:
        reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics (for planner's model, not environment)
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    # Initialize EKF trainer
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Initialize cost and planner
    cost_fn = init_cost(config, dynamics_model)
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

    states_list = [np.array(state)]
    actions_list = []

    for step in tqdm(range(1, config["total_steps"] + 1), desc=f"  {desc}"):
        # Plan with current model
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]
        actions_list.append(np.array(action).squeeze())

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        states_list.append(np.array(state))

        done = terminated or truncated
        episode_length += 1

        # Update buffer
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

        # EKF update
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)

        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"], config["buffer_size"],
                config["dim_state"], config["dim_action"],
            )
            buffer_idx = 0

    states = np.array(states_list)
    actions = np.array(actions_list)

    print(f"  Final params: theta1={train_state.params['model']['theta1']:.4f}, "
          f"theta2={train_state.params['model']['theta2']:.4f}")

    return states, actions


# ============================================================
# Main comparison
# ============================================================

def main(config_name, seed, total_steps, save_dir):
    """Run comparison between JAX and PyTorch MPC environments."""

    # Load config
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", f"{config_name}.json"
    )
    with open(config_path, "r") as f:
        config = json.load(f)

    config["seed"] = seed
    if total_steps:
        config["total_steps"] = total_steps
        config["buffer_size"] = total_steps + 10

    print(f"Config: {config_name}")
    print(f"Seed: {seed}")
    print(f"Steps: {config['total_steps']}")
    print(f"True params: theta1={config['env_params']['true_theta1']}, "
          f"theta2={config['env_params']['true_theta2']}")
    print(f"Init params: theta1={config['dynamics_params']['init_theta1']}, "
          f"theta2={config['dynamics_params']['init_theta2']}")

    key = jax.random.key(seed)

    # Get common initial state
    reset_fn, _, _ = init_env(copy.deepcopy(config))
    key, reset_key = jax.random.split(key)
    initial_state = reset_fn(reset_key)

    print(f"\nInitial state: evader=({initial_state[0]:.2f}, {initial_state[1]:.2f}), "
          f"unicycle=({initial_state[4]:.2f}, {initial_state[5]:.2f}), "
          f"heading={np.degrees(initial_state[6]):.1f}deg")

    # Run with JAX MPC
    print("\n=== Running with JAX MPC ===")
    key, run_key = jax.random.split(key)
    jax_states, jax_actions = run_active_learning(
        config, run_key, initial_state, use_torch_env=False, desc="JAX MPC"
    )

    # Run with PyTorch MPC
    print("\n=== Running with PyTorch MPC ===")
    key, run_key = jax.random.split(key)
    # Use same key for fair comparison
    run_key = jax.random.key(seed + 1000)  # Different but deterministic
    torch_states, torch_actions = run_active_learning(
        config, run_key, initial_state, use_torch_env=True, desc="PyTorch MPC"
    )

    # Compare
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON")
    print("="*60)

    # Align lengths
    min_len = min(len(jax_states), len(torch_states))
    jax_states = jax_states[:min_len]
    torch_states = torch_states[:min_len]

    # Position errors
    evader_pos_err = np.linalg.norm(
        jax_states[:, :2] - torch_states[:, :2], axis=1
    )
    unicycle_pos_err = np.linalg.norm(
        jax_states[:, 4:6] - torch_states[:, 4:6], axis=1
    )

    print(f"\nEvader position error - max: {evader_pos_err.max():.4f}, mean: {evader_pos_err.mean():.4f}")
    print(f"Unicycle position error - max: {unicycle_pos_err.max():.4f}, mean: {unicycle_pos_err.mean():.4f}")

    # Distance metrics
    jax_dist = np.sqrt(np.sum((jax_states[:, :2] - jax_states[:, 4:6])**2, axis=1))
    torch_dist = np.sqrt(np.sum((torch_states[:, :2] - torch_states[:, 4:6])**2, axis=1))

    print(f"\nJAX mean distance: {jax_dist.mean():.4f}")
    print(f"PyTorch mean distance: {torch_dist.mean():.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Trajectories
    ax = axes[0, 0]
    ax.plot(jax_states[:, 0], jax_states[:, 1], 'b-', label='Evader (JAX)', linewidth=2)
    ax.plot(torch_states[:, 0], torch_states[:, 1], 'b--', label='Evader (PyTorch)', linewidth=2)
    ax.plot(jax_states[:, 4], jax_states[:, 5], 'r-', label='Unicycle (JAX)', linewidth=2)
    ax.plot(torch_states[:, 4], torch_states[:, 5], 'r--', label='Unicycle (PyTorch)', linewidth=2)
    ax.scatter([initial_state[0]], [initial_state[1]], c='blue', s=100, marker='o', zorder=5)
    ax.scatter([initial_state[4]], [initial_state[5]], c='red', s=100, marker='o', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectories (Active Learning)')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    # Position errors
    ax = axes[0, 1]
    ax.plot(evader_pos_err, 'b-', label='Evader')
    ax.plot(unicycle_pos_err, 'r-', label='Unicycle')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position error')
    ax.set_title('Position Error (JAX vs PyTorch)')
    ax.legend()
    ax.grid(True)

    # Distance over time
    ax = axes[1, 0]
    ax.plot(jax_dist, 'g-', label='JAX', linewidth=2)
    ax.plot(torch_dist, 'g--', label='PyTorch', linewidth=2)
    ax.set_xlabel('Time step')
    ax.set_ylabel('Distance')
    ax.set_title('Evader-Unicycle Distance')
    ax.legend()
    ax.grid(True)

    # Unicycle heading
    ax = axes[1, 1]
    ax.plot(jax_states[:, 6], 'r-', label='JAX')
    ax.plot(torch_states[:, 6], 'r--', label='PyTorch')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Heading (rad)')
    ax.set_title('Unicycle Heading')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"mpc_full_workflow_comparison_seed_{seed}.png")
    plt.savefig(save_path, dpi=150)
    print(f"\nPlot saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="unicycle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=50, help="Fewer steps for faster comparison")
    parser.add_argument("--save-dir", type=str, default="./comparison_results")
    args = parser.parse_args()

    main(args.config, args.seed, args.steps, args.save_dir)
