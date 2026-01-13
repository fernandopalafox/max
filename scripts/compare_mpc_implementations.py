"""
Compare JAX and PyTorch MPC implementations.

This script wraps the PyTorch IFT_torch.py implementation to work with
JAX arrays, then compares trajectories from both implementations.
"""

import numpy as np
import torch
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from typing import Tuple

# Import PyTorch implementation
from IFT_torch import unicycle_step as torch_unicycle_step, solve_u_T, cost_T

# Import JAX implementation
import sys
sys.path.insert(0, '/home/jmilzman/Documents/max')
from max.dynamics import create_unicycle_mpc_dynamics


# ============================================================
# PyTorch MPC wrapper for JAX compatibility
# ============================================================

def create_torch_mpc_solver(
    dt: float,
    theta1: float,
    theta2: float,
    horizon: int = 2,
    weight_w: float = 0.1,
    weight_a: float = 1.0,
    weight_speed: float = 5.0,
    target_speed: float = 1.0,
    iters: int = 200,
):
    """
    Create a PyTorch-based MPC solver that takes JAX/numpy arrays.

    Returns a function: solve(x2_0, target_pos) -> u_star
    """

    def solve(x2_0: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
        """
        Solve MPC for unicycle to track target position.

        Args:
            x2_0: Initial unicycle state [px, py, alpha, v]
            target_pos: Target position [px, py]

        Returns:
            u_star: Optimal control for first timestep [w, a]
        """
        # Convert to torch tensors
        x0_torch = torch.tensor(x2_0, dtype=torch.float64)
        target_torch = torch.tensor(target_pos, dtype=torch.float64)
        theta1_torch = torch.tensor(theta1, dtype=torch.float64)
        theta2_torch = torch.tensor(theta2, dtype=torch.float64)

        # Create custom cost function matching JAX implementation
        def cost_fn(x0, u_seq, dt_val, th1, th2, p_target, **kwargs):
            """Cost function matching JAX mpc_cost structure."""
            # Rollout
            T = u_seq.shape[0]
            xs = [x0]
            x = x0
            for t in range(T):
                x = torch_unicycle_step(x, u_seq[t], dt_val, th2)
                xs.append(x)
            xs = torch.stack(xs, dim=0)

            # Terminal position error (like JAX 2-step)
            pos_err = xs[-1, :2] - p_target
            tracking = th1 * torch.sum(pos_err ** 2)

            # Control costs
            w_cost = weight_w * torch.sum(u_seq[:, 0] ** 2)
            a_cost = weight_a * torch.sum(u_seq[:, 1] ** 2)

            # Speed regulation
            v_final = xs[-1, 3]
            speed_cost = weight_speed * (v_final - target_speed) ** 2

            return tracking + w_cost + a_cost + speed_cost

        # Solve using L-BFGS
        n_u = 2
        u_flat = torch.zeros(horizon * n_u, requires_grad=True)

        opt = torch.optim.LBFGS(
            [u_flat],
            lr=1.0,
            max_iter=iters,
            line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            u_seq = u_flat.view(horizon, n_u)
            J = cost_fn(x0_torch, u_seq, dt, theta1_torch, theta2_torch, target_torch)
            J.backward()
            return J

        opt.step(closure)

        with torch.no_grad():
            u_star = u_flat.view(horizon, n_u)[0].numpy()  # Return first control only

        return u_star

    return solve


def create_jax_mpc_solver(config: dict):
    """
    Create the JAX MPC solver from the existing implementation.

    Returns the dynamics model and a solve function.
    """
    model, params = create_unicycle_mpc_dynamics(config, None, None)
    return model, params


# ============================================================
# Trajectory simulation
# ============================================================

def simulate_trajectory_torch(
    solver_fn,
    x1_init: np.ndarray,  # Evader [px, py, vx, vy]
    x2_init: np.ndarray,  # Unicycle [px, py, alpha, v]
    evader_actions: np.ndarray,  # [T, 2] evader accelerations
    dt: float,
    theta2: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate trajectory using PyTorch MPC for the unicycle.

    Returns:
        x1_traj: Evader trajectory [T+1, 4]
        x2_traj: Unicycle trajectory [T+1, 4]
        u2_traj: Unicycle controls [T, 2]
    """
    T = evader_actions.shape[0]

    x1_traj = [x1_init.copy()]
    x2_traj = [x2_init.copy()]
    u2_traj = []

    x1 = x1_init.copy()
    x2 = x2_init.copy()

    for t in range(T):
        # Unicycle targets evader's current position
        target_pos = x1[:2]

        # Solve MPC for unicycle control
        u2 = solver_fn(x2, target_pos)
        u2_traj.append(u2)

        # Update evader (double integrator)
        a1 = evader_actions[t]
        x1_new = np.zeros(4)
        x1_new[:2] = x1[:2] + x1[2:4] * dt + 0.5 * a1 * dt**2
        x1_new[2:4] = x1[2:4] + a1 * dt
        x1 = x1_new
        x1_traj.append(x1.copy())

        # Update unicycle
        x2_torch = torch.tensor(x2, dtype=torch.float64)
        u2_torch = torch.tensor(u2, dtype=torch.float64)
        x2_new = torch_unicycle_step(x2_torch, u2_torch, dt, theta2).numpy()
        x2 = x2_new
        x2_traj.append(x2.copy())

    return np.array(x1_traj), np.array(x2_traj), np.array(u2_traj)


def simulate_trajectory_jax(
    model,
    params: dict,
    x1_init: jnp.ndarray,
    x2_init: jnp.ndarray,
    evader_actions: jnp.ndarray,
    dt: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate trajectory using JAX MPC dynamics.

    Returns:
        states: Full state trajectory [T+1, 8]
    """
    T = evader_actions.shape[0]

    # Combine into full state
    state = jnp.concatenate([
        x1_init,  # [p1x, p1y, v1x, v1y]
        x2_init,  # [p2x, p2y, alpha, v]
    ])

    states = [state]

    for t in range(T):
        action = evader_actions[t]
        state = model.pred_one_step(params, state, action)
        states.append(state)

    return jnp.stack(states, axis=0)


# ============================================================
# Comparison
# ============================================================

def compare_implementations():
    """Compare JAX and PyTorch MPC implementations."""

    # Common parameters
    dt = 0.1
    theta1 = 5.0
    theta2 = 1.0
    horizon = 2  # Match JAX's 2-step horizon
    weight_w = 0.1
    weight_a = 1.0
    weight_speed = 5.0
    target_speed = 1.0

    # Initial conditions
    x1_init = np.array([0.0, 0.0, 0.0, 0.0])  # Evader at origin, stationary
    x2_init = np.array([2.0, 1.0, np.pi, 0.5])  # Unicycle offset, facing left

    # Evader actions (simple trajectory - accelerate right)
    T = 50
    evader_actions = np.zeros((T, 2))
    evader_actions[:, 0] = 0.5  # Constant x acceleration

    # Create PyTorch solver
    torch_solver = create_torch_mpc_solver(
        dt=dt,
        theta1=theta1,
        theta2=theta2,
        horizon=horizon,
        weight_w=weight_w,
        weight_a=weight_a,
        weight_speed=weight_speed,
        target_speed=target_speed,
    )

    # Create JAX solver
    config = {
        "dynamics_params": {
            "dt": dt,
            "newton_iters": 10,
            "init_theta1": theta1,
            "init_theta2": theta2,
            "weight_w": weight_w,
            "weight_a": weight_a,
            "weight_speed": weight_speed,
            "target_speed": target_speed,
        }
    }
    jax_model, jax_params = create_jax_mpc_solver(config)

    # Override params to use our theta values
    jax_params = {
        "model": {
            "theta1": jnp.array(theta1),
            "theta2": jnp.array(theta2),
        },
        "normalizer": None,
    }

    # Simulate with PyTorch
    print("Simulating with PyTorch MPC...")
    x1_torch, x2_torch, u2_torch = simulate_trajectory_torch(
        torch_solver, x1_init, x2_init, evader_actions, dt, theta2
    )

    # Simulate with JAX
    print("Simulating with JAX MPC...")
    jax_states = simulate_trajectory_jax(
        jax_model, jax_params,
        jnp.array(x1_init), jnp.array(x2_init),
        jnp.array(evader_actions), dt
    )

    # Extract JAX trajectories
    x1_jax = np.array(jax_states[:, :4])
    x2_jax = np.array(jax_states[:, 4:8])

    # Compare
    print("\n" + "="*60)
    print("TRAJECTORY COMPARISON")
    print("="*60)

    # Position errors
    x1_pos_err = np.linalg.norm(x1_torch[:, :2] - x1_jax[:, :2], axis=1)
    x2_pos_err = np.linalg.norm(x2_torch[:, :2] - x2_jax[:, :2], axis=1)

    print(f"\nEvader position error (max): {x1_pos_err.max():.6f}")
    print(f"Evader position error (mean): {x1_pos_err.mean():.6f}")
    print(f"\nUnicycle position error (max): {x2_pos_err.max():.6f}")
    print(f"Unicycle position error (mean): {x2_pos_err.mean():.6f}")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Trajectory plot
    ax = axes[0, 0]
    ax.plot(x1_torch[:, 0], x1_torch[:, 1], 'b-', label='Evader (PyTorch)', linewidth=2)
    ax.plot(x1_jax[:, 0], x1_jax[:, 1], 'b--', label='Evader (JAX)', linewidth=2)
    ax.plot(x2_torch[:, 0], x2_torch[:, 1], 'r-', label='Unicycle (PyTorch)', linewidth=2)
    ax.plot(x2_jax[:, 0], x2_jax[:, 1], 'r--', label='Unicycle (JAX)', linewidth=2)
    ax.scatter([x1_init[0]], [x1_init[1]], c='blue', s=100, marker='o', zorder=5)
    ax.scatter([x2_init[0]], [x2_init[1]], c='red', s=100, marker='o', zorder=5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Trajectories')
    ax.legend()
    ax.axis('equal')
    ax.grid(True)

    # Position error over time
    ax = axes[0, 1]
    ax.plot(x1_pos_err, 'b-', label='Evader')
    ax.plot(x2_pos_err, 'r-', label='Unicycle')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Position error')
    ax.set_title('Position Error (JAX vs PyTorch)')
    ax.legend()
    ax.grid(True)

    # Unicycle heading comparison
    ax = axes[1, 0]
    ax.plot(x2_torch[:, 2], 'r-', label='PyTorch')
    ax.plot(x2_jax[:, 2], 'r--', label='JAX')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Heading (rad)')
    ax.set_title('Unicycle Heading')
    ax.legend()
    ax.grid(True)

    # Unicycle speed comparison
    ax = axes[1, 1]
    ax.plot(x2_torch[:, 3], 'r-', label='PyTorch')
    ax.plot(x2_jax[:, 3], 'r--', label='JAX')
    ax.axhline(y=target_speed, color='k', linestyle=':', label=f'Target ({target_speed})')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Speed')
    ax.set_title('Unicycle Speed')
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.savefig('/home/jmilzman/Documents/max/comparison_results/mpc_implementation_comparison.png', dpi=150)
    print(f"\nPlot saved to comparison_results/mpc_implementation_comparison.png")
    plt.show()

    return x1_torch, x2_torch, x1_jax, x2_jax


if __name__ == "__main__":
    compare_implementations()
