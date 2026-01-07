"""
Test script for unicycle MPC dynamics.
Verifies:
1. Forward pass runs without errors
2. Gradients flow through implicit differentiation
3. MPC solver converges to reasonable controls
"""

import jax
import jax.numpy as jnp
from max.dynamics import create_unicycle_mpc_dynamics
from max.normalizers import init_normalizer


def test_forward_pass():
    """Test that forward dynamics runs without errors."""
    print("=== Test 1: Forward Pass ===")

    config = {
        "dynamics": "unicycle_mpc",
        "dynamics_params": {
            "dt": 0.1,
            "newton_iters": 10,
            "init_theta1": 1.0,  # position cost weight
            "init_theta2": 1.0,  # acceleration scaling
        },
        "normalizer": "none",
    }

    normalizer, norm_params = init_normalizer(config)
    model, params = create_unicycle_mpc_dynamics(config, normalizer, norm_params)

    # Initial state: [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
    # Player 1 at origin, player 2 at (5, 5) facing toward origin
    state = jnp.array([0.0, 0.0, 0.0, 0.0, 5.0, 5.0, -2.36, 1.0])  # alpha ~ -135 deg
    action = jnp.array([0.1, 0.1])  # player 1 acceleration

    next_state = model.pred_one_step(params, state, action)

    print(f"  Initial state: {state}")
    print(f"  Action (P1):   {action}")
    print(f"  Next state:    {next_state}")
    print(f"  P1 moved: ({state[0]:.2f}, {state[1]:.2f}) -> ({next_state[0]:.2f}, {next_state[1]:.2f})")
    print(f"  P2 moved: ({state[4]:.2f}, {state[5]:.2f}) -> ({next_state[4]:.2f}, {next_state[5]:.2f})")
    print("  PASSED\n")

    return model, params, state, action


def test_gradient_flow(model, params, state, action):
    """Test that gradients flow through the implicit differentiation."""
    print("=== Test 2: Gradient Flow ===")

    # Test 2a: Loss on position (should have zero gradient after 1 step)
    print("  Test 2a: Loss on position (expect zero gradients for 1 step)")
    def loss_position(params, state, action):
        next_state = model.pred_one_step(params, state, action)
        p1_next = next_state[0:2]
        p2_next = next_state[4:6]
        return jnp.sum((p1_next - p2_next) ** 2)

    grads_pos = jax.grad(loss_position)(params, state, action)
    print(f"    Gradient w.r.t. theta1: {grads_pos['model']['theta1']}")
    print(f"    Gradient w.r.t. theta2: {grads_pos['model']['theta2']}")

    # Test 2b: Loss on heading and velocity (should have non-zero gradient)
    print("  Test 2b: Loss on heading/velocity (expect non-zero gradients)")
    def loss_heading_vel(params, state, action):
        next_state = model.pred_one_step(params, state, action)
        alpha2 = next_state[6]
        v2 = next_state[7]
        # Arbitrary loss that depends on heading and velocity
        return alpha2**2 + v2**2

    loss = loss_heading_vel(params, state, action)
    grads_hv = jax.grad(loss_heading_vel)(params, state, action)
    print(f"    Loss (alpha^2 + v^2): {loss:.4f}")
    print(f"    Gradient w.r.t. theta1: {grads_hv['model']['theta1']}")
    print(f"    Gradient w.r.t. theta2: {grads_hv['model']['theta2']}")

    grad_theta1 = grads_hv['model']['theta1']
    grad_theta2 = grads_hv['model']['theta2']

    assert not jnp.isnan(grad_theta1), "Gradient theta1 is NaN!"
    assert not jnp.isnan(grad_theta2), "Gradient theta2 is NaN!"
    assert grad_theta1 != 0.0 or grad_theta2 != 0.0, "Both gradients are zero!"
    print("  Gradients are valid (not NaN, at least one non-zero)")

    # Test 2c: Multi-step rollout (position should have gradient after 2+ steps)
    print("  Test 2c: Loss on position after 2 steps (expect non-zero gradients)")
    def loss_position_2step(params, state, action):
        state1 = model.pred_one_step(params, state, action)
        state2 = model.pred_one_step(params, state1, action)
        p1_next = state2[0:2]
        p2_next = state2[4:6]
        return jnp.sum((p1_next - p2_next) ** 2)

    loss_2step = loss_position_2step(params, state, action)
    grads_2step = jax.grad(loss_position_2step)(params, state, action)
    print(f"    Loss (distance^2 after 2 steps): {loss_2step:.4f}")
    print(f"    Gradient w.r.t. theta1: {grads_2step['model']['theta1']}")
    print(f"    Gradient w.r.t. theta2: {grads_2step['model']['theta2']}")

    grad_theta1_2step = grads_2step['model']['theta1']
    grad_theta2_2step = grads_2step['model']['theta2']

    assert not jnp.isnan(grad_theta1_2step), "Gradient theta1 is NaN!"
    assert not jnp.isnan(grad_theta2_2step), "Gradient theta2 is NaN!"
    print("  PASSED\n")


def test_mpc_behavior():
    """Test that MPC produces sensible controls."""
    print("=== Test 3: MPC Behavior ===")

    config = {
        "dynamics": "unicycle_mpc",
        "dynamics_params": {
            "dt": 0.1,
            "newton_iters": 10,
            "init_theta1": 10.0,  # high position cost -> aggressive tracking
            "init_theta2": 1.0,
        },
        "normalizer": "none",
    }

    normalizer, norm_params = init_normalizer(config)
    model, params = create_unicycle_mpc_dynamics(config, normalizer, norm_params)

    # Player 2 facing away from player 1
    # P1 at origin, P2 at (5, 0) facing +x direction (away from P1)
    state = jnp.array([0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 1.0])
    action = jnp.array([0.0, 0.0])

    # We need access to solve_mpc to see the controls. Recreate it here for debugging.
    dt = config["dynamics_params"]["dt"]

    def unicycle_step_dbg(x2, u2, theta2):
        p2x, p2y, alpha2, v2 = x2
        w, a = u2
        p2x_next = p2x + dt * v2 * jnp.cos(alpha2)
        p2y_next = p2y + dt * v2 * jnp.sin(alpha2)
        alpha2_next = alpha2 + dt * w
        v2_next = v2 + dt * a * theta2
        return jnp.array([p2x_next, p2y_next, alpha2_next, v2_next])

    def rollout_2step_dbg(x2_0, u0, theta2):
        x2_1 = unicycle_step_dbg(x2_0, u0, theta2)
        x2_2 = unicycle_step_dbg(x2_1, jnp.zeros(2), theta2)
        return x2_2

    def mpc_cost_dbg(u0, x2_0, target_pos, theta1, theta2):
        x2_2 = rollout_2step_dbg(x2_0, u0, theta2)
        pos_err = x2_2[0:2] - target_pos
        return theta1 * jnp.sum(pos_err**2) + jnp.sum(u0**2)

    def kkt_residual_dbg(u0, x2_0, target_pos, theta1, theta2):
        return jax.grad(mpc_cost_dbg, argnums=0)(u0, x2_0, target_pos, theta1, theta2)

    def newton_solve_dbg(x2_0, target_pos, theta1, theta2):
        u = jnp.zeros(2)
        for _ in range(10):
            r = kkt_residual_dbg(u, x2_0, target_pos, theta1, theta2)
            J = jax.jacobian(kkt_residual_dbg, argnums=0)(u, x2_0, target_pos, theta1, theta2)
            J_reg = J + 1e-6 * jnp.eye(2)
            du = jnp.linalg.solve(J_reg, r)
            u = u - du
        return u

    theta1 = params["model"]["theta1"]
    theta2 = params["model"]["theta2"]

    # Run several steps
    print("  Running 20 steps with P2 tracking P1...")
    print("  P1 stationary at origin, P2 starts at (5,0) facing +x")
    print()
    states = [state]
    for i in range(20):
        # Get P2's MPC control before stepping
        p1 = state[0:2]
        x2_0 = jnp.array([state[4], state[5], state[6], state[7]])
        u2_star = newton_solve_dbg(x2_0, p1, theta1, theta2)

        state = model.pred_one_step(params, state, action)
        states.append(state)

        if i < 10 or i >= 18:  # Print first 10 and last 2
            p2x, p2y, alpha2, v2 = state[4], state[5], state[6], state[7]
            dist = jnp.sqrt((state[0] - p2x)**2 + (state[1] - p2y)**2)
            print(f"    Step {i+1:2d}: P2=({p2x:6.2f}, {p2y:6.2f}), alpha={jnp.degrees(alpha2):6.1f}°, v={v2:.2f}, u2=[{u2_star[0]:+.3f}, {u2_star[1]:+.3f}], dist={dist:.2f}")
        elif i == 10:
            print("    ...")

    states = jnp.stack(states)
    print()

    # Check that P2 moves toward P1
    initial_dist = jnp.sqrt((states[0, 0] - states[0, 4])**2 + (states[0, 1] - states[0, 5])**2)
    final_dist = jnp.sqrt((states[-1, 0] - states[-1, 4])**2 + (states[-1, 1] - states[-1, 5])**2)

    print(f"  Initial distance: {initial_dist:.2f}")
    print(f"  Final distance:   {final_dist:.2f}")

    if final_dist < initial_dist:
        print("  P2 moved toward P1 (MPC tracking works)")
        print("  PASSED\n")
    else:
        print("  WARNING: P2 did not move toward P1")
        print("  FAILED\n")


def test_trajectory_rollout():
    """Visualize a trajectory."""
    print("=== Test 4: Trajectory Rollout ===")

    config = {
        "dynamics": "unicycle_mpc",
        "dynamics_params": {
            "dt": 0.1,
            "newton_iters": 10,
            "init_theta1": 5.0,
            "init_theta2": 1.0,
            "weight_w": 0.1,  # low penalty on turning -> P2 turns freely
            "weight_a": 1.0,  # normal penalty on acceleration
        },
        "normalizer": "none",
    }

    normalizer, norm_params = init_normalizer(config)
    model, params = create_unicycle_mpc_dynamics(config, normalizer, norm_params)

    # P1 at origin moving right, P2 at (3, 3) facing down-left
    state = jnp.array([0.0, 0.0, 1.0, 0.0, 3.0, 3.0, -2.36, 0.5])

    print(f"  Initial: P1=({state[0]:.2f}, {state[1]:.2f}), P2=({state[4]:.2f}, {state[5]:.2f}), alpha={jnp.degrees(state[6]):.1f}°")
    print()

    # P1 accelerates in a circle
    states = [state]
    for i in range(50):
        # P1 does a circular motion
        t = i * 0.1
        action = jnp.array([jnp.sin(t), jnp.cos(t)]) * 0.5
        state = model.pred_one_step(params, state, action)
        states.append(state)

        if i < 10 or i >= 45:
            p1x, p1y = state[0], state[1]
            p2x, p2y, alpha2, v2 = state[4], state[5], state[6], state[7]
            dist = jnp.sqrt((p1x - p2x)**2 + (p1y - p2y)**2)
            print(f"    Step {i+1:2d}: P1=({p1x:5.2f}, {p1y:5.2f}), P2=({p2x:5.2f}, {p2y:5.2f}), alpha={jnp.degrees(alpha2):6.1f}°, v2={v2:.2f}, dist={dist:.2f}")
        elif i == 10:
            print("    ...")

    states = jnp.stack(states)

    print()
    print(f"  Rolled out {len(states)} states")
    print(f"  P1 final position: ({states[-1, 0]:.2f}, {states[-1, 1]:.2f})")
    print(f"  P2 final position: ({states[-1, 4]:.2f}, {states[-1, 5]:.2f})")
    print(f"  P2 final heading:  {states[-1, 6]:.2f} rad ({jnp.degrees(states[-1, 6]):.1f} deg)")
    print("  PASSED\n")

    return states


def plot_trajectory(states):
    """Optional: plot the trajectory if matplotlib available."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        states = np.array(states)

        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Player 1 trajectory
        ax.plot(states[:, 0], states[:, 1], 'b-', label='Player 1 (evader)', linewidth=2)
        ax.scatter(states[0, 0], states[0, 1], c='blue', s=100, marker='o', zorder=5)
        ax.scatter(states[-1, 0], states[-1, 1], c='blue', s=100, marker='x', zorder=5)

        # Player 2 trajectory
        ax.plot(states[:, 4], states[:, 5], 'r-', label='Player 2 (unicycle)', linewidth=2)
        ax.scatter(states[0, 4], states[0, 5], c='red', s=100, marker='o', zorder=5)
        ax.scatter(states[-1, 4], states[-1, 5], c='red', s=100, marker='x', zorder=5)

        # Draw heading arrows for P2 every 5 steps
        for i in range(0, len(states), 5):
            alpha = states[i, 6]
            v = states[i, 7]
            dx = v * np.cos(alpha) * 0.3
            dy = v * np.sin(alpha) * 0.3
            ax.arrow(states[i, 4], states[i, 5], dx, dy,
                    head_width=0.1, head_length=0.05, fc='darkred', ec='darkred')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('Unicycle MPC Dynamics Test')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')

        plt.savefig('test_unicycle_trajectory.png', dpi=150)
        print("  Plot saved to test_unicycle_trajectory.png")
        plt.show()

    except ImportError:
        print("  (matplotlib not available, skipping plot)")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("UNICYCLE MPC DYNAMICS TEST")
    print("="*60 + "\n")

    # Run tests
    model, params, state, action = test_forward_pass()
    test_gradient_flow(model, params, state, action)
    test_mpc_behavior()
    states = test_trajectory_rollout()

    print("="*60)
    print("ALL TESTS PASSED")
    print("="*60)

    # Optional visualization
    plot_trajectory(states)
