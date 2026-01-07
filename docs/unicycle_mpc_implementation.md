# Unicycle MPC Dynamics Implementation

## Overview

This document describes the implementation of unicycle MPC dynamics for the MAX library, based on the "Multi-Agent Active Learning" paper. The key innovation is using implicit differentiation to get gradients through a nonlinear MPC solver, enabling EKF-based online learning of opponent parameters.

## Problem Setup

### Players
- **Player 1 (Evader)**: Double integrator dynamics, controlled by the learning agent
- **Player 2 (Opponent)**: Unicycle dynamics with MPC controller, parameters unknown to Player 1

### State Space (8D)
```
[p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
  │     │    │     │    │     │     │     └── unicycle speed
  │     │    │     │    │     │     └── unicycle heading angle (radians)
  │     │    │     │    │     └── unicycle y position
  │     │    │     │    └── unicycle x position
  │     │    │     └── evader y velocity
  │     │    └── evader x velocity
  │     └── evader y position
  └── evader x position
```

### Learnable Parameters
- `theta1`: Position cost weight in unicycle's MPC objective (how aggressively it tracks)
- `theta2`: Acceleration scaling factor (how responsive the unicycle is)

## Unicycle Dynamics

The unicycle follows these discrete-time dynamics:
```
p2x_next = p2x + dt * v2 * cos(alpha2)
p2y_next = p2y + dt * v2 * sin(alpha2)
alpha2_next = alpha2 + dt * w
v2_next = v2 + dt * a * theta2
```

Where `w` is angular velocity and `a` is acceleration (the MPC control inputs).

## Two-Step MPC Horizon

### Why Two Steps?

With unicycle dynamics, control at step 0 only affects **position at step 2**, not step 1. This is because:
1. At step 0: control `u0 = [w, a]` affects heading and velocity
2. At step 1: the new heading/velocity affect position
3. At step 2: position has finally changed due to step-0 control

A 1-step horizon would have zero gradient for position tracking objectives.

### MPC Cost Function
```python
def mpc_cost(u0, x2_0, target_pos, theta1, theta2):
    x2_2 = rollout_2step(x2_0, u0, theta2)  # Rollout 2 steps
    pos_err = x2_2[0:2] - target_pos
    w, a = u0
    return theta1 * sum(pos_err**2) + weight_w * w**2 + weight_a * a**2
```

Note: `weight_w` and `weight_a` are separate because angular velocity and acceleration have different units.

## Implicit Differentiation

### The Challenge

The MPC solution `u* = argmin_u cost(u, theta)` depends on parameters `theta`. We need gradients `du*/dtheta` for:
1. EKF updates (learning theta from observations)
2. Info-gathering cost (planning to maximize information gain)

Naive approach: differentiate through Newton iterations. Problems:
- Expensive (backprop through many iterations)
- Numerically unstable

### The Solution: Implicit Function Theorem

At the optimum, the KKT conditions hold: `F(u*, theta) = grad_u cost(u*, theta) = 0`

By implicit differentiation:
```
dF/du * du*/dtheta + dF/dtheta = 0
du*/dtheta = -[dF/du]^(-1) @ dF/dtheta
```

This gives exact gradients with just one linear solve, regardless of how many Newton iterations were used.

### JAX Implementation

```python
@jax.custom_vjp
def solve_mpc(x2_0, target_pos, theta):
    """Forward pass: Newton solver."""
    theta1, theta2 = theta
    return newton_solve(x2_0, target_pos, theta1, theta2)

def solve_mpc_fwd(x2_0, target_pos, theta):
    u_star = solve_mpc(x2_0, target_pos, theta)
    return u_star, (u_star, x2_0, target_pos, theta)

def solve_mpc_bwd(res, g):
    """Backward pass: implicit differentiation."""
    u_star, x2_0, target_pos, theta = res
    theta1, theta2 = theta

    # Jacobian of KKT residual w.r.t. u (Hessian of cost)
    J_u = jax.jacobian(kkt_residual, argnums=0)(u_star, x2_0, target_pos, theta1, theta2)
    J_u_reg = J_u + 1e-6 * jnp.eye(2)

    # Solve adjoint system: lambda = J_u^{-T} @ g
    lam = jnp.linalg.solve(J_u_reg.T, g)

    # Gradient w.r.t. theta: -lambda^T @ dF/dtheta
    def kkt_theta(th):
        return kkt_residual(u_star, x2_0, target_pos, th[0], th[1])

    J_theta = jax.jacobian(kkt_theta)(theta)
    dtheta = -J_theta.T @ lam

    return (None, None, dtheta)

solve_mpc.defvjp(solve_mpc_fwd, solve_mpc_bwd)
```

## File Changes

### `max/dynamics.py`
Added `create_unicycle_mpc_dynamics()` function (~180 lines):
- `unicycle_step()`: Single step dynamics
- `rollout_2step()`: Two-step rollout for MPC
- `mpc_cost()`: Cost function with position tracking + control penalties
- `kkt_residual()`: Gradient of cost (optimality condition)
- `newton_solve()`: Newton's method to find optimal control
- `solve_mpc()`: Custom VJP wrapper for implicit differentiation
- `pred_one_step()`: Full dynamics step (evader + unicycle MPC)

### `max/environments.py`
Added `make_unicycle_mpc_env()` function:
- Creates environment with true parameters for simulation
- Reset: random positions, random heading, fixed initial speed
- Step: applies evader action, unicycle uses MPC with true params

### `configs/unicycle.json`
New configuration file with:
- `dynamics_params`: `dt`, `newton_iters`, `init_theta1`, `init_theta2`, `weight_w`, `weight_a`
- `env_params`: `true_theta1`, `true_theta2` (ground truth for simulation)
- Standard EKF trainer, iCEM planner, info-gathering cost settings

### `scripts/test_unicycle_dynamics.py`
Test script with 4 tests:
1. **Forward Pass**: Verifies dynamics run without errors
2. **Gradient Flow**: Tests gradients through implicit differentiation
   - Heading/velocity after 1 step (non-zero gradients)
   - Position after 2 steps (non-zero gradients)
3. **MPC Behavior**: Verifies unicycle tracks target
4. **Trajectory Rollout**: Visualizes multi-step behavior

### `scripts/run_unicycle_comparison.py`
Comparison script (similar to `run_lqr_comparison.py`):
- **Perfect Info**: Evader knows true theta1, theta2
- **No Learning**: Wrong params, no EKF updates
- **Passive Learning**: EKF updates, evasion-only cost
- **Active Learning**: EKF updates + info-gathering cost

## Known Limitations

### Saddle Point Issue
When the unicycle faces exactly away from the target (heading = 180° from target direction), the gradient w.r.t. angular velocity is zero. This is a saddle point where:
- Turning left or right are equally good
- The 2-step horizon MPC may choose to decelerate/reverse instead of turning

This is a limitation of the 2-step horizon, not the implicit differentiation.

### Computational Cost
Each dynamics step requires solving a Newton optimization, making it slower than closed-form LQR dynamics. Future improvements could include:
- Progress bars for long runs
- Warm-starting Newton from previous solution
- Longer MPC horizons (would require more Newton iterations)

## Usage

### Running Tests
```bash
python scripts/test_unicycle_dynamics.py
```

### Running Comparison
```bash
python scripts/run_unicycle_comparison.py --seed 42 --save-dir ./comparison_results
```

### Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_theta1` | Initial guess for position cost weight | 1.0 |
| `init_theta2` | Initial guess for acceleration scaling | 1.0 |
| `true_theta1` | True position cost weight (env only) | 5.0 |
| `true_theta2` | True acceleration scaling (env only) | 1.0 |
| `weight_w` | Penalty on angular velocity | 0.1 |
| `weight_a` | Penalty on acceleration | 1.0 |
| `newton_iters` | Newton iterations for MPC | 10 |
| `dt` | Time step | 0.1 |

## Session Progress (Jan 4, 2026)

### Bug Found: Unbounded Speed
When running the full comparison, the unicycle speed blew up to 70+ because there was no speed regulation. The evader also had unbounded velocity.

**Fix Applied**: Added speed penalty to MPC cost function:
```python
return (theta1 * jnp.sum(pos_err**2)
        + weight_w * w**2
        + weight_a * a**2
        + weight_speed * (v2_final - target_speed)**2)
```

New config parameters added:
- `weight_speed`: Penalty for deviating from target speed (default: 5.0)
- `target_speed`: Desired cruising speed (default: 1.0)

This avoids hard clipping which would cause gradient degeneracy.

### New Scripts Added

#### `scripts/analyze_trajectory.py`
Prints sampled trajectory data for debugging:
```bash
# Analyze all cases (sampled across full trajectory)
python scripts/analyze_trajectory.py --steps 40

# Filter to one case
python scripts/analyze_trajectory.py --case active --steps 50
```

#### `scripts/animate_trajectory.py`
Creates animated GIFs from saved trajectories:
```bash
# Animate one case
python scripts/animate_trajectory.py --case perfect_info

# Animate all cases
python scripts/animate_trajectory.py

# Customize fps and trail length
python scripts/animate_trajectory.py --case active_learning --fps 30 --trail 50
```

### Current Status
- Speed penalty added but **not yet tested**
- Trajectories may still be "boring" - need to investigate further

### Next Steps / TODO

1. **Re-run comparison with speed penalty**:
   ```bash
   python scripts/run_unicycle_comparison.py --seed 42
   ```

2. **Create animations to visualize behavior**:
   ```bash
   python scripts/animate_trajectory.py
   ```

3. **Analyze trajectory data**:
   ```bash
   python scripts/analyze_trajectory.py --steps 40
   ```

4. **If still boring, investigate**:
   - Is the evader moving much? (check `weight_control` - currently 10.0, might be too high)
   - Is the unicycle turning? (check heading changes)
   - Are the initial positions interesting enough?

5. **Add progress bars** to `run_unicycle_comparison.py` (still TODO - use `tqdm`)

6. **Tune parameters if needed**:
   - Lower `weight_control` to encourage evader movement
   - Adjust `weight_speed` / `target_speed` for unicycle
   - Try different `true_theta1` / `init_theta1` gaps

### Updated Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `init_theta1` | Initial guess for position cost weight | 1.0 |
| `init_theta2` | Initial guess for acceleration scaling | 1.0 |
| `true_theta1` | True position cost weight (env only) | 5.0 |
| `true_theta2` | True acceleration scaling (env only) | 1.0 |
| `weight_w` | Penalty on angular velocity | 0.1 |
| `weight_a` | Penalty on acceleration | 1.0 |
| `weight_speed` | Penalty on speed deviation | 5.0 |
| `target_speed` | Desired unicycle speed | 1.0 |
| `newton_iters` | Newton iterations for MPC | 10 |
| `dt` | Time step | 0.1 |

## References

- Multi-Agent Active Learning paper (directed information formulation)
- Implicit differentiation through optimization (Amos & Kolter, OptNet)
- JAX custom_vjp documentation
