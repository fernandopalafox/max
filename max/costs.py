"""
Minimal functional cost abstractions for max library.

Provides factory functions that return JIT-compiled closures for common cost patterns.
Follows the max library philosophy: pure functions, closures, and NamedTuples.
"""

import jax
import jax.numpy as jnp
from typing import Callable

# --- Helpers for info gathering term ---


def make_info_gathering_term(
    dynamics_fn: Callable,
    meas_noise_diag: jnp.ndarray,
) -> Callable:
    """
    Creates an information gathering (exploration bonus) term.

    Args:
        dynamics_fn: Dynamics prediction function (params, state, action) -> next_state
        meas_noise_diag: Diagonal of measurement noise covariance

    Returns:
        Function with signature (state, control, dyn_params, params_cov) -> info_gain
    """
    # Pre-compute log determinant and covariance of measurement noise
    log_det_meas_noise = jnp.sum(jnp.log(meas_noise_diag))
    meas_noise_cov = jnp.diag(meas_noise_diag)

    @jax.jit
    def info_term_fn(state, control, dyn_params, params_cov):
        """
        Computes information gain for the given state-action pair.

        Args:
            state: Current state
            control: Current control/action
            dyn_params: Dictionary containing "model" and "normalizer" params
            params_cov: Parameter covariance matrix

        Returns:
            Information gain scalar (higher = more informative)
        """

        # 1. Flatten only the learnable model parameters
        flat_model_params, unflatten_fn = jax.flatten_util.ravel_pytree(
            dyn_params["model"]
        )

        # 2. Define prediction wrapper that isolates model params from normalizer
        def pred_flat(flat_p, s, a, norm_p):
            unflat_p = unflatten_fn(flat_p)
            full_params = {"model": unflat_p, "normalizer": norm_p}
            return dynamics_fn(full_params, s, a)

        # 3. Compute Jacobian w.r.t. flattened model parameters (arg 0)
        jac_dyn = jax.jacobian(pred_flat, argnums=0)(
            flat_model_params, state, control, dyn_params["normalizer"]
        )

        # 4. Propagate parameter uncertainty to state space
        # Σ_pred = J * Σ_params * J^T + R_meas
        pred_cov = jac_dyn @ params_cov @ jac_dyn.T + meas_noise_cov

        # 5. Compute information gain (differential entropy term)
        # 0.5 * (log(det(Σ_pred)) - log(det(R_meas)))
        info_gain = 0.5 * (jnp.log(jnp.linalg.det(pred_cov)) - log_det_meas_noise)

        return info_gain

    return info_term_fn


def _stage_cost_info_gathering(
    state, control, cost_params, weight_control, weight_info, info_term_fn
):
    """
    Private helper for info gathering stage cost.
    Matches stage_cost in run_dogfight.py
    """
    # Hardcoded indices for pursuit-evasion
    p_evader = state[0:2]
    p_pursuer = state[4:6]

    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    control_cost = weight_control * jnp.sum(control**2)

    # Exploration bonus (subtracted because planners minimize cost)
    # Passed params match the structure in run_lqr.py
    exploration_term = -weight_info * info_term_fn(
        state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
    )

    # Penalize distance from origin
    origin_cost = 1.0 * jnp.sum(p_evader**2)

    return -dist_sq + control_cost + exploration_term + origin_cost


def _terminal_cost_pursuit_evasion(state):
    """
    Private helper for pursuit-evasion terminal cost.
    """
    p_evader = state[0:2]
    p_pursuer = state[4:6]
    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    return -dist_sq


def _rollout(init_state, controls, cost_params, pred_fn):
    """
    Private helper for trajectory rollout using dynamics model.
    """

    def step(carry, control):
        state, dyn_params = carry
        next_state = pred_fn(dyn_params, state, control)
        return (next_state, dyn_params), next_state

    dyn_params = cost_params["dyn_params"]
    _, states = jax.lax.scan(step, (init_state, dyn_params), controls)

    # Concatenate init_state to match dogfight shape requirements
    return jnp.concatenate([init_state[jnp.newaxis, :], states], axis=0)


def _stage_cost_linear_tracking(
    state, control, cost_params, target_point, weight_control, weight_info, info_term_fn
):
    """
    Stage cost for linear tracking: tracking error + control penalty - info bonus.

    Args:
        state: Current state (4,)
        control: Current action (2,)
        cost_params: Dict with 'dyn_params' and 'params_cov_model'
        target_point: Target state (4,)
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        info_term_fn: Information term function

    Returns:
        Stage cost scalar
    """
    # Tracking error (squared distance to target)
    tracking_error = jnp.sum((state - target_point) ** 2)

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Information bonus (subtracted because planners minimize cost)
    exploration_term = -weight_info * info_term_fn(
        state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
    )

    return tracking_error + control_cost + exploration_term


def _terminal_cost_linear_tracking(state, target_point):
    """Terminal cost: squared distance to target."""
    return jnp.sum((state - target_point) ** 2)


# --- Main factory function ---


def init_cost(config, dynamics_model):
    """
    Initializes cost function based on configuration.

    Args:
        config: Configuration dictionary containing cost_fn_params.
        dynamics_model: Object with .pred_one_step method.

    Returns:
        Cost function with signature (init_state, controls, cost_params) -> scalar
    """
    cost_type = config.get("cost_type", "info_gathering")
    print(f"🚀 Initializing cost function: {cost_type.upper()}")

    if cost_type == "info_gathering":
        # Extract params
        params = config["cost_fn_params"]
        weight_jerk = params["weight_jerk"]
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = make_info_gathering_term(
            dynamics_model.pred_one_step, meas_noise_diag
        )

        # Vectorize stage cost over the horizon
        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_info_gathering(
                s, u, cp, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory cost.
            Matches traj_cost in run_dogfight.py
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage costs (on states 0 to T-1)
            stage_costs = stage_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal cost (on state T)
            terminal = _terminal_cost_pursuit_evasion(states[-1])

            # 4. Calculate Jerk Cost (on controls)
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                # Sum over dimensions, then sum over time
                jerk = weight_jerk * jnp.sum(control_diffs**2)

            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn

    elif cost_type == "linear_tracking_info":
        # Extract params
        params = config["cost_fn_params"]
        weight_jerk = params["weight_jerk"]
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        target_point = jnp.array(params["target_point"])
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = make_info_gathering_term(
            dynamics_model.pred_one_step, meas_noise_diag
        )

        # Vectorize stage cost over the horizon
        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_linear_tracking(
                s, u, cp, target_point, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory cost for linear tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage costs (on states 0 to T-1)
            stage_costs = stage_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal cost (on state T)
            terminal = _terminal_cost_linear_tracking(states[-1], target_point)

            # 4. Calculate Jerk Cost (on controls)
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(control_diffs**2)

            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn

    else:
        raise ValueError(f"Unknown cost type: '{cost_type}'")
