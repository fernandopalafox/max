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

    Mathematical background:
        - Information gain: I(θ; s_{t+1} | s_t, a_t)
        - For Gaussians: 0.5 * [log(det(Σ_pred)) - log(det(R_meas))]
        - Where Σ_pred = J_θ * Σ_params * J_θ^T + R_meas
        - J_θ is the Jacobian of dynamics w.r.t. parameters

    Args:
        dynamics_fn: Dynamics prediction function (params, state, action) -> next_state
        meas_noise_diag: Diagonal of measurement noise covariance

    Returns:
        Function with signature (state, control, dyn_params, params_cov) -> info_gain
    """
    # Pre-compute log determinant of measurement noise
    log_det_meas_noise = jnp.sum(jnp.log(meas_noise_diag))
    meas_noise_cov = jnp.diag(meas_noise_diag)

    @jax.jit
    def info_term_fn(state, control, dyn_params, params_cov):
        """
        Computes information gain for the given state-action pair.

        Args:
            state: Current state
            control: Current control/action
            dyn_params: Current dynamics model parameters
            params_cov: Parameter covariance matrix (from EKF or other estimator)

        Returns:
            Information gain scalar (higher = more informative)
        """
        if params_cov is None:
            return 0.0

        # Flatten parameters for Jacobian computation
        flat_params, unflatten_fn = jax.flatten_util.ravel_pytree(dyn_params["model"])

        # Define prediction function with flattened parameters
        def pred_flat(flat_p, s, a):
            unflat_p = unflatten_fn(flat_p)
            full_params = {"model": unflat_p, "normalizer": dyn_params["normalizer"]}
            return dynamics_fn(full_params, s, a)

        # Compute Jacobian w.r.t. parameters
        jac_params = jax.jacobian(pred_flat, argnums=0)(flat_params, state, control)

        # Propagate parameter uncertainty to state space
        # Σ_pred = J * Σ_params * J^T + R_meas
        pred_cov = jac_params @ params_cov @ jac_params.T + meas_noise_cov

        # Compute information gain (differential entropy)
        info_gain = 0.5 * (jnp.log(jnp.linalg.det(pred_cov)) - log_det_meas_noise)

        return info_gain

    return info_term_fn

# Note: these helpers are unique to pursuit-evasion with info gathering cost

def _stage_cost_info_gathering(state, control, cost_params, weight_control, weight_info, info_term_fn):
    """Private helper for info gathering stage cost."""
    p_evader = state[0:2]
    p_pursuer = state[4:6]
    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    control_cost = weight_control * jnp.sum(control**2)
    exploration_term = -weight_info * info_term_fn(
        state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
    )
    return -dist_sq + control_cost + exploration_term


def _terminal_cost_pursuit_evasion(state):
    """Private helper for pursuit-evasion terminal cost."""
    p_evader = state[0:2]
    p_pursuer = state[4:6]
    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    return -dist_sq


def _rollout(init_state, controls, cost_params, pred_fn):
    """Private helper for trajectory rollout using dynamics model."""
    def step(carry, control):
        state, dyn_params = carry
        next_state = pred_fn(dyn_params, state, control)
        return (next_state, dyn_params), next_state

    dyn_params = cost_params["dyn_params"]
    _, states = jax.lax.scan(step, (init_state, dyn_params), controls)
    return jnp.concatenate([init_state[jnp.newaxis, :], states], axis=0)


# --- Main factory function ---

def init_cost(config, dynamics_model):
    """
    Initializes cost function based on configuration.

    This is the main factory function that follows the max library pattern
    (similar to init_env, init_dynamics, init_planner).

    Args:
        config: Configuration dictionary containing:
            - cost_type: Type of cost ("info_gathering", etc.)
            - cost_fn_params: Cost-specific parameters
        dynamics_model: Dynamics model containing pred_one_step function

    Returns:
        Cost function with signature (init_state, controls, cost_params) -> scalar

    Example:
        >>> cost_fn = init_cost(config, dynamics_model)
        >>> cost = cost_fn(init_state, controls, cost_params)
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

        # Info and stage costs
        info_term_fn = make_info_gathering_term(
            dynamics_model.pred_one_step,
            meas_noise_diag
        )
        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_info_gathering(s, u, cp, weight_control, weight_info, info_term_fn),
            in_axes=(0, 0, None)
        )

        # Total cost function
        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            states = _rollout(init_state, controls, cost_params, dynamics_model.pred_one_step)
            stage_costs = stage_cost_vmap(states[:-1], controls, cost_params)
            terminal = _terminal_cost_pursuit_evasion(states[-1])
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(control_diffs**2)
            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn
    else:
        raise ValueError(f"Unknown cost type: '{cost_type}'")
