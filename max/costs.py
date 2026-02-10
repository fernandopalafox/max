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


def make_task_aware_info_term(
    dynamics_fn: Callable,
    stage_cost_fn: Callable,
) -> Callable:
    """
    Creates an task-aware information gathering (exploration bonus) term.

    Args:
        dynamics_fn: Dynamics prediction function (params, state, action) -> next_state
        stage_cost_fn: Stage cost function (state, action) -> scalar

    Returns:
        Function with signature (state, control, dyn_params, params_cov) -> info_gain
    """

    @jax.jit
    def info_term_fn(state, control, dyn_params, params_cov):
        """
        Computes task-aware information term for the given state-action pair.

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

        # 4. Compute gradient of stage cost w.r.t. next state
        # TODO: A more accurate version would compute the gradient at the predicted next state
        grad_cost = jax.grad(stage_cost_fn, argnums=0)(
            state, control
        )

        # 5. Combine with Jacobians
        task_aware_info_cost = jnp.log(grad_cost.T @ jac_dyn @ params_cov @ jac_dyn.T @ grad_cost + 1e-6)

        return task_aware_info_cost

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


def _stage_cost_pendulum_swing_up(
    state, control, cost_params, weight_control, weight_info, info_term_fn
):
    """
    Stage cost for pendulum swing-up: goal at phi=pi with zero velocity.

    Args:
        state: [phi, phi_dot]
        control: [tau]
        cost_params: Dict with 'dyn_params' and 'params_cov_model'
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        info_term_fn: Information term function (can be None)

    Returns:
        Stage cost scalar
    """
    phi, phi_dot = state[0], state[1]

    # Swing-up cost: (phi - pi)^2 + 0.1 * phi_dot^2
    goal_cost = (phi - jnp.pi) ** 2 + 0.1 * phi_dot ** 2

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Information gathering bonus (optional)
    exploration_term = 0.0
    if info_term_fn is not None:
        exploration_term = -weight_info * info_term_fn(
            state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
        )

    return goal_cost + control_cost + exploration_term


def _terminal_cost_pendulum_swing_up(state):
    """Terminal cost: emphasize reaching inverted position."""
    phi, phi_dot = state[0], state[1]
    return (phi - jnp.pi) ** 2 + 0.1 * phi_dot ** 2


def _stage_cost_drone_state_tracking_w_info(
    state, control, cost_params, goal_state, state_weights, weight_control, weight_info, info_term_fn
):
    """
    Stage cost for drone state tracking: weighted state error + control penalty - info bonus.

    Args:
        state: [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        control: [T_1, T_2] (2,)
        cost_params: Dict with 'dyn_params' and 'params_cov_model'
        goal_state: Target state [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        state_weights: Weights for each state dimension (6,)
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        info_term_fn: Information term function (can be None)

    Returns:
        Stage cost scalar
    """
    # Weighted state tracking error
    state_error = jnp.sum(state_weights * (state - goal_state) ** 2)

    # Control penalty (penalize large thrusts)
    control_cost = weight_control * jnp.sum(control ** 2)

    # Information gathering bonus (optional)
    exploration_term = 0.0
    exploration_term = -weight_info * info_term_fn(
        state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
    )

    return state_error + control_cost + exploration_term

def _stage_cost_drone_state_tracking(
    state, control, goal_state, state_weights, weight_control
):
    """
    Stage cost for drone state tracking: weighted state error + control penalty.

    Args:
        state: [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        control: [T_1, T_2] (2,)
        goal_state: Target state [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        state_weights: Weights for each state dimension (6,)
        weight_control: Control penalty weight

    Returns:    
        Stage cost scalar
    """
    # Weighted state tracking error
    state_error = jnp.sum(state_weights * (state - goal_state) ** 2)

    # Control penalty (penalize large thrusts)
    control_cost = weight_control * jnp.sum(control ** 2)

    return state_error + control_cost


def _terminal_cost_drone_state_tracking(state, goal_state, state_weights):
    """Terminal cost: weighted squared state error to goal."""
    return jnp.sum(state_weights * (state - goal_state) ** 2)


def _stage_cost_merging_idm(
    state, control, cost_params,
    Q_diag, q_I, p_y_target, v_g, L, lane_width,
    weight_control, weight_info, info_term_fn,
):
    """
    Stage cost for merging IDM (1 vehicle): quadratic tracking
    + indicator penalties + info gathering.
    """
    # Goal: [0, p_y_target, v_g, 0, 0, 0]
    goal = jnp.zeros(6).at[1].set(p_y_target).at[2].set(v_g)

    # Quadratic tracking
    err = state - goal
    tracking_cost = jnp.sum(Q_diag * err**2)

    # Control penalty
    control_cost = weight_control * jnp.sum(control**2)

    # Indicator penalties
    ego_px, ego_py = state[0], state[1]
    idm_px = state[4]

    # 1. Collision with the single IDM vehicle
    lon_overlap = (jnp.abs(ego_px - idm_px) < L)
    lat_overlap = (
        jnp.abs(ego_py - p_y_target) < lane_width
    )
    collision = jnp.logical_and(
        lon_overlap, lat_overlap
    ).astype(jnp.float32)

    # 2. Road boundary
    road = jnp.logical_or(
        ego_py < -7.0, ego_py > 3.5
    ).astype(jnp.float32)

    # 3. Invalid merge: ego in target lane but behind IDM
    merged = (jnp.abs(ego_py - p_y_target) < 1.0)
    behind = (ego_px < idm_px)
    invalid = jnp.logical_and(
        merged, behind
    ).astype(jnp.float32)

    indicator_cost = q_I * (collision + road + invalid)

    # Information gathering term
    exploration_term = 0.0
    if info_term_fn is not None:
        exploration_term = -weight_info * info_term_fn(
            state, control,
            cost_params["dyn_params"],
            cost_params["params_cov_model"],
        )

    return (
        tracking_cost + control_cost
        + indicator_cost + exploration_term
    )


def _terminal_cost_merging_idm(
    state, Qf_diag, q_I, p_y_target, v_g, L, lane_width,
):
    """
    Terminal cost for merging IDM (1 vehicle).
    """
    goal = jnp.zeros(6).at[1].set(p_y_target).at[2].set(v_g)
    err = state - goal
    terminal_tracking = jnp.sum(Qf_diag * err**2)

    ego_px, ego_py = state[0], state[1]
    idm_px = state[4]

    lon_overlap = (jnp.abs(ego_px - idm_px) < L)
    lat_overlap = (
        jnp.abs(ego_py - p_y_target) < lane_width
    )
    collision = jnp.logical_and(
        lon_overlap, lat_overlap
    ).astype(jnp.float32)

    road = jnp.logical_or(
        ego_py < -7.0, ego_py > 3.5
    ).astype(jnp.float32)

    merged = (jnp.abs(ego_py - p_y_target) < 1.0)
    behind = (ego_px < idm_px)
    invalid = jnp.logical_and(
        merged, behind
    ).astype(jnp.float32)

    indicator_cost = q_I * (collision + road + invalid)

    return terminal_tracking + indicator_cost


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

    elif cost_type == "pendulum_swing_up_info":
        # Extract params
        params = config["cost_fn_params"]
        weight_jerk = params.get("weight_jerk", 0.0)
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        # Vectorize stage cost over the horizon
        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_pendulum_swing_up(
                s, u, cp, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory cost for pendulum swing-up.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage costs (on states 0 to T-1)
            stage_costs = stage_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal cost (on state T)
            terminal = _terminal_cost_pendulum_swing_up(states[-1])

            # 4. Calculate Jerk Cost (on controls)
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn

    elif cost_type == "merging_idm_info":
        params_cfg = config["cost_fn_params"]
        weight_jerk = params_cfg.get("weight_jerk", 0.0)
        weight_control = params_cfg["weight_control"]
        weight_info = params_cfg["weight_info"]
        meas_noise_diag = jnp.array(params_cfg["meas_noise_diag"])

        q_vs = params_cfg["q_vs"]
        q_vd = params_cfg["q_vd"]
        q_s = params_cfg["q_s"]
        q_d = params_cfg["q_d"]
        q_f_d = params_cfg["q_f_d"]
        q_I = params_cfg["q_I"]
        v_g = params_cfg["v_g"]
        p_y_target = params_cfg["p_y_target"]
        L_cost = params_cfg["L"]
        lane_width = params_cfg.get("lane_width", 2.0)

        Q_diag = jnp.array([
            q_s, q_d, q_vs, q_vd, 0., 0.,
        ])
        Qf_diag = jnp.array([
            0., q_f_d, 0., 0., 0., 0.,
        ])

        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_merging_idm(
                s, u, cp,
                Q_diag, q_I, p_y_target, v_g,
                L_cost, lane_width,
                weight_control, weight_info, info_term_fn,
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            states = _rollout(
                init_state, controls, cost_params,
                dynamics_model.pred_one_step,
            )
            stage_costs = stage_cost_vmap(
                states[:-1], controls, cost_params,
            )
            terminal = _terminal_cost_merging_idm(
                states[-1], Qf_diag, q_I,
                p_y_target, v_g, L_cost, lane_width,
            )

            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(
                    control_diffs**2
                )

            return (
                jnp.sum(stage_costs) + terminal + jerk
            )

        return cost_fn

    elif cost_type == "drone_state_tracking_info":
        # Extract params
        params = config["cost_fn_params"]
        weight_jerk = params.get("weight_jerk", 0.0)
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        goal_state = jnp.array(params["goal_state"])
        state_weights = jnp.array(params["state_weights"])
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        # Vectorize stage cost over the horizon
        stage_cost_vmap = jax.vmap(
            lambda s, u, cp: _stage_cost_drone_state_tracking_w_info(
                s, u, cp, goal_state, state_weights, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory cost for drone state tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage costs (on states 0 to T-1)
            stage_costs = stage_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal cost (on state T)
            terminal = _terminal_cost_drone_state_tracking(states[-1], goal_state, state_weights)

            # 4. Calculate Jerk Cost (on controls)
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn
    
    elif cost_type == "drone_state_tracking_info_task_aware":
        # Extract params
        params = config["cost_fn_params"]
        weight_jerk = params.get("weight_jerk", 0.0)
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        goal_state = jnp.array(params["goal_state"])
        state_weights = jnp.array(params["state_weights"])
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Vectorize stage cost over the horizon
        stage_cost_fn = lambda s, u: _stage_cost_drone_state_tracking(
            s, u, goal_state, state_weights, weight_control
        )
        stage_cost_vmap = jax.vmap(
            stage_cost_fn,
            in_axes=(0, 0),
        )

        # Create the task-aware info cost
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_task_aware_info_term(
                dynamics_model.pred_one_step,
                stage_cost_fn
            )
        else:
            info_term_fn = lambda s, u, dyn_p, cov: 0.0

        info_cost_vmap = jax.vmap(
            lambda s, u, cp: info_term_fn(
                s, u, cp["dyn_params"], cp["params_cov_model"]
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def cost_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory cost for drone state tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage costs (on states 0 to T-1)
            stage_costs = stage_cost_vmap(states[:-1], controls) + weight_info * info_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal cost (on state T)
            terminal = _terminal_cost_drone_state_tracking(states[-1], goal_state, state_weights)

            # 4. Calculate Jerk Cost (on controls)
            jerk = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_costs) + terminal + jerk

        return cost_fn

    else:
        raise ValueError(f"Unknown cost type: '{cost_type}'")
