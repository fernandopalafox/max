"""
Minimal functional reward abstractions for max library.

Provides factory functions that return JIT-compiled closures for common reward patterns.
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

        normalizing_param = params_cov.shape[0]

        return info_gain / normalizing_param

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
        grad_cost = jax.grad(stage_cost_fn, argnums=0)(
            state, control
        )

        # 5. Combine with Jacobians
        task_aware_info_cost = jnp.log(grad_cost.T @ jac_dyn @ params_cov @ jac_dyn.T @ grad_cost + 1e-6)

        return task_aware_info_cost

    return info_term_fn



def _stage_reward_info_gathering(
    state, control, cost_params, weight_control, weight_info, info_term_fn
):
    """
    Private helper for info gathering stage reward.
    reward = dist_sq - control_cost + weight_info*info - origin_cost
    """
    # Hardcoded indices for pursuit-evasion
    p_evader = state[0:2]
    p_pursuer = state[4:6]

    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    control_cost = weight_control * jnp.sum(control**2)

    # Information gathering bonus
    exploration_term = weight_info * info_term_fn(
        state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
    )

    # Penalize distance from origin
    origin_cost = 1.0 * jnp.sum(p_evader**2)

    return dist_sq - control_cost + exploration_term - origin_cost


def _terminal_reward_pursuit_evasion(state):
    """
    Private helper for pursuit-evasion terminal reward.
    """
    p_evader = state[0:2]
    p_pursuer = state[4:6]
    dist_sq = jnp.sum((p_evader - p_pursuer) ** 2)
    return dist_sq


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


def _stage_reward_linear_tracking(
    state, control, cost_params, weight_control, weight_info, info_term_fn
):
    """
    Stage reward for linear tracking: -tracking_error - control_penalty + info_bonus.

    Args:
        state: Current state (4,)
        control: Current action (2,)
        cost_params: Dict with 'dyn_params', 'params_cov_model', and 'goal_state'
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        info_term_fn: Information term function

    Returns:
        Stage reward scalar
    """
    goal_state = cost_params["goal_state"]

    # Tracking error (squared distance to target)
    tracking_error = jnp.sum((state - goal_state) ** 2)

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Information gathering bonus
    if info_term_fn is not None:
        exploration_term = weight_info * info_term_fn(
            state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
        )
    else:
        exploration_term = 0.0

    return -tracking_error - control_cost + exploration_term


def _terminal_reward_linear_tracking(state, cost_params):
    """Terminal reward: negative squared distance to target."""
    goal_state = cost_params["goal_state"]
    return -jnp.sum((state - goal_state) ** 2)


def _stage_reward_pendulum_swing_up(
    state, control, cost_params, weight_control, weight_info, info_term_fn
):
    """
    Stage reward for pendulum swing-up: goal at phi=pi with zero velocity.

    Args:
        state: [phi, phi_dot]
        control: [tau]
        cost_params: Dict with 'dyn_params' and 'params_cov_model'
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        info_term_fn: Information term function (can be None)

    Returns:
        Stage reward scalar
    """
    phi, phi_dot = state[0], state[1]

    # Swing-up cost: (phi - pi)^2 + 0.1 * phi_dot^2
    goal_cost = (phi - jnp.pi) ** 2 + 0.1 * phi_dot ** 2

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Information gathering bonus (optional)
    exploration_term = 0.0
    if info_term_fn is not None:
        exploration_term = weight_info * info_term_fn(
            state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
        )

    return -goal_cost - control_cost + exploration_term


def _terminal_reward_pendulum_swing_up(state):
    """Terminal reward: emphasize reaching inverted position."""
    phi, phi_dot = state[0], state[1]
    return -((phi - jnp.pi) ** 2 + 0.1 * phi_dot ** 2)


def _stage_reward_drone_state_tracking_w_info(
    state, control, cost_params, state_weights, weight_control, weight_info, weight_ground, info_term_fn
):
    """
    Stage reward for drone state tracking: -weighted_error - control - ground + info.

    Args:
        state: [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        control: [T_1, T_2] (2,)
        cost_params: Dict with 'dyn_params', 'params_cov_model', and 'goal_state'
        state_weights: Weights for each state dimension (6,)
        weight_control: Control penalty weight
        weight_info: Information gathering weight
        weight_ground: Ground penalty weight
        info_term_fn: Information term function (can be None)

    Returns:
        Stage reward scalar
    """
    goal_state = cost_params["goal_state"]

    # Weighted state tracking error
    state_error = jnp.sum(state_weights * (state - goal_state) ** 2)

    # Control penalty (penalize large thrusts)
    control_cost = weight_control * jnp.sum(control ** 2)

    # Ground penalty (y=0)
    ground_penalty = weight_ground * jnp.exp(-5.0 * state[1])

    # Information gathering bonus
    if info_term_fn is not None:
        exploration_term = weight_info * info_term_fn(
            state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
        )
    else:
        exploration_term = 0.0

    return -state_error - control_cost - ground_penalty + exploration_term

def _stage_reward_drone_state_tracking(
    state, control, goal_state, state_weights, weight_control, weight_ground
):
    """
    Stage reward for drone state tracking: -weighted_error - control - ground.

    Args:
        state: [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        control: [T_1, T_2] (2,)
        goal_state: Target state [p_x, p_y, phi, v_x, v_y, phi_dot] (6,)
        state_weights: Weights for each state dimension (6,)
        weight_control: Control penalty weight

    Returns:
        Stage reward scalar
    """
    # Weighted state tracking error
    state_error = jnp.sum(state_weights * (state - goal_state) ** 2)

    # Control penalty (penalize large thrusts)
    control_cost = weight_control * jnp.sum(control ** 2)

    # Ground penalty (y=0)
    ground_penalty = weight_ground * jnp.exp(-5.0 * state[1])

    return -state_error - control_cost - ground_penalty


def _terminal_reward_drone_state_tracking(state, cost_params, state_weights):
    """Terminal reward: negative weighted squared state error to goal."""
    goal_state = cost_params["goal_state"]
    return -jnp.sum(state_weights * (state - goal_state) ** 2)


def _stage_reward_merging_idm(
    state, control, cost_params,
    Q_diag, q_I, p_y_target, v_g, L, lane_width,
    weight_control, weight_info, info_term_fn,
):
    """
    Stage reward for merging IDM (1 vehicle): -tracking - ctrl - indicator + info.
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
        exploration_term = weight_info * info_term_fn(
            state, control,
            cost_params["dyn_params"],
            cost_params["params_cov_model"],
        )

    return (
        -tracking_cost - control_cost
        - indicator_cost + exploration_term
    )


def _terminal_reward_merging_idm(
    state, Qf_diag, q_I, p_y_target, v_g, L, lane_width,
):
    """
    Terminal reward for merging IDM (1 vehicle).
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

    return -terminal_tracking - indicator_cost


def _stage_reward_cheetah_velocity_tracking(
    data,  # mjx.Data
    control: jnp.ndarray,
    weight_control: float,
    heading_penalty_factor: float,
) -> float:
    """
    Stage reward for cheetah velocity maximization with flip penalty.

    Reward = forward_vel - weight_control * ||u||^2 - flip_penalty

    Args:
        data: mjx.Data (full MuJoCo physics state)
        control: 6D action torques
        weight_control: Control penalty weight
        heading_penalty_factor: Penalty for flipping (applied when |root_angle| > pi/2)

    Returns:
        Scalar reward
    """
    # Forward velocity is qvel[0] from mjx.Data
    forward_vel = data.qvel[0]

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Flip penalty: penalize when root angle exceeds pi/2
    root_angle = data.qpos[2]
    flip_penalty = heading_penalty_factor * (
        (root_angle > jnp.pi / 2).astype(jnp.float32) +
        (root_angle < -jnp.pi / 2).astype(jnp.float32)
    )

    return forward_vel - control_cost - flip_penalty


def _terminal_reward_cheetah_velocity_tracking(
    data,  # mjx.Data
) -> float:
    """Terminal reward: forward velocity."""
    forward_vel = data.qvel[0]
    return forward_vel


def _rollout_cheetah(init_data, controls, cost_params, pred_fn):
    """
    Trajectory rollout for cheetah using mjx.Data directly.

    Returns:
        data_sequence: Pytree of mjx.Data with leading time dimension
    """
    def step(carry, control):
        data, dyn_params = carry
        next_data = pred_fn(dyn_params, data, control)
        return (next_data, dyn_params), next_data

    dyn_params = cost_params["dyn_params"]
    _, data_sequence = jax.lax.scan(step, (init_data, dyn_params), controls)

    return data_sequence


def _stage_reward_cheetah_velocity_learned(
    state: jnp.ndarray,  # 17D state vector
    control: jnp.ndarray,
    weight_control: float,
    heading_penalty_factor: float,
) -> float:
    """
    Stage reward for cheetah velocity maximization using 17D state vector.

    17D state layout:
    - [0:8]: positions (rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot)
    - [8:17]: velocities (vel_x, vel_z, vel_y, vel_bthigh, vel_bshin, vel_bfoot,
                          vel_fthigh, vel_fshin, vel_ffoot)

    Reward = forward_vel - weight_control * ||u||^2 - flip_penalty
    """
    # Forward velocity is vel_x at index 8
    forward_vel = state[8]

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Flip penalty: penalize when root angle exceeds pi/2
    # Root angle (rooty) is at index 1
    root_angle = state[1]
    flip_penalty = heading_penalty_factor * (
        (root_angle > jnp.pi / 2).astype(jnp.float32) +
        (root_angle < -jnp.pi / 2).astype(jnp.float32)
    )

    return forward_vel - control_cost - flip_penalty


def _terminal_reward_cheetah_velocity_learned(
    state: jnp.ndarray,  # 17D state vector
) -> float:
    """Terminal reward: forward velocity."""
    forward_vel = state[8]  # vel_x at index 8
    return forward_vel


def _stage_reward_cheetah_velocity_learned_w_info(
    state: jnp.ndarray,
    control: jnp.ndarray,
    cost_params: dict,
    weight_control: float,
    heading_penalty_factor: float,
    weight_info: float,
    info_term_fn,
) -> float:
    """
    Stage reward for cheetah velocity maximization with info gathering bonus.

    17D state layout:
    - [0:8]: positions (rootz, rooty, bthigh, bshin, bfoot, fthigh, fshin, ffoot)
    - [8:17]: velocities (vel_x, vel_z, vel_y, vel_bthigh, vel_bshin, vel_bfoot,
                          vel_fthigh, vel_fshin, vel_ffoot)
    """
    # Forward velocity is vel_x at index 8
    forward_vel = state[8]

    # Control penalty
    control_cost = weight_control * jnp.sum(control ** 2)

    # Flip penalty
    root_angle = state[1]
    flip_penalty = heading_penalty_factor * (
        (root_angle > jnp.pi / 2).astype(jnp.float32) +
        (root_angle < -jnp.pi / 2).astype(jnp.float32)
    )

    # Info gathering bonus
    exploration_term = 0.0
    if info_term_fn is not None:
        exploration_term = weight_info * info_term_fn(
            state, control, cost_params["dyn_params"], cost_params["params_cov_model"]
        )

    return forward_vel - control_cost - flip_penalty + exploration_term


# --- Evaluation reward helpers ---


def _eval_goal_reward(state, control, cost_params, state_weights, weight_control):
    """
    Single-step goal tracking reward (no rollout).

    Args:
        state: Current state
        control: Current control action
        cost_params: Dict containing 'goal_state'
        state_weights: Weights for each state dimension
        weight_control: Control penalty weight

    Returns:
        Scalar reward for this step
    """
    goal_state = cost_params["goal_state"]
    state_error = jnp.sum(state_weights * (state - goal_state) ** 2)
    control_cost = weight_control * jnp.sum(control ** 2)
    return -state_error - control_cost


def _eval_terminal_goal_reward(state, cost_params):
    """
    Terminal distance from goal (position only), negated.

    Args:
        state: Final state
        cost_params: Dict containing 'goal_state'

    Returns:
        Negative euclidean distance from goal position
    """
    goal_state = cost_params["goal_state"]
    position = state[:2]
    goal_position = goal_state[:2]
    return -jnp.sqrt(jnp.sum((position - goal_position) ** 2))


# --- Main factory function ---


def init_reward(config, dynamics_model):
    """
    Initializes reward function based on configuration.

    Args:
        config: Configuration dictionary containing reward_fn_params.
        dynamics_model: Object with .pred_one_step method.

    Returns:
        Reward function with signature (init_state, controls, cost_params) -> scalar
    """
    reward_type = config.get("reward_type", "info_gathering")
    print(f"🚀 Initializing reward function: {reward_type.upper()}")

    if reward_type == "info_gathering":
        # Extract params
        params = config["reward_fn_params"]
        weight_jerk = params["weight_jerk"]
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = make_info_gathering_term(
            dynamics_model.pred_one_step, meas_noise_diag
        )

        # Vectorize stage reward over the horizon
        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_info_gathering(
                s, u, cp, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory reward.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on state T)
            terminal_reward = _terminal_reward_pursuit_evasion(states[-1])

            # 4. Calculate Jerk penalty (on controls)
            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(control_diffs**2)

            return jnp.sum(stage_rewards) + terminal_reward - jerk_penalty

        return reward_fn

    elif reward_type == "linear_tracking_info":
        # Extract params
        params = config["reward_fn_params"]
        weight_jerk = params["weight_jerk"]
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        # Vectorize stage reward over the horizon
        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_linear_tracking(
                s, u, cp, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory reward for linear tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on state T)
            terminal_reward = _terminal_reward_linear_tracking(states[-1], cost_params)

            # 4. Calculate Jerk penalty (on controls)
            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(control_diffs**2)

            return jnp.sum(stage_rewards) + terminal_reward - jerk_penalty

        return reward_fn

    elif reward_type == "pendulum_swing_up_info":
        # Extract params
        params = config["reward_fn_params"]
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

        # Vectorize stage reward over the horizon
        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_pendulum_swing_up(
                s, u, cp, weight_control, weight_info, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory reward for pendulum swing-up.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on state T)
            terminal_reward = _terminal_reward_pendulum_swing_up(states[-1])

            # 4. Calculate Jerk penalty (on controls)
            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_rewards) + terminal_reward - jerk_penalty

        return reward_fn

    elif reward_type == "merging_idm_info":
        params_cfg = config["reward_fn_params"]
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

        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_merging_idm(
                s, u, cp,
                Q_diag, q_I, p_y_target, v_g,
                L_cost, lane_width,
                weight_control, weight_info, info_term_fn,
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            states = _rollout(
                init_state, controls, cost_params,
                dynamics_model.pred_one_step,
            )
            stage_rewards = stage_reward_vmap(
                states[:-1], controls, cost_params,
            )
            terminal_reward = _terminal_reward_merging_idm(
                states[-1], Qf_diag, q_I,
                p_y_target, v_g, L_cost, lane_width,
            )

            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(
                    control_diffs**2
                )

            return (
                jnp.sum(stage_rewards) + terminal_reward - jerk_penalty
            )

        return reward_fn

    elif reward_type == "drone_state_tracking_info":
        # Extract params
        params = config["reward_fn_params"]
        weight_jerk = params.get("weight_jerk", 0.0)
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        weight_ground = params.get("weight_ground", 0.0)
        state_weights = jnp.array(params["state_weights"])
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Create the info term calculator (JIT-able helper)
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        # Vectorize stage reward over the horizon
        # Note: goal_state is now passed via cost_params at runtime
        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_drone_state_tracking_w_info(
                s, u, cp, state_weights, weight_control, weight_info, weight_ground, info_term_fn
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory reward for drone state tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on state T)
            terminal_reward = _terminal_reward_drone_state_tracking(states[-1], cost_params, state_weights)

            # 4. Calculate Jerk penalty (on controls)
            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_rewards) + terminal_reward - jerk_penalty

        return reward_fn

    elif reward_type == "drone_state_tracking_info_task_aware":
        # Extract params
        params = config["reward_fn_params"]
        weight_jerk = params.get("weight_jerk", 0.0)
        weight_control = params["weight_control"]
        weight_info = params["weight_info"]
        weight_ground = params.get("weight_ground", 0.0)
        goal_state = jnp.array(params["goal_state"])
        state_weights = jnp.array(params["state_weights"])
        meas_noise_diag = jnp.array(params["meas_noise_diag"])

        # Vectorize stage reward over the horizon
        stage_reward_fn = lambda s, u: _stage_reward_drone_state_tracking(
            s, u, goal_state, state_weights, weight_control, weight_ground
        )
        stage_reward_vmap = jax.vmap(
            stage_reward_fn,
            in_axes=(0, 0),
        )

        # Create the task-aware info cost
        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_task_aware_info_term(
                dynamics_model.pred_one_step,
                stage_reward_fn
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
        def reward_fn(init_state, controls, cost_params):
            """
            Calculates total trajectory reward for drone state tracking.
            """
            # 1. Rollout trajectory
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls) + weight_info * info_cost_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on state T)
            terminal_reward = _terminal_reward_drone_state_tracking(states[-1], goal_state, state_weights)

            # 4. Calculate Jerk penalty (on controls)
            jerk_penalty = 0.0
            if weight_jerk > 0:
                control_diffs = jnp.diff(controls, axis=0)
                jerk_penalty = weight_jerk * jnp.sum(control_diffs ** 2)

            return jnp.sum(stage_rewards) + terminal_reward - jerk_penalty

        return reward_fn

    elif reward_type == "cheetah_velocity_tracking":
        # Cheetah velocity tracking reward using mjx.Data directly
        params = config["reward_fn_params"]
        weight_control = params["weight_control"]
        heading_penalty_factor = params.get("heading_penalty_factor", 10.0)

        # Vectorize stage reward over the horizon
        stage_reward_vmap = jax.vmap(
            lambda d, u, cp: _stage_reward_cheetah_velocity_tracking(
                d, u, weight_control, heading_penalty_factor
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_data, controls, cost_params):
            """Total trajectory reward for cheetah velocity maximization using mjx.Data."""
            # 1. Rollout trajectory using dynamics (mjx.Data throughout)
            data_sequence = _rollout_cheetah(
                init_data, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards over all timesteps
            stage_rewards = stage_reward_vmap(data_sequence, controls, cost_params)

            # 3. Calculate terminal reward (on final state)
            final_data = jax.tree.map(lambda x: x[-1], data_sequence)
            terminal_reward = _terminal_reward_cheetah_velocity_tracking(
                final_data
            )

            return jnp.sum(stage_rewards) + terminal_reward

        return reward_fn

    elif reward_type == "cheetah_velocity_learned":
        # Cheetah velocity tracking for learned models (17D state vector)
        params = config["reward_fn_params"]
        weight_control = params["weight_control"]
        heading_penalty_factor = params.get("heading_penalty_factor", 10.0)

        stage_reward_vmap = jax.vmap(
            lambda s, u, cp: _stage_reward_cheetah_velocity_learned(
                s, u, weight_control, heading_penalty_factor
            ),
            in_axes=(0, 0, None),
        )

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            """Total trajectory reward for cheetah velocity maximization using learned model."""
            # 1. Rollout trajectory using dynamics (17D state throughout)
            states = _rollout(
                init_state, controls, cost_params, dynamics_model.pred_one_step
            )

            # 2. Calculate stage rewards (on states 0 to T-1)
            stage_rewards = stage_reward_vmap(states[:-1], controls, cost_params)

            # 3. Calculate terminal reward (on final state)
            terminal_reward = _terminal_reward_cheetah_velocity_learned(
                states[-1]
            )

            return jnp.sum(stage_rewards) + terminal_reward

        return reward_fn

    elif reward_type == "cheetah_velocity_learned_info":
        # Cheetah velocity tracking with info gathering (17D state vector)
        params = config["reward_fn_params"]
        weight_control = params["weight_control"]
        heading_penalty_factor = params.get("heading_penalty_factor", 10.0)
        weight_info = params.get("weight_info", 0.0)
        meas_noise_diag = jnp.array(params["meas_noise_diag"])
        info_steps = params.get("info_steps", None)  # None = full horizon

        info_term_fn = None
        if weight_info > 0:
            info_term_fn = make_info_gathering_term(
                dynamics_model.pred_one_step, meas_noise_diag
            )

        if info_steps is None:
            @jax.jit
            def reward_fn(init_state, controls, cost_params):
                """Cheetah velocity maximization with info gathering."""
                states = _rollout(
                    init_state, controls, cost_params, dynamics_model.pred_one_step
                )

                def scan_info(_, xu):
                    s, u = xu
                    return None, _stage_reward_cheetah_velocity_learned_w_info(
                        s, u, cost_params, weight_control,
                        heading_penalty_factor, weight_info, info_term_fn
                    )

                _, stage_rewards = jax.lax.scan(scan_info, None, (states[:-1], controls))
                terminal_reward = _terminal_reward_cheetah_velocity_learned(states[-1])
                return jnp.sum(stage_rewards) + terminal_reward
        else:
            @jax.jit
            def reward_fn(init_state, controls, cost_params):
                """Cheetah velocity maximization: info reward for first info_steps, cheap reward for rest."""
                states = _rollout(
                    init_state, controls, cost_params, dynamics_model.pred_one_step
                )

                def scan_info(_, xu):
                    s, u = xu
                    return None, _stage_reward_cheetah_velocity_learned_w_info(
                        s, u, cost_params, weight_control,
                        heading_penalty_factor, weight_info, info_term_fn
                    )

                def scan_regular(_, xu):
                    s, u = xu
                    return None, _stage_reward_cheetah_velocity_learned(
                        s, u, weight_control, heading_penalty_factor
                    )

                _, info_rewards = jax.lax.scan(
                    scan_info, None, (states[:info_steps], controls[:info_steps])
                )
                _, reg_rewards = jax.lax.scan(
                    scan_regular, None, (states[info_steps:-1], controls[info_steps:])
                )
                terminal_reward = _terminal_reward_cheetah_velocity_learned(states[-1])
                return jnp.sum(info_rewards) + jnp.sum(reg_rewards) + terminal_reward

        return reward_fn

    elif reward_type == "goal_cost":
        # Single-step evaluation reward (no rollout)
        params = config.get("reward_fn_params", {})
        state_weights = jnp.array(
            params.get("state_weights", [1.0] * config.get("dim_state", 6))
        )
        weight_control = params.get("weight_control", 0.01)

        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            # For evaluation: just compute reward at current state with first control
            control = controls[0] if controls.ndim > 1 else controls
            return _eval_goal_reward(init_state, control, cost_params, state_weights, weight_control)

        return reward_fn

    elif reward_type == "terminal_goal_cost":
        # Terminal distance reward (no rollout)
        @jax.jit
        def reward_fn(init_state, controls, cost_params):
            return _eval_terminal_goal_reward(init_state, cost_params)

        return reward_fn

    else:
        raise ValueError(f"Unknown reward type: '{reward_type}'")
