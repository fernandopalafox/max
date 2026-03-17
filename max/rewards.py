"""
Minimal functional reward abstractions for max library.

Provides factory functions that return JIT-compiled closures for cheetah reward patterns.
Follows the max library philosophy: pure functions, closures, and NamedTuples.

Two-tier API:
  - init_reward_model(config, encoder=None) -> (Reward, dict)
      New TDMPC2 path. Returns a Reward NamedTuple with predict/logits callables.
      predict(reward_params, z, action) -> scalar
      logits(reward_params, z, action)  -> (num_bins,)   [tdmpc2_learned only]

  - init_reward(config, dynamics_model) -> Callable
      Legacy path for iCEM planner / evaluator.
      Returns reward_fn(init_state, controls, cost_params) -> scalar.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, NamedTuple, Any, Optional


# ---------------------------------------------------------------------------
# Reward abstraction (TDMPC2 path)
# ---------------------------------------------------------------------------

class Reward(NamedTuple):
    predict: Callable  # (reward_params, z, action) -> scalar
    logits: Optional[Callable] = None  # (reward_params, z, action) -> (num_bins,) [tdmpc2_learned]


def init_reward_model(config: dict, encoder=None) -> tuple["Reward", dict]:
    """
    Initialize a Reward model for TDMPC2.

    Supported reward_type values:
      "tdmpc2_learned"          — learned NN reward head in latent space
      "cheetah_velocity_learned" — analytical velocity reward after decoding z

    Returns:
        (Reward, reward_params)
    """
    reward_type = config.get("reward_type", "tdmpc2_learned")
    print(f"Initializing reward model: {reward_type.upper()}")

    if reward_type == "tdmpc2_learned":
        return _init_tdmpc2_learned_reward(config)

    elif reward_type == "cheetah_velocity_learned":
        return _init_cheetah_velocity_reward_model(config, encoder)

    else:
        raise ValueError(f"Unknown reward_type for init_reward_model: {reward_type!r}")


def _init_tdmpc2_learned_reward(config: dict) -> tuple["Reward", dict]:
    """Learned NN reward head: (z, a) -> logits (num_bins,)."""
    rp = config["reward_params"]
    features = rp["features"]
    num_bins: int = rp["num_bins"]
    latent_dim: int = config["encoder_params"]["encoder_features"][-1]
    dim_a: int = config["dim_action"]

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    class _RewardHead(nn.Module):
        @nn.compact
        def __call__(self, z, a):
            x = jnp.concatenate([z, a], axis=-1)
            for feat in features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            return nn.Dense(num_bins)(x)

    reward_net = _RewardHead()
    dummy_z = jnp.ones((latent_dim,))
    dummy_a = jnp.ones((dim_a,))
    key = jax.random.key(0)
    reward_nn_params = reward_net.init(key, dummy_z, dummy_a)
    reward_params = {"reward_head": reward_nn_params}

    vmin: float = rp.get("vmin", -10.0)
    vmax: float = rp.get("vmax", 10.0)

    def logits_fn(reward_params: Any, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        return reward_net.apply(reward_params["reward_head"], z, a)

    def predict_fn(reward_params: Any, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Returns scalar reward: symexp(two_hot_inv(logits))."""
        from max.trainers import symexp, two_hot_inv
        raw_logits = logits_fn(reward_params, z, a)
        return symexp(two_hot_inv(raw_logits, vmin, vmax, num_bins))

    return Reward(predict=predict_fn, logits=logits_fn), reward_params


def _init_cheetah_velocity_reward_model(config: dict, encoder) -> tuple["Reward", dict]:
    """
    Analytical velocity reward via decoder: z -> obs -> velocity reward.
    encoder.decode(enc_params, z) must be available (passes enc_params from parameters).
    reward_params = {} (config baked in)
    """
    rp = config.get("reward_fn_params", {})
    weight_control: float = rp.get("weight_control", 0.1)
    heading_penalty_factor: float = rp.get("heading_penalty_factor", 10.0)
    from max.normalizers import init_normalizer
    normalizer, normalizer_params = init_normalizer(config)
    state_norm_params = normalizer_params["state"]

    def predict_fn(reward_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        # reward_params is ignored (use None or {} at call site)
        # enc_params not available here since reward_params carries nothing
        # This variant requires that z is already decoded externally OR that
        # the caller passes enc_params in reward_params.
        # For simplicity: reward_params = enc_params (the encoder params dict).
        if reward_params is not None and encoder is not None:
            norm_next = encoder.decode(reward_params, z)
            state = normalizer.unnormalize(state_norm_params, norm_next)
        else:
            # Fallback: treat z as state directly (16/17 dim)
            state = z
        return _stage_reward_cheetah_velocity_learned(
            state, action, weight_control, heading_penalty_factor
        )

    return Reward(predict=predict_fn, logits=None), {}


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

    # Concatenate init_state to match shape requirements
    return jnp.concatenate([init_state[jnp.newaxis, :], states], axis=0)


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
    reward_type = config.get("reward_type", "cheetah_velocity_tracking")
    print(f"🚀 Initializing reward function: {reward_type.upper()}")

    if reward_type == "cheetah_velocity_tracking":
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

    else:
        raise ValueError(f"Unknown reward type: '{reward_type}'")
