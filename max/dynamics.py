# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence, NamedTuple, Callable, Any, Optional
from max.normalizers import Normalizer, init_normalizer


# --- Base Model Definition (MLP) ---
class MLP(nn.Module):
    features: Sequence[int]

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for feat in self.features:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        return nn.Dense(state.shape[-1])(x)


# --- Container for the final dynamics model ---
class DynamicsModel(NamedTuple):
    pred_one_step: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    pred_norm_delta: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]


def _create_MLP_dynamics(
    key: jax.Array,
    dim_state: int,
    dim_action: int,
    nn_features: Sequence[int],
):
    """Creates the base MLP model and its initial parameters."""
    model = MLP(features=nn_features)
    dummy_state = jnp.ones((dim_state,))
    dummy_action = jnp.ones((dim_action,))
    params = model.init(key, dummy_state, dummy_action)
    return model, params


# --- The Higher-Order Wrapper Function ---
def create_MLP_residual_dynamics(
    key: jax.Array,
    dim_state: int,
    dim_action: int,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> DynamicsModel:
    """
    Creates and wraps a dynamics model with full input and output normalization.
    """
    model_key, key = jax.random.split(key)
    base_model, model_params = _create_MLP_dynamics(
        model_key,
        dim_state,
        dim_action,
        config["dynamics_params"]["nn_features"],
    )
    params = {"model": model_params, "normalizer": normalizer_params}

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training)."""
        norm_params = params["normalizer"]
        normalized_state = normalizer.normalize(norm_params["state"], state)
        normalized_action = normalizer.normalize(norm_params["action"], action)
        return base_model.apply(
            params["model"], normalized_state, normalized_action
        )

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the absolute next state using a residual"""
        normalized_delta = pred_norm_delta(params, state, action)
        delta = normalizer.unnormalize(
            params["normalizer"]["delta"], normalized_delta
        )
        return state + delta

    return (
        DynamicsModel(
            pred_one_step=pred_one_step, pred_norm_delta=pred_norm_delta
        ),
        params,
    )


def create_analytical_pendulum_dynamics() -> DynamicsModel:
    """
    Creates a dynamics model that is an exact analytical match for the
    dm_control pendulum environment.
    """
    # Parameters from pendulum.xml and MuJoCo defaults
    g = 9.81  # Gravity
    m = 1.0  # Mass of the pendulum bob
    l = 0.5  # Length to the center of mass of the pendulum
    damping = 0.1  # Damping on the hinge joint
    dt = 0.02  # Timestep from the XML <option>
    gear = 1.0  # Gear ratio from the actuator

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predicts the next state using the analytical equations of motion.
        This function ignores the `params` argument but keeps it for API
        compatibility with the learned model.
        """
        cos_theta, sin_theta, thetadot = state
        u = action[0]  # Control input torque

        # Recover the angle theta. We use this for the update step.
        theta = jnp.arctan2(sin_theta, cos_theta)

        # Equation of motion: m*l^2*theta_ddot = m*g*l*sin(theta) - damping*thetadot + torque
        # Rearranged for theta_ddot:
        torque = gear * u
        theta_ddot = (m * g * l * sin_theta - damping * thetadot + torque) / (
            m * l**2
        )

        # Semi-implicit Euler integration (matches MuJoCo)
        new_thetadot = thetadot + theta_ddot * dt
        new_theta = theta + new_thetadot * dt

        # New state vector
        new_cos_theta = jnp.cos(new_theta)
        new_sin_theta = jnp.sin(new_theta)

        return jnp.array([new_cos_theta, new_sin_theta, new_thetadot])

    # No trainable parameters, so return an empty dictionary
    params = {"model": [0.0], "normalizer": [0.0]}
    return DynamicsModel(pred_one_step=pred_one_step), params


# --- Probabilistic Model Definition (ProbabilisticMLP) ---
class ProbabilisticMLP(nn.Module):
    features: Sequence[int]
    output_dim: int

    @nn.compact
    def __call__(self, state, action):
        x = jnp.concatenate([state, action], axis=-1)
        for feat in self.features:
            x = nn.Dense(feat)(x)
            # The paper used swish activations, but tanh is also fine.
            x = nn.swish(x)

        # Output mean and log variance for a Gaussian distribution
        # The output dimension is 2 * state_dim
        output = nn.Dense(2 * self.output_dim)(x)
        mean, log_var = jnp.split(output, 2, axis=-1)

        # Crucial step from Appendix A.1 to prevent variance explosion/collapse [cite: 442, 448]
        # These max/min values would be learned or set as part of the model's parameters.
        # For simplicity here, we'll imagine they are passed in `params`.
        # In a full implementation, you'd add them to the train state.
        max_log_var = jnp.ones_like(mean) * 0.5
        min_log_var = jnp.ones_like(mean) * -10.0
        log_var = max_log_var - nn.softplus(max_log_var - log_var)
        log_var = min_log_var + nn.softplus(log_var - min_log_var)

        return mean, log_var


# --- Container for the final PETS dynamics model ---
class PETSDynamicsModel(NamedTuple):
    # Predicts the normalized delta distribution for a SINGLE model
    pred_norm_delta_dist: Callable[
        [Any, jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]
    ]
    # Samples a single next state given a specific bootstrap model to use
    sample_next_state: Callable[
        [Any, jax.Array, jnp.ndarray, jnp.ndarray, int], jnp.ndarray
    ]


# --- The Higher-Order Wrapper Function for the Ensemble ---
def create_probabilistic_ensemble_dynamics(
    key: jax.Array,
    dim_state: int,
    dim_action: int,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> PETSDynamicsModel:
    """
    Creates and wraps a probabilistic ensemble of dynamics models.
    """
    dynamics_params = config["dynamics_params"]
    ensemble_size = dynamics_params["ensemble_size"]
    nn_features = dynamics_params["nn_features"]
    base_model = ProbabilisticMLP(features=nn_features, output_dim=dim_state)
    keys = jax.random.split(key, ensemble_size)
    dummy_state = jnp.ones((dim_state,))
    dummy_action = jnp.ones((dim_action,))
    ensemble_params = jax.vmap(base_model.init, in_axes=(0, None, None))(
        keys, dummy_state, dummy_action
    )
    params = {"model": ensemble_params, "normalizer": normalizer_params}

    @jax.jit
    def pred_norm_delta_dist(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Predicts normalized delta distribution for a SINGLE model's params.
        """
        norm_params = params["normalizer"]
        normalized_state = normalizer.normalize(norm_params["state"], state)
        normalized_action = normalizer.normalize(norm_params["action"], action)

        mean, log_var = base_model.apply(
            {"params": params["model"]["params"]},
            normalized_state,
            normalized_action,
        )
        return mean, log_var

    vmap_pred = jax.vmap(
        pred_norm_delta_dist,
        in_axes=({"model": 0, "normalizer": None}, None, None),
    )

    @jax.jit
    def sample_next_state(
        params: Any,
        key: jax.Array,
        state: jnp.ndarray,
        action: jnp.ndarray,
        bootstrap_idx: int,
    ) -> jnp.ndarray:
        """
        Samples a single next state from one of the ensemble's models.
        """
        means, log_vars = vmap_pred(params, state, action)
        mean = means[bootstrap_idx]
        log_var = log_vars[bootstrap_idx]
        std = jnp.exp(0.5 * log_var)

        normalized_delta = (
            mean + jax.random.normal(key, shape=mean.shape) * std
        )

        delta = normalizer.unnormalize(
            params["normalizer"]["delta"], normalized_delta
        )
        return state + delta

    return (
        PETSDynamicsModel(
            pred_norm_delta_dist=pred_norm_delta_dist,
            sample_next_state=sample_next_state,
        ),
        params,
    )


def create_pursuit_evader_dynamics(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for a 2-player pursuer-evader planar system with a
    trainable single-step LQR pursuit strategy.

    - State: An 8D vector [pe_x, pe_y, ve_x, ve_y, pp_x, pp_y, vp_x, vp_y].
    - Action: A 2D vector [ae_x, ae_y] for the evader's acceleration.
    - Trainable Params:
        - `q_cholesky`: Cholesky factor of the state cost matrix Q (4x4).
        - `r_cholesky`: Cholesky factor of the control cost matrix R (2x2).
    """

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predicts the next state for the pursuer-evader system.
        """
        # --- Unpack State and Action ---
        p_evader = state[0:2]
        v_evader = state[2:4]
        p_pursuer = state[4:6]
        v_pursuer = state[6:8]
        a_evader = action.squeeze()

        # Construct full state vectors for clarity
        x_evader = jnp.concatenate([p_evader, v_evader])
        x_pursuer = jnp.concatenate([p_pursuer, v_pursuer])

        # --- Unpack Config and Construct LQR Matrices ---
        dt = config["dynamics_params"]["dt"]

        # State transition matrix (4x4)
        A = jnp.array(
            [[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )

        # Control matrix (4x2)
        B = jnp.array([[0.5 * dt**2, 0], [0, 0.5 * dt**2], [dt, 0], [0, dt]])

        # --- Construct Q and R from learnable Cholesky factors ---
        q_cholesky = params["model"]["q_cholesky"]
        r_cholesky = params["model"]["r_cholesky"]

        Q = q_cholesky @ q_cholesky.T
        R = r_cholesky @ r_cholesky.T + 1e-6 * jnp.eye(2)

        # --- Single-Step LQR Control Law (Pursuer) ---
        # Implementing Eq (10): u* = -(R + B'QB)^-1 B'Q(Ax_pursuer - x_evader)
        
        # 1. Compute the inverse term: (R + B'QB)^-1
        # shape: (2,2)
        inv_term = jnp.linalg.inv(R + B.T @ Q @ B)

        # 2. Compute the Gain component that does NOT depend on state: (R + B'QB)^-1 B'Q
        # shape: (2,4)
        Gain_prefix = inv_term @ B.T @ Q

        # 3. Compute the specific error term defined in the paper: (Ax_p - x_e)
        # The paper derivation minimizes ||Ax_p + Bu - x_e||^2, leading to this term.
        # shape: (4,)
        state_diff_term = (A @ x_pursuer) - x_evader

        # 4. Calculate optimal acceleration
        a_pursuer = -Gain_prefix @ state_diff_term

        # --- Dynamics Update (Double Integrator) ---
        next_v_evader = v_evader + a_evader * dt
        next_v_pursuer = v_pursuer + a_pursuer * dt
        next_p_evader = p_evader + 0.5 * a_evader * dt**2 + v_evader * dt
        next_p_pursuer = p_pursuer + 0.5 * a_pursuer * dt**2 + v_pursuer * dt

        return jnp.concatenate(
            [next_p_evader, next_v_evader, next_p_pursuer, next_v_pursuer],
            axis=-1,
        )

    # --- Initialize Learnable Parameters ---
    init_q_diag = jnp.array(config["dynamics_params"]["init_q_diag"])
    init_r_diag = jnp.array(config["dynamics_params"]["init_r_diag"])

    q_cholesky_init = jnp.diag(jnp.sqrt(init_q_diag))
    r_cholesky_init = jnp.diag(jnp.sqrt(init_r_diag))

    model_params = {
        "q_cholesky": q_cholesky_init,
        "r_cholesky": r_cholesky_init,
    }
    params = {"model": model_params, "normalizer": None}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=None)

    return model, params


def init_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer = None,
    normalizer_params=None,
) -> tuple[Any, Any]:
    """Initializes the appropriate dynamics model based on the configuration."""
    dynamics_type = config["dynamics"]
    print(f"🚀 Initializing dynamics model: {dynamics_type.upper()}")

    # Init None Normalizer if not provided
    if normalizer is None:
        normalizer, normalizer_params = init_normalizer(config)

    if dynamics_type == "mlp_residual":
        dim_state = config.dim_state
        dim_action = config.dim_action
        return create_MLP_residual_dynamics(
            key, dim_state, dim_action, config, normalizer, normalizer_params
        )

    elif dynamics_type == "probabilistic_ensemble":
        dim_state = config.dim_state
        dim_action = config.dim_action
        return create_probabilistic_ensemble_dynamics(
            key, dim_state, dim_action, config, normalizer, normalizer_params
        )

    elif dynamics_type == "analytical_pendulum":
        return create_analytical_pendulum_dynamics()

    elif dynamics_type == "pursuit_evader":
        return create_pursuit_evader_dynamics(
            config, normalizer, normalizer_params
        )

    else:
        raise ValueError(f"Unknown dynamics type: '{dynamics_type}'")
