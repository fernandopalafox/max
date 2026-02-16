# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
from typing import Sequence, NamedTuple, Callable, Any, Optional
from max.normalizers import Normalizer, init_normalizer


# --- Container for the final dynamics model ---
class DynamicsModel(NamedTuple):
    pred_one_step: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    pred_norm_delta: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    pred_one_step_with_info: Callable[[Any, jnp.ndarray, jnp.ndarray], tuple] = None


def create_mlp_resnet(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> DynamicsModel:
    """
    Creates an MLP ResNet dynamics model that predicts state deltas (residuals).
    Includes full input and output normalization.

    Supports ensembles: when ensemble_size > 1, initializes multiple independent
    networks and returns the mean prediction across ensemble members.
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    nn_features = config["dynamics_params"]["nn_features"]
    ensemble_size = config["dynamics_params"].get("ensemble_size", 1)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for feat in nn_features:
                x = nn.Dense(feat)(x)
                x = nn.tanh(x)
            return nn.Dense(dim_state)(x)

    base_model = MLP()
    dummy_state = jnp.ones((dim_state,))
    dummy_action = jnp.ones((dim_action,))

    # Initialize ensemble parameters
    keys = jax.random.split(key, ensemble_size)
    if ensemble_size == 1:
        model_params = base_model.init(keys[0], dummy_state, dummy_action)
    else:
        # Stack parameters across ensemble members using vmap
        model_params = jax.vmap(base_model.init, in_axes=(0, None, None))(
            keys, dummy_state, dummy_action
        )
    params = {"model": model_params, "normalizer": normalizer_params}

    def _pred_norm_delta_single(
        model_params: Any, norm_params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts normalized delta for a single model."""
        normalized_state = normalizer.normalize(norm_params["state"], state)
        normalized_action = normalizer.normalize(norm_params["action"], action)
        return base_model.apply(model_params, normalized_state, normalized_action)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training).

        For ensembles, returns the mean prediction across all members.
        """
        norm_params = params["normalizer"]
        if ensemble_size == 1:
            return _pred_norm_delta_single(
                params["model"], norm_params, state, action
            )
        else:
            # vmap over ensemble members and compute mean
            ensemble_preds = jax.vmap(
                lambda mp: _pred_norm_delta_single(mp, norm_params, state, action)
            )(params["model"])
            return jnp.mean(ensemble_preds, axis=0)

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


def create_mlp_resnet_last_layer(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> DynamicsModel:
    """
    Creates an MLP ResNet dynamics model with only the last layer trainable.

    Same architecture as create_mlp_resnet, but only the output layer
    is exposed for training. Hidden layers are randomly initialized and frozen
    (baked into the model closure).

    Supports ensembles: when ensemble_size > 1, initializes multiple independent
    networks (each with its own frozen hidden layers) and returns the mean
    prediction across ensemble members.

    Trainable Params: Only output layer weights (kernel and bias)
    Frozen Params: Hidden layer weights (baked into closure)
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    nn_features = config["dynamics_params"]["nn_features"]
    ensemble_size = config["dynamics_params"].get("ensemble_size", 1)

    class MLP(nn.Module):
        @nn.compact
        def __call__(self, state, action):
            x = jnp.concatenate([state, action], axis=-1)
            for feat in nn_features:
                x = nn.Dense(feat)(x)
                x = nn.tanh(x)
            return nn.Dense(dim_state)(x)

    base_model = MLP()
    dummy_state = jnp.ones((dim_state,))
    dummy_action = jnp.ones((dim_action,))

    # Layer naming: Dense_0, Dense_1, ..., Dense_n where n is output layer
    num_hidden = len(nn_features)
    output_layer_name = f"Dense_{num_hidden}"

    # Initialize and split parameters for each ensemble member
    keys = jax.random.split(key, ensemble_size)

    def init_and_split_params(k):
        """Initialize full params, return (frozen_params, trainable_params)."""
        full_params = base_model.init(k, dummy_state, dummy_action)

        # Extract frozen hidden layer params
        frozen = {"params": {}}
        for i in range(num_hidden):
            layer_name = f"Dense_{i}"
            frozen["params"][layer_name] = full_params["params"][layer_name]

        # Extract trainable output layer params
        trainable = {"params": {output_layer_name: full_params["params"][output_layer_name]}}

        return frozen, trainable

    if ensemble_size == 1:
        frozen_params, trainable_params = init_and_split_params(keys[0])
        # frozen_params is a single dict, trainable_params is a single dict
    else:
        # Initialize each ensemble member separately
        all_frozen = []
        all_trainable = []
        for i in range(ensemble_size):
            frozen, trainable = init_and_split_params(keys[i])
            all_frozen.append(frozen)
            all_trainable.append(trainable)

        # Stack frozen params: list of dicts -> dict with stacked arrays
        frozen_params = {"params": {}}
        for i in range(num_hidden):
            layer_name = f"Dense_{i}"
            frozen_params["params"][layer_name] = {
                "kernel": jnp.stack([f["params"][layer_name]["kernel"] for f in all_frozen], axis=0),
                "bias": jnp.stack([f["params"][layer_name]["bias"] for f in all_frozen], axis=0),
            }

        # Stack trainable params
        trainable_params = {"params": {output_layer_name: {
            "kernel": jnp.stack([t["params"][output_layer_name]["kernel"] for t in all_trainable], axis=0),
            "bias": jnp.stack([t["params"][output_layer_name]["bias"] for t in all_trainable], axis=0),
        }}}

    params = {"model": trainable_params, "normalizer": normalizer_params}

    def _pred_norm_delta_single(
        trainable_model_params: Any,
        frozen_model_params: Any,
        norm_params: Any,
        state: jnp.ndarray,
        action: jnp.ndarray,
    ) -> jnp.ndarray:
        """Predicts normalized delta for a single model."""
        normalized_state = normalizer.normalize(norm_params["state"], state)
        normalized_action = normalizer.normalize(norm_params["action"], action)

        # Reconstruct full params: frozen + trainable
        full_params = {"params": {**frozen_model_params["params"], **trainable_model_params["params"]}}
        return base_model.apply(full_params, normalized_state, normalized_action)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training).

        For ensembles, returns the mean prediction across all members.
        """
        norm_params = params["normalizer"]
        if ensemble_size == 1:
            return _pred_norm_delta_single(
                params["model"], frozen_params, norm_params, state, action
            )
        else:
            # vmap over ensemble members
            def forward_one_member(ens_idx):
                # Extract this member's trainable params
                member_trainable = {"params": {output_layer_name: {
                    "kernel": params["model"]["params"][output_layer_name]["kernel"][ens_idx],
                    "bias": params["model"]["params"][output_layer_name]["bias"][ens_idx],
                }}}
                # Extract this member's frozen params
                member_frozen = {"params": {}}
                for i in range(num_hidden):
                    layer_name = f"Dense_{i}"
                    member_frozen["params"][layer_name] = {
                        "kernel": frozen_params["params"][layer_name]["kernel"][ens_idx],
                        "bias": frozen_params["params"][layer_name]["bias"][ens_idx],
                    }
                return _pred_norm_delta_single(
                    member_trainable, member_frozen, norm_params, state, action
                )

            ensemble_preds = jax.vmap(forward_one_member)(jnp.arange(ensemble_size))
            return jnp.mean(ensemble_preds, axis=0)

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


def create_mlp_resnet_tiny_lora(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> DynamicsModel:
    """
    Creates an MLP ResNet dynamics model with TinyLoRA for efficient adaptation.

    Same architecture as create_mlp_resnet, but uses TinyLoRA to modify layer weights.
    The effective weight for each layer is: W' = W + U @ diag(Sigma) @ (sum_i v[i] * P[i]) @ V.T

    Supports ensembles: when ensemble_size > 1, initializes multiple independent
    networks and returns the mean prediction across ensemble members.

    Trainable Params: Steering vectors v for each layer (num_layers * steering_dim total)
    Frozen (in closure): For each layer: W, b, U, Sigma, V, P

    Required config["dynamics_params"]:
        - nn_features: list of hidden layer sizes
        - svd_rank: rank of SVD truncation
        - steering_dim: dimension of steering vector per layer
        - projection_seed: seed for reproducible random projections
        - ensemble_size: (optional, default 1) number of ensemble members
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    dyn_params = config["dynamics_params"]
    nn_features = dyn_params["nn_features"]
    svd_rank = dyn_params["svd_rank"]
    steering_dim = dyn_params["steering_dim"]
    projection_seed = dyn_params["projection_seed"]
    ensemble_size = dyn_params.get("ensemble_size", 1)

    # Initialize TinyLoRA layers
    # Input: state + action, Output: state delta
    frozen_layers, trainable_v_init = _init_tiny_lora_layers(
        key=key,
        nn_features=nn_features,
        input_dim=dim_state + dim_action,
        output_dim=dim_state,
        svd_rank=svd_rank,
        steering_dim=steering_dim,
        projection_seed=projection_seed,
        ensemble_size=ensemble_size,
    )

    params = {"model": trainable_v_init, "normalizer": normalizer_params}

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training).

        For ensembles, returns the mean prediction across all members.
        """
        norm_params = params["normalizer"]
        normalized_state = normalizer.normalize(norm_params["state"], state)
        normalized_action = normalizer.normalize(norm_params["action"], action)
        x = jnp.concatenate([normalized_state, normalized_action], axis=-1)

        return _tiny_lora_forward(x, params["model"], frozen_layers, ensemble_size)

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


def create_analytical_pendulum() -> DynamicsModel:
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
def create_probabilistic_ensemble(
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


def create_pursuit_evader(
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


def create_pursuit_evader_unicycle(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for a 2-player pursuer-evader planar system with a
    trainable single-step unicycle pursuit strategy.

    - State: An 8D vector [pe_x, pe_y, ve_x, ve_y, pp_x, pp_y, theta, v].
    - Action: A 2D vector [ae_x, ae_y] for the evader's acceleration.
    - Trainable Params:
        - `tracking_weight`: learnable weight for tracking error in the cost function.
    """

    dt = config["dynamics_params"]["dt"]
    T = config["dynamics_params"]["mpc_horizon"]

    def rollout_unicycle(
        x0: jnp.ndarray,
        u_seq: jnp.ndarray,
    ) -> jnp.ndarray:
                    
        def scan_fn(x, u):
            p_pursuer = x[0:2]
            speed_pursuer = x[2]
            angle_pursuer = x[3]
            a_pursuer = u[0]
            omega_pursuer = u[1]
            next_p_pursuer = p_pursuer + speed_pursuer * jnp.array([jnp.cos(angle_pursuer), jnp.sin(angle_pursuer)]) * dt
            next_speed_pursuer = speed_pursuer + a_pursuer * dt
            next_angle_pursuer = angle_pursuer + omega_pursuer * dt
            x_next = jnp.array([next_p_pursuer[0], next_p_pursuer[1], next_speed_pursuer, next_angle_pursuer])
            return x_next, x_next

        _, xs_rest = jax.lax.scan(scan_fn, x0, u_seq)
        xs = jnp.concatenate([x0[None, :], xs_rest], axis=0)
        return xs
    
    def pursuit_cost(
        u_flat: jnp.ndarray,
        init_state_pursuer: jnp.ndarray,
        evader_state: jnp.ndarray,  # Now includes velocity
        tracking_weight: float,
    ) -> float:
        u_seq = u_flat.reshape(T, 2)
        xs = rollout_unicycle(init_state_pursuer, u_seq)
        ps = xs[:, :2]

        # Predict evader trajectory (simple constant-velocity prediction)
        p_evader = evader_state[:2]
        v_evader = evader_state[2:4]
        
        # Predicted evader positions over horizon
        timesteps = jnp.arange(1, T + 1)
        predicted_evader_positions = p_evader[None, :] + v_evader[None, :] * (timesteps[:, None] * dt)
        
        # Track predicted future positions
        track = tracking_weight * jnp.sum((ps[1:] - predicted_evader_positions) ** 2)
        
        # Control regularization (smaller weight)
        control_penalty = (jnp.sum(u_seq[:, 0] ** 2) + jnp.sum(u_seq[:, 1] ** 2))

        speed_penalty = 0.1 * jnp.sum(xs[:, 2] ** 2)
        
        return track + 0.2 * control_penalty + speed_penalty

    @jax.jit
    def solve_unicycle_mpc(tracking_weight, state_evader, state_pursuer) -> jnp.ndarray:
        """
        Solve unicycle pursuit MPC using gradient descent.

        Returns:
            u_star : optimal control sequence, shape (T, 2)
            J_star : optimal cost value
        """
        u_init = jnp.zeros((T * 2,))

        # Define cost function with fixed parameters
        def cost_fn(u_flat):
            return pursuit_cost(
                u_flat, state_pursuer, state_evader, tracking_weight
            )

        # Gradient of cost function
        grad_fn = jax.grad(cost_fn)

        def gd_step(u_flat, _):
            """One step of gradient descent with fixed learning rate."""
            grad = grad_fn(u_flat)
            u_new = u_flat - config["dynamics_params"]["learning_rate"] * grad
            return u_new, None

        # Run gradient descent iterations (unrolled, fully differentiable)
        u_star_flat, _ = jax.lax.scan(gd_step, u_init, None, length=config["dynamics_params"]["max_gd_iters"])

        J_star = cost_fn(u_star_flat)
        # Compute terminal step gradient norm for monitoring MPC convergence
        grad_terminal = grad_fn(u_star_flat)
        grad_norm = jnp.linalg.norm(grad_terminal)

        return u_star_flat.reshape(T, 2), J_star, grad_norm

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
        speed_pursuer = state[6]
        angle_pursuer = state[7]
        a_evader = action.squeeze()

        # Construct full state vectors for clarity
        state_evader = jnp.concatenate([p_evader, v_evader])
        state_pursuer = jnp.concatenate([p_pursuer, jnp.atleast_1d(speed_pursuer), jnp.atleast_1d(angle_pursuer)])

        # --- learnable parameter ---
        tracking_weight = params["model"]["tracking_weight"]
 
 
        # --- Single-Step MPC Control Law (Pursuer) ---
        u_star, J_star, grad_norm = solve_unicycle_mpc(tracking_weight, state_evader, state_pursuer)
        a_pursuer = u_star[0, 0]
        omega_pursuer = u_star[0, 1]
        # --- Dynamics Update (Double Integrator + unicycle) ---
        next_p_evader = p_evader + 0.5 * a_evader * dt**2 + v_evader * dt
        next_v_evader = v_evader + a_evader * dt

        next_p_pursuer = p_pursuer + speed_pursuer * jnp.array([jnp.cos(angle_pursuer), jnp.sin(angle_pursuer)]) * dt
        next_speed_pursuer = speed_pursuer + a_pursuer * dt
        next_angle_pursuer = angle_pursuer + omega_pursuer * dt

        next_state = jnp.concatenate(
            [next_p_evader, next_v_evader, next_p_pursuer, jnp.atleast_1d(next_speed_pursuer), jnp.atleast_1d(next_angle_pursuer)],
            axis=-1,
        )
        return next_state, {"mpc_grad_norm": grad_norm, "mpc_cost": J_star}

    @jax.jit
    def pred_one_step_no_info(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Wrapper that returns only next_state for backward compatibility."""
        next_state, _ = pred_one_step(params, state, action)
        return next_state

    # --- Initialize Learnable Parameters ---
    model_params = {
        "tracking_weight": config["dynamics_params"]["init_tracking_weight"],
    }
    params = {"model": model_params, "normalizer": None}
    model = DynamicsModel(
        pred_one_step=pred_one_step_no_info,
        pred_norm_delta=None,
        pred_one_step_with_info=pred_one_step,
    )

    return model, params



def create_linear(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for a single-agent linear system with learnable A and B matrices.

    - State: A 4D vector [px, py, vx, vy].
    - Action: A 2D vector [ax, ay] for acceleration.
    - Trainable Params:
        - `A`: State transition matrix (4x4).
        - `B`: Control matrix (4x2).

    Dynamics: x_{t+1} = A @ x_t + B @ u_t
    """

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the next state using linear dynamics: x_{t+1} = A @ x + B @ u"""
        A = params["model"]["A"]
        B = params["model"]["B"]
        u = action.squeeze()  # Ensure action is (2,)
        return A @ state + B @ u

    # Initialize learnable parameters from config
    init_A = jnp.array(config["dynamics_params"]["init_A"])  # (4, 4)
    init_B = jnp.array(config["dynamics_params"]["init_B"])  # (4, 2)

    model_params = {
        "A": init_A,
        "B": init_B,
    }
    params = {"model": model_params, "normalizer": None}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=None)

    return model, params


def create_damped_pendulum(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for a damped pendulum with learnable parameters.

    State: [phi, phi_dot] (angle from down, angular velocity)
    Action: [tau] (applied torque)
    Trainable Params: b (damping), J (moment of inertia)
    Known Constants: m=1.0, g=9.81, l=1.0, dt (from config)

    Dynamics: phi_ddot = (tau - b * phi_dot - m * g * l * sin(phi)) / J
    """
    # Known constants
    m = 1.0
    g = 9.81
    l = 1.0
    dt = config["dynamics_params"]["dt"]

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the next state using damped pendulum dynamics."""
        phi, phi_dot = state[0], state[1]
        b = params["model"]["b"]
        J = params["model"]["J"]
        tau = action[0]

        phi_ddot = (tau - b * phi_dot - m * g * l * jnp.sin(phi)) / J
        phi_next = phi + phi_dot * dt
        phi_dot_next = phi_dot + phi_ddot * dt

        return jnp.array([phi_next, phi_dot_next])

    # Initialize learnable parameters from config
    init_b = config["dynamics_params"]["init_b"]
    init_J = config["dynamics_params"]["init_J"]

    model_params = {
        "b": jnp.array(init_b),
        "J": jnp.array(init_J),
    }
    params = {"model": model_params, "normalizer": None}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=None)

    return model, params


def create_planar_drone(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Planar drone model with trainable MLP residual.

    State: [p_x, p_y, phi, v_x, v_y, phi_dot] (6D)
    Action: [T_1, T_2] (2D rotor thrusts, non-negative)

    The model combines:
    - Nominal drone dynamics (known physics)
    - Trainable MLP residual predicting acceleration corrections [delta_a_x, delta_a_y, delta_alpha]

    Residual Input: [v_x, v_y, phi, wind_x, wind_y] (normalized)
    Residual Output: [delta_a_x, delta_a_y, delta_alpha] (unnormalized)

    Trainable Params: All MLP weights (hidden + output layers)
    Known Constants: mass, gravity, arm_length, inertia, dt, wind_x, wind_y (from config)
    """
    # Physical parameters from config
    dyn_params = config["dynamics_params"]
    m = dyn_params["mass"]
    g = dyn_params["gravity"]
    L = dyn_params["arm_length"]
    I = dyn_params["inertia"]
    dt = dyn_params["dt"]
    nn_features = dyn_params["nn_features"]
    wind_x_param = dyn_params.get("wind_x", 0.0)
    wind_y_param = dyn_params.get("wind_y", 0.0)

    # Create MLP for residual acceleration prediction
    # Input: [v_x, v_y, phi, wind_x, wind_y] (5D normalized)
    # Output: acceleration residuals (3D) for [delta_a_x, delta_a_y, delta_alpha]
    class ResidualMLP(nn.Module):
        features: Sequence[int]

        @nn.compact
        def __call__(self, residual_input):
            x = residual_input
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.tanh(x)
            return nn.Dense(3)(x)  # 3 acceleration residuals

    # Initialize the NN
    residual_net = ResidualMLP(features=nn_features)
    dummy_residual_input = jnp.ones((5,))  # v_x, v_y, phi, wind_x, wind_y
    nn_params = residual_net.init(key, dummy_residual_input)

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts next state using nominal dynamics + NN residual (Semi-Implicit)."""
        p_x, p_y, phi, v_x, v_y, phi_dot = state
        T_1, T_2 = action[0], action[1]

        # Total thrust
        T_total = T_1 + T_2

        # Nominal accelerations (known physics, no wind)
        a_x_nom = (1 / m) * T_total * jnp.sin(phi)
        a_y_nom = (1 / m) * T_total * jnp.cos(phi) - g
        alpha_nom = (L / I) * (T_2 - T_1)

        # Normalize residual inputs: velocities, angle, wind
        norm_state = normalizer.normalize(params["normalizer"]["state"], state)
        norm_v_x = norm_state[3]
        norm_v_y = norm_state[4]
        norm_phi = norm_state[2]

        wind_arr = jnp.array([wind_x_param, wind_y_param])
        norm_wind = normalizer.normalize(params["normalizer"]["wind"], wind_arr)

        # Build 5D residual input
        residual_input = jnp.array([norm_v_x, norm_v_y, norm_phi, norm_wind[0], norm_wind[1]])

        # Apply NN and unnormalize
        residual_normalized = residual_net.apply(params["model"], residual_input)
        residual = normalizer.unnormalize(params["normalizer"]["delta"], residual_normalized)
        delta_a_x, delta_a_y, delta_alpha = residual[0], residual[1], residual[2]

        # Total accelerations
        a_x = a_x_nom + delta_a_x
        a_y = a_y_nom + delta_a_y
        alpha = alpha_nom + delta_alpha

        # --- SEMI-IMPLICIT EULER INTEGRATION ---

        # 1. Update velocities and angular rate FIRST
        v_x_next = v_x + a_x * dt
        v_y_next = v_y + a_y * dt
        phi_dot_next = phi_dot + alpha * dt

        # 2. Update positions and angle using the NEW velocities
        p_x_next = p_x + v_x_next * dt
        p_y_next = p_y + v_y_next * dt
        phi_next = phi + phi_dot_next * dt

        return jnp.array([p_x_next, p_y_next, phi_next, v_x_next, v_y_next, phi_dot_next])

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training)."""
        next_state = pred_one_step(params, state, action)
        delta = next_state - state
        if params["normalizer"] is not None:
            return normalizer.normalize(params["normalizer"]["delta"], delta)
        return delta

    params = {"model": nn_params, "normalizer": normalizer_params}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=pred_norm_delta)

    return model, params


def create_planar_drone_last_layer(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Planar drone model with only the last MLP layer trainable.

    Same architecture as create_planar_drone, but only the output layer
    is exposed for training. Hidden layers are randomly initialized and frozen
    (baked into the model closure).

    State: [p_x, p_y, phi, v_x, v_y, phi_dot] (6D)
    Action: [T_1, T_2] (2D rotor thrusts, non-negative)

    Residual Input: [v_x, v_y, phi, wind_x, wind_y] (normalized)
    Residual Output: [delta_a_x, delta_a_y, delta_alpha] (unnormalized)

    Trainable Params: Only output layer weights
    Frozen Params: Hidden layer weights (baked into closure)
    Known Constants: mass, gravity, arm_length, inertia, dt, wind_x, wind_y (from config)
    """
    # Physical parameters from config
    dyn_params = config["dynamics_params"]
    m = dyn_params["mass"]
    g = dyn_params["gravity"]
    L = dyn_params["arm_length"]
    I = dyn_params["inertia"]
    dt = dyn_params["dt"]
    nn_features = dyn_params["nn_features"]
    wind_x_param = dyn_params.get("wind_x", 0.0)
    wind_y_param = dyn_params.get("wind_y", 0.0)

    # Create MLP for residual acceleration prediction
    # Input: [v_x, v_y, phi, wind_x, wind_y] (5D normalized)
    # Output: acceleration residuals (3D) for [delta_a_x, delta_a_y, delta_alpha]
    class ResidualMLP(nn.Module):
        features: Sequence[int]

        @nn.compact
        def __call__(self, residual_input):
            x = residual_input
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.tanh(x)
            return nn.Dense(3)(x)  # 3 acceleration residuals

    # Initialize the full NN
    residual_net = ResidualMLP(features=nn_features)
    dummy_residual_input = jnp.ones((5,))  # v_x, v_y, phi, wind_x, wind_y
    full_nn_params = residual_net.init(key, dummy_residual_input)

    # Split params: hidden layers (frozen) vs output layer (trainable)
    # Flax names layers as Dense_0, Dense_1, ..., Dense_n where n is the output
    num_hidden = len(nn_features)
    output_layer_name = f"Dense_{num_hidden}"

    # Extract frozen hidden layer params (baked into closure)
    frozen_params = {"params": {}}
    for i in range(num_hidden):
        layer_name = f"Dense_{i}"
        frozen_params["params"][layer_name] = full_nn_params["params"][layer_name]

    # Extract trainable output layer params
    trainable_params = {"params": {output_layer_name: full_nn_params["params"][output_layer_name]}}

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts next state using nominal dynamics + NN residual (Semi-Implicit)."""
        p_x, p_y, phi, v_x, v_y, phi_dot = state
        T_1, T_2 = action[0], action[1]

        # Total thrust
        T_total = T_1 + T_2

        # Nominal accelerations (known physics, no wind)
        a_x_nom = (1 / m) * T_total * jnp.sin(phi)
        a_y_nom = (1 / m) * T_total * jnp.cos(phi) - g
        alpha_nom = (L / I) * (T_2 - T_1)

        # Normalize residual inputs: velocities, angle, wind
        norm_state = normalizer.normalize(params["normalizer"]["state"], state)
        norm_v_x = norm_state[3]
        norm_v_y = norm_state[4]
        norm_phi = norm_state[2]

        wind_arr = jnp.array([wind_x_param, wind_y_param])
        norm_wind = normalizer.normalize(params["normalizer"]["wind"], wind_arr)

        # Build 5D residual input
        residual_input = jnp.array([norm_v_x, norm_v_y, norm_phi, norm_wind[0], norm_wind[1]])

        # Reconstruct full NN params: frozen (from closure) + trainable (from input)
        full_params = {"params": {**frozen_params["params"], **params["model"]["params"]}}
        residual_normalized = residual_net.apply(full_params, residual_input)
        residual = normalizer.unnormalize(params["normalizer"]["delta"], residual_normalized)
        delta_a_x, delta_a_y, delta_alpha = residual[0], residual[1], residual[2]

        # Total accelerations
        a_x = a_x_nom + delta_a_x
        a_y = a_y_nom + delta_a_y
        alpha = alpha_nom + delta_alpha

        # --- SEMI-IMPLICIT EULER INTEGRATION ---

        # 1. Update velocities and angular rate FIRST
        v_x_next = v_x + a_x * dt
        v_y_next = v_y + a_y * dt
        phi_dot_next = phi_dot + alpha * dt

        # 2. Update positions and angle using the NEW velocities
        p_x_next = p_x + v_x_next * dt
        p_y_next = p_y + v_y_next * dt
        phi_next = phi + phi_dot_next * dt

        return jnp.array([p_x_next, p_y_next, phi_next, v_x_next, v_y_next, phi_dot_next])

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training)."""
        next_state = pred_one_step(params, state, action)
        delta = next_state - state
        if params["normalizer"] is not None:
            return normalizer.normalize(params["normalizer"]["delta"], delta)
        return delta

    # Only expose trainable (output layer) params
    params = {"model": trainable_params, "normalizer": normalizer_params}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=pred_norm_delta)

    return model, params

# --- TinyLoRA ---
def _compute_svd_components(W: jnp.ndarray, svd_rank: int) -> dict:
    """
    Compute truncated SVD of weight matrix.

    Args:
        W: Weight matrix (in_features, out_features)
        svd_rank: Rank of truncation (r)

    Returns:
        Dict with U (in_features, r), Sigma (r,), V (out_features, r)
    """
    U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
    U = U_full[:, :svd_rank]
    Sigma = S_full[:svd_rank]
    V = Vh_full[:svd_rank, :].T
    return {"U": U, "Sigma": Sigma, "V": V}


def _init_random_projections(
    key: jax.Array, steering_dim: int, svd_rank: int
) -> jnp.ndarray:
    """
    Initialize frozen random projection matrices P.

    Args:
        key: JAX random key
        steering_dim: Dimension of steering vector (u)
        svd_rank: SVD rank (r)

    Returns:
        P: Random projections (steering_dim, svd_rank, svd_rank)
    """
    scale = 1.0 / jnp.sqrt(steering_dim * svd_rank)
    return jax.random.normal(key, (steering_dim, svd_rank, svd_rank)) * scale


def _init_tiny_lora_layers(
    key: jax.Array,
    nn_features: Sequence[int],
    input_dim: int,
    output_dim: int,
    svd_rank: int,
    steering_dim: int,
    projection_seed: int,
    ensemble_size: int = 1,
) -> tuple[list, dict]:
    """
    Initialize TinyLoRA layers: compute SVD and random projections for each layer.

    Args:
        key: JAX random key for MLP initialization
        nn_features: Hidden layer sizes
        input_dim: Input dimension
        output_dim: Output dimension
        svd_rank: Rank of SVD truncation
        steering_dim: Dimension of steering vector per layer
        projection_seed: Seed for reproducible random projections
        ensemble_size: Number of ensemble members (default 1 for single model)

    Returns:
        frozen_layers: List of dicts with W, b, U, Sigma, V, P for each layer.
            When ensemble_size>1, each value has shape (E, ...).
        trainable_v_init: Dict of steering vectors {"v_0": ..., "v_1": ..., ...}.
            When ensemble_size>1, each value has shape (E, steering_dim).
    """
    # Split keys for each ensemble member
    keys = jax.random.split(key, ensemble_size)

    # Generate independent projection seeds for each ensemble member
    proj_base_key = jax.random.key(projection_seed)
    proj_keys = jax.random.split(proj_base_key, ensemble_size)

    # Create base MLP class
    class BaseMLP(nn.Module):
        features: Sequence[int]

        @nn.compact
        def __call__(self, x):
            for feat in self.features:
                x = nn.Dense(feat)(x)
                x = nn.tanh(x)
            return nn.Dense(output_dim)(x)

    base_mlp = BaseMLP(features=nn_features)
    dummy_input = jnp.ones((input_dim,))
    num_layers = len(nn_features) + 1

    # Initialize all ensemble members
    all_frozen_layers = []  # List of (list of dicts), one per ensemble member
    all_v_inits = []  # List of dicts, one per ensemble member

    for ens_idx in range(ensemble_size):
        full_nn_params = base_mlp.init(keys[ens_idx], dummy_input)

        frozen_layers_single = []
        v_init_single = {}

        proj_key = proj_keys[ens_idx]

        for i in range(num_layers):
            layer_name = f"Dense_{i}"
            W = full_nn_params["params"][layer_name]["kernel"]
            b = full_nn_params["params"][layer_name]["bias"]

            svd_components = _compute_svd_components(W, svd_rank)

            proj_key, layer_key = jax.random.split(proj_key)
            P = _init_random_projections(layer_key, steering_dim, svd_rank)

            frozen_layers_single.append({
                "W": W,
                "b": b,
                "U": svd_components["U"],
                "Sigma": svd_components["Sigma"],
                "V": svd_components["V"],
                "P": P,
            })

            v_init_single[f"v_{i}"] = jnp.zeros(steering_dim, dtype=jnp.float32)

        all_frozen_layers.append(frozen_layers_single)
        all_v_inits.append(v_init_single)

    # Return structure based on ensemble size
    if ensemble_size == 1:
        # Return original structure for backward compatibility
        return all_frozen_layers[0], all_v_inits[0]
    else:
        # Stack frozen layers: list of dicts with stacked arrays
        frozen_layers_stacked = []
        for layer_idx in range(num_layers):
            stacked_layer = {
                k: jnp.stack(
                    [all_frozen_layers[e][layer_idx][k] for e in range(ensemble_size)],
                    axis=0,
                )
                for k in all_frozen_layers[0][0].keys()
            }
            frozen_layers_stacked.append(stacked_layer)

        # Stack trainable params
        trainable_v_stacked = {
            k: jnp.stack([all_v_inits[e][k] for e in range(ensemble_size)], axis=0)
            for k in all_v_inits[0].keys()
        }

        return frozen_layers_stacked, trainable_v_stacked


def _tiny_lora_forward_single(
    x: jnp.ndarray,
    params_model: dict,
    frozen_layers: list,
) -> jnp.ndarray:
    """
    Forward pass through a single TinyLoRA MLP.

    Args:
        x: Input tensor
        params_model: Dict with steering vectors {"v_0": ..., "v_1": ..., ...}
        frozen_layers: List of frozen layer dicts from _init_tiny_lora_layers

    Returns:
        Output of the TinyLoRA MLP
    """
    num_layers = len(frozen_layers)

    for i in range(num_layers):
        layer = frozen_layers[i]
        v = params_model[f"v_{i}"]

        # Compute Delta = sum_j v[j] * P[j] using einsum
        Delta = jnp.einsum("u,urk->rk", v, layer["P"])

        # Compute weight modification: U @ diag(Sigma) @ Delta @ V.T
        SigmaDelta = layer["Sigma"][:, None] * Delta
        W_delta = layer["U"] @ SigmaDelta @ layer["V"].T

        # Effective weight
        W_eff = layer["W"] + W_delta

        # Apply layer
        x = x @ W_eff + layer["b"]

        # Apply activation (tanh) for all but last layer
        if i < num_layers - 1:
            x = jnp.tanh(x)

    return x


def _tiny_lora_forward(
    x: jnp.ndarray,
    params_model: dict,
    frozen_layers: list,
    ensemble_size: int = 1,
) -> jnp.ndarray:
    """
    Forward pass through TinyLoRA MLP(s).

    Args:
        x: Input tensor
        params_model: Dict with steering vectors {"v_0": ..., "v_1": ..., ...}.
            When ensemble_size>1, each value has shape (E, steering_dim).
        frozen_layers: List of frozen layer dicts from _init_tiny_lora_layers.
            When ensemble_size>1, each value has shape (E, ...).
        ensemble_size: Number of ensemble members (default 1 for single model)

    Returns:
        Output of the TinyLoRA MLP. When ensemble_size>1, returns the mean
        across ensemble members.
    """
    if ensemble_size == 1:
        return _tiny_lora_forward_single(x, params_model, frozen_layers)

    # Ensemble forward pass using vmap
    def forward_one_member(ens_idx):
        member_params = {k: params_model[k][ens_idx] for k in params_model}
        member_frozen = [
            {k: layer[k][ens_idx] for k in layer} for layer in frozen_layers
        ]
        return _tiny_lora_forward_single(x, member_params, member_frozen)

    outputs = jax.vmap(forward_one_member)(jnp.arange(ensemble_size))
    return jnp.mean(outputs, axis=0)


def create_planar_drone_tiny_lora(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Planar drone with TinyLoRA residual for efficient online adaptation.

    Applies TinyLoRA to ALL layers of the residual MLP.
    The effective weight for each layer is: W' = W + U @ diag(Sigma) @ (sum_i v[i] * P[i]) @ V.T

    State: [p_x, p_y, phi, v_x, v_y, phi_dot] (6D)
    Action: [T_1, T_2] (2D rotor thrusts)

    Residual Input: [v_x, v_y, phi, wind_x, wind_y] (5D normalized)
    Residual Output: [delta_a_x, delta_a_y, delta_alpha] (3D unnormalized)

    Combines:
    - Nominal drone physics (gravity, thrust, torque)
    - TinyLoRA-modified MLP residual for acceleration corrections [delta_a_x, delta_a_y, delta_alpha]

    Trainable Params: Steering vectors v for each layer (num_layers * steering_dim total)
    Frozen (in closure): For each layer: W, b, U, Sigma, V, P
    Known Constants: mass, gravity, arm_length, inertia, dt, wind_x, wind_y (from config)
    """
    dyn_params = config["dynamics_params"]
    m = dyn_params["mass"]
    g = dyn_params["gravity"]
    L = dyn_params["arm_length"]
    I = dyn_params["inertia"]
    dt = dyn_params["dt"]
    nn_features = dyn_params["nn_features"]
    svd_rank = dyn_params["svd_rank"]
    steering_dim = dyn_params["steering_dim"]
    projection_seed = dyn_params["projection_seed"]
    ensemble_size = dyn_params.get("ensemble_size", 1)
    wind_x_param = dyn_params.get("wind_x", 0.0)
    wind_y_param = dyn_params.get("wind_y", 0.0)

    # Initialize TinyLoRA layers for residual MLP
    # Input: [v_x, v_y, phi, wind_x, wind_y] = 5D, Output: 3D acceleration residuals
    frozen_layers, trainable_v_init = _init_tiny_lora_layers(
        key=key,
        nn_features=nn_features,
        input_dim=5,
        output_dim=3,
        svd_rank=svd_rank,
        steering_dim=steering_dim,
        projection_seed=projection_seed,
        ensemble_size=ensemble_size,
    )

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts next state using nominal dynamics + TinyLoRA residual."""
        p_x, p_y, phi, v_x, v_y, phi_dot = state
        T_1, T_2 = action[0], action[1]

        # Total thrust
        T_total = T_1 + T_2

        # Nominal accelerations (known physics, no wind)
        a_x_nom = (1 / m) * T_total * jnp.sin(phi)
        a_y_nom = (1 / m) * T_total * jnp.cos(phi) - g
        alpha_nom = (L / I) * (T_2 - T_1)

        # Normalize residual inputs: velocities, angle, wind
        norm_state = normalizer.normalize(params["normalizer"]["state"], state)
        norm_v_x = norm_state[3]
        norm_v_y = norm_state[4]
        norm_phi = norm_state[2]

        wind_arr = jnp.array([wind_x_param, wind_y_param])
        norm_wind = normalizer.normalize(params["normalizer"]["wind"], wind_arr)

        # Build 5D residual input
        residual_input = jnp.array([norm_v_x, norm_v_y, norm_phi, norm_wind[0], norm_wind[1]])

        x = _tiny_lora_forward(residual_input, params["model"], frozen_layers, ensemble_size)

        # Unnormalize residual
        residual = normalizer.unnormalize(params["normalizer"]["delta"], x)
        delta_a_x, delta_a_y, delta_alpha = residual[0], residual[1], residual[2]

        # Total accelerations
        a_x = a_x_nom + delta_a_x
        a_y = a_y_nom + delta_a_y
        alpha = alpha_nom + delta_alpha

        # Semi-implicit Euler integration
        v_x_next = v_x + a_x * dt
        v_y_next = v_y + a_y * dt
        phi_dot_next = phi_dot + alpha * dt

        p_x_next = p_x + v_x_next * dt
        p_y_next = p_y + v_y_next * dt
        phi_next = phi + phi_dot_next * dt

        return jnp.array([p_x_next, p_y_next, phi_next, v_x_next, v_y_next, phi_dot_next])

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the normalized change in state (for training)."""
        next_state = pred_one_step(params, state, action)
        delta = next_state - state
        if params["normalizer"] is not None:
            return normalizer.normalize(params["normalizer"]["delta"], delta)
        return delta

    params = {"model": trainable_v_init, "normalizer": normalizer_params}
    model = DynamicsModel(pred_one_step=pred_one_step, pred_norm_delta=pred_norm_delta)

    return model, params


def create_merging_idm(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for highway merging with 1 IDM vehicle.

    State: [ego_px, ego_py, ego_vx, ego_vy, idm_px, idm_vx] (6D)
    Action: [ax, ay] (2D ego acceleration)
    Trainable Params: T (scalar) time headway, b (scalar) comfortable deceleration,
                      k_lat (scalar) lateral sigmoid steepness, d0 (scalar) lateral distance threshold.
    Known Constants: v0, s0, a_max, delta, L, k_lon, s_min, p_y_target, dt.
    """
    dp = config["dynamics_params"]
    dt = dp["dt"]
    v0 = dp["v0"]
    s0 = dp["s0"]
    a_max_idm = dp["a_max"]
    delta = dp["delta"]
    L = dp["L"]
    k_lon = dp["k_lon"]
    s_min = dp["s_min"]
    p_y_target = dp["p_y_target"]

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts next state using merging IDM dynamics with learnable T, b, k_lat, d0."""
        T_val = params["model"]["T"]    # scalar
        b_val = params["model"]["b"]    # scalar
        k_lat_val = params["model"]["k_lat"]  # scalar
        d0_val = params["model"]["d0"]  # scalar

        ego_px, ego_py, ego_vx, ego_vy = state[0], state[1], state[2], state[3]
        ax, ay = action[0], action[1]

        # Ego double integrator
        next_ego_px = ego_px + ego_vx * dt + 0.5 * ax * dt**2
        next_ego_py = ego_py + ego_vy * dt + 0.5 * ay * dt**2
        next_ego_vx = jnp.maximum(ego_vx + ax * dt, 0.0)
        next_ego_vy = ego_vy + ay * dt

        # Lateral proximity sigmoid
        sigma_lat = 1.0 / (1.0 + jnp.exp(k_lat_val * (jnp.abs(ego_py - p_y_target) - d0_val)))

        # IDM vehicle state
        idm_px, idm_vx = state[4], state[5]

        # No in-lane leader → infinite gap, zero approach rate
        s_lane = 1000.0
        dv_lane = 0.0

        # Gap and approach rate w.r.t. ego
        s_ego = ego_px - idm_px - L
        dv_ego = idm_vx - ego_vx

        # Blending weights
        sigma_lon = 1.0 / (1.0 + jnp.exp(-k_lon * (ego_px - idm_px)))
        alpha = sigma_lat * sigma_lon

        # Blended effective quantities
        s_eff = alpha * s_ego + (1.0 - alpha) * s_lane
        dv_eff = alpha * dv_ego + (1.0 - alpha) * dv_lane

        # Safety clamp
        s_eff = jax.nn.softplus(s_eff - s_min) + s_min

        # IDM acceleration
        s_star = s0 + idm_vx * T_val + idm_vx * dv_eff / (2.0 * jnp.sqrt(a_max_idm * b_val + 1e-8))
        a_idm = a_max_idm * (1.0 - (idm_vx / v0) ** delta - (s_star / s_eff) ** 2)

        # Update IDM vehicle
        next_idm_px = idm_px + idm_vx * dt + 0.5 * a_idm * dt**2
        next_idm_vx = jnp.maximum(idm_vx + a_idm * dt, 0.0)

        return jnp.array([
            next_ego_px, next_ego_py, next_ego_vx, next_ego_vy,
            next_idm_px, next_idm_vx,
        ])

    # Initialize learnable parameters (all scalars)
    init_T = jnp.array(dp["init_T"], dtype=jnp.float32)
    init_b = jnp.array(dp["init_b"], dtype=jnp.float32)
    init_k_lat = jnp.array(dp["init_k_lat"], dtype=jnp.float32)
    init_d0 = jnp.array(dp["init_d0"], dtype=jnp.float32)

    model_params = {
        "T": init_T,
        "b": init_b,
        "k_lat": init_k_lat,
        "d0": init_d0,
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

    if dynamics_type == "mlp_resnet":
        model, params = create_mlp_resnet(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "mlp_resnet_last_layer":
        model, params = create_mlp_resnet_last_layer(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "mlp_resnet_tiny_lora":
        model, params = create_mlp_resnet_tiny_lora(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "probabilistic_ensemble":
        dim_state = config.dim_state
        dim_action = config.dim_action
        model, params = create_probabilistic_ensemble(
            key, dim_state, dim_action, config, normalizer, normalizer_params
        )

    elif dynamics_type == "analytical_pendulum":
        model, params = create_analytical_pendulum()

    elif dynamics_type == "pursuit_evader":
        model, params = create_pursuit_evader(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "linear":
        model, params = create_linear(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "damped_pendulum":
        model, params = create_damped_pendulum(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "unicycle":
        model, params = create_pursuit_evader_unicycle(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "merging_idm":
        model, params = create_merging_idm(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "planar_drone":
        model, params = create_planar_drone(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "planar_drone_last_layer":
        model, params = create_planar_drone_last_layer(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "planar_drone_tiny_lora":
        model, params = create_planar_drone_tiny_lora(
            key, config, normalizer, normalizer_params
        )

    else:
        raise ValueError(f"Unknown dynamics type: '{dynamics_type}'")

    # Check for pretrained parameters
    pretrained_path = config.get("dynamics_params", {}).get("pretrained_params_path")
    if pretrained_path:
        with open(pretrained_path, "rb") as f:
            params = pickle.load(f)
        print(f"📦 Loaded pretrained params from {pretrained_path}")

    return model, params
