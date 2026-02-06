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
    pred_one_step_with_info: Callable[[Any, jnp.ndarray, jnp.ndarray], tuple] = None


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


def create_pursuit_evader_dynamics_unicycle(
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



def create_linear_dynamics(
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


def create_damped_pendulum_dynamics(
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


def create_merging_idm_dynamics(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for highway merging with 3 IDM vehicles.

    State: [ego_px, ego_py, ego_vx, ego_vy, v2_px, v2_vx, v3_px, v3_vx, v4_px, v4_vx] (10D)
    Action: [ax, ay] (2D ego acceleration)
    Trainable Params: T (2,) time headway, b (2,) comfortable deceleration for V3 & V4.
    Fixed Params: V2 (lead vehicle) T and b from config.
    Known Constants: v0, s0, a_max, delta, L, k_lat, d0, k_lon, s_min, p_y_target, dt.
    """
    dp = config["dynamics_params"]
    dt = dp["dt"]
    v0 = dp["v0"]
    s0 = dp["s0"]
    a_max_idm = dp["a_max"]
    delta = dp["delta"]
    L = dp["L"]
    k_lat = dp["k_lat"]
    d0 = dp["d0"]
    k_lon = dp["k_lon"]
    s_min = dp["s_min"]
    p_y_target = dp["p_y_target"]
    fixed_T_v2 = dp["fixed_T_v2"]
    fixed_b_v2 = dp["fixed_b_v2"]

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts next state using merging IDM dynamics with learnable T, b for V3/V4."""
        T_learn = params["model"]["T"]  # (2,) — V3, V4 only
        b_learn = params["model"]["b"]  # (2,)
        T_vec = jnp.concatenate([jnp.array([fixed_T_v2]), T_learn])
        b_vec = jnp.concatenate([jnp.array([fixed_b_v2]), b_learn])

        ego_px, ego_py, ego_vx, ego_vy = state[0], state[1], state[2], state[3]
        ax, ay = action[0], action[1]

        # Ego double integrator
        next_ego_px = ego_px + ego_vx * dt + 0.5 * ax * dt**2
        next_ego_py = ego_py + ego_vy * dt + 0.5 * ay * dt**2
        next_ego_vx = jnp.maximum(ego_vx + ax * dt, 0.0)
        next_ego_vy = ego_vy + ay * dt

        # Lateral proximity sigmoid (shared)
        sigma_lat = 1.0 / (1.0 + jnp.exp(k_lat * (jnp.abs(ego_py - p_y_target) - d0)))

        # IDM vehicle states
        v2_px, v2_vx = state[4], state[5]
        v3_px, v3_vx = state[6], state[7]
        v4_px, v4_vx = state[8], state[9]

        veh_px = jnp.array([v2_px, v3_px, v4_px])
        veh_vx = jnp.array([v2_vx, v3_vx, v4_vx])

        # In-lane gaps and approach rates
        s_lane = jnp.array([1000.0, v2_px - v3_px - L, v3_px - v4_px - L])
        dv_lane = jnp.array([0.0, v3_vx - v2_vx, v4_vx - v3_vx])

        # Gaps and approach rates w.r.t. ego
        s_ego = ego_px - veh_px - L
        dv_ego = veh_vx - ego_vx

        # Blending weights
        sigma_lon = 1.0 / (1.0 + jnp.exp(-k_lon * (ego_px - veh_px)))
        alpha = sigma_lat * sigma_lon

        # Blended effective quantities
        s_eff = alpha * s_ego + (1.0 - alpha) * s_lane
        dv_eff = alpha * dv_ego + (1.0 - alpha) * dv_lane

        # Safety clamp
        s_eff = jax.nn.softplus(s_eff - s_min) + s_min

        # IDM acceleration for each vehicle
        s_star = s0 + veh_vx * T_vec + veh_vx * dv_eff / (2.0 * jnp.sqrt(a_max_idm * b_vec + 1e-8))
        a_idm = a_max_idm * (1.0 - (veh_vx / v0) ** delta - (s_star / s_eff) ** 2)

        # Update IDM vehicles
        next_veh_px = veh_px + veh_vx * dt + 0.5 * a_idm * dt**2
        next_veh_vx = jnp.maximum(veh_vx + a_idm * dt, 0.0)

        return jnp.array([
            next_ego_px, next_ego_py, next_ego_vx, next_ego_vy,
            next_veh_px[0], next_veh_vx[0],
            next_veh_px[1], next_veh_vx[1],
            next_veh_px[2], next_veh_vx[2],
        ])

    # Initialize learnable parameters
    init_T = jnp.array(dp["init_T"])
    init_b = jnp.array(dp["init_b"])

    model_params = {
        "T": init_T,
        "b": init_b,
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

    elif dynamics_type == "linear":
        return create_linear_dynamics(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "damped_pendulum":
        return create_damped_pendulum_dynamics(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "unicycle":
        return create_pursuit_evader_dynamics_unicycle(
            config, normalizer, normalizer_params
        )

    elif dynamics_type == "merging_idm":
        return create_merging_idm_dynamics(
            config, normalizer, normalizer_params
        )

    else:
        raise ValueError(f"Unknown dynamics type: '{dynamics_type}'")
