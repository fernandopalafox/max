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


def create_unicycle_mpc_dynamics(
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Creates dynamics for a 2-player system where player 2 (opponent) uses
    unicycle dynamics and plans via 2-step MPC.

    Player 1 (evader): double integrator, action is acceleration
    Player 2 (opponent): unicycle, action computed via MPC

    State: 10D vector [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2, 0, 0]
           (padded to 10D for compatibility, last 2 unused)
    Action: 2D vector [a1x, a1y] for player 1's acceleration

    Learnable params:
        - theta1: position cost weight in opponent's MPC
        - theta2: role depends on config["dynamics_params"]["theta2_role"]:
            - "accel_scaling" (default): acceleration scaling in dynamics
            - "turn_penalty": angular velocity penalty weight in MPC cost
    """
    dt = config["dynamics_params"]["dt"]
    newton_iters = config["dynamics_params"].get("newton_iters", 10)
    weight_w = config["dynamics_params"].get("weight_w", 1.0)  # angular velocity penalty
    weight_a = config["dynamics_params"].get("weight_a", 1.0)  # acceleration penalty
    weight_speed = config["dynamics_params"].get("weight_speed", 0.0)  # speed deviation penalty
    target_speed = config["dynamics_params"].get("target_speed", 1.0)  # desired cruising speed
    theta2_role = config["dynamics_params"].get("theta2_role", "accel_scaling")

    # --- Unicycle helpers ---
    def unicycle_step(x2, u2, theta2):
        """
        Single unicycle step.
        x2 = [p2x, p2y, alpha2, v2]
        u2 = [w, a] (angular velocity, acceleration)
        """
        p2x, p2y, alpha2, v2 = x2
        w, a = u2

        p2x_next = p2x + dt * v2 * jnp.cos(alpha2)
        p2y_next = p2y + dt * v2 * jnp.sin(alpha2)
        alpha2_next = alpha2 + dt * w
        # theta2 only affects acceleration if theta2_role is "accel_scaling"
        if theta2_role == "accel_scaling":
            v2_next = v2 + dt * a * theta2
        else:
            v2_next = v2 + dt * a  # fixed acceleration scaling

        return jnp.array([p2x_next, p2y_next, alpha2_next, v2_next])

    def rollout_2step(x2_0, u0, theta2):
        """Rollout 2 steps. u1=0 since it doesn't affect terminal position."""
        x2_1 = unicycle_step(x2_0, u0, theta2)
        x2_2 = unicycle_step(x2_1, jnp.zeros(2), theta2)
        return x2_2

    def mpc_cost(u0, x2_0, target_pos, theta1, theta2):
        """
        Opponent's MPC cost: track player 1's position + control penalties + speed penalty.
        """
        x2_2 = rollout_2step(x2_0, u0, theta2)
        pos_err = x2_2[0:2] - target_pos
        v2_final = x2_2[3]  # final speed
        w, a = u0
        # theta2 affects turn penalty if theta2_role is "turn_penalty"
        if theta2_role == "turn_penalty":
            # Use exp(theta2) to ensure turn penalty is always positive
            turn_cost = jnp.exp(theta2) * w**2
        else:
            turn_cost = weight_w * w**2
        # Position tracking + control costs + speed regulation
        return (theta1 * jnp.sum(pos_err**2)
                + turn_cost
                + weight_a * a**2
                + weight_speed * (v2_final - target_speed)**2)

    def kkt_residual(u0, x2_0, target_pos, theta1, theta2):
        """Stationarity condition: ∇_u cost = 0"""
        return jax.grad(mpc_cost, argnums=0)(u0, x2_0, target_pos, theta1, theta2)

    # --- Newton solver ---
    def newton_solve(x2_0, target_pos, theta1, theta2):
        """Solve for optimal u0 via Newton's method."""

        def newton_step(u, _):
            r = kkt_residual(u, x2_0, target_pos, theta1, theta2)
            J = jax.jacobian(kkt_residual, argnums=0)(
                u, x2_0, target_pos, theta1, theta2
            )
            # Damped Newton with regularization for stability
            J_reg = J + 1e-6 * jnp.eye(2)
            du = jnp.linalg.solve(J_reg, r)
            return u - du, None

        u_init = jnp.zeros(2)
        u_star, _ = jax.lax.scan(newton_step, u_init, None, length=newton_iters)
        return u_star

    # --- Implicit differentiation via custom_vjp ---
    @jax.custom_vjp
    def solve_mpc(x2_0, target_pos, theta):
        """
        Solve opponent's MPC. Returns optimal control u*.
        Differentiable w.r.t. theta via implicit differentiation.
        """
        theta1, theta2 = theta
        return newton_solve(x2_0, target_pos, theta1, theta2)

    def solve_mpc_fwd(x2_0, target_pos, theta):
        u_star = solve_mpc(x2_0, target_pos, theta)
        return u_star, (u_star, x2_0, target_pos, theta)

    def solve_mpc_bwd(res, g):
        u_star, x2_0, target_pos, theta = res
        theta1, theta2 = theta

        # Jacobians of KKT residual at solution
        dF_du = jax.jacobian(kkt_residual, argnums=0)(
            u_star, x2_0, target_pos, theta1, theta2
        )

        # Jacobian w.r.t. theta = [theta1, theta2]
        def kkt_theta(th):
            return kkt_residual(u_star, x2_0, target_pos, th[0], th[1])

        dF_dtheta = jax.jacobian(kkt_theta)(theta)

        # Solve [∂F/∂u]^T v = g
        dF_du_reg = dF_du + 1e-6 * jnp.eye(2)
        v = jnp.linalg.solve(dF_du_reg.T, g)

        # ∂L/∂theta = -v^T @ ∂F/∂theta
        dtheta = -v @ dF_dtheta

        # No gradient w.r.t. x2_0 or target_pos for now
        return (None, None, dtheta)

    solve_mpc.defvjp(solve_mpc_fwd, solve_mpc_bwd)

    # --- Full dynamics step ---
    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Predicts next state for the 2-player system.

        Player 1: double integrator with input acceleration
        Player 2: unicycle with MPC-computed control
        """
        # Unpack state
        p1 = state[0:2]  # player 1 position
        v1 = state[2:4]  # player 1 velocity
        p2 = state[4:6]  # player 2 position
        alpha2 = state[6]  # player 2 heading
        v2_scalar = state[7]  # player 2 speed

        # Player 1 action
        a1 = action.squeeze()

        # Unpack learnable params
        theta1 = params["model"]["theta1"]
        theta2 = params["model"]["theta2"]
        theta = jnp.array([theta1, theta2])

        # Player 2's state for MPC
        x2_0 = jnp.array([p2[0], p2[1], alpha2, v2_scalar])

        # Player 2's target: player 1's current position
        target_pos = p1

        # Solve MPC for player 2's control
        u2_star = solve_mpc(x2_0, target_pos, theta)

        # Clamp MPC output to prevent blowup from Newton solver divergence
        max_angular_vel = 10.0  # rad/s
        max_accel = 20.0  # m/s^2
        u2_star = jnp.array([
            jnp.clip(u2_star[0], -max_angular_vel, max_angular_vel),
            jnp.clip(u2_star[1], -max_accel, max_accel)
        ])

        # --- Update player 1 (double integrator) ---
        next_v1 = v1 + a1 * dt
        next_p1 = p1 + v1 * dt + 0.5 * a1 * dt**2

        # --- Update player 2 (unicycle, single step) ---
        x2_1 = unicycle_step(x2_0, u2_star, theta2)
        next_p2 = x2_1[0:2]
        next_alpha2 = x2_1[2]
        next_v2 = x2_1[3]

        # Pack state (8D for unicycle)
        # [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
        return jnp.array([
            next_p1[0], next_p1[1],
            next_v1[0], next_v1[1],
            next_p2[0], next_p2[1],
            next_alpha2, next_v2
        ])

    # --- Initialize learnable parameters ---
    init_theta1 = config["dynamics_params"]["init_theta1"]
    init_theta2 = config["dynamics_params"]["init_theta2"]

    model_params = {
        "theta1": jnp.array(init_theta1),
        "theta2": jnp.array(init_theta2),
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

    elif dynamics_type == "unicycle_mpc":
        return create_unicycle_mpc_dynamics(
            config, normalizer, normalizer_params
        )

    else:
        raise ValueError(f"Unknown dynamics type: '{dynamics_type}'")
