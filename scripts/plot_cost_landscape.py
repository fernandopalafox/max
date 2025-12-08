import jax
import jax.numpy as jnp
import jax.flatten_util
import matplotlib.pyplot as plt
from max.dynamics import create_pursuit_evader_dynamics
from max.costs import make_info_gathering_term
from max.estimators import EKFCovArgs

config = {
    "dynamics_params": {
        "dt": 0.1,
        "init_q_diag": [1.0, 1.0, 1.0, 1.0],
        "init_r_diag": [1.0, 1.0],
    }
}

true_q_diag = jnp.array([10.0, 10.0, 1.0, 1.0])
true_r_diag = jnp.array([0.1, 0.1])
true_params = {
    "model": {
        "q_cholesky": jnp.diag(jnp.sqrt(true_q_diag)),
        "r_cholesky": jnp.diag(jnp.sqrt(true_r_diag)),
    },
    "normalizer": None,
}

dynamics_model, init_params = create_pursuit_evader_dynamics(config, None, None)

meas_noise_diag = jnp.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
info_term_fn = make_info_gathering_term(dynamics_model.pred_one_step, meas_noise_diag)

dim_state = 8
dim_action = 2
flat_params_model, unflatten_fn_model = jax.flatten_util.ravel_pytree(
    init_params["model"]
)
dim_params_model = flat_params_model.shape[0]
init_covariance = jnp.eye(dim_params_model)


@jax.jit
def parameter_dynamics_fn(params, _):
    return params


@jax.jit
def observation_fn(params, x):
    state = x[:dim_state]
    action = x[dim_state : dim_state + dim_action]
    params_model = unflatten_fn_model(params)
    params_pytree = {"model": params_model, "normalizer": None}
    pred_next_state = dynamics_model.pred_one_step(params_pytree, state, action)
    return pred_next_state - state


estimator = EKFCovArgs(
    dynamics_fn=parameter_dynamics_fn,
    observation_fn=observation_fn,
    jitter=1e-6,
)

state = jnp.array([2.0, 2.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
params_cov = init_covariance
dyn_params = init_params

proc_cov = jnp.zeros((dim_params_model, dim_params_model))
meas_cov = jnp.eye(dim_state) * 0.1

# Create a meshgrid for u_0
n_grid = 50
u_range = jnp.linspace(-2.0, 2.0, n_grid)
u1_grid, u2_grid = jnp.meshgrid(u_range, u_range)
controls_t0 = jnp.stack([u1_grid.ravel(), u2_grid.ravel()], axis=-1)

# Fixed control for t=1 (Assumed zero to isolate effect of u_0)
control_t1 = jnp.zeros(dim_action)


@jax.jit
def compute_2step_metrics(control_0):
    # --- STEP 1 (t=0 -> t=1) ---
    params_mean = {"model": unflatten_fn_model(flat_params_model), "normalizer": None}
    state_1 = dynamics_model.pred_one_step(params_mean, state, control_0)

    ekf_inp_0 = jnp.concatenate([state, control_0])
    _, cov_1, _ = estimator.estimate(
        flat_params_model,
        params_cov,
        ekf_inp_0,
        jnp.zeros(dim_state),
        proc_cov,
        meas_cov,
    )

    # --- STEP 2 (t=1 -> t=2) ---

    # 1. Calculate Information Cost (The Proxy)
    # This uses log(det(pred_cov)) internally
    info_cost_val = info_term_fn(state_1, control_t1, dyn_params, cov_1)

    # 2. Calculate EKF Log-Determinant (The Truth)
    ekf_inp_1 = jnp.concatenate([state_1, control_t1])
    _, cov_2, _ = estimator.estimate(
        flat_params_model, cov_1, ekf_inp_1, jnp.zeros(dim_state), proc_cov, meas_cov
    )

    # CHANGE: Replaced jnp.trace with slogdet
    # We use slogdet for numerical stability.
    # The sign should always be 1 for a valid Covariance matrix.
    _, logdet_val = jnp.linalg.slogdet(cov_2)

    return info_cost_val, logdet_val


# Vectorize and compute
results = jax.vmap(compute_2step_metrics)(controls_t0)
info_costs_flat, ekf_logdet_flat = results  # Renamed variable

info_costs = info_costs_flat.reshape(n_grid, n_grid)
ekf_logdets = ekf_logdet_flat.reshape(n_grid, n_grid)  # Renamed variable

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Negative Info Cost (Proxy)
# High Gain = Low Cost. We plot negative so "valleys" are good.
im0 = axes[0].imshow(
    -info_costs, extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[0].set_xlabel("u_0 (x)")
axes[0].set_ylabel("u_0 (y)")
axes[0].set_title("(-) Info Gain at t=2\n(Lower is Better)")
plt.colorbar(im0, ax=axes[0])

# Plot 2: Log Determinant of Covariance (Truth)
# Low LogDet = Low Entropy = Low Uncertainty. "Valleys" are good.
im1 = axes[1].imshow(
    ekf_logdets, extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[1].set_xlabel("u_0 (x)")
axes[1].set_ylabel("u_0 (y)")
axes[1].set_title("Log-Det of Covariance at t=2\n(Lower is Better)")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("heatmaps.png", dpi=150)
print("Saved heatmaps.png")
