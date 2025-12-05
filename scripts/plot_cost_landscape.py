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
flat_params_model, unflatten_fn_model = jax.flatten_util.ravel_pytree(init_params["model"])
dim_params_model = flat_params_model.shape[0]
init_covariance = jnp.eye(dim_params_model)

@jax.jit
def parameter_dynamics_fn(params, _):
    return params

@jax.jit
def observation_fn(params, x):
    state = x[:dim_state]
    action = x[dim_state:dim_state + dim_action]
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
    # Apply u_0. This moves us to x_1.
    # Note: The covariance update here is CONSTANT w.r.t control_0, 
    # but we must compute it to get the correct starting cov for step 2.
    
    # 1. Predict Next State x_1
    params_mean = {"model": unflatten_fn_model(flat_params_model), "normalizer": None}
    state_1 = dynamics_model.pred_one_step(params_mean, state, control_0)
    
    # 2. Update Covariance to Sigma_1
    # We use a dummy observation (zeros) because obs value doesn't affect covariance update
    ekf_inp_0 = jnp.concatenate([state, control_0])
    _, cov_1, _ = estimator.estimate(
        flat_params_model, params_cov, ekf_inp_0, jnp.zeros(dim_state), proc_cov, meas_cov
    )
    
    # --- STEP 2 (t=1 -> t=2) ---
    # Now we are at x_1. We apply fixed control_t1.
    # The pursuer reacts to x_1. THIS is where u_0 pays off.
    
    # 1. Calculate Information Cost (The Proxy)
    # We calculate the info gain of the transition x_1 -> x_2
    # Note: We pass cov_1, which is the covariance entering this step
    info_cost_val = info_term_fn(state_1, control_t1, dyn_params, cov_1)
    
    # 2. Calculate EKF Trace (The Truth)
    # We perform the actual EKF update to get Sigma_2
    ekf_inp_1 = jnp.concatenate([state_1, control_t1])
    _, cov_2, _ = estimator.estimate(
        flat_params_model, cov_1, ekf_inp_1, jnp.zeros(dim_state), proc_cov, meas_cov
    )
    trace_val = jnp.trace(cov_2)
    
    return info_cost_val, trace_val

# Vectorize and compute
results = jax.vmap(compute_2step_metrics)(controls_t0)
info_costs_flat, ekf_traces_flat = results

info_costs = info_costs_flat.reshape(n_grid, n_grid)
ekf_traces = ekf_traces_flat.reshape(n_grid, n_grid)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Negative sign on Info Cost?
# Ideally, your info_term_fn returns POSITIVE gain (higher is better).
# If your optimizer minimizes NEGATIVE gain, plot the negative here to match "valleys".
# Assuming info_term_fn returns entropy reduction (positive good):
im0 = axes[0].imshow(
    -info_costs, # Flip sign so "Good" is a valley (blue/purple)
    extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[0].set_xlabel("u_0 (x)")
axes[0].set_ylabel("u_0 (y)")
axes[0].set_title("(-) Info Gain at t=2\n(Lower is Better)")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    ekf_traces, 
    extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[1].set_xlabel("u_0 (x)")
axes[1].set_ylabel("u_0 (y)")
axes[1].set_title("Trace of Covariance at t=2\n(Lower is Better)")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("heatmaps_2step.png", dpi=150)
print("Saved heatmaps_2step.png")