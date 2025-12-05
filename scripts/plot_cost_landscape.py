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

n_grid = 50
u_range = jnp.linspace(-2.0, 2.0, n_grid)
u1_grid, u2_grid = jnp.meshgrid(u_range, u_range)
controls = jnp.stack([u1_grid.ravel(), u2_grid.ravel()], axis=-1)

@jax.jit
def compute_info_cost(control):
    return info_term_fn(state, control, dyn_params, params_cov)

info_costs = jax.vmap(compute_info_cost)(controls).reshape(n_grid, n_grid)

proc_cov = jnp.zeros((dim_params_model, dim_params_model))
meas_cov = jnp.eye(dim_state) * 0.1

@jax.jit
def compute_ekf_trace(control):
    next_state = true_params["model"]
    true_next = dynamics_model.pred_one_step(true_params, state, control)
    ekf_inp = jnp.concatenate([state, control])
    ekf_out = true_next - state
    _, cov_tp1, _ = estimator.estimate(
        flat_params_model, params_cov, ekf_inp, ekf_out, proc_cov, meas_cov
    )
    return jnp.trace(cov_tp1)

ekf_traces = jax.vmap(compute_ekf_trace)(controls).reshape(n_grid, n_grid)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im0 = axes[0].imshow(
    info_costs, extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[0].set_xlabel("u_1")
axes[0].set_ylabel("u_2")
axes[0].set_title("Information-Gathering Cost")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(
    ekf_traces, extent=[-2, 2, -2, 2], origin="lower", aspect="auto", cmap="viridis"
)
axes[1].set_xlabel("u_1")
axes[1].set_ylabel("u_2")
axes[1].set_title("Trace of EKF Covariance")
plt.colorbar(im1, ax=axes[1])

plt.tight_layout()
plt.savefig("heatmaps.png", dpi=150)
print("Saved heatmaps.png")
