"""
Benchmark: Python for loop vs jax.lax.scan for a toy MLP rollout.

Usage:
    conda run -n max python scripts/benchmark_scan.py
"""

import time
import jax
import jax.numpy as jnp
import numpy as np

# ---- Config ----
B = 256       # batch size
H = 16        # horizon
DIM_Z = 128   # latent dim
DIM_A = 6     # action dim
DIM_H = 256   # hidden dim
N_WARMUP = 3
N_BENCH = 100


# ---- Toy MLP dynamics step ----
def mlp_step(params, z, a):
    """One step of a 2-layer MLP: [z, a] -> z_next."""
    x = jnp.concatenate([z, a], axis=-1)
    x = jax.nn.relu(x @ params["W1"] + params["b1"])
    x = x @ params["W2"] + params["b2"]
    return x


def init_params(key):
    k1, k2, k3, k4 = jax.random.split(key, 4)
    scale = 0.01
    return {
        "W1": jax.random.normal(k1, (DIM_Z + DIM_A, DIM_H)) * scale,
        "b1": jnp.zeros(DIM_H),
        "W2": jax.random.normal(k2, (DIM_H, DIM_Z)) * scale,
        "b2": jnp.zeros(DIM_Z),
    }


# ---- Loop rollout ----
def loop_rollout(z0, actions, params):
    """Roll out dynamics for H steps using a Python for loop."""
    z = z0
    zs = [z0]
    for t in range(H):
        z = mlp_step(params, z, actions[:, t])
        zs.append(z)
    return jnp.stack(zs, axis=1)  # (B, H+1, DIM_Z)


# ---- Scan rollout ----
def scan_rollout(z0, actions, params):
    """Roll out dynamics for H steps using jax.lax.scan."""
    def step(z, a_t):
        z_next = mlp_step(params, z, a_t)
        return z_next, z_next

    actions_T = jnp.transpose(actions, (1, 0, 2))  # (H, B, DIM_A)
    _, z_preds = jax.lax.scan(step, z0, actions_T)  # z_preds: (H, B, DIM_Z)
    return jnp.concatenate(
        [z0[:, None, :], jnp.transpose(z_preds, (1, 0, 2))], axis=1
    )  # (B, H+1, DIM_Z)


loop_rollout_jit = jax.jit(loop_rollout)
scan_rollout_jit = jax.jit(scan_rollout)


def bench(fn, z0, actions, params, n_warmup, n_bench):
    # Warmup (triggers compilation)
    for _ in range(n_warmup):
        out = fn(z0, actions, params)
        jax.block_until_ready(out)

    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        out = fn(z0, actions, params)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t0)
    return np.array(times)


if __name__ == "__main__":
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)

    z0 = jax.random.normal(k1, (B, DIM_Z))
    actions = jax.random.normal(k2, (B, H, DIM_A))
    params = init_params(k3)

    print(f"Config: B={B}, H={H}, dim_z={DIM_Z}, dim_a={DIM_A}")
    print(f"Warmup: {N_WARMUP}x, Benchmark: {N_BENCH}x\n")

    loop_times = bench(loop_rollout_jit, z0, actions, params, N_WARMUP, N_BENCH)
    scan_times = bench(scan_rollout_jit, z0, actions, params, N_WARMUP, N_BENCH)

    print(f"loop_rollout: {loop_times.mean()*1e3:.3f} ± {loop_times.std()*1e3:.3f} ms")
    print(f"scan_rollout: {scan_times.mean()*1e3:.3f} ± {scan_times.std()*1e3:.3f} ms")
    speedup = loop_times.mean() / scan_times.mean()
    print(f"Speedup (loop/scan): {speedup:.2f}x")
