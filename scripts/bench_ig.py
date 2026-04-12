"""
Benchmark: log-det vs trace approximation for info-gain computation.
Usage: python scripts/bench_ig.py
"""
import os
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache_bench"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time, json, pickle
import jax
import jax.numpy as jnp
import jax.flatten_util

from max.dynamics import init_dynamics
from max.encoders import init_encoder

CONFIG_PATH = "configs/cheetah.json"
PRETRAINED_PATH = "data/models/cheetah/20260323_less_features/final.pkl"
BATCH_SIZE = 64
WARMUP = 3
RUNS = 20

def main():
    with open(CONFIG_PATH) as f:
        config = json.load(f)["training"]

    with open(PRETRAINED_PATH, "rb") as f:
        pretrained = pickle.load(f)["mean"]

    key = jax.random.key(0)
    key, enc_key, dyn_key = jax.random.split(key, 3)
    encoder, enc_params = init_encoder(enc_key, config, pretrained=pretrained.get("encoder"))
    dynamics, dyn_params = init_dynamics(dyn_key, config, pretrained=pretrained.get("dynamics"))

    flat_params, unflatten = jax.flatten_util.ravel_pytree(dyn_params)
    n_params = flat_params.shape[0]

    z0 = encoder.encode(enc_params, jnp.zeros(config["dim_state"]))
    latent_dim = z0.shape[0]
    dim_a = config["dim_action"]
    meas_noise_scale = config["planner"]["meas_noise_scale"]

    action_seqs = jnp.zeros((BATCH_SIZE, config["planner"]["horizon"], dim_a))
    P = jnp.eye(n_params) * 0.01
    R = meas_noise_scale * jnp.eye(latent_dim)

    print(f"n_params={n_params}, latent_dim={latent_dim}, batch={BATCH_SIZE}, meas_noise_scale={meas_noise_scale}")

    def get_J(actions):
        return jax.jacrev(lambda fp: dynamics.predict(unflatten(fp), z0, actions[0]))(flat_params)

    # --- log-det (current) ---
    def logdet_gain(actions):
        J = get_J(actions)
        S = J @ P @ J.T + R
        return 0.5 * (jnp.linalg.slogdet(S)[1] - jnp.linalg.slogdet(R)[1])

    logdet_fn = jax.jit(lambda seqs: jax.vmap(logdet_gain)(seqs))

    # --- trace approximation ---
    def trace_gain(actions):
        J = get_J(actions)
        return 0.5 / meas_noise_scale * jnp.sum(J * (J @ P))

    trace_fn = jax.jit(lambda seqs: jax.vmap(trace_gain)(seqs))

    # --- matmul only (baseline: cost of J @ P alone) ---
    def matmul_only(actions):
        J = get_J(actions)
        return jnp.sum(J @ P)  # forces J@P to be computed

    matmul_fn = jax.jit(lambda seqs: jax.vmap(matmul_only)(seqs))

    def bench(fn, seqs, name):
        for _ in range(WARMUP):
            jax.block_until_ready(fn(seqs))
        t0 = time.perf_counter()
        for _ in range(RUNS):
            jax.block_until_ready(fn(seqs))
        dt = (time.perf_counter() - t0) / RUNS
        print(f"{name:<20} {dt*1000:.1f} ms/call")
        return dt

    print()
    dt_matmul  = bench(matmul_fn,  action_seqs, "matmul only")
    dt_trace   = bench(trace_fn,   action_seqs, "trace approx")
    dt_logdet  = bench(logdet_fn,  action_seqs, "log-det (current)")

    print(f"\ntrace vs log-det speedup: {dt_logdet/dt_trace:.2f}x")
    print(f"slogdet overhead: {(dt_logdet-dt_trace)*1000:.1f} ms  ({(dt_logdet-dt_trace)/dt_logdet*100:.0f}% of log-det time)")
    print(f"J@P matmul is {dt_matmul/dt_logdet*100:.0f}% of log-det time")

    # Sanity check: are the values proportional? (ranks should match)
    gains_logdet = logdet_fn(action_seqs)
    gains_trace  = trace_fn(action_seqs)
    print(f"\nlog-det gains:  mean={float(jnp.mean(gains_logdet)):.4f}, std={float(jnp.std(gains_logdet)):.4f}")
    print(f"trace gains:    mean={float(jnp.mean(gains_trace)):.4f},  std={float(jnp.std(gains_trace)):.4f}")

if __name__ == "__main__":
    main()
