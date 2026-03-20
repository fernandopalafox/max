"""
Benchmark: Python for-loop dispatch vs jax.lax.scan for the full training step.

Usage:
    conda run -n max python scripts/benchmark_loop_vs_scan.py \\
        --config cheetah.json \\
        --n-steps 1000 \\
        --chunk-size 100

The first chunk/loop warmup triggers JIT compilation; reported throughput
excludes it so only steady-state performance is measured.
"""

import argparse
import copy
import json
import os
import time

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp

jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

from max.buffers import init_buffer
from max.critics import init_critic
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.environments import init_env
from max.planners import init_planner
from max.policies import init_policy
from max.rewards import init_reward_model
from max.rollouts import init_rollout, prefill_buffer
from max.samplers import init_sampler
from max.trainers import init_trainer


def build_rollout(config, seed_override=None):
    seed = seed_override if seed_override is not None else config["seed"]
    key = jax.random.key(seed)

    reset_fn, step_fn, get_obs_fn = init_env(config)

    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_params    = init_encoder(enc_key, config)
    dynamics,     dyn_params    = init_dynamics(dyn_key, config)
    critic,       crit_params   = init_critic(critic_key, config)
    policy,       pol_params    = init_policy(policy_key, config)
    reward_model, rew_params    = init_reward_model(config)

    parameters = {
        "mean": {
            "encoder":    enc_params,
            "dynamics":   dyn_params,
            "reward":     rew_params,
            "critic":     crit_params,
            "ema_critic": copy.deepcopy(crit_params),
            "policy":     pol_params,
        },
        "normalizer": {"q_scale": jnp.array(config["normalizer"]["critic"]["q_scale_init"])},
    }

    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        trainer_key, config, encoder, dynamics, critic, policy, reward_model, parameters
    )

    sampler = init_sampler(config["sampler"])

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(
        config,
        key=planner_key,
        encoder=encoder,
        dynamics=dynamics,
        reward=reward_model,
        critic=critic,
        policy=policy,
    )

    buffers = init_buffer(config)

    key, rollout_key = jax.random.split(key)
    rollout, rollout_state = init_rollout(
        rollout_key, config,
        reset_fn, step_fn, get_obs_fn,
        planner, planner_state,
        trainer, train_state,
        sampler,
        parameters, buffers,
    )

    sampler_cfg = config["sampler"]
    min_buffer_size = sampler_cfg.get(
        "min_buffer_size",
        sampler_cfg["batch_size"] + sampler_cfg["horizon"],
    )
    rollout_state = prefill_buffer(
        rollout_state,
        reset_fn, step_fn, get_obs_fn,
        dim_a=config["dim_action"],
        min_buffer_size=min_buffer_size,
        buffer_size=config["buffer_size"],
    )

    return rollout, rollout_state


def _block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


def bench_python_loop(rollout, rollout_state, n_steps):
    """Run n_steps via Python for-loop dispatch. First step is warmup (JIT compile)."""
    state = rollout_state

    print(f"  [loop] step 1/{n_steps}: JIT compiling full step... ", end="", flush=True)
    t_compile = time.perf_counter()
    state, _ = rollout.step_fn(state, None)
    _block(state.parameters)
    print(f"{time.perf_counter() - t_compile:.1f}s")

    remaining = n_steps - 1
    print(f"  [loop] running {remaining} steps... ", end="", flush=True)
    t0 = time.perf_counter()
    for i in range(remaining):
        state, _ = rollout.step_fn(state, None)
        if (i + 1) % 50 == 0:
            _block(state.parameters)
            elapsed = time.perf_counter() - t0
            sps = (i + 1) / elapsed
            print(f"\n  [loop]   {i+1}/{remaining} steps  ({sps:.1f} steps/s)", end="", flush=True)
    _block(state.parameters)
    elapsed = time.perf_counter() - t0
    print(f"\n  [loop] done in {elapsed:.2f}s")

    return remaining / elapsed


def bench_scan(rollout, rollout_state, n_steps, chunk_size):
    """Run n_steps via jax.lax.scan in chunks. First chunk is warmup (JIT compile)."""
    state = rollout_state

    print(f"  [scan] chunk 1 (size={chunk_size}): JIT compiling full scan... ", end="", flush=True)
    t_compile = time.perf_counter()
    state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
    _block(state.parameters)
    print(f"{time.perf_counter() - t_compile:.1f}s")

    remaining_steps = n_steps - chunk_size
    remaining_chunks = remaining_steps // chunk_size
    actual_steps = remaining_chunks * chunk_size

    print(f"  [scan] running {remaining_chunks} chunks ({actual_steps} steps)... ", end="", flush=True)
    t0 = time.perf_counter()
    for i in range(remaining_chunks):
        state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
        if (i + 1) % 5 == 0:
            _block(state.parameters)
            elapsed = time.perf_counter() - t0
            sps = (i + 1) * chunk_size / elapsed
            print(f"\n  [scan]   chunk {i+1}/{remaining_chunks}  ({sps:.1f} steps/s)", end="", flush=True)
    _block(state.parameters)
    elapsed = time.perf_counter() - t0
    print(f"\n  [scan] done in {elapsed:.2f}s")

    return actual_steps / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cheetah.json")
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--chunk-size", type=int, default=100)
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path) as f:
        config = json.load(f)["training"]

    n_steps = args.n_steps
    chunk_size = args.chunk_size

    if n_steps <= chunk_size:
        raise ValueError(f"--n-steps ({n_steps}) must be greater than --chunk-size ({chunk_size})")

    print(f"Config: {args.config}  |  n_steps={n_steps}  |  chunk_size={chunk_size}")

    t0 = time.perf_counter()
    print("\n[1/4] Building rollout state for loop trial...")
    rollout, state_loop = build_rollout(config, seed_override=0)
    print(f"      done in {time.perf_counter()-t0:.1f}s")

    t1 = time.perf_counter()
    print("[2/4] Building rollout state for scan trial...")
    _, state_scan = build_rollout(config, seed_override=0)
    print(f"      done in {time.perf_counter()-t1:.1f}s")

    print("\n[3/4] Python loop benchmark:")
    sps_loop = bench_python_loop(rollout, state_loop, n_steps)

    print("\n[4/4] Scan benchmark:")
    sps_scan = bench_scan(rollout, state_scan, n_steps, chunk_size)

    speedup = sps_scan / sps_loop
    print()
    print("=" * 60)
    print(f"Python loop:  {sps_loop:>10.1f} steps/sec  (N={n_steps})")
    print(f"Scan chunks:  {sps_scan:>10.1f} steps/sec  (N={n_steps}, chunk={chunk_size})")
    print(f"Speedup:      {speedup:>10.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
