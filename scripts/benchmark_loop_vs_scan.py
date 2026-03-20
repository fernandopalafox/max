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


def bench_python_loop(rollout, rollout_state, n_steps):
    """Run n_steps via Python for-loop dispatch. First step is warmup."""
    # Warmup: trigger JIT compilation
    state = rollout_state
    state, _ = rollout.step_fn(state, None)
    jax.block_until_ready(jax.tree_util.tree_leaves(state.parameters))

    # Measure remaining steps
    remaining = n_steps - 1
    t0 = time.perf_counter()
    for _ in range(remaining):
        state, _ = rollout.step_fn(state, None)
    jax.block_until_ready(jax.tree_util.tree_leaves(state.parameters))
    elapsed = time.perf_counter() - t0

    return remaining / elapsed


def bench_scan(rollout, rollout_state, n_steps, chunk_size):
    """Run n_steps via jax.lax.scan in chunks. First chunk is warmup."""
    # Warmup: first chunk triggers JIT compilation
    state = rollout_state
    state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(state.parameters))

    # Measure remaining chunks
    remaining_steps = n_steps - chunk_size
    remaining_chunks = remaining_steps // chunk_size
    actual_steps = remaining_chunks * chunk_size

    t0 = time.perf_counter()
    for _ in range(remaining_chunks):
        state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
    jax.block_until_ready(jax.tree_util.tree_leaves(state.parameters))
    elapsed = time.perf_counter() - t0

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
    print("Building components and pre-filling buffer...")

    # Build two independent rollout states (so they start from the same point)
    rollout, state_loop = build_rollout(config, seed_override=0)
    _,       state_scan = build_rollout(config, seed_override=0)

    print("Benchmarking Python loop (first step = warmup)...")
    sps_loop = bench_python_loop(rollout, state_loop, n_steps)

    print("Benchmarking scan chunks (first chunk = warmup)...")
    sps_scan = bench_scan(rollout, state_scan, n_steps, chunk_size)

    speedup = sps_scan / sps_loop
    print()
    print(f"Python loop:  {sps_loop:>10.1f} steps/sec  (N={n_steps}, chunk_size=N/A)")
    print(f"Scan chunks:  {sps_scan:>10.1f} steps/sec  (N={n_steps}, chunk_size={chunk_size})")
    print(f"Speedup: {speedup:.2f}x")


if __name__ == "__main__":
    main()
