"""
Three-way benchmark for the TDMPC2 training loop.

Trial 1 — original: Python for-loop, individually JIT-compiled components
  (planner.solve, env step_fn, update_buffer, sampler, trainer.train called
  separately from Python — exactly what the old train.py did)

Trial 2 — jit step + loop: entire step compiled as one jax.jit, Python for-loop
  (isolates the benefit of fusing the step vs dispatching 5 separate JIT calls)

Trial 3 — jit step + scan: entire step inside jax.lax.scan
  (eliminates Python between steps entirely)

Usage:
    conda run -n max python scripts/benchmark_loop_vs_scan.py \\
        --config cheetah.json --n-steps 200 --chunk-size 20
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

from max.buffers import init_buffer, update_buffer
from max.critics import init_critic
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.environments import init_env
from max.planners import init_planner
from max.policies import init_policy
from max.rewards import init_reward_model
from max.rollouts import init_rollout, prefill_buffer, RolloutState
from max.samplers import init_sampler
from max.trainers import init_trainer


def build_components(config, seed_override=None):
    """Initialize all components and return a Rollout, RolloutState, and raw component dict."""
    seed = seed_override if seed_override is not None else config["seed"]
    key = jax.random.key(seed)

    reset_fn, env_step_fn, get_obs_fn = init_env(config)

    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_params  = init_encoder(enc_key, config)
    dynamics,     dyn_params  = init_dynamics(dyn_key, config)
    critic,       crit_params = init_critic(critic_key, config)
    policy,       pol_params  = init_policy(policy_key, config)
    reward_model, rew_params  = init_reward_model(config)

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
        config, key=planner_key, encoder=encoder, dynamics=dynamics,
        reward=reward_model, critic=critic, policy=policy,
    )

    buffers = init_buffer(config)

    key, rollout_key = jax.random.split(key)
    rollout, rollout_state = init_rollout(
        rollout_key, config, reset_fn, env_step_fn, get_obs_fn,
        planner, planner_state, trainer, train_state, sampler, parameters, buffers,
    )

    sampler_cfg = config["sampler"]
    min_buffer_size = sampler_cfg.get(
        "min_buffer_size", sampler_cfg["batch_size"] + sampler_cfg["horizon"],
    )
    rollout_state = prefill_buffer(
        rollout_state, reset_fn, env_step_fn, get_obs_fn,
        dim_a=config["dim_action"],
        min_buffer_size=min_buffer_size,
        buffer_size=config["buffer_size"],
    )

    raw = dict(
        reset_fn=reset_fn, env_step_fn=env_step_fn, get_obs_fn=get_obs_fn,
        planner=planner, trainer=trainer, sampler=sampler,
        buffer_size=config["buffer_size"],
    )
    return rollout, rollout_state, raw


def _block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


# ---------------------------------------------------------------------------
# Trial 1: original — individually JIT-compiled components, Python for-loop
# ---------------------------------------------------------------------------

def _original_step(state: RolloutState, reset_fn, env_step_fn, get_obs_fn,
                   planner, trainer, sampler, buffer_size):
    """One step calling each JIT-compiled component separately from Python."""
    key, planner_key, train_key, sample_key, reset_key = jax.random.split(state.key, 5)

    # Plan (planner.solve_fn is @jax.jit)
    actions, new_planner_state = planner.solve(
        state.planner_state.replace(key=planner_key), state.obs, state.parameters
    )
    action = actions[0][None, :]

    # Env step (@jax.jit)
    new_mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
        state.mjx_data, state.episode_len, action
    )
    next_obs = next_obs.squeeze()
    done = bool(terminated) or bool(truncated)

    # Buffer update (@jax.jit)
    write_idx = int(state.buffer_idx) % buffer_size
    new_buffers = update_buffer(
        state.buffers, write_idx, state.obs[None, :], action, rewards, float(done)
    )
    new_buffer_idx = state.buffer_idx + jnp.int32(1)

    # Sample + train (each calls @jax.jit internally)
    train_data = sampler.sample_jit(sample_key, new_buffers, new_buffer_idx)
    new_train_state, new_parameters, _ = trainer.train(
        state.train_state, train_data, state.parameters, train_key
    )

    # Episode reset in Python
    if done:
        new_mjx_data = reset_fn(reset_key)
        final_obs = get_obs_fn(new_mjx_data).squeeze()
        new_planner_state = new_planner_state.replace(mean=jnp.zeros_like(new_planner_state.mean))
        new_episode_len = jnp.int32(0)
        new_episode_reward = jnp.float32(0.0)
    else:
        final_obs = next_obs
        new_episode_len = state.episode_len + jnp.int32(1)
        new_episode_reward = state.episode_reward + rewards[0]

    return state._replace(
        key=key, mjx_data=new_mjx_data, obs=final_obs,
        train_state=new_train_state, parameters=new_parameters,
        planner_state=new_planner_state, buffers=new_buffers,
        buffer_idx=new_buffer_idx, episode_len=new_episode_len,
        episode_reward=new_episode_reward,
    )


def bench_original(rollout_state, raw, n_steps):
    """Trial 1: bool() inside _original_step forces a CPU-GPU sync after every
    step, so this is truly sequential — exactly what the old train.py did."""
    state = rollout_state
    dot_every = max(1, n_steps // 10)

    print(f"  [original] warmup (1 step, compiles individual components)... ",
          end="", flush=True)
    t = time.perf_counter()
    state = _original_step(state, **raw)
    _block(state.parameters)
    print(f"{time.perf_counter()-t:.1f}s")

    remaining = n_steps - 1
    print(f"  [original] timing {remaining} steps ", end="", flush=True)
    t0 = time.perf_counter()
    for i in range(remaining):
        state = _original_step(state, **raw)  # bool() inside syncs every step
        if (i + 1) % dot_every == 0:
            print(".", end="", flush=True)
    _block(state.parameters)
    elapsed = time.perf_counter() - t0
    sps = remaining / elapsed
    print(f" {elapsed:.2f}s  →  {sps:.1f} steps/s")
    return sps


# ---------------------------------------------------------------------------
# Trial 2: fused jax.jit step + Python for-loop
# ---------------------------------------------------------------------------

def bench_jit_loop(rollout, rollout_state, n_steps):
    """Trial 2: full step fused into one jax.jit, called in a Python loop.
    No mid-loop syncs — dispatch all steps, block once at the end."""
    step_jit = jax.jit(rollout.step_fn)
    state = rollout_state
    dot_every = max(1, n_steps // 10)

    print(f"  [jit+loop] warmup (1 step, compiles fused step)... ", end="", flush=True)
    t = time.perf_counter()
    state, _ = step_jit(state, None)
    _block(state.parameters)
    print(f"{time.perf_counter()-t:.1f}s")

    remaining = n_steps - 1
    print(f"  [jit+loop] timing {remaining} steps ", end="", flush=True)
    t0 = time.perf_counter()
    for i in range(remaining):
        state, _ = step_jit(state, None)
        if (i + 1) % dot_every == 0:
            print(".", end="", flush=True)
    _block(state.parameters)
    elapsed = time.perf_counter() - t0
    sps = remaining / elapsed
    print(f" {elapsed:.2f}s  →  {sps:.1f} steps/s")
    return sps


# ---------------------------------------------------------------------------
# Trial 3: fused step inside jax.lax.scan
# ---------------------------------------------------------------------------

def bench_scan(rollout, rollout_state, n_steps, chunk_size):
    """Trial 3: full step inside jax.lax.scan, called in chunks from Python.
    Block between chunks (natural boundary), block once at end."""
    state = rollout_state
    dot_every = max(1, (n_steps // chunk_size) // 10)

    print(f"  [scan] warmup (1 chunk of {chunk_size} steps, compiles scan body)... ",
          end="", flush=True)
    t = time.perf_counter()
    state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
    _block(state.parameters)
    print(f"{time.perf_counter()-t:.1f}s")

    remaining_steps = n_steps - chunk_size
    remaining_chunks = remaining_steps // chunk_size
    actual_steps = remaining_chunks * chunk_size

    print(f"  [scan] timing {remaining_chunks} chunks ({actual_steps} steps) ",
          end="", flush=True)
    t0 = time.perf_counter()
    for i in range(remaining_chunks):
        state, _ = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
        if dot_every > 0 and (i + 1) % dot_every == 0:
            print(".", end="", flush=True)
    _block(state.parameters)
    elapsed = time.perf_counter() - t0
    sps = actual_steps / elapsed
    print(f" {elapsed:.2f}s  →  {sps:.1f} steps/s")
    return sps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cheetah.json")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=20)
    parser.add_argument("--log-compiles", action="store_true",
                        help="Print a line each time JAX triggers a compilation")
    args = parser.parse_args()

    if args.log_compiles:
        jax.config.update("jax_log_compiles", True)

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path) as f:
        config = json.load(f)["training"]

    n_steps = args.n_steps
    chunk_size = args.chunk_size

    if n_steps <= chunk_size:
        raise ValueError(f"--n-steps ({n_steps}) must be > --chunk-size ({chunk_size})")

    print(f"Config: {args.config}  |  n_steps={n_steps}  |  chunk_size={chunk_size}\n")

    t0 = time.perf_counter()
    print("Initializing components (x3, one per trial)...")
    rollout, state_orig, raw  = build_components(config, seed_override=0)
    _,       state_jit,  _    = build_components(config, seed_override=0)
    _,       state_scan, _    = build_components(config, seed_override=0)
    print(f"done in {time.perf_counter()-t0:.1f}s\n")

    print("[Trial 1] Original: individual JIT components, Python loop  (bool() syncs every step)")
    sps_orig = bench_original(state_orig, raw, n_steps)

    print("\n[Trial 2] Fused jax.jit step + Python loop  (async dispatch, block at end)")
    sps_jit = bench_jit_loop(rollout, state_jit, n_steps)

    print("\n[Trial 3] Fused jax.lax.scan  (block between chunks)")
    sps_scan = bench_scan(rollout, state_scan, n_steps, chunk_size)

    print()
    print("=" * 60)
    print(f"Trial 1 (original):  {sps_orig:>8.1f} steps/s")
    print(f"Trial 2 (jit+loop):  {sps_jit:>8.1f} steps/s  ({sps_jit/sps_orig:.2f}x vs original)")
    print(f"Trial 3 (jit+scan):  {sps_scan:>8.1f} steps/s  ({sps_scan/sps_orig:.2f}x vs original)")
    print("=" * 60)


if __name__ == "__main__":
    main()
