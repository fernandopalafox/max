"""
Two-way benchmark: original Python loop vs jax.lax.scan.

Trial 1 — original: mirrors main branch train.py exactly — plain Python variables,
  individually JIT-compiled components, terminated or truncated forces CPU-GPU sync
  every step.

Trial 2 — scan: jax.lax.scan over a fused step function, blocked between chunks.

Usage:
    conda run -n max python scripts/benchmark_loop_vs_scan.py \\
        --config cheetah.json --n-steps 200 --chunk-size 20
"""

import argparse
import copy
import json
import os
import time

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
from max.rollouts import init_rollout, prefill_buffer
from max.samplers import init_sampler
from max.trainers import init_trainer


def _init_common(config, seed):
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

    return key, reset_fn, env_step_fn, get_obs_fn, planner, planner_state, trainer, train_state, sampler, parameters


def _min_buf(config):
    s = config["sampler"]
    return s.get("min_buffer_size", s["batch_size"] + s["horizon"])


def build_original(config, seed=0):
    """Plain Python state for Trial 1, mirroring main branch train.py."""
    key, reset_fn, env_step_fn, get_obs_fn, planner, planner_state, trainer, train_state, sampler, parameters = \
        _init_common(config, seed)

    buffers = init_buffer(config)
    buffer_idx = 0

    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    current_obs = get_obs_fn(mjx_data).squeeze()
    episode_length = 0
    episode_total_reward = 0.0

    # Pre-fill so training runs unconditionally during the timed section
    for _ in range(_min_buf(config)):
        key, action_key, reset_key = jax.random.split(key, 3)
        action = jax.random.uniform(action_key, (1, config["dim_action"]), minval=-1.0, maxval=1.0)
        mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(mjx_data, episode_length, action)
        next_obs = next_obs.squeeze()
        done = terminated or truncated
        episode_length += 1
        buffers = update_buffer(buffers, buffer_idx, current_obs[None, :], action, rewards, float(done))
        buffer_idx += 1
        current_obs = next_obs
        if done:
            mjx_data = reset_fn(reset_key)
            current_obs = get_obs_fn(mjx_data).squeeze()
            episode_length = 0
            episode_total_reward = 0.0

    return dict(
        key=key, mjx_data=mjx_data, current_obs=current_obs,
        planner_state=planner_state, parameters=parameters, train_state=train_state,
        buffers=buffers, buffer_idx=buffer_idx,
        episode_length=episode_length, episode_total_reward=episode_total_reward,
        reset_fn=reset_fn, env_step_fn=env_step_fn, get_obs_fn=get_obs_fn,
        planner=planner, trainer=trainer, sampler=sampler,
    )


def build_scan(config, seed=0):
    """rollout + rollout_state for Trial 2."""
    key, reset_fn, env_step_fn, get_obs_fn, planner, planner_state, trainer, train_state, sampler, parameters = \
        _init_common(config, seed)

    buffers = init_buffer(config)

    key, rollout_key = jax.random.split(key)
    rollout, rollout_state = init_rollout(
        rollout_key, config, reset_fn, env_step_fn, get_obs_fn,
        planner, planner_state, trainer, train_state, sampler, parameters, buffers,
    )
    rollout_state = prefill_buffer(
        rollout_state, reset_fn, env_step_fn, get_obs_fn,
        dim_a=config["dim_action"],
        min_buffer_size=_min_buf(config),
        buffer_size=config["buffer_size"],
    )
    return rollout, rollout_state


def _block(x):
    jax.block_until_ready(jax.tree_util.tree_leaves(x))


N_WARMUP = 3


# ---------------------------------------------------------------------------
# Trial 1: original — mirrors main branch train.py
# ---------------------------------------------------------------------------

def bench_original(s, config, n_steps):
    key              = s["key"]
    mjx_data         = s["mjx_data"]
    current_obs      = s["current_obs"]
    planner_state    = s["planner_state"]
    parameters       = s["parameters"]
    train_state      = s["train_state"]
    buffers          = s["buffers"]
    buffer_idx       = s["buffer_idx"]
    episode_length   = s["episode_length"]
    episode_total_reward = s["episode_total_reward"]
    reset_fn         = s["reset_fn"]
    env_step_fn      = s["env_step_fn"]
    get_obs_fn       = s["get_obs_fn"]
    planner          = s["planner"]
    trainer          = s["trainer"]
    sampler          = s["sampler"]
    buffer_size      = config["buffer_size"]

    def run(n):
        nonlocal key, mjx_data, current_obs, planner_state, parameters, train_state
        nonlocal buffers, buffer_idx, episode_length, episode_total_reward
        for _ in range(n):
            key, planner_key = jax.random.split(key)
            planner_state = planner_state.replace(key=planner_key)
            actions, planner_state = planner.solve(planner_state, current_obs, parameters)
            action = actions[0][None, :]

            mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
                mjx_data, episode_length, action
            )
            next_obs = next_obs.squeeze()
            done = terminated or truncated
            episode_length += 1
            episode_total_reward += float(rewards[0])

            buffers = update_buffer(
                buffers, buffer_idx, current_obs[None, :], action, rewards, float(done)
            )
            buffer_idx += 1
            current_obs = next_obs

            if done:
                key, reset_key = jax.random.split(key)
                mjx_data = reset_fn(reset_key)
                current_obs = get_obs_fn(mjx_data).squeeze()
                episode_length = 0
                episode_total_reward = 0.0

            key, sample_key = jax.random.split(key)
            train_data = sampler.sample(sample_key, buffers, buffer_idx)
            if train_data is not None:
                key, train_key = jax.random.split(key)
                train_state, parameters, _ = trainer.train(train_state, train_data, parameters, train_key)

            if buffer_idx >= buffer_size:
                buffers = init_buffer(config)
                buffer_idx = 0

    print(f"  [original] warmup ({N_WARMUP} steps)... ", end="", flush=True)
    for _ in range(N_WARMUP):
        t = time.perf_counter()
        run(1)
        _block(parameters)
        print(f"{time.perf_counter()-t:.1f}s", end=" ", flush=True)
    print()

    print(f"  [original] timing {n_steps} steps... ", end="", flush=True)
    t0 = time.perf_counter()
    run(n_steps)
    _block(parameters)
    elapsed = time.perf_counter() - t0
    sps = n_steps / elapsed
    print(f"{elapsed:.1f}s  →  {sps:.1f} steps/s")
    return sps


# ---------------------------------------------------------------------------
# Trial 2: jax.lax.scan + episodes_from_buffer (mirrors train.py exactly)
# ---------------------------------------------------------------------------

import numpy as np

def _episodes_from_buffer(buffers, prev_idx, curr_idx, buffer_size):
    n = curr_idx - prev_idx
    indices = (np.arange(n) + prev_idx) % buffer_size
    dones = np.array(buffers["dones"][indices])
    rewards = np.array(buffers["rewards"][0, indices])
    episodes, ep_start = [], 0
    for i, done in enumerate(dones):
        if done == 1.0:
            episodes.append({
                "episodes/reward": float(rewards[ep_start:i+1].sum()),
                "episodes/length": int(i - ep_start + 1),
            })
            ep_start = i + 1
    return episodes


def bench_scan(rollout, rollout_state, n_steps, chunk_size, buffer_size):
    state = rollout_state
    n_chunks = n_steps // chunk_size

    print(f"  [scan] warmup ({N_WARMUP} chunks of {chunk_size} steps)... ", end="", flush=True)
    for _ in range(N_WARMUP):
        t = time.perf_counter()
        prev_idx = int(state.buffer_idx)
        state, chunk_out = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
        _block(chunk_out)
        _ = _episodes_from_buffer(state.buffers, prev_idx, int(state.buffer_idx), buffer_size)
        print(f"{time.perf_counter()-t:.1f}s", end=" ", flush=True)
    print()

    actual_steps = n_chunks * chunk_size
    print(f"  [scan] timing {n_chunks} chunks ({actual_steps} steps)... ", end="", flush=True)
    t0 = time.perf_counter()
    for _ in range(n_chunks):
        prev_idx = int(state.buffer_idx)
        state, chunk_out = jax.lax.scan(rollout.step_fn, state, None, length=chunk_size)
        _block(chunk_out)
        _ = _episodes_from_buffer(state.buffers, prev_idx, int(state.buffer_idx), buffer_size)
    elapsed = time.perf_counter() - t0
    sps = actual_steps / elapsed
    print(f"{elapsed:.1f}s  →  {sps:.1f} steps/s")
    return sps


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="cheetah.json")
    parser.add_argument("--n-steps", type=int, default=200)
    parser.add_argument("--chunk-size", type=int, default=20)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path) as f:
        config = json.load(f)["training"]

    n_steps = args.n_steps
    chunk_size = args.chunk_size

    print(f"Config: {args.config}  |  n_steps={n_steps}  |  chunk_size={chunk_size}\n")

    t0 = time.perf_counter()
    print("Initializing components (x2)...")
    s_orig         = build_original(config, seed=0)
    rollout, state = build_scan(config, seed=0)
    print(f"done in {time.perf_counter()-t0:.1f}s\n")

    print("[Trial 1] Original: individual JIT components, Python loop  (terminated or truncated syncs every step)")
    sps_orig = bench_original(s_orig, config, n_steps)

    print("\n[Trial 2] jax.lax.scan  (block between chunks)")
    sps_scan = bench_scan(rollout, state, n_steps, chunk_size, config["buffer_size"])

    print()
    print("=" * 55)
    print(f"Trial 1 (original):  {sps_orig:>8.1f} steps/s")
    print(f"Trial 2 (scan):      {sps_scan:>8.1f} steps/s  ({sps_scan/sps_orig:.2f}x)")
    print("=" * 55)


if __name__ == "__main__":
    main()
