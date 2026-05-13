# rollouts.py

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple

from max.buffers import update_buffer
from max.planners import Planner, PlannerState
from max.trainers import Trainer, TrainState
from max.samplers import Sampler


class RolloutState(NamedTuple):
    key: jax.Array
    mjx_data: any              # mjx.Data
    obs: jax.Array             # (dim_obs,) squeezed
    train_state: TrainState
    parameters: dict
    planner_state: PlannerState
    buffers: dict              # {states, actions, rewards, dones}
    buffer_idx: jnp.int32      # monotonically increasing; write at buffer_idx % buffer_size
    episode_len: jnp.int32


class StepOutputs(NamedTuple):
    train_metrics: dict


class Rollout(NamedTuple):
    step_fn: Callable   # (RolloutState, None) -> (RolloutState, StepOutputs)
    scan_fn: Callable   # jit-wrapped scan over chunk_size steps


def _make_scan_step(
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    planner: Planner,
    trainer: Trainer,
    sampler: Sampler,
    buffer_size: int,
) -> Callable:
    def step_fn(carry: RolloutState, _) -> tuple[RolloutState, StepOutputs]:
        key, planner_key, train_key, sample_key, reset_key, explore_key = jax.random.split(carry.key, 6)

        # ---- Plan ----
        actions, std, new_planner_state = planner.solve(
            carry.planner_state.replace(key=planner_key),
            carry.obs,
            carry.parameters,
        )
        # exploration: sample from converged MPPI distribution N(mean, std) matching TDMPC2 training
        action = jnp.clip(actions[0][None, :] + std[0] * jax.random.normal(explore_key, actions[0][None, :].shape), -1.0, 1.0)

        # ---- Env step ----
        new_mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
            carry.mjx_data, carry.episode_len, action
        )
        next_obs = next_obs.squeeze()
        done = terminated | truncated
        new_episode_len = carry.episode_len + jnp.int32(1)

        # ---- Buffer update (ring buffer) ----
        write_idx = carry.buffer_idx % buffer_size
        new_buffers = update_buffer(
            carry.buffers,
            write_idx,
            carry.obs[None, :],
            action,
            rewards,
            done.astype(jnp.float32),
        )
        new_buffer_idx = carry.buffer_idx + jnp.int32(1)

        # ---- Sample and train ----
        train_data = sampler.sample_jit(sample_key, new_buffers, new_buffer_idx)
        new_train_state, new_parameters, train_metrics = trainer.train(
            carry.train_state, train_data, carry.parameters, train_key
        )

        # ---- Episode reset via lax.cond ----
        def reset_branch(_):
            new_data = reset_fn(reset_key)
            new_obs = get_obs_fn(new_data).squeeze()
            new_mean = jnp.zeros_like(new_planner_state.mean)
            return new_data, new_obs, new_mean

        def keep_branch(_):
            return new_mjx_data, next_obs, new_planner_state.mean

        final_mjx_data, final_obs, final_planner_mean = jax.lax.cond(
            done, reset_branch, keep_branch, None
        )

        final_planner_state = new_planner_state.replace(mean=final_planner_mean)
        final_episode_len = jnp.where(done, jnp.int32(0), new_episode_len)

        new_carry = RolloutState(
            key=key,
            mjx_data=final_mjx_data,
            obs=final_obs,
            train_state=new_train_state,
            parameters=new_parameters,
            planner_state=final_planner_state,
            buffers=new_buffers,
            buffer_idx=new_buffer_idx,
            episode_len=final_episode_len,
        )

        step_out = StepOutputs(train_metrics=train_metrics)

        return new_carry, step_out

    return step_fn


def init_rollout(
    key: jax.Array,
    config: dict,
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    planner: Planner,
    init_planner_state: PlannerState,
    trainer: Trainer,
    init_train_state: TrainState,
    sampler: Sampler,
    init_parameters: dict,
    init_buffers: dict,
) -> tuple[Rollout, RolloutState]:
    """
    Build a scan-compatible rollout step and its initial carry state.

    Call prefill_buffer on the returned RolloutState before starting
    jax.lax.scan to ensure the buffer has enough data to train every step.
    """
    buffer_size = config["buffer_size"]

    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    obs = get_obs_fn(mjx_data).squeeze()

    init_state = RolloutState(
        key=key,
        mjx_data=mjx_data,
        obs=obs,
        train_state=init_train_state,
        parameters=init_parameters,
        planner_state=init_planner_state,
        buffers=init_buffers,
        buffer_idx=jnp.int32(0),
        episode_len=jnp.int32(0),
    )

    chunk_size = config["chunk_size"]
    step_fn = _make_scan_step(
        reset_fn, env_step_fn, get_obs_fn, planner, trainer, sampler, buffer_size
    )
    scan_fn = jax.jit(
        lambda state: jax.lax.scan(step_fn, state, None, length=chunk_size)
    )

    return Rollout(step_fn=step_fn, scan_fn=scan_fn), init_state


class CollectRolloutState(NamedTuple):
    key: jax.Array
    mjx_data: any              # mjx.Data
    obs: jax.Array             # (dim_obs,) squeezed
    parameters: dict
    planner_state: PlannerState
    buffers: dict              # {states, actions, rewards, dones}
    buffer_idx: jnp.int32      # monotonically increasing; write at buffer_idx % buffer_size
    episode_len: jnp.int32


class CollectStepOutputs(NamedTuple):
    reward: jax.Array   # scalar float, rewards[0]
    done: jax.Array     # bool scalar


class CollectRollout(NamedTuple):
    step_fn: Callable   # (CollectRolloutState, None) -> (CollectRolloutState, CollectStepOutputs)
    scan_fn: Callable   # jit-wrapped scan over chunk_size steps


def _make_collect_scan_step(
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    planner: Planner,
    buffer_size: int,
    explore: bool,
) -> Callable:
    def step_fn(carry: CollectRolloutState, _) -> tuple[CollectRolloutState, CollectStepOutputs]:
        key, planner_key, reset_key, explore_key = jax.random.split(carry.key, 4)

        # ---- Plan ----
        actions, std, new_planner_state = planner.solve(
            carry.planner_state.replace(key=planner_key),
            carry.obs,
            carry.parameters,
        )
        if explore:
            action = jnp.clip(
                actions[0][None, :] + std[0] * jax.random.normal(explore_key, actions[0][None, :].shape),
                -1.0, 1.0,
            )
        else:
            action = actions[0][None, :]

        # ---- Env step ----
        new_mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
            carry.mjx_data, carry.episode_len, action
        )
        next_obs = next_obs.squeeze()
        done = terminated | truncated
        new_episode_len = carry.episode_len + jnp.int32(1)

        # ---- Buffer update (ring buffer) ----
        write_idx = carry.buffer_idx % buffer_size
        new_buffers = update_buffer(
            carry.buffers,
            write_idx,
            carry.obs[None, :],
            action,
            rewards,
            done.astype(jnp.float32),
        )
        new_buffer_idx = carry.buffer_idx + jnp.int32(1)

        # ---- Episode reset via lax.cond ----
        def reset_branch(_):
            new_data = reset_fn(reset_key)
            new_obs = get_obs_fn(new_data).squeeze()
            new_mean = jnp.zeros_like(new_planner_state.mean)
            return new_data, new_obs, new_mean

        def keep_branch(_):
            return new_mjx_data, next_obs, new_planner_state.mean

        final_mjx_data, final_obs, final_planner_mean = jax.lax.cond(
            done, reset_branch, keep_branch, None
        )

        final_planner_state = new_planner_state.replace(mean=final_planner_mean)
        final_episode_len = jnp.where(done, jnp.int32(0), new_episode_len)

        new_carry = CollectRolloutState(
            key=key,
            mjx_data=final_mjx_data,
            obs=final_obs,
            parameters=carry.parameters,
            planner_state=final_planner_state,
            buffers=new_buffers,
            buffer_idx=new_buffer_idx,
            episode_len=final_episode_len,
        )

        return new_carry, CollectStepOutputs(reward=rewards[0], done=done)

    return step_fn


def init_collect_rollout(
    key: jax.Array,
    config: dict,
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    planner: Planner,
    init_planner_state: PlannerState,
    init_parameters: dict,
    init_buffers: dict,
) -> tuple[CollectRollout, CollectRolloutState]:
    """
    Build a scan-compatible collect step (no training) and its initial carry state.

    Call prefill_buffer on the returned CollectRolloutState before starting
    jax.lax.scan. prefill_buffer works unchanged since CollectRolloutState
    has the same field names it accesses.
    """
    buffer_size = config["buffer_size"]
    explore = config["collect"]["explore"]

    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    obs = get_obs_fn(mjx_data).squeeze()

    init_state = CollectRolloutState(
        key=key,
        mjx_data=mjx_data,
        obs=obs,
        parameters=init_parameters,
        planner_state=init_planner_state,
        buffers=init_buffers,
        buffer_idx=jnp.int32(0),
        episode_len=jnp.int32(0),
    )

    chunk_size = config["chunk_size"]
    step_fn = _make_collect_scan_step(
        reset_fn, env_step_fn, get_obs_fn, planner, buffer_size, explore
    )
    scan_fn = jax.jit(
        lambda state: jax.lax.scan(step_fn, state, None, length=chunk_size)
    )

    return CollectRollout(step_fn=step_fn, scan_fn=scan_fn), init_state


def prefill_buffer(
    rollout_state: RolloutState,
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    dim_a: int,
    prefill_buffer_size: int,
    buffer_size: int,
    action_min: float,
    action_max: float,
    planner=None,
) -> RolloutState:
    """
    Fill the replay buffer using a plain Python loop. No training.
    If planner is None, uses random actions; otherwise uses the planner.
    After this call rollout_state.buffer_idx >= prefill_buffer_size so
    jax.lax.scan can train unconditionally from the first step.
    """
    for _ in range(prefill_buffer_size):
        key, action_key, planner_key, reset_key = jax.random.split(rollout_state.key, 4)
        if planner is not None:
            actions, std, new_planner_state = planner.solve(
                rollout_state.planner_state.replace(key=planner_key),
                rollout_state.obs,
                rollout_state.parameters,
            )
            action = jnp.clip(actions[0][None, :] + std[0] * jax.random.normal(action_key, actions[0][None, :].shape), -1.0, 1.0)
            rollout_state = rollout_state._replace(planner_state=new_planner_state)
        else:
            action = jax.random.uniform(action_key, (1, dim_a), minval=action_min, maxval=action_max)

        new_mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
            rollout_state.mjx_data, rollout_state.episode_len, action
        )
        next_obs_sq = next_obs.squeeze()
        done = bool(terminated) or bool(truncated)

        write_idx = int(rollout_state.buffer_idx) % buffer_size
        new_buffers = update_buffer(
            rollout_state.buffers,
            write_idx,
            rollout_state.obs[None, :],
            action,
            rewards,
            float(done),
        )
        new_buffer_idx = rollout_state.buffer_idx + jnp.int32(1)
        new_episode_len = rollout_state.episode_len + jnp.int32(1)

        if done:
            new_mjx_data = reset_fn(reset_key)
            final_obs = get_obs_fn(new_mjx_data).squeeze()
            new_episode_len = jnp.int32(0)
        else:
            final_obs = next_obs_sq

        rollout_state = rollout_state._replace(
            key=key,
            mjx_data=new_mjx_data,
            obs=final_obs,
            buffers=new_buffers,
            buffer_idx=new_buffer_idx,
            episode_len=new_episode_len,
        )

    return rollout_state
