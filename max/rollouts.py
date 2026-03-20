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
    episode_reward: jnp.float32


class StepOutputs(NamedTuple):
    train_metrics: dict
    episode_done: jax.Array    # bool scalar; valid every step
    episode_reward: jax.Array  # float; valid only when episode_done
    episode_len: jax.Array     # int; valid only when episode_done


class Rollout(NamedTuple):
    step_fn: Callable  # (RolloutState, None) -> (RolloutState, StepOutputs)


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
        key, planner_key, train_key, sample_key, reset_key = jax.random.split(carry.key, 5)

        # ---- Plan ----
        actions, new_planner_state = planner.solve(
            carry.planner_state.replace(key=planner_key),
            carry.obs,
            carry.parameters,
        )
        action = actions[0][None, :]  # (1, dim_a)

        # ---- Env step ----
        new_mjx_data, next_obs, rewards, terminated, truncated, _ = env_step_fn(
            carry.mjx_data, carry.episode_len, action
        )
        next_obs = next_obs.squeeze()
        done = terminated | truncated
        new_episode_len = carry.episode_len + jnp.int32(1)
        new_episode_reward = carry.episode_reward + rewards[0]

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
        final_episode_reward = jnp.where(done, jnp.float32(0.0), new_episode_reward)

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
            episode_reward=final_episode_reward,
        )

        step_out = StepOutputs(
            train_metrics=train_metrics,
            episode_done=done,
            episode_reward=new_episode_reward,
            episode_len=new_episode_len,
        )

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
        episode_reward=jnp.float32(0.0),
    )

    step_fn = _make_scan_step(
        reset_fn, env_step_fn, get_obs_fn, planner, trainer, sampler, buffer_size
    )

    return Rollout(step_fn=step_fn), init_state


def prefill_buffer(
    rollout_state: RolloutState,
    reset_fn: Callable,
    env_step_fn: Callable,
    get_obs_fn: Callable,
    dim_a: int,
    min_buffer_size: int,
    buffer_size: int,
) -> RolloutState:
    """
    Fill the replay buffer with random transitions using a plain Python loop.
    No planning, no training. After this call rollout_state.buffer_idx >= min_buffer_size
    so jax.lax.scan can train unconditionally from the first step.
    """
    for _ in range(min_buffer_size):
        key, action_key, reset_key = jax.random.split(rollout_state.key, 3)
        action = jax.random.uniform(action_key, (1, dim_a), minval=-1.0, maxval=1.0)

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
        new_episode_reward = rollout_state.episode_reward + rewards[0]

        if done:
            new_mjx_data = reset_fn(reset_key)
            final_obs = get_obs_fn(new_mjx_data).squeeze()
            new_episode_len = jnp.int32(0)
            new_episode_reward = jnp.float32(0.0)
        else:
            final_obs = next_obs_sq

        rollout_state = rollout_state._replace(
            key=key,
            mjx_data=new_mjx_data,
            obs=final_obs,
            buffers=new_buffers,
            buffer_idx=new_buffer_idx,
            episode_len=new_episode_len,
            episode_reward=new_episode_reward,
        )

    return rollout_state
