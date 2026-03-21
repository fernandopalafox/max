# samplers.py

import jax
import jax.numpy as jnp
from typing import Callable, NamedTuple, Optional


class Sampler(NamedTuple):
    sample_fn: Callable
    sample_jit: Optional[Callable] = None  # JAX-pure callable for use inside jax.lax.scan

    def sample(self, key, buffer, buffer_idx) -> Optional[dict]:
        return self.sample_fn(key, buffer, buffer_idx)


def init_sampler(config: dict) -> Sampler:
    """Initializes the appropriate sampler based on the configuration."""
    sampler_type = config["type"]
    if sampler_type == "latest":
        return _create_latest_sampler()
    elif sampler_type == "trajectory_batch":
        return _create_trajectory_batch_sampler(
            config["batch_size"], config["horizon"],
            min_buffer_size=config.get("min_buffer_size"),
        )
    else:
        raise ValueError(f"Unknown sampler type: {sampler_type!r}")


def _create_latest_sampler() -> Sampler:
    """Single-transition sampler for EKF-style trainers. Key is unused.

    Skips transitions at episode boundaries (done=True at the source step)
    to avoid feeding cross-episode (s, a, s') tuples to the trainer.
    """

    def sample_fn(key, buffer, buffer_idx):
        if buffer_idx < 2:
            return None
        prev = buffer_idx - 2
        # Don't use a transition that crosses an episode boundary
        if buffer["dones"][prev] == 1.0:
            return None
        return {
            "states":      buffer["states"][0, prev:prev+1, :],
            "actions":     buffer["actions"][0, prev:prev+1, :],
            "next_states": buffer["states"][0, prev+1:prev+2, :],
        }

    return Sampler(sample_fn=sample_fn)


def _create_trajectory_batch_sampler(batch_size: int, horizon: int, min_buffer_size: int = None) -> Sampler:
    """
    Trajectory-batch sampler for latent_gd / TDMPC2-style trainers.

    Returns batches of shape:
        states:   (batch_size, horizon+1, dim_s)
        actions:  (batch_size, horizon,   dim_a)
        rewards:  (batch_size, horizon)

    Episode boundaries (dones == 1) are never spanned. The probs vector weights
    only valid windows, so a sparse buffer is handled gracefully. Callers are
    responsible for ensuring the buffer has enough data before calling (use
    prefill_buffer in rollouts.py before starting jax.lax.scan).
    """

    @jax.jit
    def _sample_jit(key, states, actions, rewards, dones, buffer_idx):
        # states:  (buffer_size, dim_s)
        # actions: (buffer_size, dim_a)
        # rewards: (buffer_size,)
        # dones:   (buffer_size,)
        n = states.shape[0]  # static (buffer_size)

        # valid[i] = True iff dones[i:i+horizon] are all 0  AND  i+horizon <= buffer_idx
        done_cumsum = jnp.concatenate([jnp.zeros(1), jnp.cumsum(dones)])  # (n+1,)
        idx = jnp.arange(n)
        end_idx = jnp.minimum(idx + horizon, n)
        window_sums = done_cumsum[end_idx] - done_cumsum[idx]
        in_range = (idx + horizon) <= buffer_idx
        valid = (window_sums == 0) & in_range
        probs = valid.astype(jnp.float32)
        probs = probs / (probs.sum() + 1e-8)

        start_indices = jax.random.choice(
            key, n, shape=(batch_size,), replace=True, p=probs
        )

        def extract_one(start):
            s = jax.lax.dynamic_slice(states,  (start, 0), (horizon + 1, states.shape[1]))
            a = jax.lax.dynamic_slice(actions, (start, 0), (horizon,     actions.shape[1]))
            r = jax.lax.dynamic_slice(rewards, (start,),   (horizon,))
            return s, a, r

        state_windows, action_windows, reward_windows = jax.vmap(extract_one)(start_indices)
        return {
            "states":  state_windows,   # (batch_size, horizon+1, dim_s)
            "actions": action_windows,  # (batch_size, horizon,   dim_a)
            "rewards": reward_windows,  # (batch_size, horizon)
        }

    def sample_fn(key, buffer, buffer_idx):
        return _sample_jit(
            key,
            buffer["states"][0],
            buffer["actions"][0],
            buffer["rewards"][0],
            buffer["dones"],
            buffer_idx,
        )

    def sample_jit_fn(key, buffer, buffer_idx):
        return _sample_jit(
            key,
            buffer["states"][0],
            buffer["actions"][0],
            buffer["rewards"][0],
            buffer["dones"],
            buffer_idx,
        )

    return Sampler(sample_fn=sample_fn, sample_jit=sample_jit_fn)
