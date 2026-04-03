# buffers.py

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict


def init_buffer(config: dict) -> Dict[str, jnp.ndarray]:
    """
    Pre-allocate JAX arrays for buffer storage.

    Returns:
        Dict containing preallocated JAX arrays:
            - states:  (num_agents, buffer_size, dim_state)
            - actions: (num_agents, buffer_size, dim_action)
            - rewards: (num_agents, buffer_size)
            - dones:   (buffer_size,)
    """
    num_agents  = config["num_agents"]
    buffer_size = config["buffer_size"]
    dim_state   = config["dim_state"]
    dim_action  = config["dim_action"]
    return {
        "states":  jnp.zeros((num_agents, buffer_size, dim_state)),
        "actions": jnp.zeros((num_agents, buffer_size, dim_action)),
        "rewards": jnp.zeros((num_agents, buffer_size)),
        "dones":   jnp.zeros(buffer_size),
    }


@jax.jit
def update_buffer(
    buffers: Dict[str, jnp.ndarray],
    buffer_idx: int,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    done: float,
) -> Dict[str, jnp.ndarray]:
    """
    Update buffers using dynamic_update_slice for better performance.

    Pure function — updates the buffer at the specified index.

    Args:
        buffers:    Current buffer dict
        buffer_idx: Index to update
        states:     States for all agents (num_agents, dim_state)
        actions:    Actions for all agents (num_agents, dim_action)
        rewards:    Rewards for all agents (num_agents,)
        done:       Done flag (shared across agents)

    Returns:
        Updated buffer dict with new transition at buffer_idx
    """
    states_s  = states[:, None, :]   # (num_agents, 1, dim_state)
    actions_s = actions[:, None, :]  # (num_agents, 1, dim_action)
    rewards_s = rewards[:, None]     # (num_agents, 1)
    done_s    = jnp.array([done])    # (1,)

    return {
        "states": jax.lax.dynamic_update_slice(
            buffers["states"], states_s, (0, buffer_idx, 0)
        ),
        "actions": jax.lax.dynamic_update_slice(
            buffers["actions"], actions_s, (0, buffer_idx, 0)
        ),
        "rewards": jax.lax.dynamic_update_slice(
            buffers["rewards"], rewards_s, (0, buffer_idx)
        ),
        "dones": jax.lax.dynamic_update_slice(
            buffers["dones"], done_s, (buffer_idx,)
        ),
    }


def episodes_from_buffer(
    buffers: Dict,
    prev_idx: int,
    curr_idx: int,
    buffer_size: int,
    partial_reward: float = 0.0,
    partial_len: int = 0,
) -> tuple[list, float, int]:
    """
    Return completed episode stats for transitions written in [prev_idx, curr_idx).

    partial_reward, partial_len: running totals for the episode that was
    in-progress at prev_idx. Pass 0 / 0 at the start of training.

    Returns:
        episodes:           list of {"episodes/reward": float, "episodes/length": int}
        new_partial_reward: reward accumulated so far in the still-incomplete episode
        new_partial_len:    steps accumulated so far in the still-incomplete episode
    """
    n = curr_idx - prev_idx
    indices = (np.arange(n) + prev_idx) % buffer_size
    dones = np.array(buffers["dones"][indices])
    rewards = np.array(buffers["rewards"][0, indices])

    episodes = []
    ep_reward = partial_reward
    ep_len = partial_len

    for done, reward in zip(dones, rewards):
        ep_reward += float(reward)
        ep_len += 1
        if done == 1.0:
            episodes.append({
                "episodes/reward": ep_reward,
                "episodes/length": ep_len,
            })
            ep_reward = 0.0
            ep_len = 0

    return episodes, ep_reward, ep_len
