# buffers.py

import jax
import jax.numpy as jnp
from typing import Dict


def init_jax_buffers(
    num_agents: int,
    buffer_size: int,
    dim_state: int,
    dim_action: int,
) -> Dict[str, jnp.ndarray]:
    """
    Pre-allocate JAX arrays for buffer storage.

    Args:
        num_agents: Number of agents
        buffer_size: Maximum number of steps to store
        dim_state: State dimension
        dim_action: Action dimension

    Returns:
        Dict containing preallocated JAX arrays:
            - states: (num_agents, buffer_size, dim_state)
            - actions: (num_agents, buffer_size, dim_action)
            - rewards: (num_agents, buffer_size)
            - dones: (buffer_size,)
            - log_pis_old: (num_agents, buffer_size)
            - values_old: (num_agents, buffer_size)
    """
    return {
        "states": jnp.zeros((num_agents, buffer_size, dim_state)),
        "actions": jnp.zeros((num_agents, buffer_size, dim_action)),
        "rewards": jnp.zeros((num_agents, buffer_size)),
        "dones": jnp.zeros(buffer_size),
        "log_pis_old": jnp.zeros((num_agents, buffer_size)),
        "values_old": jnp.zeros((num_agents, buffer_size)),
    }


@jax.jit
def update_buffer_dynamic(
    buffers: Dict[str, jnp.ndarray],
    buffer_idx: int,
    states: jnp.ndarray,
    actions: jnp.ndarray,
    rewards: jnp.ndarray,
    log_pis: jnp.ndarray,
    values: jnp.ndarray,
    done: float,
) -> Dict[str, jnp.ndarray]:
    """
    Update buffers using dynamic_update_slice for better performance.

    This pure function updates the buffer at the specified index using
    JAX's dynamic_update_slice for efficient in-place-style updates.

    Args:
        buffers: Current buffer dict
        buffer_idx: Index to update
        states: States for all agents (num_agents, dim_state)
        actions: Actions for all agents (num_agents, dim_action)
        rewards: Rewards for all agents (num_agents,)
        log_pis: Log probabilities for all agents (num_agents,)
        values: Values for all agents (num_agents,)
        done: Done flag (shared across agents)

    Returns:
        Updated buffer dict with new transition at buffer_idx
    """
    # Reshape for dynamic_update_slice (add buffer dimension)
    states = states[:, None, :]  # (num_agents, 1, dim_state)
    actions = actions[:, None, :]  # (num_agents, 1, dim_action)
    rewards = rewards[:, None]  # (num_agents, 1)
    log_pis = log_pis[:, None]  # (num_agents, 1)
    values = values[:, None]  # (num_agents, 1)
    done = jnp.array([done])  # (1,)

    buffers = {
        "states": jax.lax.dynamic_update_slice(
            buffers["states"], states, (0, buffer_idx, 0)
        ),
        "actions": jax.lax.dynamic_update_slice(
            buffers["actions"], actions, (0, buffer_idx, 0)
        ),
        "rewards": jax.lax.dynamic_update_slice(
            buffers["rewards"], rewards, (0, buffer_idx)
        ),
        "log_pis_old": jax.lax.dynamic_update_slice(
            buffers["log_pis_old"], log_pis, (0, buffer_idx)
        ),
        "values_old": jax.lax.dynamic_update_slice(
            buffers["values_old"], values, (0, buffer_idx)
        ),
        "dones": jax.lax.dynamic_update_slice(
            buffers["dones"], done, (buffer_idx,)
        ),
    }

    return buffers
