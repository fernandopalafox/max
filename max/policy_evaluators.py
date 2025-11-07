# policy_evaluators.py

import jax
import jax.numpy as jnp
import numpy as np
from typing import Any, Dict, Optional


def evaluate_policy(
    env_fns,
    policy,
    policy_params: Any,
    eval_key: jax.Array,
    n_episodes: int,
    dyn_params: Any = None,
    config: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    Evaluates a policy by running it for a fixed number of episodes.

    Uses the same RNG key for each call to ensure deterministic, reproducible
    evaluation across training runs.

    Always uses vmap over agents (even when num_agents=1).

    Args:
        env_fns: Tuple of (reset_fn, step_fn, get_obs_fn) environment functions
        policy: Policy object with select_action_deterministic method
        policy_params: Parameters of the policy (stacked along agent axis)
        eval_key: JAX random key (fixed across calls for consistency)
        n_episodes: Number of episodes to run
        dyn_params: Dynamics parameters (for model-based policies)
        config: Configuration dict containing num_agents, dim_state, and max_episode_steps (optional)

    Returns:
        Dictionary containing:
            - mean_return: Average return across episodes
            - std_return: Standard deviation of returns
            - mean_length: Average episode length
            - agent_{i}/mean_return: Per-agent average return
            - agent_{i}/std_return: Per-agent std deviation
    """

    # Unpack environment functions
    reset_fn, step_fn, get_obs_fn = env_fns

    # Detect if multi-agent from config
    num_agents = 1
    if config is not None:
        num_agents = config.get("num_agents", 1)

    episode_returns = []
    episode_lengths = []

    # Track per-agent returns (always have agent axis)
    per_agent_returns = [[] for _ in range(num_agents)]

    # Use the eval key to create episode keys deterministically
    episode_keys = jax.random.split(eval_key, n_episodes)

    # Get max episode steps from config
    max_episode_steps = (
        config.get("env_params", {}).get("max_episode_steps", 1000)
        if config
        else 1000
    )

    for episode_idx in range(n_episodes):
        # Reset environment with JAX key
        episode_key = episode_keys[episode_idx]
        state = reset_fn(episode_key)
        current_obs = get_obs_fn(state)  # Already replicated for all agents

        agent_episode_returns = np.zeros(num_agents, dtype=np.float32)
        episode_length = 0
        done = False

        while not done:
            # Use current_obs which is already replicated for all agents
            # Select actions for all agents (vmap over agents)
            actions = jax.vmap(
                policy.select_action_deterministic, in_axes=(0, 0, None)
            )(policy_params, current_obs, dyn_params)

            # Actions are already in shape (num_agents, dim_action), pass directly
            action_np = np.array(actions, dtype=np.float32)
            state, current_obs, rewards, terminated, truncated, info = (
                step_fn(state, episode_length, action_np)
            )

            done = terminated or truncated

            # Handle reward - vector (per-agent) for pursuit-evasion
            reward_array = np.array(rewards, dtype=np.float32)
            agent_episode_returns += reward_array

            episode_length += 1

        # Track per-agent returns
        for agent_idx in range(num_agents):
            per_agent_returns[agent_idx].append(
                float(agent_episode_returns[agent_idx])
            )

        # Track aggregate return (sum or mean of agent returns)
        episode_returns.append(float(np.sum(agent_episode_returns)))
        episode_lengths.append(episode_length)

    # Compute statistics
    results = {
        "mean_return": float(np.mean(episode_returns)),
        "std_return": float(np.std(episode_returns)),
        "mean_length": float(np.mean(episode_lengths)),
    }

    # Add per-agent statistics
    for agent_idx in range(num_agents):
        results[f"agent_{agent_idx}/mean_return"] = float(
            np.mean(per_agent_returns[agent_idx])
        )
        results[f"agent_{agent_idx}/std_return"] = float(
            np.std(per_agent_returns[agent_idx])
        )

    return results
