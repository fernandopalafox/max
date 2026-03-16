# dynamics_evaluators.py
"""
Evaluator infrastructure for dynamics-based task evaluation.

Provides factory functions for creating evaluators that assess learned dynamics
models by running full task rollouts and computing evaluation costs.

Uses jax.lax.scan for efficient time loops and jax.vmap for parallel episodes,
avoiding Python loop overhead and enabling GPU acceleration.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, NamedTuple

from max.normalizers import init_normalizer
from max.environments import init_env
from max.dynamics import init_dynamics
from max.planners import init_planner
from max.rewards import init_reward


class Evaluator(NamedTuple):
    """A generic container for an evaluation algorithm."""
    evaluate_fn: Callable[[dict], dict]

    def evaluate(self, dyn_params: dict) -> dict:
        """
        Evaluates the dynamics model.

        Args:
            dyn_params: The learned dynamics parameters (train_state.params format)

        Returns:
            Dictionary with evaluation metrics (accumulated costs per cost type)
        """
        return self.evaluate_fn(dyn_params)


def _resolve_evaluator_config(config: dict, param_key: str) -> dict:
    """
    Resolves config for a component, checking evaluator_params first,
    then falling back to upper-level config.

    Args:
        config: Full configuration dictionary
        param_key: Key to look for (e.g., 'env_params', 'planner_params')

    Returns:
        Resolved parameter dictionary
    """
    evaluator_params = config.get("evaluator_params", {})

    # Check if evaluator_params has this key and it's not empty
    if param_key in evaluator_params and evaluator_params[param_key]:
        return evaluator_params[param_key]

    # Fall back to upper-level config
    return config.get(param_key, {})


def _resolve_evaluator_field(config: dict, field_key: str, default: Any = None) -> Any:
    """
    Resolves a single field from evaluator_params, falling back to upper-level config.

    Args:
        config: Full configuration dictionary
        field_key: Field to look for (e.g., 'env_name', 'cost_type')
        default: Default value if not found

    Returns:
        Resolved field value
    """
    evaluator_params = config.get("evaluator_params", {})

    if field_key in evaluator_params and evaluator_params[field_key] is not None:
        return evaluator_params[field_key]

    return config.get(field_key, default)


def init_evaluator(config: dict) -> Evaluator:
    """
    Initializes a rollout evaluator.

    The evaluator always returns trajectory data along with metrics.
    Callers can choose to use or ignore the trajectory.

    Args:
        config: Full configuration dictionary. Will look for 'evaluator_params'
                section; if empty/missing, falls back to upper-level params.

    Returns:
        Evaluator instance
    """
    print("🚀 Initializing evaluator: ROLLOUT")
    return create_rollout_evaluator(config)


def create_rollout_evaluator(config: dict) -> Evaluator:
    """
    Creates a rollout evaluator that returns metrics and state trajectory.

    Uses jax.lax.scan for the time loop. Always runs full max_steps (no early termination).
    Returns trajectory data along with metrics - callers can choose to use or ignore it.

    Args:
        config: Configuration dictionary with evaluator_params section

    Returns:
        Evaluator instance where evaluate() returns dict with metrics and 'trajectory' key
    """
    evaluator_params = config.get("evaluator_params", {})

    # --- Build resolved configs for each component ---
    env_config = {**config}
    env_config["env_params"] = _resolve_evaluator_config(config, "env_params")
    env_config["env_name"] = _resolve_evaluator_field(config, "env_name")

    dynamics_config = {**config}
    dynamics_config["dynamics_params"] = _resolve_evaluator_config(config, "dynamics_params")
    dynamics_config["dynamics"] = _resolve_evaluator_field(config, "dynamics")

    planner_config = {**config}
    planner_config["planner_params"] = _resolve_evaluator_config(config, "planner_params")
    planner_config["planner_type"] = _resolve_evaluator_field(config, "planner_type")

    planning_reward_config = {**config}
    planning_reward_config["reward_fn_params"] = _resolve_evaluator_config(config, "reward_fn_params")
    planning_reward_config["reward_type"] = _resolve_evaluator_field(config, "reward_type")

    seed = evaluator_params.get("seed", config.get("seed", 42))
    key = jax.random.key(seed)

    # --- Initialize components ---
    print("  📦 Initializing evaluation environment...")
    reset_fn, step_fn, get_obs_fn = init_env(env_config)

    print("  📦 Initializing evaluation dynamics model (structure only)...")
    normalizer, norm_params = init_normalizer(dynamics_config)
    key, model_key = jax.random.split(key)
    dynamics_model, _ = init_dynamics(model_key, dynamics_config, normalizer, norm_params)

    print("  📦 Initializing planning reward function...")
    planning_reward_fn = init_reward(planning_reward_config, dynamics_model)

    print("  📦 Initializing planner...")
    key, planner_key = jax.random.split(key)
    planner, init_planner_state = init_planner(planner_config, planning_reward_fn, planner_key)

    max_steps = evaluator_params.get(
        "max_steps",
        config.get("env_params", {}).get("max_episode_steps", 200)
    )
    num_episodes = evaluator_params.get("num_episodes", 1)

    def _scan_step(carry, step_idx):
        env_state, planner_state, cost_params = carry

        # Convert env_state to state array for planner (e.g., mjx.Data -> 17D array)
        # get_obs_fn returns (1, dim_state), squeeze to (dim_state,)
        state_array = get_obs_fn(env_state).squeeze(0)

        actions, new_planner_state = planner.solve(planner_state, state_array, cost_params)
        action = actions[0]

        new_env_state, _, env_rewards, _, _, _ = step_fn(env_state, step_idx, action[None, :])

        # Return env_state (not state_array) so trajectory contains full state info
        return (new_env_state, new_planner_state, cost_params), (env_rewards[0], env_state, action)

    @jax.jit
    def _run_episode_jitted(dyn_params: dict, reset_key: jax.Array, planner_key: jax.Array):
        env_state = reset_fn(reset_key)
        planner_state = init_planner_state.replace(key=planner_key)

        cost_params = {
            "dyn_params": dyn_params,
            "params_cov_model": None,
        }

        init_carry = (env_state, planner_state, cost_params)
        (final_env_state, _, _), (all_step_rewards, all_env_states, all_actions) = jax.lax.scan(
            _scan_step, init_carry, jnp.arange(max_steps)
        )

        # Append final env_state to trajectory (JAX stacks pytrees along axis 0)
        all_env_states_with_final = jax.tree.map(
            lambda arr, final: jnp.concatenate([arr, final[None, ...]], axis=0),
            all_env_states, final_env_state
        )

        episode_reward = jnp.sum(all_step_rewards)
        return episode_reward, all_env_states_with_final, all_actions

    # Create vmapped version for multi-episode evaluation
    @jax.jit
    def _run_episodes_vmapped(dyn_params: dict, reset_keys: jax.Array, planner_keys: jax.Array):
        """Run multiple episodes in parallel using vmap."""
        vmapped_episode = jax.vmap(
            lambda rk, pk: _run_episode_jitted(dyn_params, rk, pk),
            in_axes=(0, 0)
        )
        return vmapped_episode(reset_keys, planner_keys)

    def evaluate_fn(dyn_params: dict) -> dict:
        key = jax.random.key(seed)

        if num_episodes == 1:
            reset_key, planner_key = jax.random.split(key, 2)
            episode_reward, trajectory, actions = _run_episode_jitted(dyn_params, reset_key, planner_key)

            results = {}
            results["eval/episode_reward"] = float(episode_reward)
            results["trajectory"] = jax.device_get(trajectory)
            results["actions"] = jax.device_get(actions)
        else:
            # Multi-episode: vmap over episodes
            all_keys = jax.random.split(key, num_episodes * 2)
            reset_keys = all_keys[:num_episodes]
            planner_keys = all_keys[num_episodes:]

            all_episode_rewards, all_trajectories, all_actions = _run_episodes_vmapped(
                dyn_params, reset_keys, planner_keys
            )

            results = {}
            results["eval/episode_reward"] = float(jnp.mean(all_episode_rewards))
            # Return stacked trajectories with shape (num_episodes, max_steps+1, ...)
            results["trajectory"] = jax.device_get(all_trajectories)
            results["actions"] = jax.device_get(all_actions)
            results["num_episodes"] = num_episodes

        return results

    print("  ✅ Rollout evaluator initialized.")
    return Evaluator(evaluate_fn=evaluate_fn)
