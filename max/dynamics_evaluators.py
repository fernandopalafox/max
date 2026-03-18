# dynamics_evaluators.py
"""
TDMPC2 rollout evaluator using MPPI in latent space.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, NamedTuple

from max.environments import init_env
from max.planners import (
    create_mppi_planner,
    make_tdmpc2_encode_fn,
    make_tdmpc2_trajectory_value_fn,
    make_tdmpc2_action_proposal_fn,
)


class Evaluator(NamedTuple):
    """A generic container for an evaluation algorithm."""
    evaluate_fn: Callable[[dict], dict]

    def evaluate(self, params: dict) -> dict:
        """
        Evaluates the model.

        Args:
            params: Full unified parameters dict.

        Returns:
            Dictionary with evaluation metrics and optionally 'trajectory'.
        """
        return self.evaluate_fn(params)


def init_evaluator(
    config: dict,
    encoder=None,
    dynamics=None,
    reward=None,
    critic=None,
    policy=None,
) -> Evaluator:
    """
    Initializes a TDMPC2 MPPI rollout evaluator.

    Args:
        config: Full configuration dictionary.
        encoder, dynamics, reward, critic, policy: TDMPC2 components.

    Returns:
        Evaluator instance
    """
    return _create_evaluator(config, encoder, dynamics, reward, critic, policy)


def _create_evaluator(
    config: dict,
    encoder,
    dynamics,
    reward,
    critic,
    policy,
) -> Evaluator:
    """
    TDMPC2 rollout evaluator using MPPI in latent space.

    evaluate(parameters) accepts the full unified parameters dict and passes it
    directly to the MPPI planner as cost_params.

    Evaluation config is read from evaluator_params (or falls back to top-level):
        max_steps, num_episodes, seed, env_params, planner_params
    """
    evaluator_params = config.get("evaluator_params", {})
    seed = evaluator_params.get("seed", config.get("seed", 42))
    max_steps = evaluator_params.get(
        "max_steps",
        config.get("env_params", {}).get("max_episode_steps", 200),
    )
    num_episodes = evaluator_params.get("num_episodes", 1)

    # Build env config — evaluator_params overrides take priority
    env_config = {**config}
    ep = evaluator_params.get("env_params")
    if ep:
        env_config["env_params"] = ep

    reset_fn, step_fn, get_obs_fn = init_env(env_config)

    # Build MPPI planner config — evaluator_params.planner_params overrides top-level
    mppi_config = {**config}
    ep_pp = evaluator_params.get("planner_params")
    if ep_pp:
        mppi_config["planner_params"] = ep_pp
    mppi_config["planner_type"] = "mppi"

    key = jax.random.key(seed)
    key, planner_key = jax.random.split(key)

    pp = mppi_config.get("planner_params", {})
    horizon = pp["horizon"]
    discount = pp.get("discount_factor", 0.99)
    encode_fn = make_tdmpc2_encode_fn(encoder)
    traj_value_fn = make_tdmpc2_trajectory_value_fn(
        dynamics, reward, critic, policy, horizon, discount
    )
    action_proposal_fn = make_tdmpc2_action_proposal_fn(dynamics, policy, horizon)
    planner, init_planner_state = create_mppi_planner(
        mppi_config, encode_fn, traj_value_fn, planner_key, action_proposal_fn
    )

    def _scan_step(carry, step_idx):
        env_state, planner_state, parameters = carry
        state_array = get_obs_fn(env_state).squeeze(0)
        actions, new_planner_state = planner.solve(planner_state, state_array, parameters)
        action = actions[0]  # first action of the planned sequence
        new_env_state, _, env_rewards, _, _, _ = step_fn(env_state, step_idx, action[None, :])
        return (new_env_state, new_planner_state, parameters), (env_rewards[0], env_state, action)

    @jax.jit
    def _run_episode_jitted(parameters: dict, reset_key: jax.Array, planner_key: jax.Array):
        env_state = reset_fn(reset_key)
        planner_state = init_planner_state.replace(key=planner_key)
        init_carry = (env_state, planner_state, parameters)
        (final_env_state, _, _), (all_step_rewards, all_env_states, all_actions) = jax.lax.scan(
            _scan_step, init_carry, jnp.arange(max_steps)
        )
        all_env_states_with_final = jax.tree.map(
            lambda arr, final: jnp.concatenate([arr, final[None, ...]], axis=0),
            all_env_states, final_env_state,
        )
        return jnp.sum(all_step_rewards), all_env_states_with_final, all_actions

    @jax.jit
    def _run_episodes_vmapped(parameters: dict, reset_keys: jax.Array, planner_keys: jax.Array):
        return jax.vmap(
            lambda rk, pk: _run_episode_jitted(parameters, rk, pk),
            in_axes=(0, 0),
        )(reset_keys, planner_keys)

    def evaluate_fn(parameters: dict) -> dict:
        eval_key = jax.random.key(seed)
        if num_episodes == 1:
            reset_key, pk = jax.random.split(eval_key)
            episode_reward, trajectory, actions = _run_episode_jitted(parameters, reset_key, pk)
            return {
                "eval/episode_reward": float(episode_reward),
                "trajectory": jax.device_get(trajectory),
                "actions": jax.device_get(actions),
            }
        all_keys = jax.random.split(eval_key, num_episodes * 2)
        all_rewards, all_trajs, all_acts = _run_episodes_vmapped(
            parameters, all_keys[:num_episodes], all_keys[num_episodes:]
        )
        return {
            "eval/episode_reward": float(jnp.mean(all_rewards)),
            "trajectory": jax.device_get(all_trajs),
            "actions": jax.device_get(all_acts),
            "num_episodes": num_episodes,
        }

    return Evaluator(evaluate_fn=evaluate_fn)
