# dynamics_evaluators.py
"""
Evaluator infrastructure for dynamics-based task evaluation.

Provides factory functions for creating evaluators that assess learned dynamics
models by running full task rollouts and computing evaluation costs.
"""

import jax
import jax.numpy as jnp
from typing import Any, Callable, Dict, NamedTuple

from max.normalizers import init_normalizer
from max.environments import init_env
from max.dynamics import init_dynamics
from max.planners import init_planner
from max.costs import init_cost


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
    Initializes an evaluator based on configuration.

    Args:
        config: Full configuration dictionary. Will look for 'evaluator_params'
                section; if empty/missing, falls back to upper-level params.

    Returns:
        Tuple of (Evaluator, state) where state may be None for stateless evaluators
    """
    evaluator_type = config.get("evaluator_type", "rollout")
    print(f"🚀 Initializing evaluator: {evaluator_type.upper()}")

    if evaluator_type == "rollout":
        return create_rollout_evaluator(config)
    else:
        raise ValueError(f"Unknown evaluator type: '{evaluator_type}'")


def create_rollout_evaluator(config: dict) -> tuple[Evaluator, None]:
    """
    Creates a rollout evaluator that:
    1. Initializes environment, dynamics, planner, and evaluation costs
    2. Runs rollouts using the planner with learned dynamics
    3. Computes evaluation costs at each step

    Args:
        config: Configuration dictionary with evaluator_params section

    Returns:
        Tuple of (Evaluator, None) - no persistent state needed
    """
    evaluator_params = config.get("evaluator_params", {})

    # --- Build resolved configs for each component ---
    # Environment config
    env_config = {**config}
    env_config["env_params"] = _resolve_evaluator_config(config, "env_params")
    env_config["env_name"] = _resolve_evaluator_field(config, "env_name")

    # Dynamics config
    dynamics_config = {**config}
    dynamics_config["dynamics_params"] = _resolve_evaluator_config(config, "dynamics_params")
    dynamics_config["dynamics"] = _resolve_evaluator_field(config, "dynamics")

    # Planner config
    planner_config = {**config}
    planner_config["planner_params"] = _resolve_evaluator_config(config, "planner_params")
    planner_config["planner_type"] = _resolve_evaluator_field(config, "planner_type")

    # Planning cost config (for the planner's internal optimization)
    planning_cost_config = {**config}
    planning_cost_config["cost_fn_params"] = _resolve_evaluator_config(config, "cost_fn_params")
    planning_cost_config["cost_type"] = _resolve_evaluator_field(config, "cost_type")

    # Get evaluation cost types list
    eval_cost_types = evaluator_params.get("eval_cost_types", ["goal_cost"])

    # Random key for initialization
    seed = evaluator_params.get("seed", config.get("seed", 42))
    key = jax.random.key(seed)

    # --- Initialize components ---
    print("  📦 Initializing evaluation environment...")
    reset_fn, step_fn, get_obs_fn = init_env(env_config)

    print("  📦 Initializing evaluation dynamics model (structure only)...")
    normalizer, norm_params = init_normalizer(dynamics_config)
    key, model_key = jax.random.split(key)
    dynamics_model, _ = init_dynamics(model_key, dynamics_config, normalizer, norm_params)

    print("  📦 Initializing planning cost function...")
    planning_cost_fn = init_cost(planning_cost_config, dynamics_model)

    print("  📦 Initializing planner...")
    key, planner_key = jax.random.split(key)
    planner, init_planner_state = init_planner(planner_config, planning_cost_fn, planner_key)

    print("  📦 Initializing evaluation cost function(s)...")
    eval_cost_fns = {}
    for cost_type in eval_cost_types:
        # Build config for this eval cost type
        eval_cost_config = {**config}
        eval_cost_config["cost_fn_params"] = _resolve_evaluator_config(config, "cost_fn_params")
        eval_cost_config["cost_type"] = cost_type
        eval_cost_fns[cost_type] = init_cost(eval_cost_config, dynamics_model)

    # --- Get rollout parameters ---
    max_steps = evaluator_params.get(
        "max_steps",
        config.get("env_params", {}).get("max_episode_steps", 200)
    )
    num_episodes = evaluator_params.get("num_episodes", 1)

    # Get goal state for cost_params
    goal_state = jnp.array(
        config.get("cost_fn_params", {}).get(
            "goal_state",
            jnp.zeros(config.get("dim_state", 6))
        )
    )

    # --- Create the evaluate function ---
    def _run_single_episode(
        dyn_params: dict,
        reset_key: jax.Array,
        planner_key: jax.Array,
    ) -> dict:
        """
        Runs a single evaluation episode.

        Returns:
            Dict with accumulated costs for each eval_cost_type
        """
        # Reset environment
        env_state = reset_fn(reset_key)

        # Initialize planner state
        planner_state = init_planner_state.replace(key=planner_key)

        # Initialize cost accumulators
        accumulated_costs = {cost_type: 0.0 for cost_type in eval_cost_types}

        # Build cost_params structure for planner and eval costs
        cost_params = {
            "dyn_params": dyn_params,
            "params_cov_model": None,  # Not used in evaluation
            "goal_state": goal_state,
        }

        # Run rollout loop
        for step_idx in range(max_steps):
            # Plan with current dynamics
            actions, planner_state = planner.solve(planner_state, env_state, cost_params)
            action = actions[0]  # Take first action from horizon

            # Compute evaluation costs for this step
            for cost_type, cost_fn in eval_cost_fns.items():
                step_cost = cost_fn(env_state, action[None, :], cost_params)
                accumulated_costs[cost_type] += float(step_cost)

            # Step environment
            env_state, _, _, terminated, truncated, _ = step_fn(
                env_state, step_idx, action[None, :]
            )

            # Check if episode is done
            if terminated or truncated:
                break

        return accumulated_costs

    def evaluate_fn(dyn_params: dict) -> dict:
        """
        Main evaluation function that runs multiple episodes and aggregates results.

        Args:
            dyn_params: Dynamics parameters in train_state.params format

        Returns:
            Dictionary with mean accumulated costs across episodes
        """
        key = jax.random.key(seed)

        all_costs = {cost_type: [] for cost_type in eval_cost_types}

        for ep in range(num_episodes):
            key, reset_key, planner_key = jax.random.split(key, 3)
            episode_costs = _run_single_episode(dyn_params, reset_key, planner_key)

            for cost_type in eval_cost_types:
                all_costs[cost_type].append(episode_costs[cost_type])

        # Aggregate results
        results = {}
        for cost_type in eval_cost_types:
            results[f"eval/{cost_type}"] = float(jnp.mean(jnp.array(all_costs[cost_type])))
            if num_episodes > 1:
                results[f"eval/{cost_type}_std"] = float(jnp.std(jnp.array(all_costs[cost_type])))

        return results

    print("  ✅ Rollout evaluator initialized.")
    return Evaluator(evaluate_fn=evaluate_fn)
