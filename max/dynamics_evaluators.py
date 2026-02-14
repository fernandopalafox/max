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
    elif evaluator_type == "rollout_with_trajectory":
        return create_rollout_with_trajectory_evaluator(config)
    else:
        raise ValueError(f"Unknown evaluator type: '{evaluator_type}'")


def create_rollout_evaluator(config: dict) -> Evaluator:
    """
    Creates a rollout evaluator that:
    1. Initializes environment, dynamics, planner, and evaluation costs
    2. Runs rollouts using the planner with learned dynamics
    3. Computes evaluation costs at each step

    Uses jax.lax.scan for the time loop and jax.vmap for parallel episodes,
    avoiding Python loop overhead. Always runs full max_steps (no early termination).

    Args:
        config: Configuration dictionary with evaluator_params section

    Returns:
        Evaluator instance
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
    reset_fn, step_fn, _ = init_env(env_config)

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
    # Separate stage costs from terminal costs
    # Terminal costs are only computed on the final state, not accumulated over time
    stage_cost_fn_list = []
    stage_cost_types = []
    terminal_cost_fn_list = []
    terminal_cost_types = []

    for cost_type in eval_cost_types:
        eval_cost_config = {**config}
        eval_cost_config["cost_fn_params"] = _resolve_evaluator_config(config, "cost_fn_params")
        eval_cost_config["cost_type"] = cost_type
        cost_fn = init_cost(eval_cost_config, dynamics_model)

        # Check if this is a terminal cost (by convention, terminal costs have "terminal" in the name)
        if "terminal" in cost_type.lower():
            terminal_cost_fn_list.append(cost_fn)
            terminal_cost_types.append(cost_type)
        else:
            stage_cost_fn_list.append(cost_fn)
            stage_cost_types.append(cost_type)

    num_stage_costs = len(stage_cost_fn_list)
    num_terminal_costs = len(terminal_cost_fn_list)

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

    # --- Define the scan step function ---
    def _scan_step(carry, step_idx):
        """
        Single step of the rollout, designed for jax.lax.scan.

        Args:
            carry: (env_state, planner_state, cost_params)
            step_idx: Current step index (unused but required by scan)

        Returns:
            (new_carry, step_costs) where step_costs is array of shape (num_stage_costs,)
        """
        env_state, planner_state, cost_params = carry

        # Plan with current dynamics
        actions, new_planner_state = planner.solve(planner_state, env_state, cost_params)
        action = actions[0]  # Take first action from horizon

        # Compute stage costs for this step (terminal costs are computed separately)
        if num_stage_costs > 0:
            step_costs = jnp.array([
                cost_fn(env_state, action[None, :], cost_params)
                for cost_fn in stage_cost_fn_list
            ])
        else:
            step_costs = jnp.array([])

        # Step environment (ignore termination/truncation - always run full episode)
        new_env_state, _, _, _, _, _ = step_fn(env_state, step_idx, action[None, :])

        return (new_env_state, new_planner_state, cost_params), (step_costs, new_env_state)

    def _run_single_episode(dyn_params: dict, reset_key: jax.Array, planner_key: jax.Array) -> jax.Array:
        """
        Runs a single evaluation episode using jax.lax.scan.

        Returns:
            Array of costs, shape (num_stage_costs + num_terminal_costs,)
            Stage costs are accumulated over time, terminal costs computed on final state only.
        """
        # Reset environment
        env_state = reset_fn(reset_key)

        # Initialize planner state with new key
        planner_state = init_planner_state.replace(key=planner_key)

        # Build cost_params structure
        cost_params = {
            "dyn_params": dyn_params,
            "params_cov_model": None,
            "goal_state": goal_state,
        }

        # Run rollout with scan (always full max_steps)
        init_carry = (env_state, planner_state, cost_params)
        _, (all_step_costs, all_states) = jax.lax.scan(_scan_step, init_carry, jnp.arange(max_steps))

        # all_step_costs has shape (max_steps, num_stage_costs)
        # Sum stage costs over time
        if num_stage_costs > 0:
            accumulated_stage_costs = jnp.sum(all_step_costs, axis=0)
        else:
            accumulated_stage_costs = jnp.array([])

        # Compute terminal costs on the final state only
        if num_terminal_costs > 0:
            final_state = all_states[-1]  # Last state from the rollout
            terminal_costs = jnp.array([
                cost_fn(final_state, jnp.zeros((1, config.get("dim_control", 2))), cost_params)
                for cost_fn in terminal_cost_fn_list
            ])
        else:
            terminal_costs = jnp.array([])

        # Concatenate stage and terminal costs
        return jnp.concatenate([accumulated_stage_costs, terminal_costs])

    # Vectorize over episodes (vmap over reset_key and planner_key)
    # dyn_params is shared across all episodes (in_axes=None)
    _run_episodes_vmapped = jax.vmap(
        _run_single_episode,
        in_axes=(None, 0, 0)  # dyn_params shared, keys batched
    )

    # JIT compile the entire evaluation
    @jax.jit
    def _evaluate_jitted(dyn_params: dict, reset_keys: jax.Array, planner_keys: jax.Array) -> jax.Array:
        """
        JIT-compiled evaluation over all episodes.

        Args:
            dyn_params: Dynamics parameters
            reset_keys: Keys of shape (num_episodes,) for environment resets
            planner_keys: Keys of shape (num_episodes,) for planner initialization

        Returns:
            Array of shape (num_episodes, num_cost_types)
        """
        return _run_episodes_vmapped(dyn_params, reset_keys, planner_keys)

    # Combined cost types list (stage costs first, then terminal costs)
    all_cost_types = stage_cost_types + terminal_cost_types

    def evaluate_fn(dyn_params: dict) -> dict:
        """
        Main evaluation function that runs multiple episodes in parallel.

        Args:
            dyn_params: Dynamics parameters in train_state.params format

        Returns:
            Dictionary with mean accumulated costs across episodes.
            Stage costs are accumulated over time, terminal costs computed on final state only.
        """
        # Generate keys for all episodes
        key = jax.random.key(seed)
        # Split into 2*num_episodes keys, then separate into reset and planner keys
        all_keys = jax.random.split(key, num_episodes * 2)
        reset_keys = all_keys[:num_episodes]
        planner_keys = all_keys[num_episodes:]

        # Run all episodes in parallel
        all_costs = _evaluate_jitted(dyn_params, reset_keys, planner_keys)
        # all_costs has shape (num_episodes, num_stage_costs + num_terminal_costs)

        # Aggregate results
        results = {}
        mean_costs = jnp.mean(all_costs, axis=0)
        for i, cost_type in enumerate(all_cost_types):
            results[f"eval/{cost_type}"] = float(mean_costs[i])
            if num_episodes > 1:
                results[f"eval/{cost_type}_std"] = float(jnp.std(all_costs[:, i]))

        return results

    print("  ✅ Rollout evaluator initialized.")
    return Evaluator(evaluate_fn=evaluate_fn)


def create_rollout_with_trajectory_evaluator(config: dict) -> Evaluator:
    """
    Creates a rollout evaluator that also returns the state trajectory for plotting.

    Similar to create_rollout_evaluator but:
    - Runs a single episode (no vmap)
    - Returns both metrics dict and state trajectory array

    Args:
        config: Configuration dictionary with evaluator_params section

    Returns:
        Evaluator instance where evaluate() returns dict with 'trajectory' key
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

    planning_cost_config = {**config}
    planning_cost_config["cost_fn_params"] = _resolve_evaluator_config(config, "cost_fn_params")
    planning_cost_config["cost_type"] = _resolve_evaluator_field(config, "cost_type")

    eval_cost_types = evaluator_params.get("eval_cost_types", ["goal_cost"])

    seed = evaluator_params.get("seed", config.get("seed", 42))
    key = jax.random.key(seed)

    # --- Initialize components ---
    print("  📦 Initializing evaluation environment...")
    reset_fn, step_fn, _ = init_env(env_config)

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
    stage_cost_fn_list = []
    stage_cost_types = []
    terminal_cost_fn_list = []
    terminal_cost_types = []

    for cost_type in eval_cost_types:
        eval_cost_config = {**config}
        eval_cost_config["cost_fn_params"] = _resolve_evaluator_config(config, "cost_fn_params")
        eval_cost_config["cost_type"] = cost_type
        cost_fn = init_cost(eval_cost_config, dynamics_model)

        if "terminal" in cost_type.lower():
            terminal_cost_fn_list.append(cost_fn)
            terminal_cost_types.append(cost_type)
        else:
            stage_cost_fn_list.append(cost_fn)
            stage_cost_types.append(cost_type)

    num_stage_costs = len(stage_cost_fn_list)
    num_terminal_costs = len(terminal_cost_fn_list)

    max_steps = evaluator_params.get(
        "max_steps",
        config.get("env_params", {}).get("max_episode_steps", 200)
    )

    # Resolve goal_state from evaluator_params first, then top-level
    eval_cost_fn_params = _resolve_evaluator_config(config, "cost_fn_params")
    goal_state = jnp.array(
        eval_cost_fn_params.get(
            "goal_state",
            jnp.zeros(config.get("dim_state", 6))
        )
    )

    def _scan_step(carry, step_idx):
        env_state, planner_state, cost_params = carry

        actions, new_planner_state = planner.solve(planner_state, env_state, cost_params)
        action = actions[0]

        if num_stage_costs > 0:
            step_costs = jnp.array([
                cost_fn(env_state, action[None, :], cost_params)
                for cost_fn in stage_cost_fn_list
            ])
        else:
            step_costs = jnp.array([])

        new_env_state, _, _, _, _, _ = step_fn(env_state, step_idx, action[None, :])

        return (new_env_state, new_planner_state, cost_params), (step_costs, env_state, action)

    @jax.jit
    def _run_episode_jitted(dyn_params: dict, reset_key: jax.Array, planner_key: jax.Array):
        env_state = reset_fn(reset_key)
        planner_state = init_planner_state.replace(key=planner_key)

        cost_params = {
            "dyn_params": dyn_params,
            "params_cov_model": None,
            "goal_state": goal_state,
        }

        init_carry = (env_state, planner_state, cost_params)
        (final_state, _, _), (all_step_costs, all_states, all_actions) = jax.lax.scan(
            _scan_step, init_carry, jnp.arange(max_steps)
        )

        # Append final state to trajectory
        all_states_with_final = jnp.concatenate([all_states, final_state[None, :]], axis=0)

        if num_stage_costs > 0:
            accumulated_stage_costs = jnp.sum(all_step_costs, axis=0)
        else:
            accumulated_stage_costs = jnp.array([])

        if num_terminal_costs > 0:
            terminal_costs = jnp.array([
                cost_fn(final_state, jnp.zeros((1, config.get("dim_control", 2))), cost_params)
                for cost_fn in terminal_cost_fn_list
            ])
        else:
            terminal_costs = jnp.array([])

        costs = jnp.concatenate([accumulated_stage_costs, terminal_costs])
        return costs, all_states_with_final, all_actions

    all_cost_types = stage_cost_types + terminal_cost_types

    def evaluate_fn(dyn_params: dict) -> dict:
        key = jax.random.key(seed)
        reset_key, planner_key = jax.random.split(key, 2)

        costs, trajectory, actions = _run_episode_jitted(dyn_params, reset_key, planner_key)

        results = {}
        for i, cost_type in enumerate(all_cost_types):
            results[f"eval/{cost_type}"] = float(costs[i])

        # Include trajectory and actions for plotting
        results["trajectory"] = jax.device_get(trajectory)
        results["actions"] = jax.device_get(actions)
        results["goal_state"] = jax.device_get(goal_state)

        return results

    print("  ✅ Rollout with trajectory evaluator initialized.")
    return Evaluator(evaluate_fn=evaluate_fn)
