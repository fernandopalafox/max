"""Policy abstraction for model-free and model-based control."""

from typing import Any, Callable, NamedTuple, Optional, Sequence
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct

from max.normalizers import Normalizer
from max.dynamics import DynamicsModel
from max.planners import Planner, PlannerState, CostFn


class Policy(NamedTuple):
    """A generic container for a policy."""

    select_action: Callable
    select_action_deterministic: Callable
    evaluate_value: Callable
    compute_log_prob: Callable
    compute_entropy: Callable


class PolicyState(struct.PyTreeNode):
    """State for a policy that can handle both neural networks and planners."""

    params: Any
    key: Optional[jax.Array] = None


# ============================================================================
# Actor-Critic Networks
# ============================================================================


class ActorNetwork(nn.Module):
    """Actor network that outputs mean and log_std for a Gaussian policy."""

    hidden_layers: Sequence[int]
    dim_action: int

    @nn.compact
    def __call__(self, state):
        x = state
        for feat in self.hidden_layers:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        # Output both mean and log_std
        output = nn.Dense(2 * self.dim_action)(x)
        mean, log_std = jnp.split(output, 2, axis=-1)
        return mean, log_std


class CriticNetwork(nn.Module):
    """Critic network that outputs state value."""

    hidden_layers: Sequence[int]

    @nn.compact
    def __call__(self, state):
        x = state
        for feat in self.hidden_layers:
            x = nn.Dense(feat)(x)
            x = nn.tanh(x)
        value = nn.Dense(1)(x)
        return value.squeeze(-1)


# ============================================================================
# Actor-Critic Policy Implementation
# ============================================================================


def create_actor_critic_policy(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer,
    normalizer_params: Any,
) -> tuple[Policy, PolicyState]:
    """Creates an actor-critic policy with separate actor and critic networks.

    Policy functions operate on single-agent inputs:
    - state shape: (dim_state,)
    - action shape: (dim_action,)
    - returns: scalar values/log_probs/entropy

    Parameters are always stacked along axis 0: (num_agents, ...)
    Callers must use jax.vmap to apply policy functions across agents.
    """

    policy_params = config.get("policy_params", {})
    hidden_layers = policy_params.get("hidden_layers", [64, 64])
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    num_agents = config.get("num_agents", 1)

    # Initialize networks
    actor = ActorNetwork(hidden_layers=hidden_layers, dim_action=dim_action)
    critic = CriticNetwork(hidden_layers=hidden_layers)

    # Initialize parameters
    dummy_state = jnp.zeros(dim_state)

    # Initialize separate parameters for each agent and stack them
    keys = jax.random.split(key, 2 * num_agents + 1)
    key = keys[0]
    actor_keys = keys[1 : num_agents + 1]
    critic_keys = keys[num_agents + 1 :]

    # Initialize parameters for each agent
    actor_params_list = [actor.init(k, dummy_state) for k in actor_keys]
    critic_params_list = [critic.init(k, dummy_state) for k in critic_keys]

    # Stack parameters along agent dimension
    actor_params = jax.tree.map(
        lambda *args: jnp.stack(args), *actor_params_list
    )
    critic_params = jax.tree.map(
        lambda *args: jnp.stack(args), *critic_params_list
    )

    params = {"actor": actor_params, "critic": critic_params}
    policy_state = PolicyState(params=params, key=key)

    # Define select_action function (stochastic)
    # Single agent only - callers should vmap for multi-agent
    @jax.jit
    def select_action(params, state, dyn_params, key):
        # Normalize state
        normalized_state = normalizer.normalize(
            normalizer_params["state"], state
        )

        # Forward pass through actor
        mean, log_std = actor.apply(params["actor"], normalized_state)

        std = jnp.exp(log_std)
        action = mean + std * jax.random.normal(key, shape=mean.shape)

        return action, params

    # Define select_action_deterministic function (returns mean)
    # Single agent only - callers should vmap for multi-agent
    @jax.jit
    def select_action_deterministic(params, state, dyn_params):
        # Normalize state
        normalized_state = normalizer.normalize(
            normalizer_params["state"], state
        )

        # Forward pass through actor
        mean, log_std = actor.apply(params["actor"], normalized_state)

        # Return mean action (no sampling)
        return mean

    # Define evaluate_value function
    # Single agent only - callers should vmap for multi-agent
    @jax.jit
    def evaluate_value(params, state, dyn_params):
        # Normalize state
        normalized_state = normalizer.normalize(
            normalizer_params["state"], state
        )

        # Forward pass through critic
        value = critic.apply(params["critic"], normalized_state)

        return value

    # Define compute_log_prob function
    # Single agent only - callers should vmap for multi-agent
    @jax.jit
    def compute_log_prob(params, state, action, dyn_params):
        # Normalize state
        normalized_state = normalizer.normalize(
            normalizer_params["state"], state
        )

        # Forward pass through actor to get distribution
        mean, log_std = actor.apply(params["actor"], normalized_state)
        std = jnp.exp(log_std)

        # Compute log probability of action under Gaussian distribution
        log_prob = -0.5 * jnp.sum(
            ((action - mean) / std) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
            axis=-1,
        )

        return log_prob

    # Define compute_entropy function
    # Single agent only - callers should vmap for multi-agent
    @jax.jit
    def compute_entropy(params, state, dyn_params):
        # Normalize state
        normalized_state = normalizer.normalize(
            normalizer_params["state"], state
        )

        # Forward pass through actor to get distribution
        _, log_std = actor.apply(params["actor"], normalized_state)

        # Entropy of Gaussian: H = 0.5 * dim * (1 + log(2*pi)) + sum(log_std)
        entropy = 0.5 * dim_action * (1.0 + jnp.log(2 * jnp.pi)) + jnp.sum(
            log_std
        )

        return entropy

    policy = Policy(
        select_action=select_action,
        select_action_deterministic=select_action_deterministic,
        evaluate_value=evaluate_value,
        compute_log_prob=compute_log_prob,
        compute_entropy=compute_entropy,
    )

    return policy, policy_state


# ============================================================================
# ICEM Policy Implementation
# ============================================================================


def create_icem_policy(
    key: jax.Array,
    config: Any,
    dynamics_model: DynamicsModel,
    cost_fn: CostFn,
) -> tuple[Policy, PolicyState]:
    """Creates an ICEM planner-based policy."""
    from scripts.planners import create_icem_planner

    # Create the iCEM planner
    planner, planner_state = create_icem_planner(config, cost_fn, key)

    # Store planner state as policy params
    policy_state = PolicyState(params=planner_state, key=key)

    # Define select_action function
    def select_action(params, state, dyn_params, key, deterministic=False):
        # params is a PlannerState for ICEM
        # dyn_params contains the dynamics parameters
        action_seq, new_planner_state = planner.solve(
            params, state, dyn_params, cost_fn
        )
        # Return first action and updated planner state
        first_action = action_seq[0]
        return first_action, new_planner_state

    # Define select_action_deterministic function (ICEM is already deterministic)
    def select_action_deterministic(params, state, dyn_params):
        # For ICEM, just call solve and return first action (no state update needed for eval)
        action_seq, _ = planner.solve(params, state, dyn_params, cost_fn)
        return action_seq[0]

    # Define evaluate_value function (not used for ICEM)
    def evaluate_value(_params, _state, _dyn_params):
        return 0.0

    # Define compute_log_prob function (not used for ICEM)
    def compute_log_prob(_params, _state, _action, _dyn_params):
        return 0.0

    # Define compute_entropy function (not used for ICEM)
    def compute_entropy(_params, _state, _dyn_params):
        return 0.0

    policy = Policy(
        select_action=select_action,
        select_action_deterministic=select_action_deterministic,
        evaluate_value=evaluate_value,
        compute_log_prob=compute_log_prob,
        compute_entropy=compute_entropy,
    )

    return policy, policy_state


# ============================================================================
# Policy Factory Function
# ============================================================================


def init_policy(
    key: jax.Array,
    config: Any,
    dynamics_model: DynamicsModel,
    cost_fn: CostFn,
    normalizer: Normalizer,
    normalizer_params: Any,
) -> tuple[Policy, PolicyState]:
    """Initializes the appropriate policy based on the configuration."""

    policy_type = config["policy"]
    print(f"🎯 Initializing policy: {policy_type.upper()}")

    if policy_type == "actor-critic":
        return create_actor_critic_policy(
            key, config, normalizer, normalizer_params
        )
    elif policy_type == "icem":
        return create_icem_policy(key, config, dynamics_model, cost_fn)
    else:
        raise ValueError(f"Unknown policy type: '{policy_type}'")
