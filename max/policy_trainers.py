"""Policy trainer abstraction for updating policy parameters."""

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Any, NamedTuple, Optional
from flax import struct

from max.policies import Policy
from max.dynamics import DynamicsModel


class PolicyTrainState(struct.PyTreeNode):
    """Training state for policy optimization."""

    params: Any  # For actor-critic: {'actor': ..., 'critic': ...}, for ICEM: PlannerState
    opt_state: Any = None
    key: Optional[jax.Array] = None
    # For ICEM: store latest dynamics params and cost params
    dyn_params: Any = None
    cost_params: Any = None


class PolicyTrainer(NamedTuple):
    """A generic container for a policy training algorithm."""

    train_fn: Callable[[PolicyTrainState, dict], tuple[PolicyTrainState, dict]]

    def train(
        self, train_state: PolicyTrainState, data: dict
    ) -> tuple[PolicyTrainState, dict]:
        return self.train_fn(train_state, data)


# ============================================================================
# ICEM Policy Trainer Implementation
# ============================================================================


def create_icem_policy_trainer(
    config: Any,
    policy: Policy,
) -> tuple[PolicyTrainer, PolicyTrainState]:
    """
    Creates a policy trainer for ICEM that updates the planner state with
    the newest dynamics and cost parameters.
    """

    policy_trainer_params = config.get("policy_trainer_params", {})

    def train_fn(
        train_state: PolicyTrainState, data: dict
    ) -> tuple[PolicyTrainState, dict]:
        """
        For ICEM, we just update the stored dynamics and cost parameters.
        The data dict should contain:
        - 'dyn_params': Latest dynamics model parameters
        - 'cost_params': Latest cost function parameters (if any)
        """
        new_dyn_params = data.get("dyn_params", train_state.dyn_params)
        new_cost_params = data.get("cost_params", train_state.cost_params)

        new_train_state = train_state.replace(
            dyn_params=new_dyn_params,
            cost_params=new_cost_params,
        )

        metrics = {}
        return new_train_state, metrics

    trainer = PolicyTrainer(train_fn=train_fn)

    # Initial train state doesn't have optimizers for ICEM
    # The params field will be set to the initial PlannerState when the policy is created
    initial_train_state = PolicyTrainState(
        params=None,  # Will be set later
        opt_state=None,
        key=None,
        dyn_params=None,
        cost_params=None,
    )

    return trainer, initial_train_state

# ============================================================================
# IPPO (Independent PPO) Policy Trainer Implementation
# ============================================================================


def compute_gae(
    rewards: jax.Array,
    values: jax.Array,
    dones: jax.Array,
    last_value: jax.Array,
    gamma: float,
    lam: float,
) -> tuple[jax.Array, jax.Array]:
    """
    Compute Generalized Advantage Estimation (GAE).

    Args:
        rewards: (T,) array of rewards
        values: (T,) array of value estimates
        dones: (T,) array of done flags
        last_value: scalar, value estimate for final state
        gamma: discount factor
        lam: GAE lambda parameter

    Returns:
        advantages: (T,) array of advantage estimates
        value_targets: (T,) array of value targets for training (advantages + values)
    """
    # Append last_value to values for next value calculation
    next_values = jnp.concatenate([values[1:], last_value[None]])
    next_dones = jnp.concatenate([dones[1:], jnp.array([0.0])])

    # Compute TD residuals: delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
    deltas = rewards + gamma * next_values * (1.0 - next_dones) - values

    # Compute advantages using GAE: A_t = delta_t + gamma * lambda * (1 - done) * A_{t+1}
    # Scan backwards through time
    def scan_fn(gae, t_idx):
        t = rewards.shape[0] - 1 - t_idx
        delta = deltas[t]
        next_nonterminal = 1.0 - next_dones[t]
        gae = delta + gamma * lam * next_nonterminal * gae
        return gae, gae

    _, advantages = jax.lax.scan(
        scan_fn,
        init=0.0,
        xs=jnp.arange(rewards.shape[0]),
    )

    # Reverse to get correct time order
    advantages = advantages[::-1]
    value_targets = advantages + values

    return advantages, value_targets


def create_ippo_policy_trainer(
    config: Any,
    policy: Policy,
    policy_params: Any,
    key: jax.Array,
) -> tuple[PolicyTrainer, PolicyTrainState]:
    """
    Creates an IPPO (Independent PPO) policy trainer for multi-agent actor-critic policies.

    Args:
        config: Configuration dictionary.
        policy: The policy object with compute_log_prob, etc.
        policy_params: PyTree of initial policy parameters (stacked for all agents).
        key: JAX PRNG key.

    Returns:
        tuple[PolicyTrainer, PolicyTrainState]: The trainer and the fully initialized train state.
    """

    policy_trainer_params = config.get("policy_trainer_params", {})
    actor_lr = policy_trainer_params.get("actor_lr", 3e-4)
    critic_lr = policy_trainer_params.get("critic_lr", 1e-3)
    ppo_lambda = policy_trainer_params.get("ppo_lambda", 0.95)
    ppo_gamma = policy_trainer_params.get("ppo_gamma", 0.99)
    clip_epsilon = policy_trainer_params.get("clip_epsilon", 0.2)
    n_epochs = policy_trainer_params.get("n_epochs", 4)
    mini_batch_size = policy_trainer_params.get("mini_batch_size", 64)
    entropy_coef = policy_trainer_params.get("entropy_coef", 0.01)
    value_coef = policy_trainer_params.get("value_coef", 0.5)
    max_grad_norm = policy_trainer_params.get("max_grad_norm", 0.5)

    actor_optimizer = optax.adam(actor_lr, eps=1e-5)
    critic_optimizer = optax.adam(critic_lr, eps=1e-5)

    @jax.jit
    def train_fn(
        train_state: PolicyTrainState, data: dict
    ) -> tuple[PolicyTrainState, dict]:
        """
        IPPO training step - vmap over agents.

        Expected data dict (num_agents as FIRST dimension):
        - 'states': (num_agents, H, dim_state)
        - 'actions': (num_agents, H, dim_action)
        - 'rewards': (num_agents, H) or (H,) - will broadcast if shared
        - 'dones': (num_agents, H) or (H,) - will broadcast if shared
        - 'log_pis_old': (num_agents, H)
        - 'values_old': (num_agents, H)
        - 'last_value': (num_agents,)
        """

        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        dones = data["dones"]
        log_pis_old = data["log_pis_old"]
        values_old = data["values_old"]
        last_value = data["last_value"]

        num_agents = states.shape[0]

        # advantages and value targets
        advantages, value_targets = jax.vmap(
            lambda r, v, lv: compute_gae(r, v, dones, lv, ppo_gamma, ppo_lambda),
            in_axes=(0, 0, 0)
        )(rewards, values_old, last_value)

        # Define single-agent PPO update function
        def single_agent_ppo_update(
            agent_actor_params,
            agent_critic_params,
            agent_actor_opt_state,
            agent_critic_opt_state,
            agent_states,
            agent_actions,
            agent_log_pis_old,
            agent_advantages,
            agent_value_targets,
            agent_values_old,
            agent_key,
        ):
            """
            Run PPO update for a single agent.
            All inputs have shape (H, ...) for this agent.
            """
            n_samples = agent_states.shape[0]
            n_minibatches = n_samples // mini_batch_size

            def ppo_loss_fn(
                actor_params,
                critic_params,
                batch_states,
                batch_actions,
                batch_log_pis_old,
                batch_advantages,
                batch_value_targets,
                batch_values_old,
            ):
                """Compute PPO loss for a mini-batch."""
                params = {"actor": actor_params, "critic": critic_params}

                # Compute log probabilities and values under new policy
                log_pis_new = jax.vmap(
                    lambda s, a: policy.compute_log_prob(params, s, a, None)
                )(batch_states, batch_actions)
                values_new = jax.vmap(
                    lambda s: policy.evaluate_value(params, s, None)
                )(batch_states)
                entropies = jax.vmap(
                    lambda s: policy.compute_entropy(params, s, None)
                )(batch_states)
                entropy = entropies.mean()

                # Normalize advantages per mini-batch
                advs_normed = (
                    batch_advantages - batch_advantages.mean()
                ) / (batch_advantages.std() + 1e-8)

                # PPO clipped surrogate objective
                ratio = jnp.exp(log_pis_new - batch_log_pis_old)
                clipped_ratio = jnp.clip(
                    ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon
                )
                policy_loss = -jnp.minimum(
                    ratio * advs_normed, clipped_ratio * advs_normed
                ).mean()

                # Value function loss with clipping
                value_pred_clipped = batch_values_old + jnp.clip(
                    values_new - batch_values_old, -clip_epsilon, clip_epsilon
                )
                value_loss_unclipped = (values_new - batch_value_targets) ** 2
                value_loss_clipped = (
                    value_pred_clipped - batch_value_targets
                ) ** 2
                value_loss = (
                    0.5
                    * jnp.maximum(
                        value_loss_unclipped, value_loss_clipped
                    ).mean()
                )

                # Total loss
                total_loss = (
                    policy_loss
                    + value_coef * value_loss
                    - entropy_coef * entropy
                )

                return total_loss, {
                    "policy_loss": policy_loss,
                    "value_loss": value_loss,
                    "entropy": entropy,
                    "ratio_mean": ratio.mean(),
                }

            def update_epoch(carry, _):
                """Run one full epoch through the shuffled buffer for this agent."""
                (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    key,
                ) = carry

                key, shuffle_key = jax.random.split(key)
                perm = jax.random.permutation(shuffle_key, n_samples)

                states_shuffled = agent_states[perm]
                actions_shuffled = agent_actions[perm]
                log_pis_old_shuffled = agent_log_pis_old[perm]
                advantages_shuffled = agent_advantages[perm]
                value_targets_shuffled = agent_value_targets[perm]
                values_old_shuffled = agent_values_old[perm]

                n_usable_samples = n_minibatches * mini_batch_size
                states_batched = states_shuffled[:n_usable_samples].reshape(
                    n_minibatches, mini_batch_size, -1
                )
                actions_batched = actions_shuffled[:n_usable_samples].reshape(
                    n_minibatches, mini_batch_size, -1
                )
                log_pis_old_batched = log_pis_old_shuffled[
                    :n_usable_samples
                ].reshape(n_minibatches, mini_batch_size)
                advantages_batched = advantages_shuffled[
                    :n_usable_samples
                ].reshape(n_minibatches, mini_batch_size)
                value_targets_batched = value_targets_shuffled[
                    :n_usable_samples
                ].reshape(n_minibatches, mini_batch_size)
                values_old_batched = values_old_shuffled[
                    :n_usable_samples
                ].reshape(n_minibatches, mini_batch_size)

                batched_data = (
                    states_batched,
                    actions_batched,
                    log_pis_old_batched,
                    advantages_batched,
                    value_targets_batched,
                    values_old_batched,
                )

                def update_minibatch(carry, batch_data):
                    """Update on a single mini-batch."""
                    (
                        actor_params,
                        critic_params,
                        actor_opt_state,
                        critic_opt_state,
                        key,
                    ) = carry

                    (
                        batch_states,
                        batch_actions,
                        batch_log_pis_old,
                        batch_advantages,
                        batch_value_targets,
                        batch_values_old,
                    ) = batch_data

                    (_, loss_info), grads = jax.value_and_grad(
                        lambda params: ppo_loss_fn(
                            params["actor"],
                            params["critic"],
                            batch_states,
                            batch_actions,
                            batch_log_pis_old,
                            batch_advantages,
                            batch_value_targets,
                            batch_values_old,
                        ),
                        has_aux=True,
                    )({"actor": actor_params, "critic": critic_params})

                    grads, _ = optax.clip_by_global_norm(max_grad_norm).update(
                        grads, None
                    )

                    actor_grads = grads["actor"]
                    critic_grads = grads["critic"]

                    actor_updates, actor_opt_state = actor_optimizer.update(
                        actor_grads, actor_opt_state
                    )
                    actor_params = optax.apply_updates(
                        actor_params, actor_updates
                    )

                    critic_updates, critic_opt_state = critic_optimizer.update(
                        critic_grads, critic_opt_state
                    )
                    critic_params = optax.apply_updates(
                        critic_params, critic_updates
                    )

                    return (
                        actor_params,
                        critic_params,
                        actor_opt_state,
                        critic_opt_state,
                        key,
                    ), loss_info

                carry_out, loss_infos = jax.lax.scan(
                    update_minibatch,
                    (
                        actor_params,
                        critic_params,
                        actor_opt_state,
                        critic_opt_state,
                        key,
                    ),
                    batched_data,
                )

                (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    key,
                ) = carry_out

                epoch_metrics = jax.tree.map(lambda x: x.mean(), loss_infos)

                return (
                    actor_params,
                    critic_params,
                    actor_opt_state,
                    critic_opt_state,
                    key,
                ), epoch_metrics

            # Training loop
            carry_out, epoch_metrics = jax.lax.scan(
                update_epoch,
                (
                    agent_actor_params,
                    agent_critic_params,
                    agent_actor_opt_state,
                    agent_critic_opt_state,
                    agent_key,
                ),
                jnp.arange(n_epochs),
            )

            (
                new_actor_params,
                new_critic_params,
                new_actor_opt_state,
                new_critic_opt_state,
                new_key,
            ) = carry_out

            agent_metrics = jax.tree.map(lambda x: x.mean(), epoch_metrics)

            return (
                new_actor_params,
                new_critic_params,
                new_actor_opt_state,
                new_critic_opt_state,
                new_key,
                agent_metrics,
            )

        # Vmap the single-agent PPO update over all agents
        actor_params = train_state.params["actor"]
        critic_params = train_state.params["critic"]
        actor_opt_state = train_state.opt_state["actor"]
        critic_opt_state = train_state.opt_state["critic"]

        agent_keys = jax.random.split(train_state.key, num_agents)

        # Vmap PPO update over agents
        (
            new_actor_params,
            new_critic_params,
            new_actor_opt_state,
            new_critic_opt_state,
            new_keys,
            agent_metrics,
        ) = jax.vmap(
            single_agent_ppo_update, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        )(
            actor_params,
            critic_params,
            actor_opt_state,
            critic_opt_state,
            states,
            actions,
            log_pis_old,
            advantages,
            value_targets,
            values_old,
            agent_keys,
        )

        new_params = {"actor": new_actor_params, "critic": new_critic_params}
        new_opt_state = {
            "actor": new_actor_opt_state,
            "critic": new_critic_opt_state,
        }
        new_key = new_keys[0]

        new_train_state = train_state.replace(
            params=new_params,
            opt_state=new_opt_state,
            key=new_key,
        )

        # Aggregate metrics
        metrics = jax.tree.map(lambda x: x.mean(), agent_metrics)
        metrics["mean_value_target"] = value_targets.mean()
        metrics["mean_advantage"] = advantages.mean()

        for agent_idx in range(num_agents):
            metrics[f"agent_{agent_idx}/policy_loss"] = agent_metrics[
                "policy_loss"
            ][agent_idx]
            metrics[f"agent_{agent_idx}/value_loss"] = agent_metrics[
                "value_loss"
            ][agent_idx]
            metrics[f"agent_{agent_idx}/entropy"] = agent_metrics["entropy"][
                agent_idx
            ]

        return new_train_state, metrics

    trainer = PolicyTrainer(train_fn=train_fn)

    # Init optimizer states
    actor_opt_state = jax.vmap(actor_optimizer.init)(
        policy_params["actor"]
    )
    critic_opt_state = jax.vmap(critic_optimizer.init)(
        policy_params["critic"]
    )
    opt_state = {"actor": actor_opt_state, "critic": critic_opt_state}

    # Init train state
    initial_train_state = PolicyTrainState(
        params=policy_params,
        opt_state=opt_state,
        key=key,
        dyn_params=None,
        cost_params=None,
    )

    return trainer, initial_train_state


# ============================================================================
# Policy Trainer Factory Function
# ============================================================================


def init_policy_trainer(
    config: Any,
    policy: Policy,
    policy_params: Any,
    key: jax.Array,
) -> tuple[PolicyTrainer, PolicyTrainState]:
    """Initializes the appropriate policy trainer based on the configuration."""

    policy_trainer_type = config["policy_trainer"]
    print(f"🎓 Initializing policy trainer: {policy_trainer_type.upper()}")

    if policy_trainer_type == "ippo":
        trainer, train_state = create_ippo_policy_trainer(
            config, policy, policy_params, key
        )

    elif policy_trainer_type == "icem":
        trainer, train_state = create_icem_policy_trainer(config, policy)
        train_state = train_state.replace(params=policy_params)
    else:
        raise ValueError(f"Unknown policy trainer type: '{policy_trainer_type}'")

    return trainer, train_state
