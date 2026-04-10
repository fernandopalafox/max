import jax
import jax.numpy as jnp
import jax.flatten_util
from jax import random
from functools import partial
from typing import NamedTuple, Callable, Tuple, Any
from flax import struct

Array = jnp.ndarray
RewardFn = Callable[[Array, Array, Array], float]


class PlannerState(struct.PyTreeNode):
    """Planner state for MPPI."""
    key: jax.Array
    mean: Array | None = None


class Planner(NamedTuple):
    """Generic container for a planning algorithm."""
    reward_fn: RewardFn
    solve_fn: Callable[[PlannerState, Array, Array], Tuple[Array, PlannerState]]

    def solve(
        self,
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        return self.solve_fn(state, init_env_state, cost_params)


def init_planner(
    config: Any,
    reward_fn: RewardFn = None,
    key: jax.Array = None,
    encoder=None,
    dynamics=None,
    reward=None,
    critic=None,
    policy=None,
) -> tuple[Planner, PlannerState]:
    """
    Initializes the appropriate planner based on the configuration.

    For MPPI planner: pass encoder, dynamics, reward, critic, policy, and key.
    """
    planner_type = config["planner"]["type"]

    if planner_type == "mppi":
        pp = config["planner"]
        horizon = pp["horizon"]
        discount = pp.get("discount_factor", 0.99)
        encode_fn = make_tdmpc2_encode_fn(encoder)
        traj_value_fn = make_tdmpc2_trajectory_value_fn(
            dynamics, reward, critic, policy, horizon, discount
        )
        action_proposal_fn = make_tdmpc2_action_proposal_fn(dynamics, policy, horizon)
        planner, state = create_mppi_planner(config, encode_fn, traj_value_fn, key, action_proposal_fn)
    elif planner_type == "mppi_ig":
        pp = config["planner"]
        horizon = pp["horizon"]
        discount = pp["discount_factor"]
        encode_fn = make_tdmpc2_encode_fn(encoder)
        traj_value_fn = make_tdmpc2_trajectory_value_fn_ig(
            dynamics, reward, critic, policy, horizon, discount,
            meas_noise_scale=pp["meas_noise_scale"],
            info_weight=pp["info_weight"],
        )
        action_proposal_fn = make_tdmpc2_action_proposal_fn(dynamics, policy, horizon)
        planner, state = create_mppi_planner(config, encode_fn, traj_value_fn, key, action_proposal_fn)
    else:
        raise ValueError(f"Unknown planner type: {planner_type!r}")

    return planner, state


def make_tdmpc2_encode_fn(encoder):
    """Returns encode_fn: (cost_params, obs) -> z"""
    def encode_fn(cost_params, obs):
        return encoder.encode(
            cost_params["mean"]["encoder"],
            obs,
        )
    return encode_fn


def make_tdmpc2_trajectory_value_fn(dynamics, reward, critic, policy, horizon, discount_factor):
    """Returns trajectory_value_fn: (cost_params, z0, action_seqs, key) -> values  shape (N,)"""
    def trajectory_value_fn(cost_params, z0, action_seqs, key):
        key_pi, key_q = jax.random.split(key)

        def eval_traj(actions):
            def step(z, a):
                r = reward.predict(cost_params["mean"]["reward"], z, a)
                z_next = dynamics.predict(cost_params["mean"]["dynamics"], z, a)
                return z_next, r
            z_H, rewards = jax.lax.scan(step, z0, actions)
            pi_a, _ = policy.sample(cost_params["mean"]["policy"], z_H, key_pi)
            v = jnp.mean(critic.subsample(cost_params["mean"]["critic"], z_H, pi_a, key_q))
            discounts = discount_factor ** jnp.arange(horizon)
            return jnp.dot(discounts, rewards) + (discount_factor ** horizon) * v

        return jax.vmap(eval_traj)(action_seqs)

    return trajectory_value_fn


def make_tdmpc2_trajectory_value_fn_ig(
    dynamics, reward, critic, policy, horizon, discount_factor, meas_noise_scale, info_weight
):
    """MPPI trajectory value with analytical info-gathering bonus at each rollout step.
    Terminal Q-value is task-only and untouched."""
    def trajectory_value_fn(cost_params, z0, action_seqs, key):
        key_pi, key_q = jax.random.split(key)

        def eval_traj(actions):
            def step(z, a):
                r = reward.predict(cost_params["mean"]["reward"], z, a)
                dyn_params = cost_params["mean"]["dynamics"]
                flat_params, unflatten = jax.flatten_util.ravel_pytree(dyn_params)
                J = jax.jacrev(lambda fp: dynamics.predict(unflatten(fp), z, a))(flat_params)
                P = cost_params["covariance"]
                R = meas_noise_scale * jnp.eye(z.shape[0])
                S = J @ P @ J.T + R
                info = 0.5 * (jnp.linalg.slogdet(S)[1] - jnp.linalg.slogdet(R)[1])
                z_next = dynamics.predict(dyn_params, z, a)
                return z_next, r + info_weight * info
            z_H, rewards = jax.lax.scan(step, z0, actions)
            pi_a, _ = policy.sample(cost_params["mean"]["policy"], z_H, key_pi)
            v = jnp.mean(critic.subsample(cost_params["mean"]["critic"], z_H, pi_a, key_q))
            discounts = discount_factor ** jnp.arange(horizon)
            return jnp.dot(discounts, rewards) + (discount_factor ** horizon) * v

        return jax.vmap(eval_traj)(action_seqs)

    return trajectory_value_fn


def make_tdmpc2_action_proposal_fn(dynamics, policy, horizon):
    """Returns action_proposal_fn: (cost_params, z0, key, n) -> action_seqs  shape (n, H, dim_a)"""
    def action_proposal_fn(cost_params, z0, key, n):
        def single_rollout(k):
            def step(z, k_t):
                a, _ = policy.sample(cost_params["mean"]["policy"], z, k_t)
                z_next = dynamics.predict(cost_params["mean"]["dynamics"], z, a)
                return z_next, a
            _, acts = jax.lax.scan(step, z0, jax.random.split(k, horizon))
            return acts
        return jax.vmap(single_rollout)(jax.random.split(key, n))
    return action_proposal_fn


def create_mppi_planner(
    config: Any,
    encode_fn: Callable,           # (cost_params, obs) -> z
    trajectory_value_fn: Callable, # (cost_params, z0, action_seqs, key) -> values  shape (N,)
    key: jax.Array,
    action_proposal_fn: Callable,  # (cost_params, z0, key, n) -> action_seqs  shape (n, H, dim_a)
) -> tuple[Planner, PlannerState]:
    """
    Generic MPPI planner. Reward/value logic is fully decoupled via callbacks.

    config["planner"]:
        horizon:      int
        dim_control:  int (= dim_action)
        batch_size:   int, number of trajectory samples
        num_pi_trajs: int, how many samples come from action_proposal_fn
        temperature:  float, MPPI temperature
        min_std:      float, minimum action std
        num_iterations: int
        num_elites:   int
        max_std:      float
    """
    pp = config["planner"]
    horizon: int = pp["horizon"]
    dim_a: int = pp["dim_control"]
    num_samples: int = pp["batch_size"]
    num_pi_trajs: int = pp["num_pi_trajs"]
    temperature: float = pp["temperature"]
    min_std: float = pp["min_std"]
    num_iterations: int = pp["num_iterations"]
    num_elites: int = pp["num_elites"]
    max_std: float = pp["max_std"]

    initial_mean = jnp.zeros((horizon, dim_a))
    initial_state = PlannerState(key=key, mean=initial_mean)

    @partial(jax.jit)
    def solve_fn(
        state: PlannerState,
        obs: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        key, proposal_key, iter_key = jax.random.split(state.key, 3)

        # 1. Encode observation -> latent state
        z0 = encode_fn(cost_params, obs)

        # 2. Compute policy proposals once before the iteration loop
        pi_seqs = action_proposal_fn(cost_params, z0, proposal_key, num_pi_trajs)

        n_gaussian = num_samples - num_pi_trajs

        # 3. Initialize mean (warm-started) and std (always reset to max_std)
        mean = state.mean
        std = jnp.full((horizon, dim_a), max_std)

        # 4. Iterative MPPI refinement via lax.scan
        def mppi_iteration(carry, _):
            mean, std, iter_key = carry
            iter_key, sample_key, value_key = jax.random.split(iter_key, 3)

            # Sample Gaussian trajectories around current mean
            gaussian_seqs = jnp.clip(
                mean + std * jax.random.normal(sample_key, (n_gaussian, horizon, dim_a)),
                -1.0, 1.0,
            )
            action_seqs = jnp.concatenate([pi_seqs, gaussian_seqs], axis=0)  # (N, H, dim_a)

            # Evaluate trajectories
            values = trajectory_value_fn(cost_params, z0, action_seqs, value_key)  # (N,)

            # Select top-k elites
            _, elite_idxs = jax.lax.top_k(values, num_elites)
            elite_actions = action_seqs[elite_idxs]   # (num_elites, H, dim_a)
            elite_values = values[elite_idxs]          # (num_elites,)

            # Softmax-weighted mean and std update
            score = jax.nn.softmax(temperature * (elite_values - jnp.max(elite_values)))
            new_mean = jnp.einsum("n,nha->ha", score, elite_actions)
            new_mean = jnp.clip(new_mean, -1.0, 1.0)
            diff = elite_actions - new_mean[None]
            new_std = jnp.clip(
                jnp.sqrt(jnp.einsum("n,nha->ha", score, diff ** 2) + 1e-8),
                min_std, max_std,
            )

            return (new_mean, new_std, iter_key), None

        (final_mean, _final_std, _iter_key), _ = jax.lax.scan(
            mppi_iteration, (mean, std, iter_key), None, length=num_iterations
        )

        # 5. Temporal shift for next planning step
        shifted_mean = jnp.concatenate([final_mean[1:], final_mean[-1:]], axis=0)

        new_state = state.replace(mean=shifted_mean, key=key)
        return final_mean, new_state

    return Planner(reward_fn=None, solve_fn=solve_fn), initial_state
