import jax
import jax.numpy as jnp
from jax.numpy.fft import irfft, rfftfreq
from jax import random
from functools import partial
from typing import NamedTuple, Callable, Tuple, Any
from flax import struct

# --- Type Hinting ---
Array = jnp.ndarray
# A reward function takes the initial state, a sequence of controls, and reward-specific
# parameters, and returns a scalar reward. It is responsible for rolling out the
# trajectory using the dynamics.
RewardFn = Callable[[Array, Array, Array], float]


# --- Unified Planner Abstraction ---
class PlannerState(struct.PyTreeNode):
    """A flexible planner state that can handle various planning algorithms."""

    key: jax.Array
    mean: Array | None = None  # For CEM, iCEM
    elites: Array | None = None  # For iCEM


class Planner(NamedTuple):
    """
    A generic container for a planning algorithm.
    The reward function is now part of the planner's immutable state.
    """

    reward_fn: RewardFn
    solve_fn: Callable[
        [PlannerState, Array, Array], Tuple[Array, PlannerState]
    ]

    def solve(
        self,
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        """
        Executes the planner's solve function for the pre-configured reward function.

        Args:
            state: The current state of the planner (e.g., mean, key).
            init_env_state: The initial state of the system.
            cost_params: Parameters to be passed to the reward function.

        Returns:
            A tuple containing the best control sequence and the updated planner state.
        """
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

    For iCEM/CEM/random planners: pass reward_fn and key.
    For MPPI planner: pass encoder, dynamics, reward, critic, policy, and key.
    """
    planner_type = config.get("planner_type", "icem")
    print(f"🚀 Initializing planner: {planner_type.upper()}")

    if planner_type == "cem":
        planner, state = create_cem_planner(config, reward_fn, key)
    elif planner_type == "icem":
        planner, state = create_icem_planner(config, reward_fn, key)
    elif planner_type == "random":
        planner, state = create_random_planner(config, reward_fn, key)
    elif planner_type == "mppi":
        planner, state = create_mppi_planner(
            config, encoder, dynamics, reward, critic, policy, key
        )
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    return planner, state


def create_random_planner(
    config: Any, reward_fn: RewardFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """Creates a planner that outputs random actions."""
    # The reward function is ignored but kept for API consistency.

    # Initialize state - only the key is needed.
    initial_state = PlannerState(key=key)

    @partial(jax.jit)
    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: Array,
    ) -> Tuple[Array, PlannerState]:
        """JIT-compiled solve function for the random planner."""
        # The init_env_state and cost_params are ignored but kept for API consistency.
        return _random_solve_internal(config, state)

    return Planner(reward_fn=reward_fn, solve_fn=solve_fn), initial_state


def _random_solve_internal(
    config: Any,
    state: PlannerState,
) -> Tuple[Array, PlannerState]:
    """Core logic for the random action planner."""
    # Unpack config
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    action_low = jnp.array(planner_params.get("action_low"))
    action_high = jnp.array(planner_params.get("action_high"))

    # Generate a new key for this planning step
    key, subkey = random.split(state.key)

    # Generate random actions for the entire horizon
    random_actions = random.uniform(
        subkey,
        shape=(horizon, dim_control),
        minval=action_low,
        maxval=action_high,
    )

    # Update the planner state with the new key for the next step
    new_state = state.replace(key=key)

    return random_actions, new_state


# --- CEM Implementation ---


def create_cem_planner(
    config: Any, reward_fn: RewardFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """Creates a Cross-Entropy Method (CEM) planner."""
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    unrolled_dim = horizon * dim_control
    initial_mean_val = planner_params.get("initial_mean_val", 0.0)
    initial_var_val = planner_params.get("initial_variance_val", 0.5)

    initial_mean = jnp.ones(unrolled_dim) * initial_mean_val
    initial_state = PlannerState(key=key, mean=initial_mean)
    initial_var = jnp.ones(unrolled_dim) * initial_var_val

    @partial(jax.jit)
    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: Array,
    ) -> Tuple[Array, PlannerState]:
        """JIT-compiled solve function for CEM."""
        return _cem_solve_internal(
            reward_fn, config, state, init_env_state, cost_params, initial_var
        )

    return Planner(reward_fn=reward_fn, solve_fn=solve_fn), initial_state


def _cem_solve_internal(
    reward_fn: RewardFn,
    config: Any,
    state: PlannerState,
    init_env_state: Array,
    cost_params: Array,
    initial_var: Array,  # ADDED: Receive initial_var as an argument.
) -> Tuple[Array, PlannerState]:
    """Core logic for the CEM algorithm."""
    # Unpack config
    planner_params = config.get("planner_params", {})
    batch_size = planner_params.get("batch_size")
    elit_frac = planner_params.get("elit_frac")
    learning_rate = planner_params.get("learning_rate")
    max_iter = planner_params.get("max_iter")
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    reg_cov = planner_params.get("reg_cov", 1e-6)

    num_elites = int(elit_frac * batch_size)

    # Vectorize the reward function
    vmapped_reward_fn = jax.vmap(
        lambda controls: reward_fn(
            init_env_state, controls.reshape(horizon, dim_control), cost_params
        ),
        in_axes=0,
    )

    def cem_iteration(carry, _):
        mean, var, key = carry
        key, subkey = random.split(key)

        sampling_cov = jnp.diag(var + reg_cov)
        batch = random.multivariate_normal(
            subkey, mean, sampling_cov, shape=(batch_size,)
        )

        rewards = vmapped_reward_fn(batch)
        _, elite_indices = jax.lax.top_k(rewards, k=num_elites)
        elite_samples = batch[elite_indices]

        elite_mean = jnp.mean(elite_samples, axis=0)
        elite_var = jnp.var(elite_samples, axis=0)

        new_mean = (1 - learning_rate) * mean + learning_rate * elite_mean
        new_var = (1 - learning_rate) * var + learning_rate * elite_var

        return (new_mean, new_var, key), None

    init_carry = (state.mean, initial_var, state.key)
    final_carry, _ = jax.lax.scan(
        cem_iteration, init_carry, None, length=max_iter
    )

    final_mean, _, final_key = final_carry
    best_controls = final_mean.reshape(horizon, dim_control)
    new_state = state.replace(mean=final_mean, key=final_key)

    return best_controls, new_state


# --- iCEM Implementation ---


def _create_icem_solve(config: Any, reward_fn: RewardFn) -> Callable:
    """
    Creates a JIT-compiled iCEM solve function with all parameters closed over.
    """
    # Unpack config
    planner_params = config.get("planner_params", {})
    batch_size = planner_params.get("batch_size")
    elit_frac = planner_params.get("elit_frac")
    learning_rate = planner_params.get("learning_rate")
    max_iter = planner_params.get("max_iter")
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    action_low = jnp.tile(jnp.array(planner_params.get("action_low")), horizon)
    action_high = jnp.tile(jnp.array(planner_params.get("action_high")), horizon)
    colored_noise_beta = planner_params.get("colored_noise_beta", 2.0)
    colored_noise_fmin = planner_params.get("colored_noise_fmin", 0.0)
    keep_elites_frac = planner_params.get("keep_elites_frac", 0.3)
    initial_var_val = planner_params.get("initial_variance_val", 0.5)

    # Compute derived values
    num_elites = int(elit_frac * batch_size)
    num_keep_elites = int(keep_elites_frac * num_elites)
    unrolled_dim = horizon * dim_control
    initial_var = jnp.ones(unrolled_dim) * initial_var_val

    # Pre-compute noise parameters (only depends on static config)
    # These are closed over by _generate_colored_noise below
    samples = unrolled_dim
    f = rfftfreq(samples)
    fmin_val = jnp.maximum(colored_noise_fmin, 1.0 / samples)

    # Compute s_scale with cutoff frequency handling
    s_scale = f.copy()
    ix = int(jnp.sum(s_scale < fmin_val))
    if ix > 0 and ix < len(s_scale):
        s_scale = s_scale.at[:ix].set(s_scale[ix])
    s_scale = s_scale ** (-colored_noise_beta / 2.0)

    # Compute sigma for normalization
    w = s_scale[1:].at[-1].set(s_scale[-1] * (1 + (samples % 2)) / 2.0)
    sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / samples

    # Check if samples is even (used for Nyquist handling)
    samples_even = (samples % 2 == 0)

    def _generate_colored_noise(rng):
        """Generate colored noise with pre-computed scaling (closes over s_scale, sigma)."""
        key_sr, key_si = random.split(rng)
        sr = random.normal(key=key_sr, shape=s_scale.shape) * s_scale
        si = random.normal(key=key_si, shape=s_scale.shape) * s_scale

        # DC component: set imaginary to 0, scale real by sqrt(2)
        si = si.at[0].set(0)
        sr = sr.at[0].set(sr[0] * jnp.sqrt(2))

        # Nyquist component for even-length signals
        sr, si = jax.lax.cond(
            samples_even,
            lambda: (sr.at[-1].set(sr[-1] * jnp.sqrt(2)), si.at[-1].set(0)),
            lambda: (sr, si)
        )

        s = sr + 1j * si
        return irfft(s, n=samples) / sigma

    # Vmap over batch of RNG keys
    vmapped_noise_fn = jax.vmap(_generate_colored_noise)

    @jax.jit
    def solve(
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        # Vmapped reward depends on runtime init_env_state and cost_params
        vmapped_reward_fn = jax.vmap(
            lambda ctrls: reward_fn(
                init_env_state, ctrls.reshape(horizon, dim_control), cost_params
            )
        )

        def icem_iteration(carry, i):
            mean, var, key, best_seq, best_reward, iter_elites = carry
            key, noise_key = random.split(key)

            noise_keys = random.split(noise_key, batch_size)
            samples = mean + vmapped_noise_fn(noise_keys) * jnp.sqrt(var)

            elites_to_use = jax.lax.cond(
                i == 0,
                lambda: state.elites[:num_keep_elites],
                lambda: iter_elites[:num_keep_elites],
            )
            samples = samples.at[:num_keep_elites].set(elites_to_use)
            samples = jnp.clip(samples, action_low, action_high)
            samples = jax.lax.cond(
                i == max_iter - 1,
                lambda s: s.at[-1].set(mean),
                lambda s: s,
                samples,
            )

            rewards = vmapped_reward_fn(samples)
            _, elite_indices = jax.lax.top_k(rewards, k=num_elites)
            new_iter_elites = samples[elite_indices]

            current_best_reward = rewards[elite_indices[0]]
            current_best_seq = new_iter_elites[0]
            best_seq, best_reward = jax.lax.cond(
                current_best_reward > best_reward,
                lambda: (current_best_seq, current_best_reward),
                lambda: (best_seq, best_reward),
            )

            elite_mean = jnp.mean(new_iter_elites, axis=0)
            elite_var = jnp.var(new_iter_elites, axis=0)
            new_mean = (1 - learning_rate) * mean + learning_rate * elite_mean
            new_var = (1 - learning_rate) * var + learning_rate * elite_var

            return (
                new_mean,
                new_var,
                key,
                best_seq,
                best_reward,
                new_iter_elites,
            ), None

        init_carry = (
            state.mean,
            initial_var,
            state.key,
            state.mean,
            -jnp.inf,
            jnp.zeros((num_elites, unrolled_dim)),
        )
        final_carry, _ = jax.lax.scan(
            icem_iteration, init_carry, jnp.arange(max_iter)
        )

        final_mean, _, final_key, final_best_seq, _final_best_reward, final_elites = final_carry

        # Temporal shift: roll forward by one timestep, repeat last action
        shifted_mean = jnp.concatenate(
            [final_mean[dim_control:], final_mean[-dim_control:]]
        )
        shifted_elites = jnp.concatenate(
            [final_elites[:, dim_control:], final_elites[:, -dim_control:]], axis=1
        )

        new_state = state.replace(
            mean=shifted_mean, elites=shifted_elites, key=final_key
        )
        best_controls = final_best_seq.reshape(horizon, dim_control)

        return best_controls, new_state

    return solve


def create_icem_planner(
    config: Any, reward_fn: RewardFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """Creates an Improved Cross-Entropy Method (iCEM) planner."""
    # Extract what's needed for initial state
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    batch_size = planner_params.get("batch_size")
    elit_frac = planner_params.get("elit_frac")
    initial_mean_val = planner_params.get("initial_mean_val", 0.0)

    unrolled_dim = horizon * dim_control
    num_elites = int(elit_frac * batch_size)

    # Create initial state
    initial_mean = jnp.ones(unrolled_dim) * initial_mean_val
    initial_elites = jnp.zeros((num_elites, unrolled_dim))
    initial_state = PlannerState(key=key, mean=initial_mean, elites=initial_elites)

    # Create the JIT-compiled solver with everything closed over
    _solve = _create_icem_solve(config, reward_fn)

    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        """Solve function for iCEM."""
        return _solve(state, init_env_state, cost_params)

    return Planner(reward_fn=reward_fn, solve_fn=solve_fn), initial_state


# --- MPPI (latent-space, TDMPC2-style) ---


def create_mppi_planner(
    config: Any,
    encoder,
    dynamics,
    reward,
    critic,
    policy,
    key: jax.Array,
) -> tuple[Planner, PlannerState]:
    """
    MPPI planner operating in latent space (TD-MPC2 style).

    Plans by:
      1. Encoding obs -> z0
      2. Sampling action sequences (Gaussian perturbations of current mean)
      3. Rolling out trajectories in latent space
      4. Evaluating: sum_t reward_head(z_t, a_t) + gamma^H * Q(z_H, pi(z_H))
      5. Temperature-weighted mean update
      6. Returning best action

    cost_params passed to solve() must be the full `parameters` dict
    (containing "encoder", "dynamics", "reward", "critic", "policy" keys).

    config["planner_params"]:
        horizon:      int
        dim_control:  int (= dim_action)
        batch_size:   int, number of trajectory samples
        num_pi_trajs: int, how many samples come from the policy (default 24)
        temperature:  float, MPPI temperature (default 0.5)
        min_std:      float, minimum action std (default 0.05)
        discount_factor: float, gamma for bootstrap value (default 0.99)
    """
    pp = config.get("planner_params", {})
    horizon: int = pp["horizon"]
    dim_a: int = pp.get("dim_control", config["dim_action"])
    num_samples: int = pp.get("batch_size", 512)
    num_pi_trajs: int = min(pp.get("num_pi_trajs", 24), num_samples)
    temperature: float = pp.get("temperature", 0.5)
    min_std: float = pp.get("min_std", 0.05)
    discount_factor: float = pp.get("discount_factor", 0.99)

    initial_mean = jnp.zeros((horizon, dim_a))
    initial_state = PlannerState(key=key, mean=initial_mean)

    @partial(jax.jit)
    def solve_fn(
        state: PlannerState,
        obs: Array,
        parameters: dict,  # full unified parameters dict
    ) -> Tuple[Array, PlannerState]:
        key, enc_key, sample_key, pi_key, q_key = jax.random.split(state.key, 5)

        # 1. Encode current observation
        z0 = encoder.encode(parameters["encoder"], obs)  # (latent,)

        # 2. Build action sequences
        #    - num_pi_trajs from the policy (sample H actions per trajectory)
        #    - rest: prior mean + Gaussian noise
        noise_keys = jax.random.split(sample_key, num_samples)

        def make_noisy(k):
            noise = jax.random.normal(k, (horizon, dim_a)) * jnp.maximum(min_std, 0.5)
            return jnp.clip(state.mean + noise, -1.0, 1.0)

        # Noise-based samples
        action_seqs = jax.vmap(make_noisy)(noise_keys)  # (N, H, dim_a)

        # Policy-based samples (first num_pi_trajs slots)
        def policy_rollout(k):
            def step(z, k_t):
                k_t, k_next = jax.random.split(k_t)
                a, _ = policy.sample(parameters["policy"], z, k_t)
                z_next = dynamics.infer_dynamics(parameters["dynamics"]["mean"], z, a)
                return z_next, a
            pi_keys_t = jax.random.split(k, horizon)
            _, pi_acts = jax.lax.scan(step, z0, pi_keys_t)
            return pi_acts  # (H, dim_a)

        pi_rollout_keys = jax.random.split(pi_key, num_pi_trajs)
        pi_action_seqs = jax.vmap(policy_rollout)(pi_rollout_keys)  # (num_pi_trajs, H, dim_a)
        action_seqs = action_seqs.at[:num_pi_trajs].set(pi_action_seqs)

        # 3. Evaluate trajectories
        def eval_trajectory(actions):
            """
            actions: (H, dim_a)
            Returns: total discounted return estimate
            """
            def step(z, a):
                rew = reward.predict(parameters["reward"], z, a)
                z_next = dynamics.infer_dynamics(parameters["dynamics"]["mean"], z, a)
                return z_next, rew

            z_final, stage_rewards = jax.lax.scan(step, z0, actions)

            # Bootstrap terminal value: Q(z_H, pi(z_H))
            key_pi, key_q = jax.random.split(q_key)
            pi_a_final, _ = policy.sample(parameters["policy"], z_final, key_pi)
            v_final = critic.scalar_value(
                parameters["critic"], z_final, pi_a_final, "min", key_q
            )

            # Discount schedule: gamma^0, gamma^1, ..., gamma^(H-1)
            discounts = discount_factor ** jnp.arange(horizon)
            total = jnp.sum(discounts * stage_rewards) + (discount_factor ** horizon) * v_final
            return total

        values = jax.vmap(eval_trajectory)(action_seqs)  # (N,)

        # 4. Temperature-weighted mean update
        max_val = jnp.max(values)  # numerical stability
        weights = jax.nn.softmax((values - max_val) / (temperature + 1e-8))  # (N,)
        new_mean = jnp.einsum("n,nha->ha", weights, action_seqs)  # (H, dim_a)
        new_mean = jnp.clip(new_mean, -1.0, 1.0)

        # 5. Temporal shift for next planning step
        shifted_mean = jnp.concatenate([new_mean[1:], new_mean[-1:]], axis=0)

        new_state = state.replace(mean=shifted_mean, key=key)
        return new_mean, new_state

    return Planner(reward_fn=None, solve_fn=solve_fn), initial_state