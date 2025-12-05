import jax
import jax.numpy as jnp
from jax.numpy.fft import irfft, rfftfreq
from jax import random
from functools import partial
from typing import NamedTuple, Callable, Tuple, Any
from flax import struct

# --- Type Hinting ---
Array = jnp.ndarray
# A cost function takes the initial state, a sequence of controls, and cost-specific
# parameters, and returns a scalar cost. It is responsible for rolling out the
# trajectory using the dynamics.
CostFn = Callable[[Array, Array, Array], float]


# --- Unified Planner Abstraction ---
class PlannerState(struct.PyTreeNode):
    """A flexible planner state that can handle various planning algorithms."""

    key: jax.Array
    mean: Array | None = None  # For CEM, iCEM
    elites: Array | None = None  # For iCEM


class Planner(NamedTuple):
    """
    A generic container for a planning algorithm.
    The cost function is now part of the planner's immutable state.
    """

    cost_fn: CostFn
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
        Executes the planner's solve function for the pre-configured cost function.

        Args:
            state: The current state of the planner (e.g., mean, key).
            init_env_state: The initial state of the system.
            cost_params: Parameters to be passed to the cost function.

        Returns:
            A tuple containing the best control sequence and the updated planner state.
        """
        return self.solve_fn(state, init_env_state, cost_params)


def init_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """
    Initializes the appropriate planner based on the configuration.
    The cost function is now a required argument for initialization.
    """
    planner_type = config.get("planner_type", "icem")
    print(f"🚀 Initializing planner: {planner_type.upper()}")

    if planner_type == "cem":
        planner, state = create_cem_planner(config, cost_fn, key)
    elif planner_type == "icem":
        planner, state = create_icem_planner(config, cost_fn, key)
    elif planner_type == "random":
        planner, state = create_random_planner(config, cost_fn, key)
    # elif planner_type == "ilqr":
    #     planner, state = create_ilqr_planner(config, cost_fn, key) # Future extension
    else:
        raise ValueError(f"Unknown planner type: {planner_type}")

    return planner, state


def create_random_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """Creates a planner that outputs random actions."""
    # The cost function is ignored but kept for API consistency.

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

    return Planner(cost_fn=cost_fn, solve_fn=solve_fn), initial_state


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
    config: Any, cost_fn: CostFn, key: jax.Array
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
            cost_fn, config, state, init_env_state, cost_params, initial_var
        )

    return Planner(cost_fn=cost_fn, solve_fn=solve_fn), initial_state


def _cem_solve_internal(
    cost_fn: CostFn,
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

    # Vectorize the cost function
    vmapped_cost_fn = jax.vmap(
        lambda controls: cost_fn(
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

        costs = vmapped_cost_fn(batch)
        _, elite_indices = jax.lax.top_k(-costs, k=num_elites)
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


def create_icem_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """Creates an Improved Cross-Entropy Method (iCEM) planner."""
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon")
    dim_control = planner_params.get("dim_control")
    batch_size = planner_params.get("batch_size")
    elit_frac = planner_params.get("elit_frac")
    unrolled_dim = horizon * dim_control
    num_elites = int(elit_frac * batch_size)
    initial_mean_val = planner_params.get("initial_mean_val", 0.0)
    initial_var_val = planner_params.get("initial_variance_val", 0.5)

    initial_mean = jnp.ones(unrolled_dim) * initial_mean_val
    initial_elites = jnp.zeros((num_elites, unrolled_dim))
    initial_state = PlannerState(
        key=key, mean=initial_mean, elites=initial_elites
    )
    initial_var = jnp.ones(unrolled_dim) * initial_var_val

    @partial(jax.jit)
    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: Array,
    ) -> Tuple[Array, PlannerState]:
        """JIT-compiled solve function for iCEM."""
        return _icem_solve_internal(
            cost_fn, config, state, init_env_state, cost_params, initial_var
        )

    return Planner(cost_fn=cost_fn, solve_fn=solve_fn), initial_state


def _icem_solve_internal(
    cost_fn: CostFn,
    config: Any,
    state: PlannerState,
    init_env_state: Array,
    cost_params: Array,
    initial_var: Array,
) -> Tuple[Array, PlannerState]:
    """Core logic for the iCEM algorithm."""
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

    num_elites = int(elit_frac * batch_size)
    num_keep_elites = int(keep_elites_frac * num_elites)
    unrolled_dim = horizon * dim_control

    # Vectorize cost and noise functions
    vmapped_cost_fn = jax.vmap(
        lambda ctrls: cost_fn(
            init_env_state, ctrls.reshape(horizon, dim_control), cost_params
        )
    )
    vmapped_noise_fn = jax.vmap(
        lambda k: powerlaw_psd_gaussian(
            exponent=colored_noise_beta,
            size=(unrolled_dim,),
            rng=k,
            fmin=colored_noise_fmin,
        )
    )

    def icem_iteration(carry, i):
        mean, var, key, best_seq, best_cost, iter_elites = carry
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

        costs = vmapped_cost_fn(samples)
        _, elite_indices = jax.lax.top_k(-costs, k=num_elites)
        new_iter_elites = samples[elite_indices]

        current_best_cost = costs[elite_indices[0]]
        current_best_seq = new_iter_elites[0]
        best_seq, best_cost = jax.lax.cond(
            current_best_cost < best_cost,
            lambda: (current_best_seq, current_best_cost),
            lambda: (best_seq, best_cost),
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
            best_cost,
            new_iter_elites,
        ), None

    init_carry = (
        state.mean,
        initial_var,
        state.key,
        state.mean,
        jnp.inf,
        jnp.zeros((num_elites, unrolled_dim)),
    )
    final_carry, _ = jax.lax.scan(
        icem_iteration, init_carry, jnp.arange(max_iter)
    )

    final_mean, _, final_key, final_best_seq, _, final_elites = final_carry
    new_state = state.replace(
        mean=final_mean, elites=final_elites, key=final_key
    )
    best_controls = final_best_seq.reshape(horizon, dim_control)

    return best_controls, new_state


# --- iCEM Helper Function ---


@partial(jax.jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(
    exponent: float, size: tuple, rng: jax.Array, fmin: float = 0
) -> Array:
    """Generates Gaussian noise with a power-law power spectral density."""
    samples = size[-1]
    f = rfftfreq(samples)
    fmin = jnp.maximum(fmin, 1.0 / samples)

    s_scale = f
    ix = jnp.sum(s_scale < fmin)

    def cutoff(x, idx):
        x_idx = jax.lax.dynamic_slice(x, (idx,), (1,))
        y = jnp.ones_like(x) * x_idx
        mask = jnp.arange(x.shape[0]) < idx
        return jnp.where(mask, y, x)

    s_scale = jax.lax.cond(
        jnp.logical_and(ix > 0, ix < len(s_scale)),
        cutoff,
        lambda x, idx: x,
        s_scale,
        ix,
    )
    s_scale = s_scale ** (-exponent / 2.0)

    w = s_scale[1:]
    w = w.at[-1].set(w[-1] * (1 + (samples % 2)) / 2.0)
    sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / samples

    size_list = list(size)
    size_list[-1] = len(f)
    dims_to_add = len(size_list) - 1
    s_scale_shaped = s_scale[(jnp.newaxis,) * dims_to_add + (Ellipsis,)]

    key_sr, key_si = random.split(rng)
    sr = random.normal(key=key_sr, shape=tuple(size_list)) * s_scale_shaped
    si = random.normal(key=key_si, shape=tuple(size_list)) * s_scale_shaped

    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * jnp.sqrt(2))

    def make_real_if_even(sr_in, si_in):
        si_out = si_in.at[..., -1].set(0)
        sr_out = sr_in.at[..., -1].set(sr_in[..., -1] * jnp.sqrt(2))
        return sr_out, si_out

    sr, si = jax.lax.cond(
        samples % 2 == 0,
        make_real_if_even,
        lambda sr_in, si_in: (sr_in, si_in),
        sr,
        si,
    )

    s = sr + 1j * si
    y = irfft(s, n=samples, axis=-1) / sigma
    return y
