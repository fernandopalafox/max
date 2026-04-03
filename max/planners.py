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

    Args:
        config: Configuration dictionary
        cost_fn: Cost function (may have dynamics_model attribute for A*)
        key: JAX random key
    """
    planner_type = config.get("planner_type", "icem")
    print(f"🚀 Initializing planner: {planner_type.upper()}")

    if planner_type == "cem":
        planner, state = create_cem_planner(config, cost_fn, key)
    elif planner_type == "icem":
        planner, state = create_icem_planner(config, cost_fn, key)
    elif planner_type == "random":
        planner, state = create_random_planner(config, cost_fn, key)
    elif planner_type == "astar":
        planner, state = create_astar_planner(config, cost_fn, key)
    elif planner_type == "mpvi":
        planner, state = create_mpvi_planner(config, cost_fn, key)
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


def _create_icem_solve(config: Any, cost_fn: CostFn) -> Callable:
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
        # Vmapped cost depends on runtime init_env_state and cost_params
        vmapped_cost_fn = jax.vmap(
            lambda ctrls: cost_fn(
                init_env_state, ctrls.reshape(horizon, dim_control), cost_params
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


# --- MPVI (Model Predictive Value Iteration) Implementation ---


def create_mpvi_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """
    Creates a Model Predictive Value Iteration planner for discrete gridworld.

    Performs finite-horizon backward DP over the full grid using the learned
    dynamics model, then extracts an action sequence by forward-rolling the
    time-indexed policy from the current state.

    Uses cost_fn.stage_cost_fn (if available) for per-(state, action) costs,
    which automatically includes info gathering when weight_info > 0.

    Fully JIT-compiled: no callbacks, no Python loops.

    Args:
        config: Configuration dictionary with planner_params
        cost_fn: Cost function (with dynamics_model and optionally stage_cost_fn)
        key: JAX random key
    """
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon", 25)
    dim_control = planner_params.get("dim_control", 1)
    grid_size = planner_params.get("grid_size", 10)
    weight_time = planner_params.get("weight_time", 0.1)

    if not hasattr(cost_fn, 'dynamics_model'):
        raise ValueError("MPVI planner requires cost_fn to have dynamics_model attribute")

    dynamics_model = cost_fn.dynamics_model
    pred_one_step_fn = dynamics_model.pred_one_step
    num_actions = 4
    num_cells = grid_size * grid_size

    initial_state = PlannerState(key=key)

    # Pre-compute all (x,y) coordinates and action arrays (static)
    xs, ys = jnp.meshgrid(jnp.arange(grid_size), jnp.arange(grid_size))
    all_xy = jnp.stack([xs, ys], axis=-1).reshape(num_cells, 2).astype(jnp.float32)
    all_actions = jnp.arange(num_actions, dtype=jnp.float32)

    def _predict_transition(dyn_params, state_xy, action_scalar):
        """Predict next state for one (state, action) pair, rounded to grid."""
        action = jnp.array([action_scalar])
        discrete_action = jnp.round(jnp.clip(action, 0.0, 3.0))
        next_state = pred_one_step_fn(dyn_params, state_xy, discrete_action)
        next_xy = jnp.clip(jnp.round(next_state), 0, grid_size - 1)
        return next_xy

    # Double vmap: over all cells x all actions -> (num_cells, num_actions, 2)
    _predict_all_actions = jax.vmap(_predict_transition, in_axes=(None, None, 0))
    _predict_all_cells_actions = jax.vmap(_predict_all_actions, in_axes=(None, 0, None))

    # If cost_fn exposes a per-(state, action) stage cost, use it directly.
    # This captures weight_info, info_term_fn, etc. from cost_fn's closure,
    # so MPVI doesn't need to know about info gathering at all.
    if hasattr(cost_fn, 'stage_cost_fn'):
        _stage_cost = cost_fn.stage_cost_fn

        def _sc_single(state_xy, action_scalar, cost_params):
            action = jnp.array([action_scalar])
            return _stage_cost(state_xy, action, cost_params)

        _sc_all_actions = jax.vmap(_sc_single, in_axes=(None, 0, None))
        _sc_all_cells_actions = jax.vmap(_sc_all_actions, in_axes=(0, None, None))
        use_cost_fn_stage = True
        print("  MPVI: using cost_fn.stage_cost_fn for per-(state, action) costs")
    else:
        use_cost_fn_stage = False
        print("  MPVI: no stage_cost_fn on cost_fn, using Manhattan + time fallback")

    # If cost_fn exposes a terminal cost, use it for V initialization.
    # This ensures V is scaled consistently with stage costs (e.g., weight_goal).
    if hasattr(cost_fn, 'terminal_cost_fn'):
        _terminal_cost = cost_fn.terminal_cost_fn
        _tc_all_cells = jax.vmap(_terminal_cost, in_axes=(0, None))
        use_cost_fn_terminal = True
        print("  MPVI: using cost_fn.terminal_cost_fn for V initialization")
    else:
        use_cost_fn_terminal = False
        print("  MPVI: no terminal_cost_fn on cost_fn, using raw Manhattan for V init")

    @jax.jit
    def solve(state, init_env_state, cost_params):
        dyn_params = cost_params["dyn_params"]
        goal_state = jnp.array(cost_params["goal_state"], dtype=jnp.float32)

        # Phase 1: Batch transition computation for all (cell, action) pairs
        next_states = _predict_all_cells_actions(dyn_params, all_xy, all_actions)
        next_x = jnp.clip(jnp.round(next_states[..., 0]).astype(jnp.int32), 0, grid_size - 1)
        next_y = jnp.clip(jnp.round(next_states[..., 1]).astype(jnp.int32), 0, grid_size - 1)

        # Phase 2: V initialization (terminal cost over all cells)
        if use_cost_fn_terminal:
            # Use cost_fn's terminal cost (includes weight_goal scaling)
            V_flat = _tc_all_cells(all_xy, cost_params)
            V = V_flat.reshape(grid_size, grid_size)
        else:
            # Fallback: raw Manhattan distance
            manhattan = jnp.sum(jnp.abs(all_xy - goal_state[None, :]), axis=1)
            V = manhattan.reshape(grid_size, grid_size)

        # Reshape transition indices to (grid_size, grid_size, num_actions)
        next_x_grid = next_x.reshape(grid_size, grid_size, num_actions)
        next_y_grid = next_y.reshape(grid_size, grid_size, num_actions)

        # Compute per-(state, action) stage costs
        if use_cost_fn_stage:
            # Use cost_fn's stage cost (includes info gathering if weight_info > 0)
            stage_costs_flat = _sc_all_cells_actions(all_xy, all_actions, cost_params)
            stage_costs_sa = stage_costs_flat.reshape(grid_size, grid_size, num_actions)
        else:
            # Fallback: Manhattan + time penalty (no info term)
            manhattan_fb = jnp.sum(jnp.abs(all_xy - goal_state[None, :]), axis=1)
            base_stage_costs = (manhattan_fb + weight_time).reshape(grid_size, grid_size)
            stage_costs_sa = base_stage_costs[..., None] * jnp.ones((1, 1, num_actions))

        # Phase 3: Backward DP via scan (reverse) -> (horizon, grid_size, grid_size) policy
        def backward_step(V, _):
            V_next = V[next_y_grid, next_x_grid]         # (10, 10, 4)
            Q = stage_costs_sa + V_next                    # (10, 10, 4)
            policy = jnp.argmin(Q, axis=-1)                # (10, 10)
            V_new = jnp.min(Q, axis=-1)                    # (10, 10)
            return V_new, policy

        _, all_policies = jax.lax.scan(backward_step, V, None, length=horizon, reverse=True)

        # Phase 4: Forward action extraction via scan
        def forward_step(xy, policy_t):
            x_int = jnp.clip(jnp.round(xy[0]).astype(jnp.int32), 0, grid_size - 1)
            y_int = jnp.clip(jnp.round(xy[1]).astype(jnp.int32), 0, grid_size - 1)
            action = policy_t[y_int, x_int].astype(jnp.float32)
            action_arr = jnp.array([action])
            discrete_action = jnp.round(jnp.clip(action_arr, 0.0, 3.0))
            next_xy = pred_one_step_fn(dyn_params, xy, discrete_action)
            next_xy = jnp.clip(jnp.round(next_xy), 0, grid_size - 1)
            return next_xy, action

        _, actions_seq = jax.lax.scan(forward_step, init_env_state, all_policies)
        actions = actions_seq.reshape(horizon, dim_control)

        new_state = state.replace(key=jax.random.split(state.key)[1])
        return actions, new_state

    return Planner(cost_fn=cost_fn, solve_fn=solve), initial_state


# --- A* Implementation ---


def create_astar_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
) -> tuple[Planner, PlannerState]:
    """
    Creates an A* planner for discrete gridworld navigation.

    Uses the learned dynamics model to predict transitions and searches
    for an optimal path using Manhattan distance heuristic.

    Uses jax.pure_callback to allow Python control flow within JIT-compiled code.

    Args:
        config: Configuration dictionary
        cost_fn: Cost function (with dynamics_model attribute)
        key: JAX random key
    """
    initial_state = PlannerState(key=key)

    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon", 50)
    dim_control = planner_params.get("dim_control", 1)

    # Get dynamics model from cost function
    if not hasattr(cost_fn, 'dynamics_model'):
        raise ValueError("A* planner requires cost_fn to have dynamics_model attribute")

    dynamics_model = cost_fn.dynamics_model

    # Capture the prediction function in the closure
    pred_one_step_fn = dynamics_model.pred_one_step

    # Diagnostic counter for A* callback
    _astar_call_count = [0]
    _last_param_norm = [None]

    # Create a pure Python function that will be called via callback
    def _astar_search_callback(init_env_state_flat, goal_state_flat, dyn_params_flat):
        """Pure Python A* search using learned model (called from JIT via callback)."""
        # Diagnostic: check if params are changing between calls
        _astar_call_count[0] += 1
        param_leaves = jax.tree.leaves(dyn_params_flat["model"])
        current_norm = sum(float(jnp.linalg.norm(v)) for v in param_leaves)
        if _astar_call_count[0] <= 5 or _astar_call_count[0] % 100 == 0:
            prev = _last_param_norm[0]
            delta_str = f", delta={current_norm - prev:.8f}" if prev is not None else ""
            #print(f"[A* call #{_astar_call_count[0]}] param_norm={current_norm:.6f}{delta_str}")
        _last_param_norm[0] = current_norm

        # Convert flat arrays to positions
        start_x = int(round(float(init_env_state_flat[0])))
        start_y = int(round(float(init_env_state_flat[1])))
        goal_x = int(round(float(goal_state_flat[0])))
        goal_y = int(round(float(goal_state_flat[1])))

        start = (start_x, start_y)
        goal = (goal_x, goal_y)

        # A* search using model predictions
        path = _astar_search_model_based(
            start, goal, pred_one_step_fn, dyn_params_flat, horizon
        )

        # Convert path to actions
        if len(path) <= 0:
            print(f"A* failed: no path from {start} to {goal}")
            actions = jnp.zeros((horizon, dim_control))
        elif len(path) == 1:
            # Already at goal or only one waypoint - no movement needed
            # Use action 0 (arbitrary, agent stays at goal)
            actions = jnp.zeros((horizon, dim_control))
            if start != goal:
                pass
                #print(f"A* {start}->{goal}: already at goal")
        else:
            actions_list = []
            for i in range(len(path) - 1):
                curr = path[i]
                next_pos = path[i + 1]
                action = _position_diff_to_action(curr, next_pos)
                actions_list.append(action)

            # Pad with "stay" actions (action 0)
            while len(actions_list) < horizon:
                actions_list.append(0.0)

            actions = jnp.array(actions_list[:horizon]).reshape(horizon, dim_control)
            # Debug: print first few actions
            action_str = ', '.join([f"{int(a)}" for a in actions_list[:min(5, len(actions_list))]])
            #print(f"A* {start}->{goal}: path_len={len(path)}, first_actions=[{action_str}]")
        return actions

    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        """A* solve function using pure_callback for JIT compatibility."""
        goal_state = jnp.array(cost_params["goal_state"])
        dyn_params = cost_params["dyn_params"]

        # Use pure_callback to call Python A* from within JIT
        result_shape = jax.ShapeDtypeStruct((horizon, dim_control), jnp.float32)
        actions = jax.pure_callback(
            _astar_search_callback,
            result_shape,
            init_env_state,
            goal_state,
            dyn_params,
        )

        # Update key for consistency
        new_key = jax.random.split(state.key)[1]
        new_state = state.replace(key=new_key)

        return actions, new_state

    return Planner(cost_fn=cost_fn, solve_fn=solve_fn), initial_state


def _astar_solve_internal(
    config: Any,
    state: PlannerState,
    init_env_state: Array,
    cost_params: dict,
) -> Tuple[Array, PlannerState]:
    """
    A* search for discrete gridworld navigation.

    The dynamics model predicts continuous states, which are rounded
    to discrete grid positions for planning.

    Note: This function uses Python control flow (not JIT-compatible),
    but materializes JAX tracers to concrete values before A* search.
    """
    planner_params = config.get("planner_params", {})
    horizon = planner_params.get("horizon", 50)
    dim_control = planner_params.get("dim_control", 1)

    # Extract goal from cost_params
    goal_state = jnp.array(cost_params["goal_state"])

    # Get maze layout from config
    maze_layout = config.get("env_params", {}).get("maze_layout", None)

    # Convert JAX arrays to concrete Python values (to avoid tracing issues)
    # Use .item() to materialize the tracer into a concrete scalar
    start_x = int(round(float(init_env_state[0].item())))
    start_y = int(round(float(init_env_state[1].item())))
    goal_x = int(float(goal_state[0].item()))
    goal_y = int(float(goal_state[1].item()))

    start = (start_x, start_y)
    goal = (goal_x, goal_y)

    # A* search with maze layout
    path = _astar_search_grid(start, goal, maze_layout, horizon)

    # Convert path to actions
    if len(path) <= 1:
        # No path found or already at goal
        actions = jnp.zeros((horizon, dim_control))
    else:
        actions_list = []
        for i in range(len(path) - 1):
            curr = path[i]
            next_pos = path[i + 1]
            action = _position_diff_to_action(curr, next_pos)
            actions_list.append(action)

        # Pad with "stay" actions (action 0) if path is shorter than horizon
        while len(actions_list) < horizon:
            actions_list.append(0.0)

        actions = jnp.array(actions_list[:horizon]).reshape(horizon, dim_control)

    # Update key for randomness (even though A* is deterministic)
    new_key = jax.random.split(state.key)[1]
    new_state = state.replace(key=new_key)

    return actions, new_state


def _manhattan_distance(pos, goal):
    """Calculate Manhattan distance heuristic."""
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])


def _get_neighbors(pos, maze_layout=None):
    """
    Get valid neighboring grid cells based on maze layout.

    Uses bitmask encoding from maze_layout:
    - Bit 0 (value 1): can move up (y+1)
    - Bit 1 (value 2): can move down (y-1)
    - Bit 2 (value 4): can move left (x-1)
    - Bit 3 (value 8): can move right (x+1)

    Args:
        pos: (x, y) current position
        maze_layout: 2D array of bitmasks (None = assume all moves valid)

    Returns:
        List of valid (x, y) neighbor positions
    """
    x, y = pos
    neighbors = []

    # If no maze layout provided, assume all moves within bounds are valid
    if maze_layout is None:
        grid_size = 10
        if y + 1 < grid_size:
            neighbors.append((x, y + 1))
        if y - 1 >= 0:
            neighbors.append((x, y - 1))
        if x - 1 >= 0:
            neighbors.append((x - 1, y))
        if x + 1 < grid_size:
            neighbors.append((x + 1, y))
        return neighbors

    # Get bitmask for current cell
    if 0 <= y < len(maze_layout) and 0 <= x < len(maze_layout[0]):
        bitmask = maze_layout[y][x]
    else:
        return []  # Out of bounds

    # Check each direction using bitmask
    # Action 0: up (y+1) - bit 0
    if (bitmask & 1) and y + 1 < len(maze_layout):
        neighbors.append((x, y + 1))

    # Action 1: down (y-1) - bit 1
    if (bitmask & 2) and y - 1 >= 0:
        neighbors.append((x, y - 1))

    # Action 2: left (x-1) - bit 2
    if (bitmask & 4) and x - 1 >= 0:
        neighbors.append((x - 1, y))

    # Action 3: right (x+1) - bit 3
    if (bitmask & 8) and x + 1 < len(maze_layout[0]):
        neighbors.append((x + 1, y))

    return neighbors


def _position_diff_to_action(curr, next_pos):
    """Convert position difference to action index."""
    dx = next_pos[0] - curr[0]
    dy = next_pos[1] - curr[1]

    if dy == 1:  # up
        return 0.0
    elif dy == -1:  # down
        return 1.0
    elif dx == -1:  # left
        return 2.0
    elif dx == 1:  # right
        return 3.0
    else:  # no movement
        return 0.0


def _astar_search_grid(start, goal, maze_layout=None, max_steps=50):
    """
    A* search on a discrete grid with obstacle awareness.

    Uses the maze layout bitmasks to determine valid transitions.
    If no maze layout is provided, assumes all cells are navigable.

    Args:
        start: (x, y) starting position
        goal: (x, y) goal position
        maze_layout: 2D array of bitmasks indicating valid moves
        max_steps: maximum path length

    Returns:
        List of (x, y) positions from start to goal
    """
    import heapq

    # Priority queue: (f_score, g_score, position, path)
    open_set = [(0 + _manhattan_distance(start, goal), 0, start, [start])]
    closed_set = set()

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)

        # Check if reached goal
        if current == goal:
            return path

        # Check if path is too long
        if g_score >= max_steps:
            continue

        # Skip if already visited
        if current in closed_set:
            continue

        closed_set.add(current)

        # Explore neighbors (respects maze obstacles)
        for neighbor in _get_neighbors(current, maze_layout):
            if neighbor in closed_set:
                continue

            new_g_score = g_score + 1  # Cost to reach neighbor
            new_h_score = _manhattan_distance(neighbor, goal)
            new_f_score = new_g_score + new_h_score
            new_path = path + [neighbor]

            heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))

    # No path found, return empty path
    return []


def _astar_search_model_based(start, goal, pred_one_step_fn, dyn_params, max_steps=50):
    """
    A* search using learned dynamics model to determine valid transitions.

    For each potential action, queries the model to see if movement is possible.
    If predicted next state ≈ current state, the action is blocked (wall).

    Args:
        start: (x, y) starting position
        goal: (x, y) goal position
        pred_one_step_fn: Function (params, state, action) -> next_state
        dyn_params: Model parameters
        max_steps: Maximum path length

    Returns:
        List of (x, y) positions from start to goal
    """
    import heapq

    # Priority queue: (f_score, g_score, position, path)
    open_set = [(0 + _manhattan_distance(start, goal), 0, start, [start])]
    closed_set = set()

    while open_set:
        f_score, g_score, current, path = heapq.heappop(open_set)

        # Check if reached goal
        if current == goal:
            return path

        # Check if path is too long
        if g_score >= max_steps:
            continue

        # Skip if already visited
        if current in closed_set:
            continue

        closed_set.add(current)

        # Explore neighbors using model predictions
        for neighbor in _get_neighbors_model_based(current, pred_one_step_fn, dyn_params):
            if neighbor in closed_set:
                continue

            new_g_score = g_score + 1
            new_h_score = _manhattan_distance(neighbor, goal)
            new_f_score = new_g_score + new_h_score
            new_path = path + [neighbor]

            heapq.heappush(open_set, (new_f_score, new_g_score, neighbor, new_path))

    # No path found, return start only
    return [start]


def _get_neighbors_model_based(pos, pred_one_step_fn, dyn_params):
    """
    Get valid neighboring cells using learned model predictions.

    For each action {0,1,2,3} (up, down, left, right):
    - Query model: next_state = pred_one_step(params, current_state, action)
    - If next_state ≈ current_state → blocked (wall)
    - If next_state ≠ current_state → valid neighbor

    Args:
        pos: (x, y) current position
        pred_one_step_fn: Model prediction function
        dyn_params: Model parameters

    Returns:
        List of valid (x, y) neighbor positions
    """
    x, y = pos
    current_state = jnp.array([float(x), float(y)])
    neighbors = []

    # Test each action: 0=up, 1=down, 2=left, 3=right
    action_deltas = {
        0: (0, 1),   # up: y+1
        1: (0, -1),  # down: y-1
        2: (-1, 0),  # left: x-1
        3: (1, 0),   # right: x+1
    }

    # Debug: print model predictions for each action
    debug_preds = []
    for action_idx, (dx, dy) in action_deltas.items():
        action = jnp.array([float(action_idx)])

        # Query model
        predicted_next_state = pred_one_step_fn(dyn_params, current_state, action)

        # Round to grid coordinates
        pred_x = int(round(float(predicted_next_state[0])))
        pred_y = int(round(float(predicted_next_state[1])))

        debug_preds.append(f"a{action_idx}->({pred_x},{pred_y})")

        # If prediction differs from current position, it's a valid move
        # (model learned this action causes movement)
        if (pred_x, pred_y) != (x, y):
            neighbors.append((pred_x, pred_y))

    # Occasionally print debug info (first few cells only)
    if (x + y) < 3 and len(neighbors) > 0:
        pass
        #print(f"  Model @ ({x},{y}): {', '.join(debug_preds)} -> {len(neighbors)} valid neighbors")

    return neighbors

    return solve


def create_icem_planner(
    config: Any, cost_fn: CostFn, key: jax.Array
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
    _solve = _create_icem_solve(config, cost_fn)

    def solve_fn(
        state: PlannerState,
        init_env_state: Array,
        cost_params: dict,
    ) -> Tuple[Array, PlannerState]:
        """Solve function for iCEM."""
        return _solve(state, init_env_state, cost_params)

    return Planner(cost_fn=cost_fn, solve_fn=solve_fn), initial_state