from jax import numpy as jnp
from jax import jit, grad, jacfwd
from jax.lax import scan
from functools import partial
from jax.numpy import sqrt, newaxis
from jax.numpy.fft import irfft, rfftfreq

# disable jitting
import jax

# jax.config.update("jax_disable_jit", True)


class LQR:
    def __init__(self, cost, dynamics):
        self.Q = cost.Q  # (horizon+1, dim_state, dim_state)
        self.R = cost.R  # (horizon, dim_control, dim_control)
        self.A = dynamics.A  # (horizon, dim_state, dim_state)
        self.B = dynamics.B  # (horizon, dim_state, dim_control)

    @partial(jit, static_argnums=0)
    def solve(self, init_state, init_controls):
        horizon = init_controls.shape[0]

        # Backward pass
        def scan_step(S_t, idx):
            Q_tm1 = self.Q[idx - 1]
            R_tm1 = self.R[idx - 1]
            A_tm1 = self.A[idx - 1]
            B_tm1 = self.B[idx - 1]

            B_tm1_S_t = B_tm1.T @ S_t @ B_tm1
            K_tm1 = -jnp.linalg.solve(R_tm1 + B_tm1_S_t, B_tm1.T @ S_t @ A_tm1)
            A_B_K_tm1 = A_tm1 + B_tm1 @ K_tm1
            S_tm1 = (
                Q_tm1 + K_tm1.T @ R_tm1 @ K_tm1 + A_B_K_tm1.T @ S_t @ A_B_K_tm1
            )

            return S_tm1, K_tm1

        rev_idxs = jnp.arange(horizon, 0, -1)
        _, K_rev = scan(scan_step, self.Q[horizon], rev_idxs)

        Ks = jnp.flip(K_rev, axis=0)

        # Forward pass
        # Return state and controls
        def forward_step(state_t, idx):
            K_t = Ks[idx]
            control_t = K_t @ state_t
            state_tp1 = self.A[idx] @ state_t + self.B[idx] @ control_t
            return state_tp1, (state_tp1, control_t)

        _, (states_minus_init, controls) = scan(
            forward_step, init_state, jnp.arange(horizon)
        )
        states = jnp.vstack([init_state, states_minus_init])

        success_flag = True  # TODO: Add checks for success
        stats = {"Ks": Ks}  # TODO : Add more statistics if needed

        return (states, controls), (success_flag, stats)


class iLQR:
    def __init__(
        self,
        cost,
        dynamics,
        horizon,
        dims,
        rel_cost_decrease_threshold=1e-2,
        feedforward_norm_threshold=1.0,
        max_iterations=50,
        alpha_init=1.0,
        alpha_min=1e-3,
        eta=0.2,
        beta=0.5,
        reg_param_init=1e-6,
        reg_param_scaling=10.0,
        reg_param_max=1e-1,
    ):
        self.cost = cost["stage"]
        self.terminal_cost = cost["terminal"]
        self.total_cost = cost["traj"]
        self.dynamics = dynamics
        self.horizon = horizon
        self.dim_state = dims["state"]
        self.dim_control = dims["control"]

        # Iteration parameters
        self.rel_cost_decrease_threshold = rel_cost_decrease_threshold
        self.feedforward_norm_threshold = feedforward_norm_threshold
        self.max_iterations = max_iterations

        # Line-search parameters
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.eta = eta
        self.beta = beta

        # Numerical params
        self.reg_param_init = reg_param_init
        self.reg_param_scaling = reg_param_scaling
        self.reg_param_max = reg_param_max

        # Approximations
        self.q_terminal = jit(grad(self.terminal_cost, argnums=0))
        self.Q_terminal = jit(jacfwd(self.q_terminal, argnums=0))

        q_fn = jax.grad(self.cost, argnums=0)
        r_fn = jax.grad(self.cost, argnums=1)
        jac_dyn_x = jax.jacfwd(self.dynamics, argnums=0)
        jac_dyn_u = jax.jacfwd(self.dynamics, argnums=1)
        jac_q_x = jax.jacfwd(q_fn, argnums=0)
        jac_r = jax.jacfwd(r_fn, argnums=1)
        jac_q_u = jax.jacfwd(q_fn, argnums=1)

        def _all_derivs(x, u):
            A = jac_dyn_x(x, u)
            B = jac_dyn_u(x, u)
            q = q_fn(x, u)
            r = r_fn(x, u)
            Q = jac_q_x(x, u)
            R = jac_r(x, u)
            H = jac_q_u(x, u)
            return A, B, Q, R, H, q, r

        self.all_derivs = jit(jax.vmap(_all_derivs, in_axes=(0, 0)))

        # Rollout functions
        def rollout(init_state, controls_nom):
            def step(init_state, idx):
                control_t = controls_nom[idx]
                next_state = self.dynamics(init_state, control_t)
                return next_state, (next_state, control_t)

            _, (states_minus_init, controls) = scan(
                step, init_state, jnp.arange(self.horizon)
            )

            return (
                jnp.vstack([init_state, states_minus_init]),
                controls,
            )  # TODO: check if vstack is efficient

        def rollout_with_policy(states_nom, controls_nom, Ks, ds):
            def step_with_policy(state_t, idx):
                K_t = Ks[idx]
                d_t = ds[idx]
                control_t_nom = controls_nom[idx]
                state_t_nom = states_nom[idx]
                control_t = control_t_nom + K_t @ (state_t - state_t_nom) + d_t
                state_tp1 = self.dynamics(state_t, control_t)
                return state_tp1, (state_tp1, control_t)

            init_state = states_nom[0]
            _, (states_minus_init, controls) = scan(
                step_with_policy, init_state, jnp.arange(self.horizon)
            )
            states = jnp.vstack(
                [init_state, states_minus_init]
            )  # TODO: check if vstack is efficient

            return states, controls

        self.rollout = jit(rollout)
        self.rollout_with_policy = jit(rollout_with_policy)

        # Line search function
        # TODO Don't carry trajectory
        def line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
        ):
            def cond_fun(carry):
                (alpha, _, _, _, pass_test) = carry
                return (alpha > self.alpha_min) & (~pass_test)

            def body_fun(carry):
                (alpha, _, _, _, _) = carry
                states, controls = self.rollout_with_policy(
                    states_nom, controls_nom, Ks, ds * alpha
                )
                cost = self.total_cost(states, controls)
                pass_test = (
                    cost <= cost_0 - self.eta * alpha * pred_cost_decrease
                )
                alpha_next = jax.lax.select(
                    pass_test, alpha, alpha * self.beta
                )
                return (alpha_next, states, controls, cost, pass_test)

            return jax.lax.while_loop(
                cond_fun,
                body_fun,
                (self.alpha_init, states_nom, controls_nom, 0.0, False),
            )

        self.line_search = jit(line_search)

    @partial(jit, static_argnums=0)
    def convergence_conditions(self, carry):
        _, it, _, _, _, _, stats = carry
        (
            alpha,
            cost_0,
            cost,
            cost_decr,
            pred_cost_decrease,
            norm_ds,
            cond_R_BSB,
            line_search_ok,
        ) = stats[it - 1]

        # Exit if actual drop is approx zero and
        cost_drop = cost_0 - cost
        cost_small_and_norm_small = (jnp.abs(cost_drop) < 1e-10) & (
            norm_ds < self.feedforward_norm_threshold
        )

        # Relative reduction less than threshold but greater than zero
        rel_drop = (cost_0 - cost) / cost_0
        is_drop_acceptable = (rel_drop > 0.0) & (
            rel_drop < self.rel_cost_decrease_threshold
        )

        # Norm of the feedforward term is below threshold
        is_norm_acceptable = norm_ds < self.feedforward_norm_threshold

        return (
            is_drop_acceptable & is_norm_acceptable
        ) | cost_small_and_norm_small

    @partial(jax.jit, static_argnums=0)
    def continue_iteration(self, carry):
        _, it, _, _, _, _, stats = carry

        converged = jax.lax.cond(
            it > 0,
            lambda x: self.convergence_conditions(x),
            lambda _: False,  # Always false for it=0
            carry,
        )

        # for it>0 grab ls_ok, otherwise default to True
        ls_ok = jax.lax.cond(
            it > 0,
            lambda _: stats[it - 1, 7] > 0.5,
            lambda _: True,
            None,
        )

        iter_limit_reached = it >= self.max_iterations

        # continue only if not yet converged AND the last line-search was OK
        keep_going = (~converged) & ls_ok & (~iter_limit_reached)

        return keep_going

    @partial(jit, static_argnums=0)
    def scan_step(self, carry, approximations):
        reg_param, S_t, s_t, total_cost_decr, _ = carry
        A_tm1, B_tm1, Q_tm1, R_tm1, H_tm1, q_tm1, r_tm1 = approximations

        # Policy
        R_BSB_tm1 = (
            R_tm1 + B_tm1.T @ S_t @ B_tm1 + jnp.eye(R_tm1.shape[0]) * reg_param
        )
        #
        K_tm1 = -jnp.linalg.solve(R_BSB_tm1, H_tm1.T + B_tm1.T @ S_t @ A_tm1)
        d_tm1 = -jnp.linalg.solve(R_BSB_tm1, B_tm1.T @ s_t + r_tm1)

        # Value function
        Z_tm1 = A_tm1 + B_tm1 @ K_tm1
        S_tm1 = (
            Q_tm1
            + K_tm1.T @ R_tm1 @ K_tm1
            + H_tm1 @ K_tm1
            + K_tm1.T @ H_tm1.T
            + Z_tm1.T @ S_t @ Z_tm1
        )
        s_tm1 = (
            q_tm1
            + K_tm1.T @ r_tm1
            + (K_tm1.T @ R_tm1.T + H_tm1) @ d_tm1
            + Z_tm1.T @ (S_t.T @ B_tm1 @ d_tm1 + s_t)
        )

        # Predicted cost decrease
        cost_decrease = 0.5 * d_tm1.T @ R_BSB_tm1 @ d_tm1
        total_cost_decr += cost_decrease

        return (
            (reg_param, S_tm1, s_tm1, total_cost_decr, R_BSB_tm1),
            (K_tm1, d_tm1),
        )

    @partial(jit, static_argnums=0)
    def ilqr_step(self, carry):
        reg_param, it, states_nom, controls_nom, _, _, stats = carry
        # Backward pass
        approximations = self.all_derivs(states_nom[:-1], controls_nom)
        (_, _, _, pred_cost_decrease, R_BSB_tm1), (Ks, ds) = scan(
            self.scan_step,
            (
                reg_param,
                self.Q_terminal(states_nom[-1]),
                self.q_terminal(states_nom[-1]),
                0.0,
                jnp.zeros((self.dim_control, self.dim_control)),
            ),
            approximations,
            reverse=True,
        )

        # Line-search
        # TODO: only if not converged
        cost_0 = self.total_cost(states_nom, controls_nom)
        alpha, states, controls, cost, ls_ok = self.line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
        )

        stats = stats.at[it].set(
            [
                alpha,
                cost_0,
                cost,
                cost_0 - cost,
                pred_cost_decrease,
                jnp.linalg.norm(ds),
                jnp.linalg.cond(R_BSB_tm1),
                ls_ok,
            ]
        )

        return reg_param, it + 1, states, controls, Ks, ds, stats

    @partial(jit, static_argnums=0)
    def solve_body(
        self,
        reg_param,
        init_state,
        init_controls,
    ):
        # TODO: Return feedback terms as well (Ks, ds)

        # Nominal trajectory
        states_nom, controls_nom = self.rollout(init_state, init_controls)

        # Run iLQR
        num_stats = 8
        carry_0 = (
            reg_param,
            0,
            states_nom,
            controls_nom,
            jnp.zeros((self.horizon, self.dim_control, self.dim_state)),  # Ks
            jnp.zeros((self.horizon, self.dim_control)),  # ds
            jnp.zeros((self.max_iterations, num_stats)),
        )
        carry_final = jax.lax.while_loop(
            self.continue_iteration,
            self.ilqr_step,
            carry_0,
        )

        converged = self.convergence_conditions(carry_final)

        return (carry_final[2], carry_final[3]), (
            converged,
            carry_final[6],
        )

    def solve(self, init_state, init_controls):
        reg_param = self.reg_param_init
        while reg_param < self.reg_param_max:
            print(f"Trying reg_param={reg_param:.2e}")
            (states, controls), (success_flag, stats) = self.solve_body(
                reg_param, init_state, init_controls
            )
            if success_flag:
                self.print_stats(stats)
                return (states, controls), (success_flag, stats)
            else:
                self.print_stats(stats)
                reg_param *= self.reg_param_scaling

        return (states, controls), (False, stats)

    def print_stats(
        self,
        stats,
        headers=["α", "J0", "J1", "ΔJ", "predΔJ", "‖d‖", "cond"],
        num_fmt=".2e",
    ):
        # 1) pull into Python
        stats = stats.tolist()

        # 2) find last “ok?”==True row (ok flag is assumed to be the last element of each row)
        last_ok = 0
        for i, row in enumerate(stats):
            if row[-1]:
                last_ok = i
        stats = stats[: last_ok + 2]

        # 3) drop the ok flag column from each row
        stats = [row[:-1] for row in stats]

        # 4) prepend iteration index
        indexed = [[i + 1] + row for i, row in enumerate(stats)]
        headers = ["it"] + headers

        # 5) stringify each cell, but format `it` as integer
        str_rows = []
        for row in indexed:
            srow = []
            for j, v in enumerate(row):
                if j == 0:
                    srow.append(f"{int(v)}")
                else:
                    srow.append(f"{v:{num_fmt}}")
            str_rows.append(srow)

        # 6) compute column widths
        cols = list(zip(headers, *str_rows))
        widths = [max(len(item) for item in col) for col in cols]

        # 7) build format and divider
        row_fmt = " | ".join(f"{{:>{w}}}" for w in widths)
        sep = "-" * (sum(widths) + 3 * (len(widths) - 1))

        # 8) print it all
        print(row_fmt.format(*headers))
        print(sep)
        for r in str_rows:
            print(row_fmt.format(*r))


class iLQRAdaptive:
    """iLQR with variable dynamics parameters"""

    def __init__(
        self,
        cost,
        dynamics,
        horizon,
        dims,
        rel_cost_decrease_threshold=1e-2,
        feedforward_norm_threshold=1.0,
        max_iterations=50,
        alpha_init=1.0,
        alpha_min=1e-3,
        eta=0.2,
        beta=0.5,
        reg_param_init=1e-6,
        reg_param_scaling=10.0,
        reg_param_max=1e-1,
    ):
        self.cost = cost["stage"]
        self.terminal_cost = cost["terminal"]
        self.total_cost = cost["traj"]
        self.dynamics = dynamics
        self.horizon = horizon
        self.dim_state = dims["state"]
        self.dim_control = dims["control"]

        # Iteration parameters
        self.rel_cost_decrease_threshold = rel_cost_decrease_threshold
        self.feedforward_norm_threshold = feedforward_norm_threshold
        self.max_iterations = max_iterations

        # Line-search parameters
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.eta = eta
        self.beta = beta

        # Numerical params
        self.reg_param_init = reg_param_init
        self.reg_param_scaling = reg_param_scaling
        self.reg_param_max = reg_param_max

        # Approximations
        self.q_terminal = jit(grad(self.terminal_cost, argnums=0))
        self.Q_terminal = jit(jacfwd(self.q_terminal, argnums=0))

        q_fn = jax.grad(self.cost, argnums=0)
        r_fn = jax.grad(self.cost, argnums=1)
        jac_dyn_x = jax.jacfwd(self.dynamics, argnums=0)
        jac_dyn_u = jax.jacfwd(self.dynamics, argnums=1)
        jac_q_x = jax.jacfwd(q_fn, argnums=0)
        jac_r = jax.jacfwd(r_fn, argnums=1)
        jac_q_u = jax.jacfwd(q_fn, argnums=1)

        def _all_derivs(x, u, dynamics_params):
            A = jac_dyn_x(x, u, dynamics_params)
            B = jac_dyn_u(x, u, dynamics_params)
            q = q_fn(x, u)
            r = r_fn(x, u)
            Q = jac_q_x(x, u)
            R = jac_r(x, u)
            H = jac_q_u(x, u)
            return A, B, Q, R, H, q, r

        self.all_derivs = jit(jax.vmap(_all_derivs, in_axes=(0, 0, None)))

        # Rollout functions
        def rollout(init_state, controls_nom, dynamics_params):
            def step(init_state, idx):
                control_t = controls_nom[idx]
                next_state = self.dynamics(
                    init_state, control_t, dynamics_params
                )
                return next_state, (next_state, control_t)

            _, (states_minus_init, controls) = scan(
                step, init_state, jnp.arange(self.horizon)
            )

            return (
                jnp.vstack([init_state, states_minus_init]),
                controls,
            )  # TODO: check if vstack is efficient

        def rollout_with_policy(
            states_nom, controls_nom, Ks, ds, dynamics_params
        ):
            def step_with_policy(state_t, idx):
                K_t = Ks[idx]
                d_t = ds[idx]
                control_t_nom = controls_nom[idx]
                state_t_nom = states_nom[idx]
                control_t = control_t_nom + K_t @ (state_t - state_t_nom) + d_t
                state_tp1 = self.dynamics(state_t, control_t, dynamics_params)
                return state_tp1, (state_tp1, control_t)

            init_state = states_nom[0]
            _, (states_minus_init, controls) = scan(
                step_with_policy, init_state, jnp.arange(self.horizon)
            )
            states = jnp.vstack(
                [init_state, states_minus_init]
            )  # TODO: check if vstack is efficient

            return states, controls

        self.rollout = jit(rollout)
        self.rollout_with_policy = jit(rollout_with_policy)

        # Line search function
        # TODO Don't carry trajectory
        def line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
            dynamics_params,
        ):
            def cond_fun(carry):
                (alpha, _, _, _, pass_test) = carry
                return (alpha > self.alpha_min) & (~pass_test)

            def body_fun(carry):
                (alpha, _, _, _, _) = carry
                states, controls = self.rollout_with_policy(
                    states_nom, controls_nom, Ks, ds * alpha, dynamics_params
                )
                cost = self.total_cost(states, controls)
                pass_test = (
                    cost <= cost_0 - self.eta * alpha * pred_cost_decrease
                )
                alpha_next = jax.lax.select(
                    pass_test, alpha, alpha * self.beta
                )
                return (alpha_next, states, controls, cost, pass_test)

            return jax.lax.while_loop(
                cond_fun,
                body_fun,
                (self.alpha_init, states_nom, controls_nom, 0.0, False),
            )

        self.line_search = jit(line_search)

    @partial(jit, static_argnums=0)
    def convergence_conditions(self, carry):
        _, it, _, _, _, _, stats, _ = carry
        (
            alpha,
            cost_0,
            cost,
            cost_decr,
            pred_cost_decrease,
            norm_ds,
            cond_R_BSB,
            line_search_ok,
        ) = stats[it - 1]

        # Exit if actual drop is approx zero and
        cost_drop = cost_0 - cost
        cost_small_and_norm_small = (jnp.abs(cost_drop) < 1e-10) & (
            norm_ds < self.feedforward_norm_threshold
        )

        # Relative reduction less than threshold but greater than zero
        rel_drop = (cost_0 - cost) / cost_0
        is_drop_acceptable = (rel_drop > 0.0) & (
            rel_drop < self.rel_cost_decrease_threshold
        )

        # Norm of the feedforward term is below threshold
        is_norm_acceptable = norm_ds < self.feedforward_norm_threshold

        return (
            is_drop_acceptable & is_norm_acceptable
        ) | cost_small_and_norm_small

    @partial(jax.jit, static_argnums=0)
    def continue_iteration(self, carry):
        _, it, _, _, _, _, stats, _ = carry

        converged = jax.lax.cond(
            it > 0,
            lambda x: self.convergence_conditions(x),
            lambda _: False,  # Always false for it=0
            carry,
        )

        # for it>0 grab ls_ok, otherwise default to True
        ls_ok = jax.lax.cond(
            it > 0,
            lambda _: stats[it - 1, 7] > 0.5,
            lambda _: True,
            None,
        )

        iter_limit_reached = it >= self.max_iterations

        # continue only if not yet converged AND the last line-search was OK
        keep_going = (~converged) & ls_ok & (~iter_limit_reached)

        return keep_going

    @partial(jit, static_argnums=0)
    def scan_step(self, carry, approximations):
        reg_param, S_t, s_t, total_cost_decr, _ = carry
        A_tm1, B_tm1, Q_tm1, R_tm1, H_tm1, q_tm1, r_tm1 = approximations

        # Policy
        R_BSB_tm1 = (
            R_tm1 + B_tm1.T @ S_t @ B_tm1 + jnp.eye(R_tm1.shape[0]) * reg_param
        )
        #
        K_tm1 = -jnp.linalg.solve(R_BSB_tm1, H_tm1.T + B_tm1.T @ S_t @ A_tm1)
        d_tm1 = -jnp.linalg.solve(R_BSB_tm1, B_tm1.T @ s_t + r_tm1)

        # Value function
        Z_tm1 = A_tm1 + B_tm1 @ K_tm1
        S_tm1 = (
            Q_tm1
            + K_tm1.T @ R_tm1 @ K_tm1
            + H_tm1 @ K_tm1
            + K_tm1.T @ H_tm1.T
            + Z_tm1.T @ S_t @ Z_tm1
        )
        s_tm1 = (
            q_tm1
            + K_tm1.T @ r_tm1
            + (K_tm1.T @ R_tm1.T + H_tm1) @ d_tm1
            + Z_tm1.T @ (S_t.T @ B_tm1 @ d_tm1 + s_t)
        )

        # Predicted cost decrease
        cost_decrease = 0.5 * d_tm1.T @ R_BSB_tm1 @ d_tm1
        total_cost_decr += cost_decrease

        return (
            (reg_param, S_tm1, s_tm1, total_cost_decr, R_BSB_tm1),
            (K_tm1, d_tm1),
        )

    @partial(jit, static_argnums=0)
    def ilqr_step(self, carry):
        (
            reg_param,
            it,
            states_nom,
            controls_nom,
            _,
            _,
            stats,
            dynamics_params,
        ) = carry
        # Backward pass
        approximations = self.all_derivs(
            states_nom[:-1], controls_nom, dynamics_params
        )
        (_, _, _, pred_cost_decrease, R_BSB_tm1), (Ks, ds) = scan(
            self.scan_step,
            (
                reg_param,
                self.Q_terminal(states_nom[-1]),
                self.q_terminal(states_nom[-1]),
                0.0,
                jnp.zeros((self.dim_control, self.dim_control)),
            ),
            approximations,
            reverse=True,
        )

        # Line-search
        # TODO: only if not converged
        cost_0 = self.total_cost(states_nom, controls_nom)
        alpha, states, controls, cost, ls_ok = self.line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
            dynamics_params,
        )

        stats = stats.at[it].set(
            [
                alpha,
                cost_0,
                cost,
                cost_0 - cost,
                pred_cost_decrease,
                jnp.linalg.norm(ds),
                jnp.linalg.cond(R_BSB_tm1),
                ls_ok,
            ]
        )

        return (
            reg_param,
            it + 1,
            states,
            controls,
            Ks,
            ds,
            stats,
            dynamics_params,
        )

    @partial(jit, static_argnums=0)
    def solve_body(
        self, reg_param, init_state, init_controls, dynamics_params
    ):
        # TODO: Return feedback terms as well (Ks, ds)

        # Nominal trajectory
        states_nom, controls_nom = self.rollout(
            init_state, init_controls, dynamics_params
        )

        # Run iLQR
        num_stats = 8
        carry_0 = (
            reg_param,
            0,
            states_nom,
            controls_nom,
            jnp.zeros((self.horizon, self.dim_control, self.dim_state)),  # Ks
            jnp.zeros((self.horizon, self.dim_control)),  # ds
            jnp.zeros((self.max_iterations, num_stats)),
            dynamics_params,
        )
        carry_final = jax.lax.while_loop(
            self.continue_iteration,
            self.ilqr_step,
            carry_0,
        )

        converged = self.convergence_conditions(carry_final)

        return (carry_final[2], carry_final[3]), (
            converged,
            carry_final[6],
        )

    def solve(self, init_state, init_controls, dynamics_params, verbose=False):
        reg_param = self.reg_param_init
        while reg_param < self.reg_param_max:
            print(f"Trying reg_param={reg_param:.2e}")
            (states, controls), (success_flag, stats) = self.solve_body(
                reg_param, init_state, init_controls, dynamics_params
            )
            if success_flag:
                self.print_stats(stats) if verbose else None
                return (states, controls), (success_flag, stats)
            else:
                self.print_stats(stats) if verbose else None
                reg_param *= self.reg_param_scaling

        return (states, controls), (False, stats)

    def print_stats(
        self,
        stats,
        headers=["α", "J0", "J1", "ΔJ", "predΔJ", "‖d‖", "cond"],
        num_fmt=".2e",
    ):
        # 1) pull into Python
        stats = stats.tolist()

        # 2) find last “ok?”==True row (ok flag is assumed to be the last element of each row)
        last_ok = 0
        for i, row in enumerate(stats):
            if row[-1]:
                last_ok = i
        stats = stats[: last_ok + 2]

        # 3) drop the ok flag column from each row
        stats = [row[:-1] for row in stats]

        # 4) prepend iteration index
        indexed = [[i + 1] + row for i, row in enumerate(stats)]
        headers = ["it"] + headers

        # 5) stringify each cell, but format `it` as integer
        str_rows = []
        for row in indexed:
            srow = []
            for j, v in enumerate(row):
                if j == 0:
                    srow.append(f"{int(v)}")
                else:
                    srow.append(f"{v:{num_fmt}}")
            str_rows.append(srow)

        # 6) compute column widths
        cols = list(zip(headers, *str_rows))
        widths = [max(len(item) for item in col) for col in cols]

        # 7) build format and divider
        row_fmt = " | ".join(f"{{:>{w}}}" for w in widths)
        sep = "-" * (sum(widths) + 3 * (len(widths) - 1))

        # 8) print it all
        print(row_fmt.format(*headers))
        print(sep)
        for r in str_rows:
            print(row_fmt.format(*r))


class iLQRAdaptiveAugmented:
    """iLQR with variable dynamics and an augmented lagrangian formulation"""

    def __init__(
        self,
        cost,
        dynamics,
        constraints,
        horizon,
        dims,
        rel_cost_decrease_threshold=1e-2,
        feedforward_norm_threshold=1.0,
        max_iterations=50,
        alpha_init=1.0,
        alpha_min=1e-3,
        eta=0.2,
        beta=0.5,
        reg_param_init=1e-6,
        reg_param_scaling=10.0,
        reg_param_max=1e-1,
        constraint_violation_threshold=1e-3,
        dual_convergence_threshold=1e-3,
        penalty_scaling=1.2,
        max_iterations_AL=50,
    ):
        self.cost = cost["stage"]
        self.terminal_cost = cost["terminal"]
        self.total_cost = cost["traj"]
        self.dynamics = dynamics
        self.constraints = constraints
        self.horizon = horizon
        self.dim_state = dims["state"]
        self.dim_control = dims["control"]
        self.dim_constraints = dims["constraints"]

        # Iteration parameters
        self.rel_cost_decrease_threshold = rel_cost_decrease_threshold
        self.feedforward_norm_threshold = feedforward_norm_threshold
        self.max_iterations = max_iterations

        # Line-search parameters
        self.alpha_init = alpha_init
        self.alpha_min = alpha_min
        self.eta = eta
        self.beta = beta

        # Numerical params
        self.reg_param_init = reg_param_init
        self.reg_param_scaling = reg_param_scaling
        self.reg_param_max = reg_param_max

        # Augmented Lagrangian parameters
        self.constraint_violation_threshold = constraint_violation_threshold
        self.dual_convergence_threshold = dual_convergence_threshold
        self.penalty_scaling = penalty_scaling
        self.max_iterations_AL = max_iterations_AL

        # Approximations
        self.q_terminal = jit(grad(self.terminal_cost, argnums=0))
        self.Q_terminal = jit(jacfwd(self.q_terminal, argnums=0))

        q_fn = jax.grad(self.cost, argnums=0)
        r_fn = jax.grad(self.cost, argnums=1)
        jac_dyn_x = jax.jacfwd(self.dynamics, argnums=0)
        jac_dyn_u = jax.jacfwd(self.dynamics, argnums=1)
        jac_q_x = jax.jacfwd(q_fn, argnums=0)
        jac_r = jax.jacfwd(r_fn, argnums=1)
        jac_q_u = jax.jacfwd(q_fn, argnums=1)

        def _all_derivs(x, u, dynamics_params, cost_params):
            A = jac_dyn_x(x, u, dynamics_params)
            B = jac_dyn_u(x, u, dynamics_params)
            q = q_fn(x, u, cost_params)
            r = r_fn(x, u, cost_params)
            Q = jac_q_x(x, u, cost_params)
            R = jac_r(x, u, cost_params)
            H = jac_q_u(x, u, cost_params)
            return A, B, Q, R, H, q, r

        self.all_derivs = jit(
            jax.vmap(_all_derivs, in_axes=(0, 0, None, None))
        )

        # Rollout functions
        def rollout(init_state, controls_nom, dynamics_params):
            def step(init_state, idx):
                control_t = controls_nom[idx]
                next_state = self.dynamics(
                    init_state, control_t, dynamics_params
                )
                return next_state, (next_state, control_t)

            _, (states_minus_init, controls) = scan(
                step, init_state, jnp.arange(self.horizon)
            )

            return (
                jnp.vstack([init_state, states_minus_init]),
                controls,
            )  # TODO: check if vstack is efficient

        def rollout_with_policy(
            states_nom, controls_nom, Ks, ds, dynamics_params
        ):
            def step_with_policy(state_t, idx):
                K_t = Ks[idx]
                d_t = ds[idx]
                control_t_nom = controls_nom[idx]
                state_t_nom = states_nom[idx]
                control_t = control_t_nom + K_t @ (state_t - state_t_nom) + d_t
                state_tp1 = self.dynamics(state_t, control_t, dynamics_params)
                return state_tp1, (state_tp1, control_t)

            init_state = states_nom[0]
            _, (states_minus_init, controls) = scan(
                step_with_policy, init_state, jnp.arange(self.horizon)
            )
            states = jnp.vstack(
                [init_state, states_minus_init]
            )  # TODO: check if vstack is efficient

            return states, controls

        self.rollout = jit(rollout)
        self.rollout_with_policy = jit(rollout_with_policy)

        # Line search function
        # TODO Don't carry trajectory
        def line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
            dynamics_params,
            cost_params,
        ):
            def cond_fun(carry):
                (alpha, _, _, _, pass_test, _) = carry
                return (alpha > self.alpha_min) & (~pass_test)

            def body_fun(carry):
                (alpha, _, _, _, _, cost_params) = carry
                states, controls = self.rollout_with_policy(
                    states_nom, controls_nom, Ks, ds * alpha, dynamics_params
                )
                cost = self.total_cost(states, controls, cost_params)
                pass_test = (
                    cost <= cost_0 - self.eta * alpha * pred_cost_decrease
                )
                alpha_next = jax.lax.select(
                    pass_test, alpha, alpha * self.beta
                )
                return (
                    alpha_next,
                    states,
                    controls,
                    cost,
                    pass_test,
                    cost_params,
                )

            return jax.lax.while_loop(
                cond_fun,
                body_fun,
                (
                    self.alpha_init,
                    states_nom,
                    controls_nom,
                    0.0,
                    False,
                    cost_params,
                ),
            )

        self.line_search = jit(line_search)

    @partial(jit, static_argnums=0)
    def convergence_conditions(self, carry):
        _, it, _, _, _, _, stats, _, _ = carry
        (
            alpha,
            cost_0,
            cost,
            cost_decr,
            pred_cost_decrease,
            norm_ds,
            cond_R_BSB,
            line_search_ok,
        ) = stats[it - 1]

        # Exit if actual drop is approx zero and
        cost_drop = cost_0 - cost
        cost_small_and_norm_small = (jnp.abs(cost_drop) < 1e-10) & (
            norm_ds < self.feedforward_norm_threshold
        )

        # Relative reduction less than threshold but greater than zero
        rel_drop = (cost_0 - cost) / cost_0
        is_drop_acceptable = (rel_drop > 0.0) & (
            rel_drop < self.rel_cost_decrease_threshold
        )

        # Norm of the feedforward term is below threshold
        is_norm_acceptable = norm_ds < self.feedforward_norm_threshold

        return (
            is_drop_acceptable & is_norm_acceptable
        ) | cost_small_and_norm_small

    @partial(jax.jit, static_argnums=0)
    def continue_iteration(self, carry):
        _, it, _, _, _, _, stats, _, _ = carry

        converged = jax.lax.cond(
            it > 0,
            lambda x: self.convergence_conditions(x),
            lambda _: False,  # Always false for it=0
            carry,
        )

        # for it>0 grab ls_ok, otherwise default to True
        ls_ok = jax.lax.cond(
            it > 0,
            lambda _: stats[it - 1, 7] > 0.5,
            lambda _: True,
            None,
        )

        iter_limit_reached = it >= self.max_iterations

        # continue only if not yet converged AND the last line-search was OK
        keep_going = (~converged) & ls_ok & (~iter_limit_reached)

        return keep_going

    @partial(jit, static_argnums=0)
    def scan_step(self, carry, approximations):
        reg_param, S_t, s_t, total_cost_decr, _ = carry
        A_tm1, B_tm1, Q_tm1, R_tm1, H_tm1, q_tm1, r_tm1 = approximations

        # Policy
        R_BSB_tm1 = (
            R_tm1 + B_tm1.T @ S_t @ B_tm1 + jnp.eye(R_tm1.shape[0]) * reg_param
        )
        #
        K_tm1 = -jnp.linalg.solve(R_BSB_tm1, H_tm1.T + B_tm1.T @ S_t @ A_tm1)
        d_tm1 = -jnp.linalg.solve(R_BSB_tm1, B_tm1.T @ s_t + r_tm1)

        # Value function
        Z_tm1 = A_tm1 + B_tm1 @ K_tm1
        S_tm1 = (
            Q_tm1
            + K_tm1.T @ R_tm1 @ K_tm1
            + H_tm1 @ K_tm1
            + K_tm1.T @ H_tm1.T
            + Z_tm1.T @ S_t @ Z_tm1
        )
        s_tm1 = (
            q_tm1
            + K_tm1.T @ r_tm1
            + (K_tm1.T @ R_tm1.T + H_tm1) @ d_tm1
            + Z_tm1.T @ (S_t.T @ B_tm1 @ d_tm1 + s_t)
        )

        # Predicted cost decrease
        cost_decrease = 0.5 * d_tm1.T @ R_BSB_tm1 @ d_tm1
        total_cost_decr += cost_decrease

        return (
            (reg_param, S_tm1, s_tm1, total_cost_decr, R_BSB_tm1),
            (K_tm1, d_tm1),
        )

    @partial(jit, static_argnums=0)
    def ilqr_step(self, carry):
        (
            reg_param,
            it,
            states_nom,
            controls_nom,
            _,
            _,
            stats,
            dynamics_params,
            cost_params,
        ) = carry
        # Backward pass
        approximations = self.all_derivs(
            states_nom[:-1], controls_nom, dynamics_params, cost_params
        )
        (_, _, _, pred_cost_decrease, R_BSB_tm1), (Ks, ds) = scan(
            self.scan_step,
            (
                reg_param,
                self.Q_terminal(states_nom[-1], cost_params),
                self.q_terminal(states_nom[-1], cost_params),
                0.0,
                jnp.zeros((self.dim_control, self.dim_control)),
            ),
            approximations,
            reverse=True,
        )

        # Line-search
        # TODO: only if not converged
        cost_0 = self.total_cost(states_nom, controls_nom, cost_params)
        alpha, states, controls, cost, ls_ok, cost_params = self.line_search(
            states_nom,
            controls_nom,
            Ks,
            ds,
            cost_0,
            pred_cost_decrease,
            dynamics_params,
            cost_params,
        )

        stats = stats.at[it].set(
            [
                alpha,
                cost_0,
                cost,
                cost_0 - cost,
                pred_cost_decrease,
                jnp.linalg.norm(ds),
                jnp.linalg.cond(R_BSB_tm1),
                ls_ok,
            ]
        )

        return (
            reg_param,
            it + 1,
            states,
            controls,
            Ks,
            ds,
            stats,
            dynamics_params,
            cost_params,
        )

    @partial(jit, static_argnums=0)
    def solve_body(
        self,
        reg_param,
        init_state,
        init_controls,
        dynamics_params,
        cost_params,
    ):
        # TODO: Return feedback terms as well (Ks, ds)

        # Nominal trajectory
        states_nom, controls_nom = self.rollout(
            init_state, init_controls, dynamics_params
        )

        # Run iLQR
        num_stats = 8
        carry_0 = (
            reg_param,
            0,
            states_nom,
            controls_nom,
            jnp.zeros((self.horizon, self.dim_control, self.dim_state)),  # Ks
            jnp.zeros((self.horizon, self.dim_control)),  # ds
            jnp.zeros((self.max_iterations, num_stats)),
            dynamics_params,
            cost_params,
        )
        carry_final = jax.lax.while_loop(
            self.continue_iteration,
            self.ilqr_step,
            carry_0,
        )

        converged = self.convergence_conditions(carry_final)

        Ks = carry_final[4]

        return (
            (carry_final[2], carry_final[3]),
            (
                converged,
                carry_final[6],
            ),
            Ks,
        )

    def solve_ilqr(
        self,
        init_state,
        init_controls,
        dynamics_params,
        cost_params,
        verbose=False,
    ):
        reg_param = self.reg_param_init
        while reg_param < self.reg_param_max:
            print(f"Trying reg_param={reg_param:.2e}") if verbose else None
            (states, controls), (success_flag, stats), Ks = self.solve_body(
                reg_param,
                init_state,
                init_controls,
                dynamics_params,
                cost_params,
            )
            if success_flag:
                self.print_stats(stats) if verbose else None
                return (states, controls), (success_flag, stats), Ks
            else:
                self.print_stats(stats) if verbose else None
                reg_param *= self.reg_param_scaling

        return (states, controls), (False, stats), Ks

    @partial(jit, static_argnums=0)
    def convergence_conditions_AL(
        self, cost_params_old, cost_params_new, const_viol_new
    ):
        # check dual convergence
        lagrange_multipliers_old = cost_params_old[1:]
        lagrange_multipliers_new = cost_params_new[1:]
        dual_convergence_ok = (
            jnp.linalg.norm(
                lagrange_multipliers_new - lagrange_multipliers_old
            )
            < self.dual_convergence_threshold
        )

        constraint_violation = (
            const_viol_new < self.constraint_violation_threshold
        )

        return dual_convergence_ok & constraint_violation

    @partial(jit, static_argnums=0)
    def constraint_violation(self, states, controls):
        return jnp.linalg.norm(self.constraints(states, controls))

    def solve(
        self,
        init_state,
        init_controls,
        dynamics_params,
        cost_params_init,
        verbose=False,
    ):
        # Initial solve. If good enough we can skip the AL iterations
        print("Running initial iLQR solve...")
        (states_new, controls_new), (success_flag, stats), Ks = (
            self.solve_ilqr(
                init_state,
                init_controls,
                dynamics_params,
                cost_params_init,
                verbose,
            )
        )
        const_viol_new = self.constraint_violation(states_new, controls_new)
        if const_viol_new < self.constraint_violation_threshold:
            if verbose:
                print(f"Constraint violation: {const_viol_new:.2e}")
                print(
                    "Initial iLQR solution is feasible, skipping AL iterations."
                )
            return (states_new, controls_new), (success_flag, stats), Ks
        else:
            if verbose:
                print(
                    f"Constraint violation: {const_viol_new:.2e}. "
                    "Running augmented Lagrangian iterations."
                )

        # Run augmented Lagrangian iterations
        iter_counter = 0
        cost_params_old = cost_params_init
        cost_params_new = cost_params_init
        while (
            not self.convergence_conditions_AL(
                cost_params_old, cost_params_new, const_viol_new
            )
        ) & (iter_counter < self.max_iterations_AL):
            controls_old = controls_new
            cost_params_old = cost_params_new
            const_viol_old = const_viol_new

            ((states_new, controls_new), (success_flag, stats), Ks) = (
                self.solve_ilqr(
                    init_state,
                    controls_old,
                    dynamics_params,
                    cost_params_old,
                    verbose,
                )
            )

            const_viol_new = self.constraint_violation(
                states_new, controls_new
            )
            penalty_param = cost_params_old[0]
            lagrange_multipliers = cost_params_old[
                1 : self.dim_constraints + 1
            ]

            # Update lagrange multipliers
            lagrange_multipliers += penalty_param * self.constraints(
                states_new, controls_new
            )

            # Update penalty parameters
            if (
                const_viol_new > const_viol_old * 0.8
            ):  # TODO: make this a parameter
                penalty_param *= self.penalty_scaling

            cost_params_new = jnp.concatenate(
                (
                    jnp.array([penalty_param]),
                    lagrange_multipliers,
                    cost_params_old[self.dim_constraints + 1 :],
                )
            )

            iter_counter += 1

            if verbose:
                lm_delta = jnp.linalg.norm(
                    lagrange_multipliers
                    - cost_params_old[1 : self.dim_constraints + 1]
                )
                print(
                    f"AL iter {iter_counter}: "
                    f"penalty param={penalty_param:.2e}, "
                    f"LM delta={lm_delta:.3e}, "
                    f"const. viol.={const_viol_new:.3e}"
                )

        if verbose:
            penalty_param = cost_params_new[0]
            lagrange_multipliers = cost_params_new[
                1 : self.dim_constraints + 1
            ]
            final_const_viol = self.constraint_violation(
                states_new, controls_new
            )
            print(f"AL loop done after {iter_counter} iters.")
            print(f"Final penalty param: {penalty_param:.2e}")
            print(f"Final LMs: {lagrange_multipliers}")
            print(f"Final const. viol.: {final_const_viol:.2e}")

        return (
            (states_new, controls_new),
            (success_flag, stats),
            Ks,
        )

    def print_stats(
        self,
        stats,
        headers=["α", "J0", "J1", "ΔJ", "predΔJ", "‖d‖", "cond"],
        num_fmt=".2e",
    ):
        # 1) pull into Python
        stats = stats.tolist()

        # 2) find last “ok?”==True row (ok flag is assumed to be the last element of each row)
        last_ok = 0
        for i, row in enumerate(stats):
            if row[-1]:
                last_ok = i
        stats = stats[: last_ok + 2]

        # 3) drop the ok flag column from each row
        stats = [row[:-1] for row in stats]

        # 4) prepend iteration index
        indexed = [[i + 1] + row for i, row in enumerate(stats)]
        headers = ["it"] + headers

        # 5) stringify each cell, but format `it` as integer
        str_rows = []
        for row in indexed:
            srow = []
            for j, v in enumerate(row):
                if j == 0:
                    srow.append(f"{int(v)}")
                else:
                    srow.append(f"{v:{num_fmt}}")
            str_rows.append(srow)

        # 6) compute column widths
        cols = list(zip(headers, *str_rows))
        widths = [max(len(item) for item in col) for col in cols]

        # 7) build format and divider
        row_fmt = " | ".join(f"{{:>{w}}}" for w in widths)
        sep = "-" * (sum(widths) + 3 * (len(widths) - 1))

        # 8) print it all
        print(row_fmt.format(*headers))
        print(sep)
        for r in str_rows:
            print(row_fmt.format(*r))


class CrossEntropyPlanner:
    def __init__(
        self,
        cost_fn,
        batch_size,
        elit_frac,
        learning_rate,
        max_iter,
        dims,
        horizon,
        reg_cov=1e-6,
    ):

        self.cost_fn = cost_fn
        self.batch_size = batch_size
        self.elit_frac = elit_frac
        self.num_elites = int(elit_frac * batch_size)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.dim_control = dims["control"]
        self.horizon = horizon
        self.reg_cov = reg_cov * jnp.ones(self.dim_control * self.horizon)

        def cost_w_roll(init_state, unrolled_controls, cost_params):
            controls = unrolled_controls.reshape(
                self.horizon, self.dim_control
            )
            return self.cost_fn(init_state, controls, cost_params)

        self.cost_fn_vmap = jit(
            jax.vmap(
                cost_w_roll,
                in_axes=(None, 0, None),
                out_axes=0,
            )
        )

    def step_fn(self, init_state, init_mean, var, cost_params, key):
        key, subkey = jax.random.split(key)
        sampling_cov = jnp.diag(var + self.reg_cov)
        batch = jax.random.multivariate_normal(
            subkey, init_mean, sampling_cov, shape=(self.batch_size,)
        )

        # Elites
        costs = self.cost_fn_vmap(init_state, batch, cost_params)
        _, elite_indices = jax.lax.top_k(-costs, k=self.num_elites)
        elite_samples = batch[elite_indices]

        # New mean and variance from elite samples
        elite_mean = jnp.mean(elite_samples, axis=0)
        elite_var = jnp.var(elite_samples, axis=0)

        # Update mean and variance (not covariance)
        new_mean = (
            1 - self.learning_rate
        ) * init_mean + self.learning_rate * elite_mean
        new_var = (
            1 - self.learning_rate
        ) * var + self.learning_rate * elite_var

        return new_mean, new_var, key

    @partial(jit, static_argnums=0)
    def solve(
        self,
        init_state,
        init_controls,
        init_var,
        cost_params,
        key,
    ):
        init_mean = init_controls.reshape(-1)

        def body_fn(carry, _):
            init_mean, var, key = carry
            new_mean, new_cov, key = self.step_fn(
                init_state, init_mean, var, cost_params, key
            )
            return (new_mean, new_cov, key), None

        init_carry = (init_mean, init_var, key)
        final_carry, _ = jax.lax.scan(
            body_fn, init_carry, None, length=self.max_iter
        )

        final_mean, final_var, _ = final_carry

        final_controls = final_mean.reshape(self.horizon, self.dim_control)

        return final_controls, final_var, True


class CrossEntropyPlannerOld:
    def __init__(
        self,
        cost_fn,
        batch_size,
        elit_frac,
        learning_rate,
        max_iter,
        dims,
        horizon,
        reg_cov=1e-6,
    ):

        self.cost_fn = cost_fn
        self.batch_size = batch_size
        self.elit_frac = elit_frac
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.dim_control = dims["control"]
        self.horizon = horizon
        self.reg_cov = reg_cov * jnp.eye(self.dim_control * self.horizon)

        def cost_w_roll(init_state, unrolled_controls, cost_params):
            controls = unrolled_controls.reshape(
                self.horizon, self.dim_control
            )
            return self.cost_fn(init_state, controls, cost_params)

        self.cost_fn_vmap = jit(
            jax.vmap(
                cost_w_roll,
                in_axes=(None, 0, None),
                out_axes=0,
            )
        )

    def step_fn(self, init_state, init_mean, init_cov, cost_params, key):
        # Sample batch size using the mean and covariance
        key, subkey = jax.random.split(key)
        batch = jax.random.multivariate_normal(
            subkey, init_mean, init_cov, shape=(self.batch_size,)
        )

        # Compute cost for each sample
        costs = self.cost_fn_vmap(init_state, batch, cost_params)

        # Sort costs and select elite samples
        sorted_indices = jnp.argsort(costs)
        num_elite = int(self.elit_frac * self.batch_size)
        elite_indices = sorted_indices[:num_elite]
        elite_samples = batch[elite_indices]

        # New mean and covariance from elite samples
        new_mean = jnp.mean(elite_samples, axis=0)
        new_cov = jnp.cov(elite_samples, rowvar=False) + self.reg_cov

        # Average
        new_mean = (
            1 - self.learning_rate
        ) * init_mean + self.learning_rate * new_mean
        new_cov = (
            1 - self.learning_rate
        ) * init_cov + self.learning_rate * new_cov

        return new_mean, new_cov, key

    def solve(
        self,
        init_state,
        init_controls,
        init_cov,
        cost_params,
        key,
        verbose=True,
    ):
        init_mean = init_controls.reshape(-1)

        def body_fn(carry, _):
            init_mean, init_cov, key = carry
            new_mean, new_cov, key = self.step_fn(
                init_state, init_mean, init_cov, cost_params, key
            )
            return (new_mean, new_cov, key), None

        init_carry = (init_mean, init_cov, key)
        final_carry, _ = jax.lax.scan(
            body_fn, init_carry, None, length=self.max_iter
        )

        final_mean, final_cov, _ = final_carry

        final_controls = final_mean.reshape(self.horizon, self.dim_control)

        if verbose:
            print(
                "Initial cost:",
                self.cost_fn(init_state, init_controls, cost_params),
            )
            print(
                "Final cost:",
                self.cost_fn(init_state, final_controls, cost_params),
            )

        return final_controls, final_cov, True


@partial(jit, static_argnums=(0, 1, 3))
def powerlaw_psd_gaussian(
    exponent: float, size: int, rng: jax.random.PRNGKey, fmin: float = 0
):
    """Gaussian (1/f)**beta noise generation."""
    try:
        size = list(size)
    except TypeError:
        size = [size]

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
    sigma = 2 * sqrt(jnp.sum(w**2)) / samples

    size[-1] = len(f)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    key_sr, key_si = jax.random.split(rng)
    sr = jax.random.normal(key=key_sr, shape=s_scale.shape) * s_scale
    si = jax.random.normal(key=key_si, shape=s_scale.shape) * s_scale

    si = si.at[..., 0].set(0)
    sr = sr.at[..., 0].set(sr[..., 0] * sqrt(2))

    def make_real_if_even(sr_in, si_in):
        si_out = si_in.at[..., -1].set(0)
        sr_out = sr_in.at[..., -1].set(sr_in[..., -1] * sqrt(2))
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


class ImprovedCrossEntropyPlanner:
    def __init__(
        self,
        cost_fn,
        batch_size,
        elit_frac,
        learning_rate,
        max_iter,
        dims,
        horizon,
        action_low=-1.0,
        action_high=1.0,
        colored_noise_beta=2.0,
        colored_noise_fmin=0.0,
        keep_elites_frac=0.3,
    ):
        self.cost_fn = cost_fn
        self.batch_size = batch_size
        self.elit_frac = elit_frac
        self.num_elites = int(elit_frac * batch_size)
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.dim_control = dims["control"]
        self.horizon = horizon
        self.action_low = action_low
        self.action_high = action_high
        self.colored_noise_beta = colored_noise_beta
        self.colored_noise_fmin = colored_noise_fmin
        self.keep_elites_frac = keep_elites_frac
        self.num_keep_elites = int(keep_elites_frac * self.num_elites)
        self.unrolled_dim = self.dim_control * self.horizon

        def cost_w_roll(init_state, unrolled_controls, cost_params):
            controls = unrolled_controls.reshape(
                self.horizon, self.dim_control
            )
            return self.cost_fn(init_state, controls, cost_params)

        self.cost_fn_vmap = jit(
            jax.vmap(cost_w_roll, in_axes=(None, 0, None), out_axes=0)
        )

        self.noise_fn_vmap = jit(
            jax.vmap(
                lambda k: powerlaw_psd_gaussian(
                    exponent=self.colored_noise_beta,
                    size=self.unrolled_dim,
                    rng=k,
                    fmin=self.colored_noise_fmin,
                ),
                in_axes=0,
            )
        )

        def get_pfe(pfe):
            return pfe[: self.num_keep_elites].reshape(
                self.num_keep_elites, self.unrolled_dim
            )

        self.get_pfe = jit(get_pfe)

    @partial(jit, static_argnums=0)
    def solve(
        self,
        init_state,
        init_controls,
        init_var,
        cost_params,
        key,
        prev_final_elites,
    ):
        # The mean `μ` is assumed to be pre-shifted outside the solver. (Algorithm 1, Line 8)
        init_mean = init_controls.reshape(-1)

        def body_fn(carry, i):
            mean, var, key, best_sequence, best_cost, iter_elites = carry
            key, noise_key = jax.random.split(key)

            # Sample from colored noise distribution. (Algorithm 1, Line 13)
            noise_keys = jax.random.split(noise_key, self.batch_size)
            colored_noise = self.noise_fn_vmap(noise_keys)
            samples = mean + colored_noise * jnp.sqrt(var)

            # This corresponds to "keep elites" memory. (Algorithm 1, Line 14-17)
            # Select which elites to use based on the iteration number.
            elites_to_use = jax.lax.cond(
                i == 0,
                lambda: prev_final_elites[: self.num_keep_elites],
                lambda: iter_elites[: self.num_keep_elites],
            )
            samples = samples.at[: self.num_keep_elites].set(elites_to_use)

            # Clip actions to their boundaries. (Algorithm 1, Line 13)
            samples = jnp.clip(samples, self.action_low, self.action_high)

            # Add the mean to the pool of samples in the last iteration. (Algorithm 1, Line 19)
            samples = jax.lax.cond(
                i == self.max_iter - 1,
                lambda s: s.at[-1].set(mean),
                lambda s: s,
                samples,
            )

            # Evaluate costs and find new elites.
            costs = self.cost_fn_vmap(init_state, samples, cost_params)
            _, elite_indices = jax.lax.top_k(-costs, k=self.num_elites)
            new_iter_elites = samples[elite_indices]

            # Update best sequence to enable executing the best action. (Algorithm 1, Line 24)
            current_best_cost = costs[elite_indices[0]]
            current_best_sequence = new_iter_elites[0]
            best_sequence, best_cost = jax.lax.cond(
                current_best_cost < best_cost,
                lambda: (current_best_sequence, current_best_cost),
                lambda: (best_sequence, best_cost),
            )

            # Fit a new Gaussian distribution to the elite set. (Algorithm 1, Line 22)
            elite_mean = jnp.mean(new_iter_elites, axis=0)
            elite_var = jnp.var(new_iter_elites, axis=0)
            new_mean = (
                1 - self.learning_rate
            ) * mean + self.learning_rate * elite_mean
            new_var = (
                1 - self.learning_rate
            ) * var + self.learning_rate * elite_var

            return (
                new_mean,
                new_var,
                key,
                best_sequence,
                best_cost,
                new_iter_elites,
            ), None

        init_carry = (
            init_mean,
            init_var,
            key,
            init_mean,
            jnp.inf,
            jnp.zeros((self.num_elites, self.unrolled_dim)),
        )
        final_carry, _ = jax.lax.scan(
            body_fn, init_carry, jnp.arange(self.max_iter)
        )

        _, final_var, _, final_best_sequence, _, final_elites = final_carry
        final_controls = final_best_sequence.reshape(
            self.horizon, self.dim_control
        )

        return final_controls, final_var, final_elites, True
