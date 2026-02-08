import jax
import jax.numpy as jnp


class EKF:
    def __init__(
        self,
        dynamics_fn,
        observation_fn,
        process_cov_fn,
        observation_cov_fn,
        jitter=1e-6,
    ):
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        self.process_cov_fn = process_cov_fn
        self.observation_cov_fn = observation_cov_fn
        self.jitter = jitter

        self._ekf_jit = jax.jit(self._ekf_fn)

    def _ekf_fn(self, mean_t, cov_t, control_t, obs_tp1):
        stats = {}

        # Propagate mean
        mean_tp1_apriori = self.dynamics_fn(mean_t, control_t)

        # Propagate covariance
        jac_dyn = jax.jacrev(self.dynamics_fn, argnums=0)(mean_t, control_t)

        cov_tp1_apriori = jac_dyn @ cov_t @ jac_dyn.T + self.process_cov_fn(
            mean_tp1_apriori
        )

        jac_obs = jax.jacrev(self.observation_fn, argnums=0)(
            mean_tp1_apriori, control_t
        )

        innovation_cov = (
            jac_obs @ cov_tp1_apriori @ jac_obs.T
            + self.observation_cov_fn(mean_tp1_apriori)
        )
        kalman_gain = (
            cov_tp1_apriori
            @ jac_obs.T
            @ jnp.linalg.inv(
                innovation_cov + self.jitter * jnp.eye(innovation_cov.shape[0])
            )
        )
        eye_cov = jnp.eye(cov_t.shape[0])
        cov_tp1 = (eye_cov - kalman_gain @ jac_obs) @ cov_tp1_apriori

        cov_tp1 = 0.5 * (cov_tp1 + cov_tp1.T)

        # Correct mean and cov if observation is available
        if obs_tp1 is None:
            return mean_tp1_apriori, cov_tp1, stats
        else:
            innovation = obs_tp1 - self.observation_fn(
                mean_tp1_apriori, control_t
            )

            mean_tp1 = mean_tp1_apriori + kalman_gain @ innovation

            return mean_tp1, cov_tp1, stats

    def estimate(self, mean_t, cov_t, control_t, obs_tp1=None):
        return self._ekf_jit(mean_t, cov_t, control_t, obs_tp1)


# TODO: Cleanup input. Can use control pytree w/ proc and meas cov
class EKFCovArgs:
    def __init__(
        self,
        dynamics_fn,
        observation_fn,
        jitter=1e-6,
    ):
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        self.jitter = jitter

        self._ekf_jit = jax.jit(self._ekf_fn)

    def _ekf_fn(
        self, mean_t, cov_t, control_t, obs_tp1, proc_cov_t, meas_cov_t
    ):
        stats = {}

        # Propagate mean
        mean_tp1_apriori = self.dynamics_fn(mean_t, control_t)

        # Propagate covariance
        jac_dyn = jax.jacrev(self.dynamics_fn, argnums=0)(mean_t, control_t)

        cov_tp1_apriori = jac_dyn @ cov_t @ jac_dyn.T + proc_cov_t

        jac_obs = jax.jacrev(self.observation_fn, argnums=0)(
            mean_tp1_apriori, control_t
        )

        innovation_cov = jac_obs @ cov_tp1_apriori @ jac_obs.T + meas_cov_t
        kalman_gain = (
            cov_tp1_apriori
            @ jac_obs.T
            @ jnp.linalg.inv(
                innovation_cov + self.jitter * jnp.eye(innovation_cov.shape[0])
            )
        )
        eye_cov = jnp.eye(cov_t.shape[0])
        cov_tp1 = (eye_cov - kalman_gain @ jac_obs) @ cov_tp1_apriori

        cov_tp1 = 0.5 * (cov_tp1 + cov_tp1.T)

        innovation = obs_tp1 - self.observation_fn(mean_tp1_apriori, control_t)

        mean_tp1 = mean_tp1_apriori + kalman_gain @ innovation

        return mean_tp1, cov_tp1, stats

    def estimate(
        self, mean_t, cov_t, control_t, obs_tp1, proc_cov_t, meas_cov_t
    ):
        return self._ekf_jit(
            mean_t, cov_t, control_t, obs_tp1, proc_cov_t, meas_cov_t
        )
class LOFI:
    def __init__(
        self,
        dynamics_fn,
        observation_fn,
        observation_cov_fn,
        jitter=1e-6,
        compute_diagnostics=False,  # Set to True for debugging, False for speed
    ):
        self.dynamics_fn = dynamics_fn
        self.observation_fn = observation_fn
        self.observation_cov_fn = observation_cov_fn
        self.jitter = jitter
        self.compute_diagnostics = compute_diagnostics

        # JIT-compile the main loop with L as static argument
        # Also make compute_diagnostics static to allow branch elimination
        self._ekf_jit = jax.jit(self._ekf_fn, static_argnums=(7, 8))

    @staticmethod
    def _vec_pinv(vec):
        """Safe element-wise inverse for vectors."""
        return jnp.where(vec != 0, 1.0 / vec, 0.0)
    
    def _add_jitter(self, arr):
        return jnp.where(arr == 0, self.jitter, arr)
    
    @staticmethod
    def _jacrev_2d(f, x, p):
        return jnp.atleast_2d(jax.jacrev(f, argnums=0)(x, p))
    
    @staticmethod
    def _fast_svd(M):
        U, S, _ = jnp.linalg.svd(M.T @ M, full_matrices=False, hermitian=True)
        U = M @ (U * LOFI._vec_pinv(jnp.sqrt(S)))
        S = jnp.sqrt(S)
        return U, S

    """
    LOFI PREDICT FUNCTION (Algorithm 2): Computes a priori estimates of model params (mean) and covariance
    Params:
        mu: the previous parameters of the model (previous mean of the gaussian)
        Upsilon: the previous diagonal component of the model's covariance matrix
        W: the previous low-rank component of the model's covariance matrix
        x: the current state
        gamma: the parameter decay factor
        q: the current process noise to avoid covariance collapse
        h: the observation function (the actual nn output)
    Returns: tuple with predicted mean, covariance components, and next state
    """
    def predict(self, mu, Upsilon, W, x, gamma, q, h):
        # Line 2 of algorithm
        mu_pred = gamma * mu

        # Line 3 of algorithm
        Upsilon_pred = self._vec_pinv(gamma**2 * self._vec_pinv(Upsilon) + q)

        # Useful computations to store for following steps
        Uinv = self._vec_pinv(Upsilon)

        # Line 4 of algorithm, sum split into temp vars for clarity
        C_t_1 = jnp.eye(W.shape[1])
        C_t_2 = q * W.T @ (Upsilon_pred[:, None] * (Uinv[:, None] * W))
        C_t = jnp.linalg.pinv(C_t_1 + C_t_2)
        C_t = self._add_jitter(C_t)

        # Line 5 of algorithm
        chol_Ct = jnp.linalg.cholesky(C_t)
        W_pred = gamma * (Upsilon_pred[:, None] * Uinv[:, None] * W) @ chol_Ct
        
        # Line 6 of algorithm
        y_hat = h(mu_pred, x)

        # Line 7 of algorithm (return)
        return mu_pred, Upsilon_pred, W_pred, y_hat
    
    """
    LOFI UPDATE FUNCTION (Algorithm 3): Updates the parameters and covariance components based on a priori estimates
    NOTE: L is a static argument for jitting purposes (used to index arrays)
    Params:
        mu_pred: the a priori estimate of nn parameters (mean of the multivariate gaussian)
        Upsilon_pred: the a priori estimate of diagonal covariance component
        W_pred: the a priori estimate of low-rank covariance component
        x: the initial state
        y: the ground-truth next state
        y_hat: the estimate of the next state
        h: the observation function (the actual nn)
        L: the number of low-rank terms kept in low-rank matrix
        h_V_fn: covariance of the observation function
    Returns: tuple with new mean and new covariance matrix components (bundled as a dict)
    """
    def update(self, mu_pred, Upsilon_pred, W_pred, x, y, y_hat, h, L, h_V_fn):

        # Line 2 of algorithm
        R_t = h_V_fn(x, mu_pred)
        # Line 3 of algorithm
        L_t = jnp.linalg.cholesky(R_t)
        # Line 4 of algorithm, modified to match original code
        A_t = jnp.linalg.lstsq(L_t, jnp.eye(y_hat.shape[0]))[0].T

        # Line 5 of algorithm
        H_t = self._jacrev_2d(h, mu_pred, x) + self.jitter

        # Line 6 of algorithm
        W_tilde = jnp.hstack([W_pred, (H_t.T @ A_t.T)])

        # Useful computations to store for following steps
        Uinv_pred = self._vec_pinv(Upsilon_pred)

        # Line 7 of algorithm
        L_tilde = W_tilde.shape[1]
        G_t = jnp.linalg.pinv(jnp.eye(L_tilde) + W_tilde.T @ (Uinv_pred[:, None] * W_tilde))

        # Line 8 of algorithm
        Ct = H_t.T @ (A_t.T @ A_t)
        # Line 9 of algorithm, sum split into temp vars for clarity
        K_t_1 = Uinv_pred[:, None] * Ct
        K_t_2 = Uinv_pred[:, None] * (W_tilde @ G_t @ W_tilde.T @ (Uinv_pred[:, None] * Ct))
        K_t = K_t_1 - K_t_2  # actual kalman gain matrix

        # Line 10 of algorithm
        mu = mu_pred + K_t @ (y - y_hat)

        # Line 11 of algorithm
        U_tilde, Lambda_tilde = self._fast_svd(W_tilde)
        # Line 12 of algorithm
        U_t = U_tilde[:, :L]
        # Line 13 of algorithm
        W_t = U_t * Lambda_tilde[:L][None, :]

        # Line 14 of algorithm
        Lambda_drop = Lambda_tilde[L:]
        U_drop = U_tilde[:, L:]
        # Line 15 of algorithm
        W_drop = U_drop * Lambda_drop[None, :]
        # Line 16 of algorithm
        # Original (inefficient): diag_add = jnp.sum(W_drop @ W_drop.T, axis=1)
        # We only need the diagonal of W_drop @ W_drop.T, which is the row-wise squared norm.
        diag_add = jnp.sum(W_drop * W_drop, axis=1)
        Upsilon = Upsilon_pred + diag_add

        # Protection against numerical issues: prevent Upsilon from collapsing
        # Stronger adaptive floor to prevent covariance collapse
        alpha_floor = 2e-6
        floor_val = jnp.maximum(self.jitter, alpha_floor * jnp.median(Upsilon))
        Upsilon = jnp.maximum(Upsilon, floor_val)

        # Line 17 of algorithm (return)
        return mu, {"Upsilon": Upsilon, "W": W_t}, H_t, K_t


    """
    LOFI MAIN LOOP (Algorithm 1, modified): Runs one iteration of the LOFI main loop algorithm.
    This function is jittable, and is used in jax.lax.scan in the LOFI trainer class.
    Params:
        mu: the previous parameters of the model (previous mean of the gaussian)
        Upsilon: the previous diagonal component of the model's covariance matrix
        W: the previous low-rank component of the model's covariance matrix
        x: the current state
        y: the ground-truth next state
        gamma: the parameter decay factor
        L: the number of low-rank terms kept in low-rank matrix
    Returns: the next mean (model params) and covariance matrix components (bundled together as a dict)
    """
    def _ekf_fn(self, mu, Upsilon, W, x, y, gamma, q, L, compute_diagnostics):
        """LOFI main loop (Algorithm 1): One step of predict + update."""
        # Predict step
        mu_pred, Upsilon_pred, W_pred, y_pred = self.predict(
            mu, Upsilon, W, x, gamma, q, self.observation_fn
        )
        
        # Update step
        mu_next, cov_next, jac, kalman_gain = self.update(
            mu_pred, Upsilon_pred, W_pred, x, y, y_pred, 
            self.observation_fn, L, self.observation_cov_fn
        )
        
        # Diagnostics (only compute if flag is True - branch eliminated by JIT when False)
        if compute_diagnostics:
            diagnostics = {
                'upsilon_min': jnp.min(Upsilon),
                'upsilon_max': jnp.max(Upsilon),
                'upsilon_mean': jnp.mean(Upsilon),
                'w_norm': jnp.linalg.norm(W),
                'mu_norm': jnp.linalg.norm(mu),
                'condition_upsilon': jnp.max(Upsilon) / (jnp.min(Upsilon) + 1e-10),
                'upsilon_pred_min': jnp.min(Upsilon_pred),
                'jacobian_norm': jnp.linalg.norm(jac),
                'kalman_gain_norm': jnp.linalg.norm(kalman_gain),
            }
        else:
            # Return empty dict to maintain consistent output structure
            diagnostics = {}
        
        return mu_next, cov_next, y_pred, diagnostics

    def estimate(self, mu, cov_t: dict, x, y, gamma, q, L, epoch=0):
        """Public interface for LOFI estimation step."""
        Upsilon = cov_t["Upsilon"]
        W = cov_t["W"]
        return self._ekf_jit(mu, Upsilon, W, x, y, gamma, q, L, self.compute_diagnostics)