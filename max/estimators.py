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
