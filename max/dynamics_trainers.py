# trainers.py

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Any, NamedTuple, Optional
from max.dynamics import DynamicsModel
from flax import struct
from flax.traverse_util import path_aware_map
import jax.flatten_util
from max.estimators import EKFCovArgs, EKFEfficient
from max.normalizers import STANDARD_NORMALIZER


class TrainState(struct.PyTreeNode):
    """A flexible training state that can handle both GD and EKF trainers."""

    params: Any
    opt_state: Any = None
    covariance: Optional[jax.Array] = None
    key: Optional[jax.Array] = None


class Trainer(NamedTuple):
    """A generic container for a training algorithm."""

    train_fn: Callable[[Any, jnp.ndarray, dict], Any]

    def train(self, train_state: TrainState, data: dict, **kwargs) -> Any:
        return self.train_fn(train_state, data, **kwargs)


# --- Specific Trainer Implementations ---


def init_trainer(
    config: Any,
    dynamics_model: DynamicsModel,
    init_params: Any,
    key: jax.Array,
) -> tuple[Trainer, TrainState]:
    """Initializes the appropriate trainer based on the configuration."""

    # TODO: Figure out if passing the key this way is the best approach
    trainer_type = config["trainer"]
    print(f"🚀 Initializing trainer: {trainer_type.upper()}")

    if trainer_type == "gd":
        trainer, train_state = create_gradient_descent_trainer(
            config, dynamics_model, init_params
        )
    elif trainer_type == "latent_gd":
        trainer, train_state = create_latent_trainer(
            config, dynamics_model, init_params
        )
    elif trainer_type == "ekf":
        trainer, train_state = create_EKF_trainer(
            config, dynamics_model, init_params
        )
    elif trainer_type == "ekf_efficient":
        trainer, train_state = create_EKF_efficient_trainer(
            config, dynamics_model, init_params
        )
    else:
        raise ValueError(f"Unknown trainer type: {trainer_type}")

    return trainer, train_state


def create_gradient_descent_trainer(
    config: Any,
    dynamics_model: DynamicsModel,
    init_params: Any,
) -> tuple[Trainer, TrainState]:
    """
    Creates a trainer that updates 'model' parameters with Adam and freezes
    'normalizer' parameters using optax.multi_transform.

    Supports multi-step autoregressive loss via trainer_params.multistep_horizon.
    """
    trainer_params = config.get("trainer_params", {})
    learning_rate = trainer_params.get("learning_rate", 3e-4)
    multistep_horizon = trainer_params.get("multistep_horizon", 1)

    partition_optimizers = {
        "model": optax.adam(learning_rate),
        "normalizer": optax.set_to_zero(),
    }
    mask = path_aware_map(
        lambda path, _: path[0],
        init_params,
    )
    optimizer = optax.multi_transform(partition_optimizers, mask)
    opt_state = optimizer.init(init_params)
    train_state = TrainState(params=init_params, opt_state=opt_state)

    normalizer = STANDARD_NORMALIZER
    vmap_normalize = jax.vmap(normalizer.normalize, in_axes=(None, 0))

    if multistep_horizon == 1:
        # Single-step loss (original behavior)
        vmap_pred_norm_delta = jax.vmap(
            dynamics_model.pred_norm_delta, in_axes=(None, 0, 0)
        )

        @jax.jit
        def loss_fn(params: Any, data: dict) -> float:
            """Computes Mean Squared Error loss in normalized delta space."""
            states, actions, true_next_states = (
                data["states"],
                data["actions"],
                data["next_states"],
            )
            pred_norm_delta = vmap_pred_norm_delta(params, states, actions)
            target_delta = true_next_states - states
            norm_params = params["normalizer"]["delta"]
            target_norm_delta = vmap_normalize(norm_params, target_delta)
            return jnp.mean((pred_norm_delta - target_norm_delta) ** 2)
    else:
        # Multi-step autoregressive loss
        print(f"  Using multi-step loss with horizon={multistep_horizon}")

        def rollout_one(params, init_state, action_seq, true_states):
            """Autoregressive rollout for single sample, returns MSE."""
            def step(state, action):
                next_state = dynamics_model.pred_one_step(params, state, action)
                return next_state, next_state

            _, pred_states = jax.lax.scan(step, init_state, action_seq)

            # Compute deltas: pred vs true
            all_states = jnp.concatenate([init_state[None], pred_states], axis=0)
            pred_deltas = all_states[1:] - all_states[:-1]
            true_deltas = true_states[1:] - true_states[:-1]

            # Normalize and compute MSE
            pred_norm = vmap_normalize(params["normalizer"]["delta"], pred_deltas)
            true_norm = vmap_normalize(params["normalizer"]["delta"], true_deltas)
            return jnp.mean((pred_norm - true_norm) ** 2)

        vmap_rollout = jax.vmap(rollout_one, in_axes=(None, 0, 0, 0))

        @jax.jit
        def loss_fn(params: Any, data: dict) -> float:
            """Multi-step autoregressive loss in normalized delta space."""
            states = data["states"]    # (batch, H+1, dim_s)
            actions = data["actions"]  # (batch, H, dim_a)
            init_states = states[:, 0]
            losses = vmap_rollout(params, init_states, actions, states)
            return jnp.mean(losses)

    @jax.jit
    def train_step(
        train_state: TrainState, data: dict
    ) -> tuple[TrainState, float]:
        """Performs a single gradient descent update"""
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params, data)
        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        new_train_state = train_state.replace(
            params=new_params, opt_state=new_opt_state
        )
        return new_train_state, loss

    def train_fn(
        train_state: TrainState, data: dict, **kwargs
    ) -> tuple[TrainState, float]:
        return train_step(train_state, data)

    return Trainer(train_fn=train_fn), train_state


def create_latent_trainer(
    config: Any,
    dynamics_model: DynamicsModel,
    init_params: Any,
) -> tuple[Trainer, TrainState]:
    """
    Trains encoder+dynamics via a consistency loss (predicted latent vs. stop-gradient
    target latent) and the decoder in isolation (stop-gradient latent input).
    Uses temporal decay λ^t to weight near-horizon errors more.
    """
    trainer_params = config.get("trainer_params", {})
    learning_rate = trainer_params.get("learning_rate", 3e-4)
    temporal_coefficient = trainer_params.get("temporal_coefficient", 0.5)

    partition_optimizers = {
        "model": optax.adam(learning_rate),
        "normalizer": optax.set_to_zero(),
    }
    mask = path_aware_map(lambda path, _: path[0], init_params)
    optimizer = optax.multi_transform(partition_optimizers, mask)
    opt_state = optimizer.init(init_params)
    train_state = TrainState(params=init_params, opt_state=opt_state)

    normalizer = STANDARD_NORMALIZER

    def rollout_one(params, init_state, action_seq, true_states):
        """
        Autoregressive rollout for a single sample.
        Returns combined consistency + decoder loss with temporal decay.
        """
        norm_params = params["normalizer"]
        H = action_seq.shape[0]

        # Precompute target normalized states and deltas
        norm_true_states = jax.vmap(normalizer.normalize, in_axes=(None, 0))(
            norm_params["encoder"], true_states
        )
        # target latents: encode all true states (stop-gradient applied in loss)
        target_zs = jax.vmap(dynamics_model.encode, in_axes=(None, 0))(
            params, norm_true_states
        )

        # Initial latent
        norm_init = normalizer.normalize(norm_params["encoder"], init_state)
        z0 = dynamics_model.encode(params, norm_init)

        def step(z, t_action):
            t, action = t_action
            norm_action = normalizer.normalize(norm_params["action"], action)
            z_next = dynamics_model.infer_dynamics(params, z, norm_action)
            return z_next, (t, z_next)

        _, (timesteps, pred_zs) = jax.lax.scan(
            step, z0, (jnp.arange(H), action_seq)
        )

        # Temporal decay weights: λ^(t+1) for t in 0..H-1
        weights = temporal_coefficient ** (timesteps + 1)  # (H,)

        # Consistency loss: (pred_z - sg(target_z))^2 * λ^t
        sg_target_zs = jax.lax.stop_gradient(target_zs[1:])  # (H, latent_dim)
        consistency_errs = jnp.mean((pred_zs - sg_target_zs) ** 2, axis=-1)  # (H,)
        consistency_loss = jnp.sum(weights * consistency_errs) / jnp.sum(weights)

        # Decoder loss: (decode(sg(pred_z)) - norm_true_state)^2 * λ^t
        sg_pred_zs = jax.lax.stop_gradient(pred_zs)  # (H, latent_dim)
        pred_norm_next_states = jax.vmap(dynamics_model.decode, in_axes=(None, 0))(
            params, sg_pred_zs
        )
        decoder_errs = jnp.mean((pred_norm_next_states - norm_true_states[1:]) ** 2, axis=-1)  # (H,)
        decoder_loss = jnp.sum(weights * decoder_errs) / jnp.sum(weights)

        return consistency_loss + decoder_loss

    vmap_rollout = jax.vmap(rollout_one, in_axes=(None, 0, 0, 0))

    @jax.jit
    def loss_fn(params: Any, data: dict) -> float:
        states = data["states"]    # (batch, H+1, dim_s)
        actions = data["actions"]  # (batch, H, dim_a)
        init_states = states[:, 0]
        losses = vmap_rollout(params, init_states, actions, states)
        return jnp.mean(losses)

    @jax.jit
    def train_step(
        train_state: TrainState, data: dict
    ) -> tuple[TrainState, float]:
        loss, grads = jax.value_and_grad(loss_fn)(train_state.params, data)
        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        new_train_state = train_state.replace(
            params=new_params, opt_state=new_opt_state
        )
        return new_train_state, loss

    def train_fn(
        train_state: TrainState, data: dict, **kwargs
    ) -> tuple[TrainState, float]:
        return train_step(train_state, data)

    return Trainer(train_fn=train_fn), train_state


def create_EKF_trainer(
    config: Any,
    dynamics_model: DynamicsModel,
    init_params: Any,
) -> tuple[Trainer, TrainState]:
    """
    Creates a trainer that updates model parameters using an EKF for online
    learning (batch size of 1).
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    learning_params = config.get("trainer_params", {})
    learning_rate = learning_params.get("learning_rate", 3e-4)
    weight_decay = learning_params.get("weight_decay", 1.0)
    proc_noise_init = learning_params.get("proc_noise_init", 1e-4)
    proc_noise_decay = learning_params.get("proc_noise_decay", 0.999)
    proc_noise_floor = learning_params.get("proc_noise_floor", 1e-6)
    jitter = learning_params.get("jitter", 1e-6)
    init_cov_scale = learning_params.get("init_cov_scale", 1.0)

    flat_params_model, unflatten_fn_model = jax.flatten_util.ravel_pytree(
        init_params["model"]
    )
    _, unflatten_fn_norm = jax.flatten_util.ravel_pytree(
        init_params["normalizer"]
    )
    dim_params_model = flat_params_model.shape[0]

    init_covariance = jnp.eye(dim_params_model) * init_cov_scale
    train_state = TrainState(params=init_params, covariance=init_covariance)

    @jax.jit
    def parameter_dynamics_fn(params, _):
        """Parameter state transition function with optional weight decay."""
        return params * weight_decay

    @jax.jit
    def observation_fn(params, x):
        state = x[:dim_state]
        action = x[dim_state : dim_state + dim_action]
        flat_params_norm = x[dim_state + dim_action :]
        params_model = unflatten_fn_model(params)
        params_norm = unflatten_fn_norm(flat_params_norm)
        params_pytree = {"model": params_model, "normalizer": params_norm}
        pred_next_state = dynamics_model.pred_one_step(
            params_pytree, state, action
        )
        return pred_next_state - state

    estimator = EKFCovArgs(
        dynamics_fn=parameter_dynamics_fn,
        observation_fn=observation_fn,
        jitter=jitter,
    )

    def process_cov_fn(step):
        noise_val = jnp.clip(
            proc_noise_init * (proc_noise_decay**step),
            a_min=proc_noise_floor,
        )
        return jnp.eye(dim_params_model) * noise_val

    def observation_cov_fn(_):
        return jnp.eye(dim_state) / learning_rate

    @jax.jit
    def train_step(
        train_state: TrainState, data: dict, step_idx: int
    ) -> tuple[TrainState, float]:
        """Performs a single EKF update on one data point."""

        state = jnp.squeeze(data["states"], axis=0)
        action = jnp.squeeze(data["actions"], axis=0)
        next_state = jnp.squeeze(data["next_states"], axis=0)

        flat_params_norm, _ = jax.flatten_util.ravel_pytree(
            train_state.params["normalizer"]
        )

        ekf_inp = jnp.concatenate([state, action, flat_params_norm], axis=-1)
        ekf_out = next_state - state

        proc_cov = process_cov_fn(step_idx)
        meas_cov = observation_cov_fn(step_idx)
        flat_params_model, _ = jax.flatten_util.ravel_pytree(
            train_state.params["model"]
        )
        flat_params_model_new, cov_params_model_new, _ = estimator.estimate(
            flat_params_model,
            train_state.covariance,
            ekf_inp,
            ekf_out,
            proc_cov,
            meas_cov,
        )

        ekf_pred = observation_fn(flat_params_model_new, ekf_inp)
        loss = jnp.mean((ekf_out - ekf_pred) ** 2)  # "Loss" is innovation

        params_new = {
            "model": unflatten_fn_model(flat_params_model_new),
            "normalizer": train_state.params["normalizer"],
        }
        new_train_state = train_state.replace(
            params=params_new, covariance=cov_params_model_new
        )
        return new_train_state, loss

    def train_fn(
        train_state: TrainState, data: dict, **kwargs
    ) -> tuple[TrainState, float]:
        batch_size = data["states"].shape[0]
        if batch_size != 1:
            raise ValueError(
                f"EKF trainer only supports a batch size of 1 for online learning. "
                f"Received batch of size {batch_size}."
            )

        step_idx = kwargs.get("step", 0)
        return train_step(train_state, data, step_idx)

    return Trainer(train_fn=train_fn), train_state


def create_EKF_efficient_trainer(
    config: Any,
    dynamics_model: DynamicsModel,
    init_params: Any,
) -> tuple[Trainer, TrainState]:
    """
    Creates a trainer that updates model parameters using an EKFEfficient for
    online learning (batch size of 1).

    This is a simplified EKF that assumes identity dynamics (no process noise)
    and uses a fixed measurement covariance.
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    learning_params = config.get("trainer_params", {})
    learning_rate = learning_params.get("learning_rate", 3e-4)
    jitter = learning_params.get("jitter", 1e-6)
    init_cov_scale = learning_params.get("init_cov_scale", 1.0)

    flat_params_model, unflatten_fn_model = jax.flatten_util.ravel_pytree(
        init_params["model"]
    )
    _, unflatten_fn_norm = jax.flatten_util.ravel_pytree(
        init_params["normalizer"]
    )
    dim_params_model = flat_params_model.shape[0]

    init_covariance = jnp.eye(dim_params_model) * init_cov_scale
    train_state = TrainState(params=init_params, covariance=init_covariance)

    @jax.jit
    def parameter_dynamics_fn(params, _):
        """Identity parameter dynamics (no weight decay)."""
        return params

    @jax.jit
    def observation_fn(params, x):
        state = x[:dim_state]
        action = x[dim_state : dim_state + dim_action]
        flat_params_norm = x[dim_state + dim_action :]
        params_model = unflatten_fn_model(params)
        params_norm = unflatten_fn_norm(flat_params_norm)
        params_pytree = {"model": params_model, "normalizer": params_norm}
        pred_next_state = dynamics_model.pred_one_step(
            params_pytree, state, action
        )
        return pred_next_state - state

    meas_cov = jnp.eye(dim_state) / learning_rate

    estimator = EKFEfficient(
        dynamics_fn=parameter_dynamics_fn,
        observation_fn=observation_fn,
        meas_cov=meas_cov,
        jitter=jitter,
    )

    @jax.jit
    def train_step(
        train_state: TrainState, data: dict
    ) -> tuple[TrainState, float]:
        """Performs a single EKF update on one data point."""

        state = jnp.squeeze(data["states"], axis=0)
        action = jnp.squeeze(data["actions"], axis=0)
        next_state = jnp.squeeze(data["next_states"], axis=0)

        flat_params_norm, _ = jax.flatten_util.ravel_pytree(
            train_state.params["normalizer"]
        )

        ekf_inp = jnp.concatenate([state, action, flat_params_norm], axis=-1)
        ekf_out = next_state - state

        flat_params_model, _ = jax.flatten_util.ravel_pytree(
            train_state.params["model"]
        )
        flat_params_model_new, cov_params_model_new, _ = estimator.estimate(
            flat_params_model,
            train_state.covariance,
            ekf_inp,
            ekf_out,
        )

        # Prequential Error
        pre_fit_pred = observation_fn(flat_params_model, ekf_inp)
        pre_fit_loss = jnp.mean((ekf_out - pre_fit_pred) ** 2)

        params_new = {
            "model": unflatten_fn_model(flat_params_model_new),
            "normalizer": train_state.params["normalizer"],
        }
        new_train_state = train_state.replace(
            params=params_new, covariance=cov_params_model_new
        )
        return new_train_state, pre_fit_loss

    def train_fn(
        train_state: TrainState, data: dict, **kwargs
    ) -> tuple[TrainState, float]:
        batch_size = data["states"].shape[0]
        if batch_size != 1:
            raise ValueError(
                f"EKF efficient trainer only supports a batch size of 1 for online learning. "
                f"Received batch of size {batch_size}."
            )

        return train_step(train_state, data)

    return Trainer(train_fn=train_fn), train_state

