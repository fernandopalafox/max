# trainers.py

import jax
import jax.numpy as jnp
import optax
from typing import Callable, Any, NamedTuple, Optional
from max.dynamics import DynamicsModel, PETSDynamicsModel
from flax import struct
from flax.traverse_util import path_aware_map
import jax.flatten_util
from max.estimators import EKFCovArgs
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
    print(f"Initializing trainer: {trainer_type.upper()}")

    if trainer_type == "gd":
        trainer, train_state = create_gradient_descent_trainer(
            config, dynamics_model, init_params
        )
    elif trainer_type == "ekf":
        trainer, train_state = create_EKF_trainer(
            config, dynamics_model, init_params
        )
    elif trainer_type == "pets":
        trainer, train_state = create_probabilistic_ensemble_trainer(
            config, dynamics_model, init_params, key
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
    """
    trainer_params = config.get("trainer_params", {})
    learning_rate = trainer_params.get("learning_rate", 3e-4)

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

    vmap_pred_norm_delta = jax.vmap(
        dynamics_model.pred_norm_delta, in_axes=(None, 0, 0)
    )
    normalizer = STANDARD_NORMALIZER
    vmap_normalize = jax.vmap(normalizer.normalize, in_axes=(None, 0))

    @jax.jit
    def loss_fn(params: Any, data: dict) -> float:
        """Computes Mean Squared Error loss with normalized targets"""
        states, actions, true_next_states = (
            data["states"],
            data["actions"],
            data["next_states"],
        )
        pred_norm_deltas = vmap_pred_norm_delta(params, states, actions)
        true_deltas = true_next_states - states
        norm_params = params["normalizer"]
        true_norm_deltas = vmap_normalize(norm_params["delta"], true_deltas)
        return jnp.mean((true_norm_deltas - pred_norm_deltas) ** 2)

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


def create_probabilistic_ensemble_trainer(
    config: Any,
    dynamics_model: PETSDynamicsModel,
    init_params: Any,
    key: jax.Array,
) -> tuple[Trainer, TrainState]:
    """
    Creates a trainer for the Probabilistic Ensemble (PE) model.

    This trainer uses:
    1. Bootstrapping to create unique datasets for each model in the ensemble.
    2. Gaussian Negative Log-Likelihood loss to train the probabilistic outputs.
    """
    trainer_params = config.get("trainer_params", {})
    learning_rate = trainer_params.get("learning_rate", 1e-3)
    ensemble_size = config["dynamics_params"]["ensemble_size"]

    partition_optimizers = {
        "model": optax.adam(learning_rate),
        "normalizer": optax.set_to_zero(),
    }
    mask = path_aware_map(lambda path, _: path[0], init_params)
    optimizer = optax.multi_transform(partition_optimizers, mask)
    opt_state = optimizer.init(init_params)
    train_state = TrainState(params=init_params, opt_state=opt_state, key=key)

    vmap_pred_dist = jax.vmap(
        dynamics_model.pred_norm_delta_dist, in_axes=(None, 0, 0)
    )
    normalizer = STANDARD_NORMALIZER
    vmap_normalize = jax.vmap(normalizer.normalize, in_axes=(None, 0))

    def loss_fn(model_params: Any, static_params: Any, data: dict) -> float:
        """
        Computes the Gaussian NLL loss for a SINGLE model.
        """
        params = {
            "model": model_params,
            "normalizer": static_params["normalizer"],
        }

        states, actions, true_next_states = (
            data["states"],
            data["actions"],
            data["next_states"],
        )
        pred_means, pred_log_vars = vmap_pred_dist(params, states, actions)
        true_deltas = true_next_states - states
        norm_params = params["normalizer"]
        true_norm_deltas = vmap_normalize(norm_params["delta"], true_deltas)
        inv_vars = jnp.exp(-pred_log_vars)
        mse_term = jnp.sum(
            jnp.square(pred_means - true_norm_deltas) * inv_vars, axis=-1
        )
        log_det_term = jnp.sum(pred_log_vars, axis=-1)
        loss = jnp.mean(mse_term + log_det_term)
        return loss

    grad_fn = jax.value_and_grad(loss_fn, argnums=0)
    vmap_grad_fn = jax.vmap(
        grad_fn,
        in_axes=(0, None, {"states": 0, "actions": 0, "next_states": 0}),
    )

    @jax.jit
    def train_step(
        train_state: TrainState, bootstrapped_data: dict
    ) -> tuple[TrainState, float]:
        """Performs a single gradient descent update for the entire ensemble."""
        model_params = train_state.params["model"]
        static_params = {"normalizer": train_state.params["normalizer"]}

        losses, model_grads = vmap_grad_fn(
            model_params, static_params, bootstrapped_data
        )

        grads = {
            "model": model_grads,
            "normalizer": train_state.params["normalizer"],
        }

        updates, new_opt_state = optimizer.update(
            grads, train_state.opt_state, train_state.params
        )
        new_params = optax.apply_updates(train_state.params, updates)
        new_train_state = train_state.replace(
            params=new_params, opt_state=new_opt_state
        )

        total_loss = jnp.mean(losses)
        return new_train_state, total_loss

    def train_fn(
        train_state: TrainState, data: dict, **kwargs
    ) -> tuple[TrainState, float]:
        """The public training function that handles bootstrapping."""
        key, bootstrap_key = jax.random.split(train_state.key)
        batch_size = data["states"].shape[0]

        bootstrap_indices = jax.random.randint(
            bootstrap_key,
            shape=(ensemble_size, batch_size),
            minval=0,
            maxval=batch_size,
        )
        bootstrapped_data = jax.tree_map(lambda x: x[bootstrap_indices], data)
        new_train_state, loss = train_step(train_state, bootstrapped_data)
        return new_train_state.replace(key=key), loss

    return Trainer(train_fn=train_fn), train_state
