# trainers_grad_capture.py
"""
TDMPC2 trainer variant that accumulates gradient sums for subspace alignment analysis.

Tracks per-step for each Dense kernel in the dynamics MLP:
  G_sum_unscaled: sum of raw backprop gradients (before clip/optimizer)
  G_sum_scaled:   sum of Adam-scaled updates from wm_optimizer (before lr, after Adam ratio)

Initial weights are captured externally (in the training script) before the first step,
since they don't change and don't need to live in the JAX state.
"""

import jax
import jax.numpy as jnp
import optax
from flax import struct
from typing import Any, Callable, NamedTuple

from max.encoders import Encoder
from max.critics import Critic
from max.policies import Policy
from max.trainers import Trainer
from max.utilities import symlog, two_hot, soft_ce, ema_update


class GradCaptureTrainState(struct.PyTreeNode):
    opt_state: Any
    g_sum_unscaled: Any  # dict: {layer_name: kernel_sum_array}
    g_sum_scaled: Any    # dict: {layer_name: kernel_sum_array}


def init_grad_capture_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, GradCaptureTrainState]:
    """
    TD-MPC2 trainer that accumulates gradient sums for dynamics Dense kernels.

    Identical to init_tdmpc2_trainer in all optimizer/loss logic.
    Adds gradient accumulation buffers to TrainState.
    """
    tp = config["trainer"]
    lr: float = tp["lr"]
    encoder_lr: float = tp["encoder_lr"]
    policy_lr: float = tp["policy_lr"]
    grad_clip_norm: float = tp["grad_clip_norm"]
    H: int = tp["horizon"]
    discount_factor: float = tp["discount_factor"]
    temporal_decay: float = tp["temporal_decay"]
    ema_decay: float = tp["ema_decay"]
    consistency_coef: float = tp["consistency_coef"]
    reward_coef: float = tp["reward_coef"]
    value_coef: float = tp["value_coef"]
    entropy_coef: float = tp["entropy_coef"]

    dim_action: int = config["dim_action"]

    critic_cfg = config["critic"]
    num_bins: int = critic_cfg["num_bins"]
    vmin: float = critic_cfg["vmin"]
    vmax: float = critic_cfg["vmax"]
    num_ensemble: int = critic_cfg["num_ensemble"]

    reward_cfg = config["reward"]
    rew_num_bins: int = reward_cfg["num_bins"]
    rew_vmin: float = reward_cfg["vmin"]
    rew_vmax: float = reward_cfg["vmax"]

    # --- Identify Dense layer names in dynamics MLP ---
    dyn_params = init_params["mean"]["dynamics"]["params"]
    dense_names = sorted([k for k in dyn_params if k.startswith("Dense_")])

    # --- Optimizers (identical to init_tdmpc2_trainer) ---
    def _make_labels(params: dict) -> dict:
        mean_labels = {
            k: jax.tree_util.tree_map(lambda _: k, v)
            for k, v in params["mean"].items()
        }
        return {
            "mean": mean_labels,
            "normalizer": jax.tree_util.tree_map(lambda _: "normalizer", params["normalizer"]),
        }

    partition_optimizers = {
        "encoder":    optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(encoder_lr)),
        "dynamics":   optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr)),
        "reward":     optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr)),
        "critic":     optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr)),
        "ema_critic": optax.set_to_zero(),
        "policy":     optax.set_to_zero(),
        "normalizer": optax.set_to_zero(),
    }
    param_labels = _make_labels(init_params)
    wm_optimizer = optax.multi_transform(partition_optimizers, param_labels)

    pi_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(policy_lr, eps=1e-5),
    )

    wm_opt_state = wm_optimizer.init(init_params)
    pi_opt_state = pi_optimizer.init(init_params["mean"]["policy"])

    # Initialize gradient accumulator buffers (zeros matching kernel shapes)
    g_sum_init = {
        name: jnp.zeros_like(dyn_params[name]["kernel"])
        for name in dense_names
    }

    train_state = GradCaptureTrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
        g_sum_unscaled=g_sum_init,
        g_sum_scaled={name: jnp.zeros_like(dyn_params[name]["kernel"]) for name in dense_names},
    )

    # --- Vmapped helpers ---
    encode_single = encoder.encode
    encode_batch = jax.vmap(encode_single, in_axes=(None, 0))
    infer_batch = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))
    rew_logits_fn = reward.logits
    rew_logits_batch = jax.vmap(rew_logits_fn, in_axes=(None, 0, 0))
    sample_fn = policy.sample
    sample_batch = jax.vmap(sample_fn, in_axes=(None, 0, 0))
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    # --- Loss functions (identical to init_tdmpc2_trainer) ---

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        obs = batch["states"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        B = obs.shape[0]

        key, pi_key, q_key = jax.random.split(key, 3)

        obs_next_flat = obs[:, 1:].reshape(B * H, -1)
        z_next_flat_sg = jax.lax.stop_gradient(
            encode_batch(params["mean"]["encoder"], obs_next_flat)
        )

        pi_keys_flat = jax.random.split(pi_key, B * H)
        next_actions_flat, _ = sample_batch(
            params["mean"]["policy"], z_next_flat_sg, pi_keys_flat
        )

        q_keys_flat = jax.random.split(q_key, B * H)
        q_sampled_flat = jax.vmap(
            critic.subsample, in_axes=(None, 0, 0, 0)
        )(params["mean"]["ema_critic"], z_next_flat_sg, next_actions_flat, q_keys_flat)
        q_min_flat = jnp.min(q_sampled_flat, axis=-1)
        q_min = q_min_flat.reshape(B, H)

        td_targets = jax.lax.stop_gradient(rewards + discount_factor * q_min)

        z0 = encode_batch(params["mean"]["encoder"], obs[:, 0])

        consistency_loss = jnp.zeros(())
        reward_loss = jnp.zeros(())
        q_loss = jnp.zeros(())

        zs = [z0]
        z = z0

        for t in range(H):
            w = temporal_decay ** t
            a_t = actions[:, t]

            rew_logits = rew_logits_batch(params["mean"]["reward"], z, a_t)
            rew_targets = two_hot_batch_r(symlog(rewards[:, t]))
            reward_loss = reward_loss + w * jnp.mean(soft_ce(rew_logits, rew_targets))

            q_logits_all = critic.logits(params["mean"]["critic"], z, a_t)
            td_target_th = two_hot_batch_c(symlog(td_targets[:, t]))
            q_loss_all = jax.vmap(soft_ce, in_axes=(0, None))(q_logits_all, td_target_th)
            q_loss = q_loss + w * jnp.sum(jnp.mean(q_loss_all, axis=-1))

            z_pred = infer_batch(params["mean"]["dynamics"], z, a_t)
            z_real = jax.lax.stop_gradient(
                encode_batch(params["mean"]["encoder"], obs[:, t + 1])
            )

            consistency_loss = consistency_loss + w * jnp.mean((z_pred - z_real) ** 2)

            zs.append(z_pred)
            z = z_pred

        zs_stacked = jnp.stack(zs, axis=1)

        consistency_loss = consistency_loss / H
        reward_loss = reward_loss / H
        q_loss = q_loss / (H * num_ensemble)

        total_loss = (
            consistency_coef * consistency_loss
            + reward_coef * reward_loss
            + value_coef * q_loss
        )
        metrics = {
            "losses/consistency": consistency_loss,
            "losses/reward":      reward_loss,
            "losses/value":       q_loss,
        }
        return total_loss, (metrics, zs_stacked)

    def policy_loss_fn(
        policy_params: dict,
        critic_params_sg: dict,
        zs_sg: jnp.ndarray,
        key: jax.Array,
        q_scale: jnp.ndarray,
    ):
        B = zs_sg.shape[0]
        policy_loss = jnp.zeros(())
        avg_qs = []

        for t in range(H + 1):
            z_t = zs_sg[:, t, :]
            key, sample_key, subkey = jax.random.split(key, 3)
            sample_keys = jax.random.split(sample_key, B)

            actions, log_probs = sample_batch(policy_params, z_t, sample_keys)

            avg_q = critic.value(critic_params_sg, z_t, actions, subkey)
            avg_qs.append(avg_q)

            entropy = -log_probs
            scaled_entropy = entropy * dim_action
            step_objective = (avg_q + entropy_coef * scaled_entropy) / q_scale
            policy_loss = policy_loss - (temporal_decay ** t) * jnp.mean(step_objective)

        avg_qs_stacked = jnp.stack(avg_qs, axis=1)
        metrics = {"losses/policy": policy_loss, "losses/entropy": jnp.mean(-log_probs)}
        return policy_loss, (metrics, avg_qs_stacked)

    @jax.jit
    def train_step(
        train_state: GradCaptureTrainState,
        batch: dict,
        parameters: dict,
        key: jax.Array,
    ) -> tuple[GradCaptureTrainState, dict, dict]:
        key, wm_key, pi_key = jax.random.split(key, 3)

        # ---- Step 1: World-model backward ----
        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(parameters, batch, wm_key)

        # Capture raw (unscaled) dynamics Dense kernel gradients before optimizer
        g_unscaled = {
            name: wm_grads["mean"]["dynamics"]["params"][name]["kernel"]
            for name in dense_names
        }

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )

        # Capture Adam-scaled updates for dynamics Dense kernels.
        # wm_updates["mean"]["dynamics"]["params"][name]["kernel"] = -lr * m_hat/(sqrt(v_hat)+eps)
        # Scale doesn't affect SVD subspaces, so we accumulate directly.
        g_scaled = {
            name: wm_updates["mean"]["dynamics"]["params"][name]["kernel"]
            for name in dense_names
        }

        # Accumulate gradient sums
        new_g_sum_unscaled = jax.tree_util.tree_map(
            jnp.add, train_state.g_sum_unscaled, g_unscaled
        )
        new_g_sum_scaled = jax.tree_util.tree_map(
            jnp.add, train_state.g_sum_scaled, g_scaled
        )

        parameters = optax.apply_updates(parameters, wm_updates)

        # ---- Step 2: Policy backward ----
        zs_sg = jax.lax.stop_gradient(zs)
        critic_params_sg = jax.lax.stop_gradient(parameters["mean"]["critic"])

        q_scale = parameters["normalizer"]["q_scale"]

        (_, (pi_metrics, avg_qs)), pi_grads = jax.value_and_grad(
            policy_loss_fn, argnums=0, has_aux=True
        )(parameters["mean"]["policy"], critic_params_sg, zs_sg, pi_key, q_scale)

        pi_updates, new_pi_opt = pi_optimizer.update(
            pi_grads, train_state.opt_state["policy"]
        )
        parameters = parameters | {
            "mean": parameters["mean"] | {
                "policy": optax.apply_updates(parameters["mean"]["policy"], pi_updates)
            }
        }

        # ---- Step 3: EMA target critic update ----
        parameters = parameters | {
            "mean": parameters["mean"] | {
                "ema_critic": ema_update(parameters["mean"]["ema_critic"], parameters["mean"]["critic"], ema_decay)
            }
        }

        # ---- Update running Q scale ----
        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
            g_sum_unscaled=new_g_sum_unscaled,
            g_sum_scaled=new_g_sum_scaled,
        )

        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state
