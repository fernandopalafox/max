# trainers.py
"""
TDMPC2 trainer implementation.

Two-optimizer structure:
  - World-model optimizer: encoder, dynamics, reward head, critic (separate LRs)
  - Policy optimizer: policy network

Both optimizers operate on the same `parameters` dict via optax.multi_transform
(world-model) and a direct adam (policy).

References: TD-MPC2 (Hansen et al. 2023), tdmpc2.py / world_model.py
"""

import jax
import jax.numpy as jnp
import jax.flatten_util
import optax
from flax import struct
from typing import Any, Callable, NamedTuple

from max.encoders import Encoder
from max.critics import Critic
from max.policies import Policy
from max.utilities import symlog, symexp, two_hot, two_hot_inv, soft_ce, ema_update
from max.estimators import EKFEfficient


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class TrainState(struct.PyTreeNode):
    """Training state holding both optimizer states."""
    opt_state: Any          # {"world_model": wm_opt_state, "policy": pi_opt_state}


# ---------------------------------------------------------------------------
# Trainer container
# ---------------------------------------------------------------------------

class Trainer(NamedTuple):
    """Generic trainer container."""
    train_fn: Callable  # (train_state, batch, parameters, key) -> (train_state, parameters, metrics)

    def train(self, train_state: TrainState, batch: dict, parameters: dict, key: jax.Array):
        return self.train_fn(train_state, batch, parameters, key)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def init_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Initialize a trainer based on config["trainer"].

    Currently supported: "tdmpc2", "ogd"
    """
    trainer_type = config["trainer"]["type"]

    if trainer_type == "tdmpc2":
        return init_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    if trainer_type == "meta_tdmpc2":
        return init_meta_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    if trainer_type == "fomaml_tdmpc2":
        return init_fomaml_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    if trainer_type == "adam_fomaml_tdmpc2":
        return init_adam_fomaml_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    if trainer_type == "ekf_fomaml_tdmpc2":
        return init_ekf_fomaml_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    if trainer_type == "ogd":
        return init_ogd_trainer(key, config, encoder, dynamics, init_params)
    if trainer_type == "ekf_efficient":
        return init_ekf_efficient_trainer(key, config, encoder, dynamics, init_params)
    if trainer_type == "ekf_batch":
        return init_ekf_batch_trainer(key, config, encoder, dynamics, init_params)
    if trainer_type == "gd_lora_xs":
        return init_gd_lora_xs_trainer(key, config, encoder, dynamics, init_params)

    raise ValueError(f"Unknown trainer: {trainer_type!r}")


# ---------------------------------------------------------------------------
# TDMPC2 trainer
# ---------------------------------------------------------------------------

def init_tdmpc2_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Initialize TD-MPC2 trainer.

    config["trainer"]:
        lr:               float, world-model LR (dynamics, reward, critic)
        encoder_lr:       float, encoder LR (default 0.3 * lr)
        policy_lr:        float, policy LR
        grad_clip_norm:   float, gradient clipping (default 20)
        horizon:          int, rollout horizon H
        discount_factor:  float, gamma
        temporal_decay:   float, rho^t weighting per timestep
        ema_decay:        float, EMA coefficient for target critic
        consistency_coef: float
        reward_coef:      float
        value_coef:       float
        entropy_coef:     float
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

    # critic distributional params
    critic_cfg = config["critic"]
    num_bins: int = critic_cfg["num_bins"]
    vmin: float = critic_cfg["vmin"]
    vmax: float = critic_cfg["vmax"]
    num_ensemble: int = critic_cfg["num_ensemble"]

    reward_cfg = config["reward"]
    rew_num_bins: int = reward_cfg["num_bins"]
    rew_vmin: float = reward_cfg["vmin"]
    rew_vmax: float = reward_cfg["vmax"]

    # --- Optimizers ---
    # World-model optimizer: per-component learning rates, EMA/policy/normalizer frozen
    def _make_labels(params: dict) -> dict:
        """Assign optimizer label to every leaf based on component key under params["mean"]."""
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
        "policy":     optax.set_to_zero(),   # updated by policy optimizer only
        "normalizer": optax.set_to_zero(),   # frozen
    }
    param_labels = _make_labels(init_params)
    wm_optimizer = optax.multi_transform(partition_optimizers, param_labels)

    # Policy optimizer: separate adam with eps=1e-5
    pi_optimizer = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adam(policy_lr, eps=1e-5),
    )

    wm_opt_state = wm_optimizer.init(init_params)
    pi_opt_state = pi_optimizer.init(init_params["mean"]["policy"])

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
    )

    # --- Vmapped helpers ---
    # Single obs -> z
    encode_single = encoder.encode
    # Batched: (B, dim_s) -> (B, latent)
    encode_batch = jax.vmap(encode_single, in_axes=(None, 0))

    # Dynamics: (mean_params, (B, latent), (B, dim_a)) -> (B, latent)
    infer_batch = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))

    # Reward logits: reward.logits for tdmpc2_learned
    rew_logits_fn = reward.logits  # (reward_params, z, a) -> (num_bins,) logits
    rew_logits_batch = jax.vmap(rew_logits_fn, in_axes=(None, 0, 0))

    # Critic value: already handles batch via vmap inside, but need batch over B
    # value(params, z, a) where z/a can be batched -> (num_ensemble, B, num_bins)
    # We use it directly.

    # Policy sample (batched)
    # Signature: sample(policy_params, z, key) -> (action, log_prob)
    sample_fn = policy.sample
    sample_batch = jax.vmap(sample_fn, in_axes=(None, 0, 0))

    # two_hot over batch: (B,) scalars -> (B, num_bins) targets
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    # --- Loss functions ---

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        """
        World-model loss: consistency + reward + Q-value.
        Returns (total_loss, (metrics_dict, zs))
        where zs: (B, H+1, latent) latent rollout states (for policy loss).
        """
        obs = batch["states"]     # (B, H+1, dim_s)
        actions = batch["actions"]  # (B, H, dim_a)
        rewards = batch["rewards"]  # (B, H)
        B = obs.shape[0]

        # ---- 1. TD targets (stop_gradient) ----
        key, pi_key, q_key = jax.random.split(key, 3)

        # Encode obs[:, 1:] -> (B, H, latent)
        # Flatten to (B*H, dim_s), encode, reshape
        obs_next_flat = obs[:, 1:].reshape(B * H, -1)
        z_next_flat_sg = jax.lax.stop_gradient(
            encode_batch(params["mean"]["encoder"], obs_next_flat)
        )  # (B*H, latent)

        # Sample next actions from current policy
        pi_keys_flat = jax.random.split(pi_key, B * H)
        next_actions_flat, _ = sample_batch(
            params["mean"]["policy"], z_next_flat_sg, pi_keys_flat
        )  # (B*H, dim_a)

        # Q target: min of 2 random ensemble members on EMA critic
        q_keys_flat = jax.random.split(q_key, B * H)
        q_sampled_flat = jax.vmap(
            critic.subsample, in_axes=(None, 0, 0, 0)
        )(params["mean"]["ema_critic"], z_next_flat_sg, next_actions_flat, q_keys_flat)  # (B*H, num_subsample)
        q_min_flat = jnp.min(q_sampled_flat, axis=-1)                            # (B*H,)
        q_min = q_min_flat.reshape(B, H)

        td_targets = jax.lax.stop_gradient(rewards + discount_factor * q_min)  # (B, H)

        # ---- 2. Latent rollout ----
        z0 = encode_batch(params["mean"]["encoder"], obs[:, 0])  # (B, latent)

        consistency_loss = jnp.zeros(())
        reward_loss = jnp.zeros(())
        q_loss = jnp.zeros(())

        zs = [z0]
        z = z0

        for t in range(H):
            w = temporal_decay ** t
            a_t = actions[:, t]  # (B, dim_a)

            # Reward and Q use CURRENT latent z_t (before dynamics step)
            rew_logits = rew_logits_batch(params["mean"]["reward"], z, a_t)  # (B, rew_num_bins)
            rew_targets = two_hot_batch_r(symlog(rewards[:, t]))  # (B, rew_num_bins)
            reward_loss = reward_loss + w * jnp.mean(soft_ce(rew_logits, rew_targets))

            # Q loss over ALL ensemble members
            q_logits_all = critic.logits(params["mean"]["critic"], z, a_t)  # (num_ens, B, num_bins)
            td_target_th = two_hot_batch_c(symlog(td_targets[:, t]))  # (B, num_bins)
            q_loss_all = jax.vmap(soft_ce, in_axes=(0, None))(q_logits_all, td_target_th)  # (num_ensemble, B)
            q_loss = q_loss + w * jnp.sum(jnp.mean(q_loss_all, axis=-1))

            # Dynamics step: z_t -> z_{t+1}
            z_pred = infer_batch(params["mean"]["dynamics"], z, a_t)  # (B, latent)
            z_real = jax.lax.stop_gradient(
                encode_batch(params["mean"]["encoder"], obs[:, t + 1])
            )  # (B, latent)

            # Consistency loss: mean over all elements (matches F.mse_loss semantics)
            consistency_loss = consistency_loss + w * jnp.mean((z_pred - z_real) ** 2)

            zs.append(z_pred)
            z = z_pred

        zs_stacked = jnp.stack(zs, axis=1)  # (B, H+1, latent)

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
        """
        Policy loss over H latent states with temporal_decay weighting.
        Gradients only flow through policy_params.
        """
        B = zs_sg.shape[0]
        policy_loss = jnp.zeros(())
        avg_qs = []

        for t in range(H + 1):
            z_t = zs_sg[:, t, :]  # (B, latent_dim)
            key, sample_key, subkey = jax.random.split(key, 3)
            sample_keys = jax.random.split(sample_key, B)

            actions, log_probs = sample_batch(policy_params, z_t, sample_keys)  # (B, dim_a), (B,)

            avg_q = critic.value(critic_params_sg, z_t, actions, subkey)  # (B,)
            avg_qs.append(avg_q)

            entropy = -log_probs  # (B,)
            scaled_entropy = entropy * dim_action  # scale by action_dim
            step_objective = (avg_q + entropy_coef * scaled_entropy) / q_scale  # (B,)
            policy_loss = policy_loss - (temporal_decay ** t) * jnp.mean(step_objective)

        avg_qs_stacked = jnp.stack(avg_qs, axis=1)  # (B, H+1)
        metrics = {"losses/policy": policy_loss, "losses/entropy": jnp.mean(-log_probs)}
        return policy_loss, (metrics, avg_qs_stacked)

    @jax.jit
    def train_step(
        train_state: TrainState,
        batch: dict,
        parameters: dict,
        key: jax.Array,
    ) -> tuple[TrainState, dict, dict]:
        key, wm_key, pi_key = jax.random.split(key, 3)

        # ---- Step 1: World-model backward ----
        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(parameters, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

        # ---- Step 2: Policy backward (separate, detached zs) ----
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

        # ---- Update running Q scale (IQR from t=0 column) ----
        # tau = 1 - ema_decay: official uses tau=0.01 for both target critic and RunningScale
        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
        )

        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state


# ---------------------------------------------------------------------------
# MetaTDMPC2 trainer
# ---------------------------------------------------------------------------

def init_meta_tdmpc2_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    First-order MAML variant of TDMPC2 for meta-learning an adaptable dynamics
    adapter initialization.

    Per meta-batch trajectory, a single naive-GD step adapts the dynamics
    adapter ψ on the first transition (s0, a0, s1), then the regular TDMPC2
    losses (consistency + reward + Q) are computed on the remaining H-step
    meta-test sequence using the per-trajectory adapted adapter ψ'. Gradients
    flow through the inner GD step, so ψ is trained to be adaptable.

    The dynamics must be an adapter variant (dense_last_layer, dense_lora_xs,
    dense_tiny_lora); only params["mean"]["dynamics"] (= ψ) is inner-adapted.
    Requires sampler.horizon = trainer.horizon + 1.

    config["trainer"]: same fields as tdmpc2 plus:
        meta_lr_inner:  float, inner-loop naive GD step size α for ψ adaptation
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
    meta_lr_inner: float = tp["meta_lr_inner"]

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

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
    )

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch  = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))
    rew_logits_batch = jax.vmap(reward.logits, in_axes=(None, 0, 0))
    sample_batch = jax.vmap(policy.sample, in_axes=(None, 0, 0))
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        obs_full = batch["states"]        # (B, H+2, dim_s)
        actions_full = batch["actions"]   # (B, H+1, dim_a)
        rewards_full = batch["rewards"]   # (B, H+1)

        # ---- Inner adaptation: one GD step on R params only ----
        # P and Q (subspace) are held fixed in the inner loop; only R matrices are adapted.
        # Second-order gradients flow through this for full MAML.
        enc_params_sg = jax.lax.stop_gradient(params["mean"]["encoder"])
        dyn_params = params["mean"]["dynamics"]
        adapter = dyn_params["adapter"]
        r_params = {k: v for k, v in adapter.items() if k.startswith("R_")}

        def inner_loss(r):
            full_dyn = dyn_params | {"adapter": adapter | r}
            z0 = encode_batch(enc_params_sg, obs_full[:, 0])
            z1_tgt = jax.lax.stop_gradient(encode_batch(enc_params_sg, obs_full[:, 1]))
            return jnp.mean((infer_batch(full_dyn, z0, actions_full[:, 0]) - z1_tgt) ** 2)

        r_grad = jax.grad(inner_loss)(r_params)
        r_adapted = jax.tree_util.tree_map(
            lambda p, g: p - meta_lr_inner * g, r_params, r_grad
        )
        psi_adapted = dyn_params | {"adapter": adapter | r_adapted}

        # ---- Meta-test views ----
        obs = obs_full[:, 1:]
        actions = actions_full[:, 1:]
        rewards = rewards_full[:, 1:]
        B = obs.shape[0]

        # ---- TD targets (stop_gradient) ----
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
        q_min = jnp.min(q_sampled_flat, axis=-1).reshape(B, H)
        td_targets = jax.lax.stop_gradient(rewards + discount_factor * q_min)

        # ---- Latent rollout with shared ψ' ----
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

            z_pred = infer_batch(psi_adapted, z, a_t)
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
    def train_step(train_state, batch, parameters, key):
        key, wm_key, pi_key = jax.random.split(key, 3)

        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(parameters, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

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

        parameters = parameters | {
            "mean": parameters["mean"] | {
                "ema_critic": ema_update(parameters["mean"]["ema_critic"], parameters["mean"]["critic"], ema_decay)
            }
        }

        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
        )

        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state


# ---------------------------------------------------------------------------
# FOMAML-TDMPC2 trainer
# ---------------------------------------------------------------------------

def init_fomaml_tdmpc2_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    First-Order MAML (FOMAML) variant of MetaTDMPC2.

    Drops second-order terms by treating the adapted adapter ψ' as a constant
    w.r.t. the initial ψ. Unlike the full meta trainer, adaptation uses a single
    gradient step over the entire batch of first transitions (not per-trajectory),
    so ψ' has the same shape as ψ. The outer forward then uses shared ψ' exactly
    like vanilla TDMPC2. A straight-through estimator routes d_loss/d_ψ' back to
    ψ, making the cost only ~2× vanilla (inner grad + outer grad), not 3-4×.

    Config fields identical to meta_tdmpc2.
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
    meta_lr_inner: float = tp["meta_lr_inner"]

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

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
    )

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch  = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))
    rew_logits_batch = jax.vmap(reward.logits, in_axes=(None, 0, 0))
    sample_batch = jax.vmap(policy.sample, in_axes=(None, 0, 0))
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        obs = batch["states"][:, 1:]      # (B, H+1, dim_s) meta-test states
        actions = batch["actions"][:, 1:] # (B, H, dim_a)
        rewards = batch["rewards"][:, 1:] # (B, H)
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
        q_min = jnp.min(q_sampled_flat, axis=-1).reshape(B, H)
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

    def policy_loss_fn(policy_params, critic_params_sg, zs_sg, key, q_scale):
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
            step_objective = (avg_q + entropy_coef * entropy * dim_action) / q_scale
            policy_loss = policy_loss - (temporal_decay ** t) * jnp.mean(step_objective)
        avg_qs_stacked = jnp.stack(avg_qs, axis=1)
        metrics = {"losses/policy": policy_loss, "losses/entropy": jnp.mean(-log_probs)}
        return policy_loss, (metrics, avg_qs_stacked)

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        key, wm_key, pi_key = jax.random.split(key, 3)

        # Inner adaptation: one GD step on R params only.
        # P and Q (subspace) are held fixed in the inner loop; only R matrices are adapted.
        enc_params_sg = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        dyn_params = parameters["mean"]["dynamics"]
        adapter = dyn_params["adapter"]
        r_params = {k: v for k, v in adapter.items() if k.startswith("R_")}

        def inner_loss(r):
            full_dyn = dyn_params | {"adapter": adapter | r}
            z0 = encode_batch(enc_params_sg, batch["states"][:, 0])
            z1_tgt = jax.lax.stop_gradient(encode_batch(enc_params_sg, batch["states"][:, 1]))
            z_pred = infer_batch(full_dyn, z0, batch["actions"][:, 0])
            return jnp.mean((z_pred - z1_tgt) ** 2)

        inner_grad = jax.grad(inner_loss)(r_params)
        r_adapted = jax.tree_util.tree_map(
            lambda p, g: p - meta_lr_inner * g, r_params, inner_grad
        )

        # Straight-through: forward uses adapted R values, backward routes to original R.
        r_fomaml = jax.tree_util.tree_map(
            lambda ra, p: jax.lax.stop_gradient(ra) + (p - jax.lax.stop_gradient(p)),
            r_adapted, r_params,
        )
        psi_fomaml = dyn_params | {"adapter": adapter | r_fomaml}

        # Substitute ψ_fomaml into params for the outer loss — same structure as vanilla.
        params_fomaml = parameters | {
            "mean": parameters["mean"] | {"dynamics": psi_fomaml}
        }

        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(params_fomaml, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

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

        parameters = parameters | {
            "mean": parameters["mean"] | {
                "ema_critic": ema_update(parameters["mean"]["ema_critic"], parameters["mean"]["critic"], ema_decay)
            }
        }

        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
        )
        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state


# ---------------------------------------------------------------------------
# Adam-FOMAML-TDMPC2 trainer
# ---------------------------------------------------------------------------

def init_adam_fomaml_tdmpc2_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Adam-FOMAML variant of FOMAML-TDMPC2.

    Identical to init_fomaml_tdmpc2_trainer except the inner adaptation step
    uses a single Adam update from zero state instead of plain SGD.  At t=1
    the bias-correction terms cancel, giving:
        r' = r - lr * grad / (|grad| + eps)   (approximately sign(grad))

    Config fields identical to fomaml_tdmpc2.
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
    meta_lr_inner: float = tp["meta_lr_inner"]

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

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
    )

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch  = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))
    rew_logits_batch = jax.vmap(reward.logits, in_axes=(None, 0, 0))
    sample_batch = jax.vmap(policy.sample, in_axes=(None, 0, 0))
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    def adam_inner_step(r_params, inner_grad, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        # Single Adam step from zero state — bias correction cancels perfectly at t=1.
        m_hat = inner_grad
        v_hat = jax.tree_util.tree_map(lambda g: g ** 2, inner_grad)
        return jax.tree_util.tree_map(
            lambda p, m, v: p - lr * m / (jnp.sqrt(v) + eps),
            r_params, m_hat, v_hat,
        )

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        obs = batch["states"][:, 1:]
        actions = batch["actions"][:, 1:]
        rewards = batch["rewards"][:, 1:]
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
        q_min = jnp.min(q_sampled_flat, axis=-1).reshape(B, H)
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

    def policy_loss_fn(policy_params, critic_params_sg, zs_sg, key, q_scale):
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
            step_objective = (avg_q + entropy_coef * entropy * dim_action) / q_scale
            policy_loss = policy_loss - (temporal_decay ** t) * jnp.mean(step_objective)
        avg_qs_stacked = jnp.stack(avg_qs, axis=1)
        metrics = {"losses/policy": policy_loss, "losses/entropy": jnp.mean(-log_probs)}
        return policy_loss, (metrics, avg_qs_stacked)

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        key, wm_key, pi_key = jax.random.split(key, 3)

        enc_params_sg = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        dyn_params = parameters["mean"]["dynamics"]
        adapter = dyn_params["adapter"]
        r_params = {k: v for k, v in adapter.items() if k.startswith("R_")}

        def inner_loss(r):
            full_dyn = dyn_params | {"adapter": adapter | r}
            z0 = encode_batch(enc_params_sg, batch["states"][:, 0])
            z1_tgt = jax.lax.stop_gradient(encode_batch(enc_params_sg, batch["states"][:, 1]))
            z_pred = infer_batch(full_dyn, z0, batch["actions"][:, 0])
            return jnp.mean((z_pred - z1_tgt) ** 2)

        inner_grad = jax.grad(inner_loss)(r_params)
        r_adapted = adam_inner_step(r_params, inner_grad, meta_lr_inner)

        r_fomaml = jax.tree_util.tree_map(
            lambda ra, p: jax.lax.stop_gradient(ra) + (p - jax.lax.stop_gradient(p)),
            r_adapted, r_params,
        )
        psi_fomaml = dyn_params | {"adapter": adapter | r_fomaml}

        params_fomaml = parameters | {
            "mean": parameters["mean"] | {"dynamics": psi_fomaml}
        }

        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(params_fomaml, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

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

        parameters = parameters | {
            "mean": parameters["mean"] | {
                "ema_critic": ema_update(parameters["mean"]["ema_critic"], parameters["mean"]["critic"], ema_decay)
            }
        }

        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
        )
        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state


# ---------------------------------------------------------------------------
# EKF-FOMAML-TDMPC2 trainer
# ---------------------------------------------------------------------------

def init_ekf_fomaml_tdmpc2_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    critic: Critic,
    policy: Policy,
    reward,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    FOMAML variant where the inner adaptation step uses EKFEfficient instead of SGD.

    Inner loop: EKFEfficient updates only the R adapter params, using each trajectory's
    first transition (z0, a0) → z1 as the observation. Covariance is reset to P0 at
    every training step (fresh prior per episode). Sequential EKF updates are applied
    over the B trajectories in the batch via jax.lax.scan.

    The obs_fn is constructed once at trainer init. Current backbone params are passed
    through the EKF control vector x = [z_t, a_t, flat_full_dyn] so the pre-built
    EKFEfficient instance remains valid across outer steps. EKFEfficient only
    differentiates w.r.t. flat_r (argnums=0), so the backbone in x is free.

    Outer loop: identical to fomaml_tdmpc2 (TDMPC2 loss with straight-through estimator).

    Config fields (trainer block):
        lr:             EKF measurement noise scale: meas_cov = I / lr
        jitter:         EKF regularisation on innovation covariance diagonal
        init_cov_scale: initial R covariance = I * init_cov_scale (reset each step)
        outer_lr:       world-model Adam LR (dynamics, reward, critic)
        encoder_lr, policy_lr, grad_clip_norm, horizon, discount_factor,
        temporal_decay, ema_decay, consistency_coef, reward_coef,
        value_coef, entropy_coef: same as fomaml_tdmpc2
    """
    tp = config["trainer"]
    lr: float = tp["lr"]
    jitter: float = tp["jitter"]
    init_cov_scale: float = tp["init_cov_scale"]
    outer_lr: float = tp["outer_lr"]
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

    # ---- EKF setup ----
    dummy_z = encoder.encode(init_params["mean"]["encoder"], jnp.zeros(config["dim_state"]))
    latent_dim: int = dummy_z.shape[0]

    flat_r_init, unflatten_r = jax.flatten_util.ravel_pytree(
        {k: v for k, v in init_params["mean"]["dynamics"]["adapter"].items() if k.startswith("R_")}
    )
    dim_r: int = flat_r_init.shape[0]
    _, unflatten_dyn = jax.flatten_util.ravel_pytree(init_params["mean"]["dynamics"])

    B: int = config["sampler"]["batch_size"]

    meas_cov = jnp.eye(latent_dim) / lr
    P0 = jnp.eye(dim_r) * init_cov_scale

    # obs_fn takes all B first transitions stacked and returns the mean prediction (shape D).
    # This matches FOMAML's averaging semantics: one EKF step using the full batch,
    # equivalent to a single Kalman update on the mean innovation across trajectories.
    def observation_fn(flat_r, x):
        z0 = x[:B * latent_dim].reshape(B, latent_dim)
        a0 = x[B * latent_dim:B * latent_dim + B * dim_action].reshape(B, dim_action)
        flat_full_dyn = x[B * latent_dim + B * dim_action:]
        full_dyn = unflatten_dyn(flat_full_dyn)
        r = unflatten_r(flat_r)
        full_dyn_adapted = full_dyn | {"adapter": full_dyn["adapter"] | r}
        preds = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))(full_dyn_adapted, z0, a0)
        return jnp.mean(preds, axis=0)  # (D,)

    estimator = EKFEfficient(
        dynamics_fn=lambda params, _: params,
        observation_fn=observation_fn,
        meas_cov=meas_cov,
        jitter=jitter,
    )

    # ---- Optimizers ----
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
        "dynamics":   optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(outer_lr)),
        "reward":     optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(outer_lr)),
        "critic":     optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(outer_lr)),
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

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
    )

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch  = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))
    rew_logits_batch = jax.vmap(reward.logits, in_axes=(None, 0, 0))
    sample_batch = jax.vmap(policy.sample, in_axes=(None, 0, 0))
    two_hot_batch_c = jax.vmap(lambda x: two_hot(x, vmin, vmax, num_bins))
    two_hot_batch_r = jax.vmap(lambda x: two_hot(x, rew_vmin, rew_vmax, rew_num_bins))

    def wm_loss_fn(params: dict, batch: dict, key: jax.Array):
        obs = batch["states"][:, 1:]      # (B, H+1, dim_s) meta-test states
        actions = batch["actions"][:, 1:] # (B, H, dim_a)
        rewards = batch["rewards"][:, 1:] # (B, H)
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
        q_min = jnp.min(q_sampled_flat, axis=-1).reshape(B, H)
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

    def policy_loss_fn(policy_params, critic_params_sg, zs_sg, key, q_scale):
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
            step_objective = (avg_q + entropy_coef * entropy * dim_action) / q_scale
            policy_loss = policy_loss - (temporal_decay ** t) * jnp.mean(step_objective)
        avg_qs_stacked = jnp.stack(avg_qs, axis=1)
        metrics = {"losses/policy": policy_loss, "losses/entropy": jnp.mean(-log_probs)}
        return policy_loss, (metrics, avg_qs_stacked)

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        key, wm_key, pi_key = jax.random.split(key, 3)

        # Inner adaptation: single EKF step on R params using all B first transitions.
        # The obs_fn returns mean prediction over the batch, so obs_dim = D (not B*D).
        # This matches FOMAML's single averaged gradient step exactly.
        enc_params_sg = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        dyn_params = parameters["mean"]["dynamics"]
        adapter = dyn_params["adapter"]
        r_params = {k: v for k, v in adapter.items() if k.startswith("R_")}

        flat_r, _ = jax.flatten_util.ravel_pytree(r_params)
        flat_full_dyn, _ = jax.flatten_util.ravel_pytree(dyn_params)

        z0 = encode_batch(enc_params_sg, batch["states"][:, 0])  # (B, D)
        z1 = jax.lax.stop_gradient(
            encode_batch(enc_params_sg, batch["states"][:, 1])
        )  # (B, D)
        a0 = batch["actions"][:, 0]  # (B, dim_a)

        x = jnp.concatenate([z0.ravel(), a0.ravel(), flat_full_dyn])
        z1_mean = jnp.mean(z1, axis=0)  # (D,) — mean target, matches FOMAML averaging

        flat_r_adapted, _, _ = estimator._ekf_fn(flat_r, P0, x, z1_mean)
        r_adapted = unflatten_r(flat_r_adapted)

        # Straight-through: forward uses adapted R values, backward routes to original R.
        r_fomaml = jax.tree_util.tree_map(
            lambda ra, p: jax.lax.stop_gradient(ra) + (p - jax.lax.stop_gradient(p)),
            r_adapted, r_params,
        )
        psi_fomaml = dyn_params | {"adapter": adapter | r_fomaml}

        params_fomaml = parameters | {
            "mean": parameters["mean"] | {"dynamics": psi_fomaml}
        }

        (wm_loss_total, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(params_fomaml, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

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

        parameters = parameters | {
            "mean": parameters["mean"] | {
                "ema_critic": ema_update(parameters["mean"]["ema_critic"], parameters["mean"]["critic"], ema_decay)
            }
        }

        scale_tau = 1.0 - ema_decay
        iqr = jnp.maximum(jnp.percentile(avg_qs[:, 0], 75) - jnp.percentile(avg_qs[:, 0], 25), 1.0)
        new_q_scale = (1.0 - scale_tau) * q_scale + scale_tau * iqr
        parameters = parameters | {"normalizer": parameters["normalizer"] | {"q_scale": new_q_scale}}

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
        )
        all_metrics = {**wm_metrics, **pi_metrics, "losses/world_model": wm_loss_total}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state


# ---------------------------------------------------------------------------
# OGD (online gradient descent) trainer
# ---------------------------------------------------------------------------

def init_ogd_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Online gradient descent trainer: one Adam step on dynamics params per call,
    using consistency loss only.

    config["trainer"]:
        lr:             float, Adam learning rate
        horizon:        int, rollout horizon H
        temporal_decay: float, λ^t weight per timestep
        grad_clip_norm: float, gradient clipping (default 20)
    """
    tp = config["trainer"]
    lr: float = tp["lr"]
    H: int = tp["horizon"]
    temporal_decay: float = tp["temporal_decay"]
    grad_clip_norm: float = tp.get("grad_clip_norm", 20.0)

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))

    optimizer = optax.chain(optax.clip_by_global_norm(grad_clip_norm), optax.adam(lr))
    opt_state = optimizer.init(init_params["mean"]["dynamics"])
    train_state = TrainState(opt_state=opt_state)

    def consistency_loss_fn(dyn_params, enc_params, batch):
        obs = batch["states"]      # (B, H+1, dim_s)
        actions = batch["actions"]  # (B, H, dim_a)

        z = encode_batch(enc_params, obs[:, 0])
        loss = jnp.zeros(())
        for t in range(H):
            w = temporal_decay ** t
            z_pred = infer_batch(dyn_params, z, actions[:, t])
            z_real = jax.lax.stop_gradient(encode_batch(enc_params, obs[:, t + 1]))
            loss = loss + w * jnp.mean((z_pred - z_real) ** 2)
            z = z_pred
        return loss / H

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        enc_params = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        loss, grads = jax.value_and_grad(consistency_loss_fn)(
            parameters["mean"]["dynamics"], enc_params, batch
        )
        updates, new_opt = optimizer.update(grads, train_state.opt_state)
        new_dyn_params = optax.apply_updates(parameters["mean"]["dynamics"], updates)
        new_parameters = parameters | {
            "mean": parameters["mean"] | {"dynamics": new_dyn_params}
        }
        new_train_state = train_state.replace(opt_state=new_opt)
        return new_train_state, new_parameters, {"losses/consistency": loss}

    trainer = Trainer(train_fn=train_step)
    return trainer, train_state


# ---------------------------------------------------------------------------
# GD LoRA-XS adapter trainer
# ---------------------------------------------------------------------------

def init_gd_lora_xs_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Online GD adapter-only trainer for LoRA-XS finetuning experiments.

    Adapts only R_ params in dynamics["adapter"] via one plain GD step per
    transition. Encoder is stop-gradiented. Everything else unchanged.
    Compatible with the "latest" sampler (batch_size=1, horizon=1).

    config["trainer"]:
        meta_lr_inner: float, GD step size
    """
    meta_lr_inner: float = config["trainer"]["meta_lr_inner"]

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))
    infer_batch  = jax.vmap(dynamics.predict, in_axes=(None, 0, 0))

    train_state = TrainState(opt_state={})

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        enc_params = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        dyn_params = parameters["mean"]["dynamics"]
        adapter    = dyn_params["adapter"]
        r_params   = {k: v for k, v in adapter.items() if k.startswith("R_")}

        def loss_fn(r):
            full_dyn = dyn_params | {"adapter": adapter | r}
            z0     = encode_batch(enc_params, batch["states"][:, 0])
            z1_tgt = jax.lax.stop_gradient(
                encode_batch(enc_params, batch["states"][:, 1])
            )
            return jnp.mean(
                (infer_batch(full_dyn, z0, batch["actions"][:, 0]) - z1_tgt) ** 2
            )

        loss, r_grad = jax.value_and_grad(loss_fn)(r_params)
        r_new = jax.tree_util.tree_map(
            lambda p, g: p - meta_lr_inner * g, r_params, r_grad
        )
        new_dyn    = dyn_params | {"adapter": adapter | r_new}
        new_params = parameters | {"mean": parameters["mean"] | {"dynamics": new_dyn}}
        return train_state, new_params, {"losses/consistency": loss}

    return Trainer(train_fn=train_step), train_state


# ---------------------------------------------------------------------------
# EKF-efficient trainer
# ---------------------------------------------------------------------------

def init_ekf_efficient_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Online Bayesian dynamics adaptation via EKF, operating in latent space.

    Adapts only dynamics parameters; encoder is frozen (stop-gradient).
    Covariance is stored in parameters["covariance"] (initialised here by
    mutating init_params, which is the same dict as `parameters` in train.py).

    config["trainer"]:
        lr:             float, scales measurement covariance as eye(latent_dim) / lr
        jitter:         float, regularisation added to innovation covariance diagonal
        init_cov_scale: float, initial parameter covariance = eye(dim_dyn) * init_cov_scale
    """
    tp = config["trainer"]
    learning_rate: float = tp["lr"]
    jitter: float = tp["jitter"]
    init_cov_scale: float = tp["init_cov_scale"]

    dim_action: int = config["dim_action"]

    # Latent dimension via a single forward pass of the encoder
    dummy_z = encoder.encode(init_params["mean"]["encoder"], jnp.zeros(config["dim_state"]))
    latent_dim: int = dummy_z.shape[0]

    # Flatten dynamics params to learn their structure
    flat_dyn_init, unflatten_fn_dyn = jax.flatten_util.ravel_pytree(
        init_params["mean"]["dynamics"]
    )
    dim_dyn_params: int = flat_dyn_init.shape[0]

    # Initialise covariance directly in the parameters dict
    init_params["covariance"] = jnp.eye(dim_dyn_params) * init_cov_scale
    init_trace = float(dim_dyn_params * init_cov_scale)

    # Observation function: predict z_{t+1} from [z_t | a_t] using dynamics params
    def observation_fn(flat_dyn_params, x):
        z_t = x[:latent_dim]
        a_t = x[latent_dim:]
        return dynamics.predict(unflatten_fn_dyn(flat_dyn_params), z_t, a_t)

    meas_cov = jnp.eye(latent_dim) / learning_rate

    estimator = EKFEfficient(
        dynamics_fn=lambda params, _: params,  # identity: no process noise
        observation_fn=observation_fn,
        meas_cov=meas_cov,
        jitter=jitter,
    )

    train_state = TrainState(opt_state=None)

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        enc_params = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        z_t = encoder.encode(enc_params, batch["states"][0, 0])
        z_next = jax.lax.stop_gradient(encoder.encode(enc_params, batch["states"][0, 1]))
        a_t = batch["actions"][0, 0]

        ekf_inp = jnp.concatenate([z_t, a_t])  # (latent_dim + dim_a,)
        ekf_out = z_next                         # (latent_dim,)

        flat_dyn, _ = jax.flatten_util.ravel_pytree(parameters["mean"]["dynamics"])
        cov = parameters["covariance"]

        flat_dyn_new, cov_new, _ = estimator.estimate(flat_dyn, cov, ekf_inp, ekf_out)

        # Prequential (pre-fit) loss
        loss = jnp.mean((ekf_out - observation_fn(flat_dyn, ekf_inp)) ** 2)

        new_parameters = parameters | {
            "mean": parameters["mean"] | {"dynamics": unflatten_fn_dyn(flat_dyn_new)},
            "covariance": cov_new,
        }
        cov_trace = jnp.trace(cov_new)
        return train_state, new_parameters, {
            "losses/pred_error": loss,
            "losses/cov_trace": cov_trace,
            "losses/cov_trace_delta": cov_trace - init_trace,
            "losses/cov_diag_min": jnp.min(jnp.diag(cov_new)),
        }

    trainer = Trainer(train_fn=train_step)
    return trainer, train_state


# ---------------------------------------------------------------------------
# EKF-batch trainer
# ---------------------------------------------------------------------------

def init_ekf_batch_trainer(
    key: jax.Array,
    config: dict,
    encoder: Encoder,
    dynamics,
    init_params: dict,
) -> tuple[Trainer, TrainState]:
    """
    Batched EKF dynamics adaptation using the trajectory_batch sampler.

    Processes each trajectory in the batch sequentially via jax.lax.scan,
    applying one EKFEfficient update per trajectory.  Sequential updates are
    mathematically equivalent to a single batch update because trajectories
    are independent given theta.  This keeps the observation dimension at
    M = H * D (e.g. 192) rather than B * H * D (e.g. 3072), making the
    Jacobian computation and the (M, M) innovation covariance solve fast.

    The diagonal measurement covariance encodes temporal decay:
        R_t = (1 / (lr * lambda^t)) * I_D   for step t in [0, H).

    config["trainer"]:
        lr:             float, base learning rate (scales R^{-1})
        horizon:        int, autoregressive rollout length H
        temporal_decay: float, lambda — per-step decay in measurement trust
        jitter:         float, regularisation on innovation covariance diagonal
        init_cov_scale: float, initial P = eye(N) * init_cov_scale

    config["sampler"]:
        batch_size:     int, number of trajectories B processed sequentially
    """
    tp = config["trainer"]
    lr: float = tp["lr"]
    H: int = tp["horizon"]
    temporal_decay: float = tp["temporal_decay"]
    jitter: float = tp["jitter"]
    init_cov_scale: float = tp["init_cov_scale"]

    B: int = config["sampler"]["batch_size"]

    dummy_z = encoder.encode(init_params["mean"]["encoder"], jnp.zeros(config["dim_state"]))
    latent_dim: int = dummy_z.shape[0]

    flat_dyn_init, unflatten_fn = jax.flatten_util.ravel_pytree(
        init_params["mean"]["dynamics"]
    )
    N: int = flat_dyn_init.shape[0]

    init_params["covariance"] = jnp.eye(N) * init_cov_scale
    init_trace = float(N * init_cov_scale)

    # Single-trajectory measurement covariance: (H*D, H*D) diagonal
    steps = jnp.arange(H)
    meas_var_per_step = 1.0 / (lr * temporal_decay ** steps)
    meas_cov = jnp.diag(jnp.repeat(meas_var_per_step, latent_dim))

    encode_batch = jax.vmap(encoder.encode, in_axes=(None, 0))

    def observation_fn(flat_theta, control):
        """Single-trajectory rollout: M = H * D outputs."""
        dyn_params = unflatten_fn(flat_theta)
        z0 = control["z0"]       # (D,)
        acts = control["actions"] # (H, dim_a)

        def step_fn(z, a):
            z_next = dynamics.predict(dyn_params, z, a)
            return z_next, z_next

        _, preds = jax.lax.scan(step_fn, z0, acts)
        return preds.reshape(-1)  # (H*D,)

    estimator = EKFEfficient(
        dynamics_fn=lambda params, _: params,
        observation_fn=observation_fn,
        meas_cov=meas_cov,
        jitter=jitter,
    )

    train_state = TrainState(opt_state=None)

    @jax.jit
    def train_step(train_state, batch, parameters, key):
        enc_params = jax.lax.stop_gradient(parameters["mean"]["encoder"])
        obs = batch["states"]
        actions = batch["actions"]

        obs_flat = obs.reshape(B * (H + 1), -1)
        z_all = jax.lax.stop_gradient(encode_batch(enc_params, obs_flat))
        z_all = z_all.reshape(B, H + 1, latent_dim)

        flat_theta_init, _ = jax.flatten_util.ravel_pytree(parameters["mean"]["dynamics"])
        P_init = parameters["covariance"]

        # Sequential EKF updates over the batch via scan
        def update_one(carry, i):
            flat_theta, P = carry
            control_i = {"z0": z_all[i, 0], "actions": actions[i]}
            Y_i = z_all[i, 1:].reshape(-1)
            flat_theta_new, P_new, _ = estimator.estimate(flat_theta, P, control_i, Y_i)
            return (flat_theta_new, P_new), None

        (flat_theta_new, P_new), _ = jax.lax.scan(
            update_one, (flat_theta_init, P_init), jnp.arange(B)
        )

        # Consistency loss matching OGD: (1/H) * sum_t [ lambda^t * mean_{B,D}(err^2) ]
        def predict_single(flat_theta, z0, acts):
            dyn_params = unflatten_fn(flat_theta)
            def step_fn(z, a):
                z_next = dynamics.predict(dyn_params, z, a)
                return z_next, z_next
            _, preds = jax.lax.scan(step_fn, z0, acts)
            return preds  # (H, D)

        preds = jax.vmap(predict_single, in_axes=(None, 0, 0))(
            flat_theta_init, z_all[:, 0], actions
        )  # (B, H, D)
        sq_err = (preds - z_all[:, 1:]) ** 2                           # (B, H, D)
        weights = temporal_decay ** jnp.arange(H)                      # (H,)
        loss = jnp.sum(weights * jnp.mean(sq_err, axis=(0, 2))) / H    # scalar

        new_parameters = parameters | {
            "mean": parameters["mean"] | {"dynamics": unflatten_fn(flat_theta_new)},
            "covariance": P_new,
        }
        return train_state, new_parameters, {
            "losses/consistency": loss,
            "losses/cov_trace": jnp.trace(P_new),
            "losses/cov_trace_delta": jnp.trace(P_new) - init_trace,
            "losses/cov_diag_min": jnp.min(jnp.diag(P_new)),
        }

    trainer = Trainer(train_fn=train_step)
    return trainer, train_state
