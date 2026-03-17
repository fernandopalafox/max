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
import optax
from flax import struct
from typing import Any, Callable, NamedTuple

from max.encoders import Encoder
from max.critics import Critic
from max.policies import Policy


# ---------------------------------------------------------------------------
# Math helpers (ported from TDMPC2 math.py to JAX)
# ---------------------------------------------------------------------------

def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric exp: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def two_hot(x: jnp.ndarray, vmin: float, vmax: float, num_bins: int) -> jnp.ndarray:
    """
    Encode scalar x as a two-hot distribution over `num_bins` evenly-spaced bins.

    Args:
        x:       scalar or shape (...)
        vmin:    minimum bin value
        vmax:    maximum bin value
        num_bins: number of bins

    Returns:
        soft label vector of shape (..., num_bins)
    """
    bins = jnp.linspace(vmin, vmax, num_bins)
    x_clipped = jnp.clip(x, vmin, vmax)
    # Lower bin index
    k = jnp.floor(
        (x_clipped - vmin) / (vmax - vmin + 1e-8) * (num_bins - 1)
    ).astype(jnp.int32)
    k = jnp.clip(k, 0, num_bins - 2)
    lower_val = bins[k]
    upper_val = bins[k + 1]
    upper_weight = (x_clipped - lower_val) / (upper_val - lower_val + 1e-8)
    upper_weight = jnp.clip(upper_weight, 0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    # Build one-hot-like label — needs to be done without dynamic scatters for JIT
    lower_one_hot = jax.nn.one_hot(k, num_bins)
    upper_one_hot = jax.nn.one_hot(k + 1, num_bins)
    # Broadcast weights to match trailing dim
    lower_w = jnp.expand_dims(lower_weight, axis=-1)
    upper_w = jnp.expand_dims(upper_weight, axis=-1)
    return lower_w * lower_one_hot + upper_w * upper_one_hot


def two_hot_inv(logits: jnp.ndarray, vmin: float, vmax: float, num_bins: int) -> jnp.ndarray:
    """
    Convert categorical logits to scalar Q value (expectation over bins).

    Args:
        logits: (..., num_bins)
    Returns:
        (...,)
    """
    bins = jnp.linspace(vmin, vmax, num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    return jnp.sum(probs * bins, axis=-1)


def soft_ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """
    Soft cross-entropy: -sum(targets * log_softmax(logits), axis=-1).

    Args:
        logits:  (..., num_bins)
        targets: (..., num_bins)
    Returns:
        (...,) scalar per sample
    """
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)


def ema_update(old: Any, new: Any, decay: float) -> Any:
    """Exponential moving average: new_val = decay * old + (1 - decay) * new."""
    return jax.tree_util.tree_map(
        lambda o, n: decay * o + (1.0 - decay) * n, old, new
    )


# ---------------------------------------------------------------------------
# Training state
# ---------------------------------------------------------------------------

class TrainState(struct.PyTreeNode):
    """Training state holding both optimizer states and running Q-scale."""
    opt_state: Any          # {"world_model": wm_opt_state, "policy": pi_opt_state}
    scale: jnp.ndarray      # shape (2,) = [mean, std] for Q normalization in policy loss


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

    Currently supported: "tdmpc2"
    """
    trainer_type = config.get("trainer", "tdmpc2")
    print(f"Initializing trainer: {trainer_type.upper()}")

    if trainer_type == "tdmpc2":
        return init_tdmpc2_trainer(
            key, config, encoder, dynamics, critic, policy, reward, init_params
        )
    else:
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

    config["trainer_params"]:
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
    tp = config["trainer_params"]
    lr: float = tp["lr"]
    encoder_lr: float = tp.get("encoder_lr", lr * 0.3)
    policy_lr: float = tp["policy_lr"]
    grad_clip_norm: float = tp.get("grad_clip_norm", 20.0)
    H: int = tp["horizon"]
    discount_factor: float = tp["discount_factor"]
    temporal_decay: float = tp["temporal_decay"]
    ema_decay: float = tp["ema_decay"]
    consistency_coef: float = tp["consistency_coef"]
    reward_coef: float = tp["reward_coef"]
    value_coef: float = tp["value_coef"]
    entropy_coef: float = tp["entropy_coef"]

    # critic distributional params
    critic_cfg = config["critic_params"]
    num_bins: int = critic_cfg["num_bins"]
    vmin: float = critic_cfg["vmin"]
    vmax: float = critic_cfg["vmax"]
    num_ensemble: int = critic_cfg.get("num_ensemble", 5)

    reward_cfg = config["reward_params"]
    rew_num_bins: int = reward_cfg.get("num_bins", num_bins)
    rew_vmin: float = reward_cfg.get("vmin", vmin)
    rew_vmax: float = reward_cfg.get("vmax", vmax)

    # --- Optimizers ---
    # World-model optimizer: per-component learning rates, EMA/policy/normalizer frozen
    def _make_labels(params: dict) -> dict:
        """Assign optimizer label to every leaf based on top-level key."""
        result = {}
        for top_key, subtree in params.items():
            result[top_key] = jax.tree_util.tree_map(lambda _: top_key, subtree)
        return result

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
    pi_opt_state = pi_optimizer.init(init_params["policy"])

    train_state = TrainState(
        opt_state={"world_model": wm_opt_state, "policy": pi_opt_state},
        scale=jnp.array([0.0, 1.0]),  # [mean, std] for Q normalization
    )

    # --- Vmapped helpers ---
    # Single obs -> z
    encode_single = encoder.encode
    # Batched: (B, dim_s) -> (B, latent)
    encode_batch = jax.vmap(encode_single, in_axes=(None, 0))
    # Decode: (B, latent) -> (B, dim_s) in normalized space
    decode_batch = jax.vmap(encoder.decode, in_axes=(None, 0))

    # Dynamics: (dyn_params, (B, latent), (B, dim_a)) -> (B, latent)
    infer_batch = jax.vmap(dynamics.infer_dynamics, in_axes=(None, 0, 0))

    # Reward logits: reward.logits expected for tdmpc2_learned
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
            encode_batch(params["encoder"], obs_next_flat)
        )  # (B*H, latent)

        # Sample next actions from current policy
        pi_keys_flat = jax.random.split(pi_key, B * H)
        next_actions_flat, _ = sample_batch(
            params["policy"], z_next_flat_sg, pi_keys_flat
        )  # (B*H, dim_a)

        # Q target: min of 2 random ensemble members on EMA critic
        q_keys_flat = jax.random.split(q_key, B * H)
        q_min_flat = jax.vmap(
            lambda z, a, k: critic.scalar_value(
                params["ema_critic"], z, a, "min", k
            )
        )(z_next_flat_sg, next_actions_flat, q_keys_flat)  # (B*H,)
        q_min = q_min_flat.reshape(B, H)

        td_targets = jax.lax.stop_gradient(rewards + discount_factor * q_min)  # (B, H)

        # ---- 2. Latent rollout ----
        z0 = encode_batch(params["encoder"], obs[:, 0])  # (B, latent)

        consistency_loss = jnp.zeros(())
        reward_loss = jnp.zeros(())
        q_loss = jnp.zeros(())

        zs = [z0]
        z = z0

        for t in range(H):
            w = temporal_decay ** t
            a_t = actions[:, t]  # (B, dim_a)

            # Dynamics step
            z_pred = infer_batch(params["dynamics"]["mean"], z, a_t)  # (B, latent)
            z_real = jax.lax.stop_gradient(
                encode_batch(params["encoder"], obs[:, t + 1])
            )  # (B, latent)

            # Consistency loss (MSE in latent space)
            consistency_loss = consistency_loss + w * jnp.mean(
                jnp.sum((z_pred - z_real) ** 2, axis=-1)
            )

            # Reward loss (distributional)
            rew_logits = rew_logits_batch(params["reward"], z_pred, a_t)  # (B, rew_num_bins)
            rew_targets = two_hot_batch_r(symlog(rewards[:, t]))  # (B, rew_num_bins)
            reward_loss = reward_loss + w * jnp.mean(soft_ce(rew_logits, rew_targets))

            # Q loss over ALL ensemble members
            q_logits_all = critic.value(params["critic"], z_pred, a_t)  # (num_ens, B, num_bins)
            td_target_th = two_hot_batch_c(symlog(td_targets[:, t]))     # (B, num_bins)
            for qi in range(num_ensemble):
                q_loss = q_loss + w * jnp.mean(soft_ce(q_logits_all[qi], td_target_th))

            zs.append(z_pred)
            z = z_pred

        zs_stacked = jnp.stack(zs, axis=1)  # (B, H+1, latent)

        total_loss = (
            consistency_coef * consistency_loss
            + reward_coef * reward_loss
            + value_coef * q_loss
        )
        metrics = {
            "consistency_loss": consistency_loss,
            "reward_loss": reward_loss,
            "q_loss": q_loss,
        }
        return total_loss, (metrics, zs_stacked)

    def pi_loss_fn(
        policy_params: dict,
        critic_params_sg: dict,
        zs_sg: jnp.ndarray,
        key: jax.Array,
        scale: jnp.ndarray,
    ):
        """
        Policy loss over all H+1 latent states with temporal_decay weighting.
        Gradients only flow through policy_params.
        """
        B = zs_sg.shape[0]
        H1 = zs_sg.shape[1]  # H+1

        key, pi_key, q_key = jax.random.split(key, 3)

        # Sample actions for all (B, H+1) latent states
        flat_z = zs_sg.reshape(B * H1, -1)
        flat_pi_keys = jax.random.split(pi_key, B * H1)
        flat_actions, flat_log_probs = sample_batch(
            policy_params, flat_z, flat_pi_keys
        )  # (B*H1, dim_a), (B*H1,)

        # Average Q of 2 random ensemble members (stop_gradient on critic)
        flat_q_keys = jax.random.split(q_key, B * H1)
        flat_avg_q = jax.vmap(
            lambda z, a, k: critic.scalar_value(critic_params_sg, z, a, "avg", k)
        )(flat_z, flat_actions, flat_q_keys)  # (B*H1,)

        avg_q = flat_avg_q.reshape(B, H1)            # (B, H+1)
        log_probs = flat_log_probs.reshape(B, H1)    # (B, H+1)

        # Q normalization with running scale
        scale_mean, scale_std = scale[0], scale[1]
        avg_q_norm = (avg_q - scale_mean) / (scale_std + 1e-8)

        # Temporal decay weights
        temporal_weights = temporal_decay ** jnp.arange(H1)  # (H+1,)

        # pi_loss = -(entropy_coef * entropy + Q_normalized) * weights
        entropy = -log_probs  # (B, H+1)
        pi_loss = (-(entropy_coef * entropy + avg_q_norm) * temporal_weights).mean()

        metrics = {
            "policy_loss": pi_loss,
            "policy_entropy": entropy.mean(),
        }
        return pi_loss, (metrics, avg_q)

    @jax.jit
    def train_step(
        train_state: TrainState,
        batch: dict,
        parameters: dict,
        key: jax.Array,
    ) -> tuple[TrainState, dict, dict]:
        key, wm_key, pi_key = jax.random.split(key, 3)

        # ---- Step 1: World-model backward ----
        (wm_loss, (wm_metrics, zs)), wm_grads = jax.value_and_grad(
            wm_loss_fn, has_aux=True
        )(parameters, batch, wm_key)

        wm_updates, new_wm_opt = wm_optimizer.update(
            wm_grads, train_state.opt_state["world_model"], parameters
        )
        parameters = optax.apply_updates(parameters, wm_updates)

        # ---- Step 2: Policy backward (separate, detached zs) ----
        zs_sg = jax.lax.stop_gradient(zs)
        critic_params_sg = jax.lax.stop_gradient(parameters["critic"])

        (pi_loss, (pi_metrics, avg_q)), pi_grads = jax.value_and_grad(
            pi_loss_fn, argnums=0, has_aux=True
        )(parameters["policy"], critic_params_sg, zs_sg, pi_key, train_state.scale)

        pi_updates, new_pi_opt = pi_optimizer.update(
            pi_grads, train_state.opt_state["policy"]
        )
        parameters = parameters | {
            "policy": optax.apply_updates(parameters["policy"], pi_updates)
        }

        # ---- Step 3: EMA target critic update ----
        parameters = parameters | {
            "ema_critic": ema_update(parameters["ema_critic"], parameters["critic"], ema_decay)
        }

        # ---- Update running Q scale ----
        q_mean = jnp.mean(avg_q)
        q_std = jnp.std(avg_q)
        new_scale = jnp.array([
            0.99 * train_state.scale[0] + 0.01 * q_mean,
            0.99 * train_state.scale[1] + 0.01 * q_std,
        ])

        new_train_state = train_state.replace(
            opt_state={"world_model": new_wm_opt, "policy": new_pi_opt},
            scale=new_scale,
        )

        all_metrics = {**wm_metrics, **pi_metrics, "wm_loss": wm_loss, "pi_loss": pi_loss}
        return new_train_state, parameters, all_metrics

    def train_fn(train_state, batch, parameters, key):
        return train_step(train_state, batch, parameters, key)

    trainer = Trainer(train_fn=train_fn)
    return trainer, train_state
