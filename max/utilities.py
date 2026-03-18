# utilities.py
"""Shared math primitives for TDMPC2."""

import jax
import jax.numpy as jnp
from typing import Any


def mish(x: jnp.ndarray) -> jnp.ndarray:
    """Mish activation: x * tanh(softplus(x))."""
    return x * jnp.tanh(jax.nn.softplus(x))


def simnorm(x: jnp.ndarray, dim_v: int, tau: float = 1.0) -> jnp.ndarray:
    """SimNorm: reshape into groups of dim_v, softmax each group."""
    shape = x.shape
    L = shape[-1] // dim_v
    x = x.reshape(*shape[:-1], L, dim_v)
    x = jax.nn.softmax(x / tau, axis=-1)
    return x.reshape(shape)


def symlog(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric log: sign(x) * log(1 + |x|)."""
    return jnp.sign(x) * jnp.log1p(jnp.abs(x))


def symexp(x: jnp.ndarray) -> jnp.ndarray:
    """Symmetric exp: sign(x) * (exp(|x|) - 1)."""
    return jnp.sign(x) * jnp.expm1(jnp.abs(x))


def two_hot(x: jnp.ndarray, vmin: float, vmax: float, num_bins: int) -> jnp.ndarray:
    """Encode scalar x as a two-hot distribution over num_bins evenly-spaced bins."""
    bins = jnp.linspace(vmin, vmax, num_bins)
    x_clipped = jnp.clip(x, vmin, vmax)
    k = jnp.floor(
        (x_clipped - vmin) / (vmax - vmin + 1e-8) * (num_bins - 1)
    ).astype(jnp.int32)
    k = jnp.clip(k, 0, num_bins - 2)
    lower_val = bins[k]
    upper_val = bins[k + 1]
    upper_weight = (x_clipped - lower_val) / (upper_val - lower_val + 1e-8)
    upper_weight = jnp.clip(upper_weight, 0.0, 1.0)
    lower_weight = 1.0 - upper_weight
    lower_one_hot = jax.nn.one_hot(k, num_bins)
    upper_one_hot = jax.nn.one_hot(k + 1, num_bins)
    lower_w = jnp.expand_dims(lower_weight, axis=-1)
    upper_w = jnp.expand_dims(upper_weight, axis=-1)
    return lower_w * lower_one_hot + upper_w * upper_one_hot


def two_hot_inv(logits: jnp.ndarray, vmin: float, vmax: float, num_bins: int) -> jnp.ndarray:
    """Convert categorical logits to scalar value (expectation over symexp-transformed bins)."""
    bins = jnp.linspace(vmin, vmax, num_bins)
    probs = jax.nn.softmax(logits, axis=-1)
    return symexp(jnp.sum(probs * bins, axis=-1))


def soft_ce(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    """Soft cross-entropy: -sum(targets * log_softmax(logits), axis=-1)."""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return -jnp.sum(targets * log_probs, axis=-1)


def ema_update(old: Any, new: Any, decay: float) -> Any:
    """Exponential moving average: decay * old + (1 - decay) * new."""
    return jax.tree_util.tree_map(
        lambda o, n: decay * o + (1.0 - decay) * n, old, new
    )


def count_parameters(params: Any) -> int:
    """Count total number of scalar parameters in a pytree."""
    return sum(x.size for x in jax.tree_util.tree_leaves(params))
