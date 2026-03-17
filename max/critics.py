# critics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Callable, Any

from max.utilities import mish, two_hot_inv


class Critic(NamedTuple):
    value: Callable         # (critic_params, z, a) -> logits  shape (...batch dims...) prepended with (num_ensemble,)
    scalar_value: Callable  # (critic_params, z, a, return_type, key) -> scalar Q


def init_critic(key: jax.Array, config: dict) -> tuple["Critic", dict]:
    """
    Initialize Q-function ensemble.

    config["critic_params"]:
        features:     list[int], MLP hidden sizes
        num_ensemble: int, number of Q-network ensemble members (default 5)
        num_bins:     int, distributional bins (default 101)
        vmin:         float, minimum return (default -10)
        vmax:         float, maximum return (default 10)
        dropout:      float, dropout rate on first hidden layer (default 0.01)

    Returns:
        (Critic, critic_params) where critic_params = {"ensemble": vmapped_flax_params}
    """
    critic_cfg = config["critic_params"]
    features = critic_cfg["features"]
    num_ensemble: int = critic_cfg.get("num_ensemble", 5)
    num_bins: int = critic_cfg.get("num_bins", 101)
    vmin: float = critic_cfg.get("vmin", -10.0)
    vmax: float = critic_cfg.get("vmax", 10.0)
    dropout: float = critic_cfg.get("dropout", 0.01)

    class _QNet(nn.Module):
        @nn.compact
        def __call__(self, z, a, training: bool = False):
            x = jnp.concatenate([z, a], axis=-1)
            for i, feat in enumerate(features):
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
                if i == 0 and dropout > 0.0:
                    x = nn.Dropout(rate=dropout, deterministic=not training)(x)
            return nn.Dense(num_bins)(x)

    q_net = _QNet()

    latent_dim: int = config["encoder_params"]["encoder_features"][-1]
    dim_a: int = config["dim_action"]
    dummy_z = jnp.ones((latent_dim,))
    dummy_a = jnp.ones((dim_a,))

    # Initialize num_ensemble independent Q-networks
    ens_keys = jax.random.split(key, num_ensemble)
    ensemble_params = jax.vmap(lambda k: q_net.init(k, dummy_z, dummy_a))(ens_keys)

    critic_params = {"ensemble": ensemble_params}

    def value(critic_params: Any, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """
        Returns raw logits.
        z/a can be unbatched (latent,)/(dim_a,) -> returns (num_ensemble, num_bins)
        or batched (B, latent)/(B, dim_a)       -> returns (num_ensemble, B, num_bins)
        """
        def single_forward(params):
            return q_net.apply(params, z, a)
        return jax.vmap(single_forward)(critic_params["ensemble"])

    def scalar_value(
        critic_params: Any,
        z: jnp.ndarray,
        a: jnp.ndarray,
        return_type: str,
        key: jax.Array,
    ) -> jnp.ndarray:
        """
        Returns scalar Q estimate(s).
        return_type:
          "min" - pick 2 random ensemble members, return min (anti-overestimation for TD targets)
          "avg" - pick 2 random ensemble members, return average (for policy gradient)
          "all" - return all ensemble Q-values (num_ensemble, ...)
        """
        logits = value(critic_params, z, a)               # (num_ensemble, ..., num_bins)
        q_vals = two_hot_inv(logits, vmin, vmax, num_bins)  # (num_ensemble, ...)

        if return_type == "all":
            return q_vals

        key, k1, k2 = jax.random.split(key, 3)
        idx1 = jax.random.randint(k1, shape=(), minval=0, maxval=num_ensemble)
        idx2 = jax.random.randint(k2, shape=(), minval=0, maxval=num_ensemble)

        if return_type == "min":
            return jnp.minimum(q_vals[idx1], q_vals[idx2])
        elif return_type == "avg":
            return 0.5 * (q_vals[idx1] + q_vals[idx2])
        else:
            raise ValueError(f"Unknown return_type: {return_type!r}")

    return Critic(value=value, scalar_value=scalar_value), critic_params
