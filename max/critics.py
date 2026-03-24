# critics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Callable, Any

from max.utilities import mish, two_hot_inv


class Critic(NamedTuple):
    value: Callable       # (critic_params, z, a) -> logits  shape (...batch dims...) prepended with (num_ensemble,)
    subsample: Callable   # (critic_params, z, a, key, n=2) -> (n, ...) Q values (raw, no min/mean)


def init_critic(key: jax.Array, config: dict, pretrained: dict = None) -> tuple["Critic", dict]:
    """
    Initialize Q-function ensemble.

    config["critic"]:
        type:          str, critic type (e.g. "ensemble")
        features:      list[int], MLP hidden sizes
        num_ensemble:  int, number of Q-network ensemble members
        num_bins:      int, distributional bins
        vmin:          float, minimum return
        vmax:          float, maximum return
        dropout:       float, dropout rate on first hidden layer
        num_subsample: int, ensemble members to subsample for TD targets

    Returns:
        (Critic, critic_params) where critic_params = {"ensemble": vmapped_flax_params}

    If pretrained is provided, it is used as the initial parameters (fully trainable).
    """
    critic_type = config["critic"]["type"]
    if critic_type == "ensemble":
        return _init_ensemble_critic(key, config, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown critic type: {critic_type!r}")


def _init_ensemble_critic(key: jax.Array, config: dict, pretrained: dict = None) -> tuple["Critic", dict]:
    critic_cfg = config["critic"]
    features = critic_cfg["features"]
    num_ensemble: int = critic_cfg["num_ensemble"]
    num_bins: int = critic_cfg["num_bins"]
    vmin: float = critic_cfg["vmin"]
    vmax: float = critic_cfg["vmax"]
    dropout: float = critic_cfg["dropout"]
    num_subsample: int = critic_cfg["num_subsample"]

    # TODO: fix dropout!

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

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_a: int = config["dim_action"]
    dummy_z = jnp.ones((latent_dim,))
    dummy_a = jnp.ones((dim_a,))

    if pretrained is not None:
        critic_params = pretrained
    else:
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

    def subsample(
        critic_params: Any,
        z: jnp.ndarray,
        a: jnp.ndarray,
        key: jax.Array,
    ) -> jnp.ndarray:
        """
        Pick num_subsample random ensemble members, return their scalar Q values (no aggregation).
        z/a unbatched -> returns (num_subsample,); z/a batched (B,...) -> returns (num_subsample, B).
        Caller decides how to aggregate (min for TD targets, mean for policy/planner).
        """
        logits = value(critic_params, z, a)                        # (num_ensemble, [B,] num_bins)
        q_vals = two_hot_inv(logits, vmin, vmax, num_bins)          # (num_ensemble, [B])
        idxs = jax.random.choice(key, num_ensemble, shape=(num_subsample,), replace=False)
        return q_vals[idxs]                                         # (num_subsample, [B])

    return Critic(value=value, subsample=subsample), critic_params
