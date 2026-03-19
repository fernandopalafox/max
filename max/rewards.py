"""
TDMPC2 reward head: learned NN in latent space.

Provides:
    init_reward_model(config) -> (Reward, reward_params)
        Reward.predict(params, z, action) -> scalar
        Reward.logits(params, z, action)  -> (num_bins,)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Callable, NamedTuple, Any

from max.utilities import mish, two_hot_inv


class Reward(NamedTuple):
    predict: Callable  # (reward_params, z, action) -> scalar
    logits: Callable   # (reward_params, z, action) -> (num_bins,)


def init_reward_model(config: dict) -> tuple["Reward", dict]:
    """
    Initialize the TDMPC2 learned reward head.

    config["reward"]:
        type:      str, reward type (e.g. "mlp")
        features:  list[int], MLP hidden sizes
        num_bins:  int, distributional bins
        vmin:      float, minimum reward value
        vmax:      float, maximum reward value

    Returns:
        (Reward, reward_params) where reward_params are flat Flax params.
    """
    reward_type = config["reward"]["type"]
    if reward_type == "mlp":
        return _init_mlp_reward(config)
    else:
        raise ValueError(f"Unknown reward type: {reward_type!r}")


def _init_mlp_reward(config: dict) -> tuple["Reward", dict]:
    rp = config["reward"]
    features = rp["features"]
    num_bins: int = rp["num_bins"]
    vmin: float = rp["vmin"]
    vmax: float = rp["vmax"]

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_a: int = config["dim_action"]

    class _RewardHead(nn.Module):
        @nn.compact
        def __call__(self, z, a):
            x = jnp.concatenate([z, a], axis=-1)
            for feat in features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
            return nn.Dense(num_bins)(x)

    reward_net = _RewardHead()
    dummy_z = jnp.ones((latent_dim,))
    dummy_a = jnp.ones((dim_a,))
    reward_params = reward_net.init(jax.random.key(0), dummy_z, dummy_a)

    def logits_fn(reward_params: Any, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        return reward_net.apply(reward_params, z, a)

    def predict_fn(reward_params: Any, z: jnp.ndarray, a: jnp.ndarray) -> jnp.ndarray:
        """Returns scalar reward estimate."""
        return two_hot_inv(logits_fn(reward_params, z, a), vmin, vmax, num_bins)

    return Reward(predict=predict_fn, logits=logits_fn), reward_params
