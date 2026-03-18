# encoders.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
from typing import NamedTuple, Callable, Any, Sequence

from max.normalizers import Normalizer
from max.utilities import mish, simnorm


class Encoder(NamedTuple):
    encode: Callable  # (enc_params, norm_params, obs) -> z


def init_encoder(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
) -> tuple["Encoder", dict]:
    """
    Initialize encoder.

    config["encoder_params"]:
        encoder_features: list[int], e.g. [256, 128] (last entry = latent_dim)
        simnorm_dim_v:    int, simplex dimension V (latent_dim must be divisible by V)
        simnorm_tau:      float, softmax temperature (default 1.0)
        pretrained_params_path: optional path to load pretrained params (pkl with "model" key)

    Returns:
        (Encoder, enc_params) where enc_params are the NN params only (no embedded normalizer).
    """
    dim_state = config["dim_state"]
    enc_cfg = config["encoder_params"]
    encoder_features: Sequence[int] = enc_cfg["encoder_features"]
    simnorm_dim_v: int = enc_cfg["simnorm_dim_v"]
    simnorm_tau: float = enc_cfg["simnorm_tau"]

    latent_dim = encoder_features[-1]
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )

    class _Encoder(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in encoder_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
            x = nn.Dense(encoder_features[-1])(x)
            x = nn.LayerNorm()(x)
            return simnorm(x, simnorm_dim_v, simnorm_tau)

    encoder_net = _Encoder()

    dummy_norm_state = jnp.ones((dim_state,))

    pretrained_path = enc_cfg.get("pretrained_params_path")
    if pretrained_path:
        with open(pretrained_path, "rb") as f:
            pretrained = pickle.load(f)
        enc_params = pretrained["encoder"]["encoder"]
    else:
        key, k1 = jax.random.split(key)
        enc_params = encoder_net.init(k1, dummy_norm_state)

    def encode(enc_params: Any, norm_params: Any, obs: jnp.ndarray) -> jnp.ndarray:
        norm_obs = normalizer.normalize(norm_params["encoder"], obs)
        return encoder_net.apply(enc_params, norm_obs)

    return Encoder(encode=encode), enc_params
