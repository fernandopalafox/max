# encoders.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
from typing import NamedTuple, Callable, Any, Sequence

from max.normalizers import Normalizer
from max.utilities import mish, simnorm


class Encoder(NamedTuple):
    encode: Callable  # (enc_params, obs) -> z
    decode: Callable  # (enc_params, z) -> norm_obs


def init_encoder(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: dict,
) -> tuple["Encoder", dict]:
    """
    Initialize encoder/decoder pair.

    config["encoder_params"]:
        encoder_features: list[int], e.g. [256, 128] (last entry = latent_dim)
        decoder_features: list[int], e.g. [512, 256]
        simnorm_dim_v:    int, simplex dimension V (latent_dim must be divisible by V)
        simnorm_tau:      float, softmax temperature (default 1.0)
        pretrained_params_path: optional path to load pretrained params (pkl with "model" key)

    Returns:
        (Encoder, enc_params) where:
            enc_params = {
                "encoder":    flax_encoder_params,
                "decoder":    flax_decoder_params,
                "normalizer": state_norm_params,
            }
    """
    dim_state = config["dim_state"]
    enc_cfg = config["encoder_params"]
    encoder_features: Sequence[int] = enc_cfg["encoder_features"]
    decoder_features: Sequence[int] = enc_cfg["decoder_features"]
    simnorm_dim_v: int = enc_cfg["simnorm_dim_v"]
    simnorm_tau: float = enc_cfg.get("simnorm_tau", 1.0)

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

    class _Decoder(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in decoder_features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
            return nn.Dense(dim_state)(x)

    encoder_net = _Encoder()
    decoder_net = _Decoder()

    dummy_norm_state = jnp.ones((dim_state,))
    dummy_z = jnp.ones((latent_dim,))

    pretrained_path = enc_cfg.get("pretrained_params_path")
    if pretrained_path:
        with open(pretrained_path, "rb") as f:
            pretrained = pickle.load(f)
        encoder_nn_params = pretrained["encoder"]["encoder"]
        decoder_nn_params = pretrained["encoder"]["decoder"]
        print(f"Loaded pretrained encoder/decoder from {pretrained_path}")
    else:
        key, k1, k2 = jax.random.split(key, 3)
        encoder_nn_params = encoder_net.init(k1, dummy_norm_state)
        decoder_nn_params = decoder_net.init(k2, dummy_z)

    enc_params = {
        "encoder": encoder_nn_params,
        "decoder": decoder_nn_params,
        "normalizer": normalizer_params["state"],
    }

    def encode(enc_params: Any, obs: jnp.ndarray) -> jnp.ndarray:
        norm_obs = normalizer.normalize(enc_params["normalizer"], obs)
        return encoder_net.apply(enc_params["encoder"], norm_obs)

    def decode(enc_params: Any, z: jnp.ndarray) -> jnp.ndarray:
        """Returns normalized (decoder-space) observation."""
        return decoder_net.apply(enc_params["decoder"], z)

    return Encoder(encode=encode, decode=decode), enc_params
