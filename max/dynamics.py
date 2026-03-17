# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
from typing import NamedTuple, Callable, Any

from max.normalizers import Normalizer, init_normalizer
from max.utilities import mish, simnorm


class Dynamics(NamedTuple):
    predict: Callable  # (mean_params, z, action) -> next_z


def init_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer = None,
    normalizer_params: dict = None,
    encoder=None,  # accepted for API compatibility, unused
) -> tuple[Dynamics, dict]:
    """
    Dispatcher — reads config.get("dynamics", "dense").

    Supported variants: "dense", "dense_lora".

    Returns:
        (Dynamics, dyn_params) where dyn_params = {"mean": <trainable params>}.
        Dynamics.predict(dyn_params["mean"], z, action) -> z_next
            action: raw (un-normalized); normalization baked in via closure.
    """
    variant = config.get("dynamics", "dense")
    print(f"Initializing dynamics: {variant.upper()}")

    if normalizer is None:
        normalizer, normalizer_params = init_normalizer(config)

    if variant == "dense":
        return _init_dense_dynamics(key, config, normalizer, normalizer_params)

    if variant == "dense_lora":
        return _init_lora_dynamics(key, config, normalizer, normalizer_params)

    raise ValueError(f"Unknown dynamics: {variant!r}")


def _init_dense_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer,
    normalizer_params: dict,
) -> tuple[Dynamics, dict]:
    """
    Dense MLP dynamics: NormedLinear blocks with SimNorm final activation.

    config["dynamics_params"]:
        dynamics_features: list[int], MLP hidden+output sizes (last = latent_dim)
        simnorm_dim_v:     int, simplex dimension V
        simnorm_tau:       float, softmax temperature (default 1.0)

    Returns dyn_params = {"mean": flax_params}.
    """
    dyn_cfg = config["dynamics_params"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg.get("simnorm_tau", 1.0)

    latent_dim: int = config["encoder_params"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]
    action_norm_params = normalizer_params["action"]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )

    class _DynamicsNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
            x = nn.Dense(features[-1])(x)
            x = nn.LayerNorm()(x)
            return simnorm(x, simnorm_dim_v, simnorm_tau)

    dynamics_net = _DynamicsNet()
    dummy_x = jnp.ones((latent_dim + dim_action,))
    key, k1 = jax.random.split(key)
    mean_params = dynamics_net.init(k1, dummy_x)

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        norm_action = normalizer.normalize(action_norm_params, action)
        return dynamics_net.apply(mean_params, jnp.concatenate([z, norm_action], axis=-1))

    return Dynamics(predict=predict), {"mean": mean_params}


def _init_lora_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer,
    normalizer_params: dict,
) -> tuple[Dynamics, dict]:
    """
    LoRA-XS dynamics: frozen pretrained MLP with trainable low-rank R matrices.

    Loads a pretrained dense dynamics checkpoint and computes truncated SVD of
    each Dense layer weight. Only the R matrices (one r×r matrix per layer) are
    trainable. Effective weight: W_eff = W + U @ diag(Sigma) @ R @ V^T.

    Param count: num_layers * r^2 (e.g. 3 layers, r=16 → 768 params).

    config["dynamics_params"]:
        dynamics_features:      list[int], must match pretrained architecture
        simnorm_dim_v, simnorm_tau: same as dense variant
        svd_rank:               int, LoRA rank r (default 32)
        r_init_std:             float, std for R init (default 1e-5)
        pretrained_params_path: str, path to pretrained latent_dynamics pkl

    Returns dyn_params = {"mean": {"R_0": ..., "R_1": ..., ...}}.
    """
    dyn_cfg = config["dynamics_params"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg.get("simnorm_tau", 1.0)
    svd_rank: int = dyn_cfg.get("svd_rank", 32)
    r_init_std: float = dyn_cfg.get("r_init_std", 1e-5)
    pretrained_path: str = dyn_cfg["pretrained_params_path"]

    latent_dim: int = config["encoder_params"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]
    action_norm_params = normalizer_params["action"]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    with open(pretrained_path, "rb") as f:
        pretrained = pickle.load(f)
    pretrained_dyn = pretrained["dynamics"]["mean"]["params"]
    print(f"Loaded pretrained dynamics from {pretrained_path}")

    # Build frozen layer dicts and initialize R matrices
    frozen_layers = []
    R_init = {}
    for i in range(len(features)):
        W = pretrained_dyn[f"Dense_{i}"]["kernel"]
        b = pretrained_dyn[f"Dense_{i}"]["bias"]
        ln_scale = pretrained_dyn[f"LayerNorm_{i}"]["scale"]
        ln_bias = pretrained_dyn[f"LayerNorm_{i}"]["bias"]

        U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
        frozen_layers.append({
            "W": W, "b": b,
            "U": U_full[:, :svd_rank],
            "Sigma": S_full[:svd_rank],
            "V": Vh_full[:svd_rank, :].T,
            "ln_scale": ln_scale, "ln_bias": ln_bias,
        })

        key, k = jax.random.split(key)
        R_init[f"R_{i}"] = jax.random.normal(k, (svd_rank, svd_rank)) * r_init_std

    n_layers = len(features)

    def _forward(R_params: Any, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(frozen_layers):
            R = R_params[f"R_{i}"]
            W_eff = layer["W"] + layer["U"] @ (layer["Sigma"][:, None] * R) @ layer["V"].T
            x = x @ W_eff + layer["b"]
            x = layer["ln_scale"] * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + layer["ln_bias"]
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        norm_action = normalizer.normalize(action_norm_params, action)
        return _forward(mean_params, jnp.concatenate([z, norm_action], axis=-1))

    return Dynamics(predict=predict), {"mean": R_init}
