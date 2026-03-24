# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Callable, Any

from max.normalizers import Normalizer
from max.utilities import mish, simnorm


class Dynamics(NamedTuple):
    predict: Callable  # (mean_params, z, action) -> next_z


def init_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer = None,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    Dispatcher — reads config["dynamics"]["type"].

    Supported variants: "dense", "dense_lora", "dense_last_layer".

    Returns:
        (Dynamics, dyn_params) where dyn_params are the trainable params directly.
        Dynamics.predict(dyn_params, z, action) -> z_next
            action: raw (un-normalized).

    If pretrained is provided, it is used as the initial parameters (fully trainable).
    For dense_lora, pretrained should be the dense dynamics Flax params to build LoRA on top of.
    """
    variant = config["dynamics"]["type"]

    if variant == "dense":
        return _init_dense_dynamics(key, config, pretrained=pretrained)
    if variant == "dense_lora":
        return _init_lora_dynamics(key, config, pretrained=pretrained)
    if variant == "dense_last_layer":
        return _init_dense_last_layer_dynamics(key, config, pretrained=pretrained)

    raise ValueError(f"Unknown dynamics: {variant!r}")


def _init_dense_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    Dense MLP dynamics: NormedLinear blocks with SimNorm final activation.

    config["dynamics"]:
        type:              str, "dense"
        dynamics_features: list[int], MLP hidden+output sizes (last = latent_dim)
        simnorm_dim_v:     int, simplex dimension V
        simnorm_tau:       float, softmax temperature (default 1.0)

    Returns dyn_params = flax_params directly.
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]

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
    if pretrained is not None:
        mean_params = pretrained
    else:
        key, k1 = jax.random.split(key)
        mean_params = dynamics_net.init(k1, dummy_x)

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_net.apply(mean_params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), mean_params


def _init_dense_last_layer_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    Dense MLP dynamics with only the last layer (Dense + LayerNorm) trainable.

    All earlier layers are frozen and closed over at construction time.
    If pretrained is provided, all layers are initialized from it; otherwise random init.
    Either way, only the last-layer params are returned as trainable.

    config["dynamics"]:
        type:              str, "dense_last_layer"
        dynamics_features: list[int], MLP hidden+output sizes (last = latent_dim)
        simnorm_dim_v:     int, simplex dimension V
        simnorm_tau:       float, softmax temperature (default 1.0)

    Returns dyn_params = {"kernel", "bias", "ln_scale", "ln_bias"} for the last layer.
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]

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

    if pretrained is not None:
        all_params = pretrained
    else:
        key, k1 = jax.random.split(key)
        all_params = dynamics_net.init(k1, dummy_x)

    n = len(features)
    last_i = n - 1
    p = all_params["params"]

    # Freeze all but the last layer by closing over them
    frozen = [
        {
            "W": p[f"Dense_{i}"]["kernel"],
            "b": p[f"Dense_{i}"]["bias"],
            "ln_scale": p[f"LayerNorm_{i}"]["scale"],
            "ln_bias": p[f"LayerNorm_{i}"]["bias"],
        }
        for i in range(last_i)
    ]

    last_params = {
        "kernel":   p[f"Dense_{last_i}"]["kernel"],
        "bias":     p[f"Dense_{last_i}"]["bias"],
        "ln_scale": p[f"LayerNorm_{last_i}"]["scale"],
        "ln_bias":  p[f"LayerNorm_{last_i}"]["bias"],
    }

    def _forward(last_p: Any, x: jnp.ndarray) -> jnp.ndarray:
        for layer in frozen:
            x = x @ layer["W"] + layer["b"]
            x = layer["ln_scale"] * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + layer["ln_bias"]
            x = mish(x)
        x = x @ last_p["kernel"] + last_p["bias"]
        x = last_p["ln_scale"] * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + last_p["ln_bias"]
        return simnorm(x, simnorm_dim_v, simnorm_tau)

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), last_params


def _init_lora_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    LoRA-XS dynamics: frozen pretrained MLP with trainable low-rank R matrices.

    pretrained should be the dense dynamics Flax params (e.g. from a prior dense run).
    The dense weights are baked into the forward pass via SVD decomposition.
    Only the R matrices (one r×r matrix per layer) are trainable.
    Effective weight: W_eff = W + U @ diag(Sigma) @ R @ V^T.

    Param count: num_layers * r^2 (e.g. 3 layers, r=16 → 768 params).

    config["dynamics"]:
        type:              str, "dense_lora"
        dynamics_features: list[int], must match pretrained architecture
        simnorm_dim_v, simnorm_tau: same as dense variant
        svd_rank:          int, LoRA rank r (default 32)
        r_init_std:        float, std for R init (default 1e-5)

    Returns dyn_params = {"R_0": ..., "R_1": ..., ...} directly.
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]
    svd_rank: int = dyn_cfg["svd_rank"]
    r_init_std: float = dyn_cfg["r_init_std"]

    latent_dim: int = config["encoder"]["encoder_features"][-1]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    pretrained_dyn = pretrained["params"]

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
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), R_init
