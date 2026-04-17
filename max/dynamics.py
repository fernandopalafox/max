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

    Supported variants: "dense", "dense_lora_xs", "dense_last_layer".

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
    if variant == "dense_lora_xs":
        return _init_lora_xs_dynamics(key, config, pretrained=pretrained)
    if variant == "dense_last_layer":
        return _init_dense_last_layer_dynamics(key, config, pretrained=pretrained)
    if variant == "dense_tiny_lora":
        return _init_tiny_lora_dynamics(key, config, pretrained=pretrained)

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

    if config["dynamics"]["frozen"]:
        def predict(params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
            return dynamics_net.apply(mean_params, jnp.concatenate([z, action], axis=-1))
        return Dynamics(predict=predict), {}

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

    # Frozen prefix params (all layers except the last), closed over at construction time.
    # The last layer params are returned as trainable.
    dense_last = f"Dense_{last_i}"
    ln_last    = f"LayerNorm_{last_i}"
    frozen_prefix = {k: v for k, v in p.items() if k not in (dense_last, ln_last)}

    last_params = {
        "kernel":   p[dense_last]["kernel"],
        "bias":     p[dense_last]["bias"],
        "ln_scale": p[ln_last]["scale"],
        "ln_bias":  p[ln_last]["bias"],
    }

    def _forward(last_p: Any, x: jnp.ndarray) -> jnp.ndarray:
        # Reconstruct full Flax params dict so the forward pass is numerically
        # identical to _init_dense_dynamics (same LayerNorm implementation).
        full_params = {
            "params": {
                **frozen_prefix,
                dense_last: {"kernel": last_p["kernel"], "bias": last_p["bias"]},
                ln_last:    {"scale": last_p["ln_scale"], "bias": last_p["ln_bias"]},
            }
        }
        return dynamics_net.apply(full_params, x)

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), last_params


def _init_lora_xs_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    LoRA-XS dynamics with learnable projection subspace (P, Q) and freeze control.

    Effective weight: W_eff = W + P @ R @ Q^T
    (code convention: x @ W, so W is (d_in, d_out), P is (d_in, rank), Q is (d_out, rank))

    P and Q are trainable parameters that co-adapt with W during meta-learning,
    keeping the adapter subspace aligned with the backbone. Only R is adapted
    in the inner loop and at deployment time.

    All adapter parameters (P, Q, R) are stored flat under the "adapter" key.

    config["dynamics"]:
        type:              str, "dense_lora_xs"
        dynamics_features: list[int]
        simnorm_dim_v, simnorm_tau: same as dense variant
        rank:              int, bottleneck rank r
        r_init_std:        float, std for R init (P and Q init via SVD of W)
        adapt_layers:      list[int], indices of layers to adapt
        frozen_backbone:   bool — true: W/b/ln in closure; false: in params["backbone"]
        frozen_adapter:    bool — must be false; raises if true
        freeze_subspace:   bool
            false (pretraining): P, Q init via SVD of W, placed in params["adapter"] as learnable
            true  (finetuning):  P, Q baked into closure — from pretrained["adapter"] if available,
                                 else computed via SVD of backbone W (e.g. dense TDMPC2 baseline)

    Returned params["mean"]["dynamics"]:
        frozen_backbone=false, freeze_subspace=false -> {"backbone": {...}, "adapter": {P_i, Q_i, R_i, ...}}
        frozen_backbone=true,  freeze_subspace=true  -> {"adapter": {R_i, ...}}
        frozen_backbone=true,  freeze_subspace=false -> {"adapter": {P_i, Q_i, R_i, ...}}
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]
    rank: int = dyn_cfg["rank"]
    r_init_std: float = dyn_cfg["r_init_std"]
    adapt_layers: set = set(dyn_cfg["adapt_layers"])
    frozen_backbone: bool = dyn_cfg["frozen_backbone"]
    frozen_adapter: bool = dyn_cfg["frozen_adapter"]
    freeze_subspace: bool = dyn_cfg["freeze_subspace"]

    if frozen_adapter:
        raise ValueError("frozen_adapter=true is invalid for dense_lora_xs — nothing to adapt.")

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    # Build initial backbone weights.
    if pretrained is not None and "backbone" in pretrained:
        # Loading from a LoRA-XS checkpoint (backbone dict with W/b/ln_scale/ln_bias per layer).
        def _get_layer(i):
            bl = pretrained["backbone"][f"layer_{i}"]
            return bl["W"], bl["b"], bl["ln_scale"], bl["ln_bias"]
    elif pretrained is not None and "params" in pretrained:
        # Loading from a dense Flax checkpoint.
        pretrained_dyn = pretrained["params"]
        def _get_layer(i):
            return (
                pretrained_dyn[f"Dense_{i}"]["kernel"],
                pretrained_dyn[f"Dense_{i}"]["bias"],
                pretrained_dyn[f"LayerNorm_{i}"]["scale"],
                pretrained_dyn[f"LayerNorm_{i}"]["bias"],
            )
    else:
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

        key, k_init = jax.random.split(key)
        init_p = _DynamicsNet().init(k_init, jnp.ones((latent_dim + dim_action,)))["params"]
        def _get_layer(i):
            return (
                init_p[f"Dense_{i}"]["kernel"],
                init_p[f"Dense_{i}"]["bias"],
                init_p[f"LayerNorm_{i}"]["scale"],
                init_p[f"LayerNorm_{i}"]["bias"],
            )

    frozen_layers = []
    backbone_params = {}
    adapter_params = {}

    # Compute input dims per layer for P/Q shapes.
    in_dims = [latent_dim + dim_action] + list(features[:-1])

    for i in range(len(features)):
        W, b, ln_scale, ln_bias = _get_layer(i)

        layer = {"adapted": i in adapt_layers}

        if frozen_backbone:
            layer.update({"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias})
        else:
            backbone_params[f"layer_{i}"] = {
                "W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias
            }

        if i in adapt_layers:
            if freeze_subspace:
                if pretrained is not None and "adapter" in pretrained:
                    # Load P, Q from a pretrained LoRA-XS checkpoint (MAML/FOMAML).
                    P = pretrained["adapter"][f"P_{i}"]
                    Q = pretrained["adapter"][f"Q_{i}"]
                else:
                    # No pretrained adapter (e.g. dense TDMPC2 baseline): init via SVD
                    # of the backbone W. Absorb Σ into P to match original LoRA-XS scaling.
                    U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
                    P = U_full[:, :rank] * S_full[:rank]
                    Q = Vh_full[:rank, :].T
                layer.update({"P": P, "Q": Q})
            else:
                # Learnable subspace (pretraining): init P, Q via SVD of backbone W,
                # absorbing Σ into P so initial scaling matches original LoRA-XS.
                U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
                adapter_params[f"P_{i}"] = U_full[:, :rank] * S_full[:rank]
                adapter_params[f"Q_{i}"] = Vh_full[:rank, :].T

            if pretrained is not None and "adapter" in pretrained and f"R_{i}" in pretrained["adapter"]:
                adapter_params[f"R_{i}"] = pretrained["adapter"][f"R_{i}"]
            else:
                key, kr = jax.random.split(key)
                adapter_params[f"R_{i}"] = jax.random.normal(kr, (rank, rank)) * r_init_std

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward(params: Any, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(frozen_layers):
            if frozen_backbone:
                W = layer["W"]
                b = layer["b"]
                ln_scale = layer["ln_scale"]
                ln_bias = layer["ln_bias"]
            else:
                bl = params["backbone"][f"layer_{i}"]
                W = bl["W"]
                b = bl["b"]
                ln_scale = bl["ln_scale"]
                ln_bias = bl["ln_bias"]

            if layer["adapted"]:
                if freeze_subspace:
                    P, Q = layer["P"], layer["Q"]
                else:
                    P = params["adapter"][f"P_{i}"]
                    Q = params["adapter"][f"Q_{i}"]
                R = params["adapter"][f"R_{i}"]
                W_eff = W + P @ R @ Q.T
                x = x @ W_eff + b
            else:
                x = x @ W + b

            x = ln_scale * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + ln_bias
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    dyn_params = {}
    if not frozen_backbone:
        dyn_params["backbone"] = backbone_params
    dyn_params["adapter"] = adapter_params

    return Dynamics(predict=predict), dyn_params


def _init_tiny_lora_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    TinyLoRA dynamics: frozen pretrained MLP with trainable steering vectors.

    Like LoRA-XS but replaces the r×r trainable R matrix with a steering vector
    v of dimension steering_dim combined with frozen random projection matrices P:
        Delta = einsum("u,urk->rk", v, P)   # (r, r)
        W_eff = W + U @ diag(Sigma) @ Delta @ V^T

    P is fixed at init: shape (steering_dim, r, r), scale 1/sqrt(steering_dim * r).
    v is initialized to zeros so W_eff = W at construction time.

    Param count: num_layers * steering_dim (e.g. 4 layers, s=8 → 32 params).

    config["dynamics"]:
        type:              str, "dense_tiny_lora"
        dynamics_features: list[int], must match pretrained architecture
        simnorm_dim_v, simnorm_tau: same as dense variant
        svd_rank:          int, LoRA rank r
        steering_dim:      int, dimension of steering vector s
        projection_seed:   int, seed for frozen random projections P

    Returns dyn_params = {"v_0": ..., "v_1": ..., ...} for adapted layers only.
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]
    svd_rank: int = dyn_cfg["svd_rank"]
    steering_dim: int = dyn_cfg["steering_dim"]
    projection_seed: int = dyn_cfg["projection_seed"]
    adapt_layers: set = set(dyn_cfg["adapt_layers"])

    latent_dim: int = config["encoder"]["encoder_features"][-1]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    pretrained_dyn = pretrained["params"]
    proj_key = jax.random.key(projection_seed)

    frozen_layers = []
    v_init = {}
    for i in range(len(features)):
        W = pretrained_dyn[f"Dense_{i}"]["kernel"]
        b = pretrained_dyn[f"Dense_{i}"]["bias"]
        ln_scale = pretrained_dyn[f"LayerNorm_{i}"]["scale"]
        ln_bias = pretrained_dyn[f"LayerNorm_{i}"]["bias"]

        layer = {"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias, "adapted": i in adapt_layers}

        if i in adapt_layers:
            U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
            proj_key, pk = jax.random.split(proj_key)
            P = jax.random.normal(pk, (steering_dim, svd_rank, svd_rank)) / jnp.sqrt(steering_dim * svd_rank)
            layer.update({
                "U": U_full[:, :svd_rank],
                "Sigma": S_full[:svd_rank],
                "V": Vh_full[:svd_rank, :].T,
                "P": P,
            })
            v_init[f"v_{i}"] = jnp.zeros(steering_dim)

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward(v_params: Any, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(frozen_layers):
            if layer["adapted"]:
                v = v_params[f"v_{i}"]
                Delta = jnp.einsum("u,urk->rk", v, layer["P"])
                W_eff = layer["W"] + layer["U"] @ (layer["Sigma"][:, None] * Delta) @ layer["V"].T
                x = x @ W_eff + layer["b"]
            else:
                x = x @ layer["W"] + layer["b"]
            x = layer["ln_scale"] * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + layer["ln_bias"]
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), v_init
