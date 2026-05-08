# dynamics_gradsvd.py
"""
LoRA-XS dynamics variant that initializes A/B from SVD of the average scaled gradient
(Adam updates accumulated during pretraining) rather than SVD of the pretrained weight.

Identical to Case 2 of _init_lora_xs_dynamics in dynamics.py except the SVD source.
Adapts from a dense pretrained checkpoint: backbone frozen, A/B frozen from gradient SVD,
fresh R trainable. Effective weight: W_eff = W + B @ R @ A^T.
"""

import jax
import jax.numpy as jnp
from typing import Any

from max.dynamics import Dynamics
from max.utilities import mish, simnorm


def init_gradsvd_lora_xs_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict,
    grad_avg: dict,
) -> tuple[Dynamics, dict]:
    """
    LoRA-XS dynamics initialized from SVD of scaled gradient rather than SVD of weights.

    Args:
        pretrained: dense Flax checkpoint with "params" key
            ({"params": {"Dense_i": {"kernel": ..., "bias": ...}, "LayerNorm_i": {...}, ...}})
        grad_avg: dict mapping "Dense_i" -> gradient array (g_sum_scaled from grad_capture.pkl).
            SVD singular vectors are scale-invariant so using the sum directly is equivalent
            to using the per-step average.

    Returns:
        (Dynamics, {"adapter": {"R_0": ..., "R_1": ..., ...}})
    """
    dyn_cfg = config["dynamics"]
    features = dyn_cfg["dynamics_features"]
    simnorm_dim_v: int = dyn_cfg["simnorm_dim_v"]
    simnorm_tau: float = dyn_cfg["simnorm_tau"]
    rank: int = dyn_cfg["rank"]
    r_init_std: float = dyn_cfg["r_init_std"]
    adapt_layers: set = set(dyn_cfg["adapt_layers"])

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_action: int = config["dim_action"]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    p = pretrained["params"]

    frozen_layers = []
    adapter_params = {}

    for i in range(len(features)):
        W = p[f"Dense_{i}"]["kernel"]
        b = p[f"Dense_{i}"]["bias"]
        ln_scale = p[f"LayerNorm_{i}"]["scale"]
        ln_bias = p[f"LayerNorm_{i}"]["bias"]

        layer = {
            "adapted": i in adapt_layers,
            "W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias,
        }

        if i in adapt_layers:
            G = jnp.array(grad_avg[f"Dense_{i}"])
            U, S, Vh = jnp.linalg.svd(G, full_matrices=False)
            layer["A"] = Vh[:rank, :].T       # (d_out, rank)
            layer["B"] = U[:, :rank] * S[:rank]  # (d_in, rank)

            key, kr = jax.random.split(key)
            adapter_params[f"R_{i}"] = jax.random.normal(kr, (rank, rank)) * r_init_std

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward_full(params: Any, x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        bh_activations = {}
        for i, layer in enumerate(frozen_layers):
            W, b, ln_scale, ln_bias = layer["W"], layer["b"], layer["ln_scale"], layer["ln_bias"]

            if layer["adapted"]:
                A = layer["A"]
                B = layer["B"]
                R = params["adapter"][f"R_{i}"]
                bh_activations[i] = x @ B
                x = x @ (W + B @ R @ A.T) + b
            else:
                x = x @ W + b

            x = ln_scale * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + ln_bias
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x, bh_activations

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        out, _ = _forward_full(mean_params, jnp.concatenate([z, action], axis=-1))
        return out

    def predict_with_activations(
        mean_params: Any, z: jnp.ndarray, action: jnp.ndarray
    ) -> tuple[jnp.ndarray, dict]:
        return _forward_full(mean_params, jnp.concatenate([z, action], axis=-1))

    dyn_params = {"adapter": adapter_params}
    return Dynamics(predict=predict, predict_with_activations=predict_with_activations), dyn_params
