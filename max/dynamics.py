# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Callable, Any, Optional

from max.normalizers import Normalizer
from max.utilities import mish, simnorm


class Dynamics(NamedTuple):
    predict: Callable                                    # (mean_params, z, action) -> next_z
    predict_with_activations: Optional[Callable] = None # (mean_params, z, action) -> (next_z, {layer_idx: Bh})


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
    if variant == "dense_lora":
        return _init_lora_dynamics(key, config, pretrained=pretrained)

    raise ValueError(f"Unknown dynamics: {variant!r}")


def create_animals_dynamics(config):
    """Analytical dominance contest dynamics with estimated θ = [q̂, K̂].

    Returns (Dynamics, init_dyn_params) where init_dyn_params = {"q": q̂, "K": K̂}.
    Dynamics.predict(params, state, action) -> next_state
    """
    q_hat_init, K_hat_init = config["ekf"]["theta_hat_init"]
    init_dyn_params = {"q": jnp.array(q_hat_init), "K": jnp.array(K_hat_init)}
    action_min = config["environment"]["action_min"]
    action_max = config["environment"]["action_max"]

    def predict(params, state, action):
        q_hat = params["q"]
        K_hat = params["K"]
        x1, x2 = state[0], state[1]
        u = action_min + (action[0] + 1.0) * 0.5 * (action_max - action_min)
        return jnp.array([x1 + u, x2 + K_hat * (q_hat - x1)])

    return Dynamics(predict=predict), init_dyn_params


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


def _init_lora_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    LoRA dynamics. Effective weight: W_eff = W + B @ A^T

    pretrained must be a dense Flax checkpoint (has "params" key).
    Backbone frozen (closure). Only adapter {A_i, B_i} is trainable.
    B init: Kaiming uniform. A init: zeros → delta = 0 at init.
    Returns {"adapter": {A_i, B_i, ...}}
    """
    dyn_cfg = config["dynamics"]
    features      = dyn_cfg["dynamics_features"]
    simnorm_dim_v = dyn_cfg["simnorm_dim_v"]
    simnorm_tau   = dyn_cfg["simnorm_tau"]
    rank          = dyn_cfg["rank"]
    adapt_layers  = set(dyn_cfg["adapt_layers"])

    p = pretrained["params"]
    frozen_layers  = []
    adapter_params = {}

    for i in range(len(features)):
        W     = p[f"Dense_{i}"]["kernel"]
        b     = p[f"Dense_{i}"]["bias"]
        ln_s  = p[f"LayerNorm_{i}"]["scale"]
        ln_b  = p[f"LayerNorm_{i}"]["bias"]
        layer = {"W": W, "b": b, "ln_scale": ln_s, "ln_bias": ln_b, "adapted": i in adapt_layers}

        if i in adapt_layers:
            d_in, d_out = W.shape
            key, kb = jax.random.split(key)
            adapter_params[f"A_{i}"] = jnp.zeros((d_out, rank))
            adapter_params[f"B_{i}"] = jax.nn.initializers.he_uniform()(kb, (d_in, rank))

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward(params, x):
        for i, layer in enumerate(frozen_layers):
            if layer["adapted"]:
                A = params["adapter"][f"A_{i}"]
                B = params["adapter"][f"B_{i}"]
                x = x @ (layer["W"] + B @ A.T) + layer["b"]
            else:
                x = x @ layer["W"] + layer["b"]
            x = layer["ln_scale"] * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + layer["ln_bias"]
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x

    def predict(params, z, action):
        return _forward(params, jnp.concatenate([z, action], axis=-1))

    return Dynamics(predict=predict), {"adapter": adapter_params}


def _init_lora_xs_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    LoRA-XS dynamics. Effective weight: W_eff = W + A @ R @ B^T

    Exactly three cases, determined by pretrained:

    Case 1 — pretrained is None (pretraining from scratch):
        Backbone and adapter (A, B, R) all trainable.
        Returns {"backbone": {...}, "adapter": {A_i, B_i, R_i, ...}}

    Case 2 — pretrained is a dense Flax checkpoint ("params" key, e.g. baseline_new):
        Backbone frozen. A, B frozen via SVD of pretrained W. Fresh R trainable.
        Returns {"adapter": {R_i, ...}}

    Case 3 — pretrained is a LoRA-XS checkpoint ("backbone" key):
        Backbone frozen. A, B frozen from pretrained adapter. Fresh R trainable.
        Returns {"adapter": {R_i, ...}}

    Any other pretrained structure raises an error.
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

    if pretrained is None:
        case = 1
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
            return init_p[f"Dense_{i}"]["kernel"], init_p[f"Dense_{i}"]["bias"], \
                   init_p[f"LayerNorm_{i}"]["scale"], init_p[f"LayerNorm_{i}"]["bias"]
    elif "params" in pretrained:
        case = 2
        p = pretrained["params"]
        def _get_layer(i):
            return p[f"Dense_{i}"]["kernel"], p[f"Dense_{i}"]["bias"], \
                   p[f"LayerNorm_{i}"]["scale"], p[f"LayerNorm_{i}"]["bias"]
    elif "backbone" in pretrained:
        case = 3
        def _get_layer(i):
            bl = pretrained["backbone"][f"layer_{i}"]
            return bl["W"], bl["b"], bl["ln_scale"], bl["ln_bias"]
    else:
        raise ValueError(
            "pretrained for dense_lora_xs must be None, a dense Flax checkpoint "
            "('params' key), or a LoRA-XS checkpoint ('backbone' key)."
        )

    frozen_layers = []
    backbone_params = {}
    adapter_params = {}

    for i in range(len(features)):
        W, b, ln_scale, ln_bias = _get_layer(i)
        layer = {"adapted": i in adapt_layers}

        if case == 1:
            backbone_params[f"layer_{i}"] = {"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias}
        else:
            layer.update({"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias})

        if i in adapt_layers:
            if case == 1:
                U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
                adapter_params[f"A_{i}"] = Vh[:rank, :].T          # (d_out, rank) — paper's A
                adapter_params[f"B_{i}"] = U[:, :rank] * S[:rank]  # (d_in,  rank) — paper's B^T
            elif case == 2:
                U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
                layer["A"] = Vh[:rank, :].T
                layer["B"] = U[:, :rank] * S[:rank]
            else:
                layer["A"] = pretrained["adapter"][f"A_{i}"]
                layer["B"] = pretrained["adapter"][f"B_{i}"]

            key, kr = jax.random.split(key)
            adapter_params[f"R_{i}"] = jax.random.normal(kr, (rank, rank)) * r_init_std

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward_full(params: Any, x: jnp.ndarray) -> tuple[jnp.ndarray, dict]:
        bh_activations = {}
        for i, layer in enumerate(frozen_layers):
            if case == 1:
                bl = params["backbone"][f"layer_{i}"]
                W, b, ln_scale, ln_bias = bl["W"], bl["b"], bl["ln_scale"], bl["ln_bias"]
            else:
                W, b, ln_scale, ln_bias = layer["W"], layer["b"], layer["ln_scale"], layer["ln_bias"]

            if layer["adapted"]:
                A = params["adapter"][f"A_{i}"] if case == 1 else layer["A"]
                B = params["adapter"][f"B_{i}"] if case == 1 else layer["B"]
                R = params["adapter"][f"R_{i}"]
                bh_activations[i] = x @ B  # bottleneck activation: z = Bh in paper notation
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

    dyn_params = {}
    if case == 1:
        dyn_params["backbone"] = backbone_params
    dyn_params["adapter"] = adapter_params

    return Dynamics(predict=predict, predict_with_activations=predict_with_activations), dyn_params


def _init_tiny_lora_dynamics(
    key: jax.Array,
    config: Any,
    pretrained: dict = None,
) -> tuple[Dynamics, dict]:
    """
    TinyLoRA dynamics. Effective weight: W_eff = W + U @ diag(Sigma) @ Delta @ V^T
    where Delta = einsum("u,urk->rk", v, P).

    Three cases, determined by pretrained (mirrors LoRA-XS):

    Case 1 — pretrained is None (pretraining from scratch):
        Backbone and adapter (U, Sigma, V, P, v) all trainable. U, Sigma, V init
        from SVD of randomly-initialized W; P drawn from projection_seed; v = 0.
        Returns {"backbone": {...}, "adapter": {U_i, Sigma_i, V_i, P_i, v_i, ...}}

    Case 2 — pretrained is a dense Flax checkpoint ("params" key):
        Backbone frozen. U, Sigma, V frozen via SVD of pretrained W. P frozen
        from projection_seed. Fresh v = 0 trainable.
        Returns {"adapter": {v_i, ...}}

    Case 3 — pretrained is a TinyLoRA checkpoint ("backbone" key):
        Backbone, U, Sigma, V, P all frozen from the pretrained checkpoint.
        v initialized from pretrained["adapter"][f"v_{i}"] (preserves pretrain signal).
        Returns {"adapter": {v_i, ...}}
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
    dim_action: int = config["dim_action"]

    assert features[-1] == latent_dim, (
        f"dynamics_features[-1]={features[-1]} must equal latent_dim={latent_dim}"
    )

    if pretrained is None:
        case = 1
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
            return init_p[f"Dense_{i}"]["kernel"], init_p[f"Dense_{i}"]["bias"], \
                   init_p[f"LayerNorm_{i}"]["scale"], init_p[f"LayerNorm_{i}"]["bias"]
    elif "params" in pretrained:
        case = 2
        p = pretrained["params"]
        def _get_layer(i):
            return p[f"Dense_{i}"]["kernel"], p[f"Dense_{i}"]["bias"], \
                   p[f"LayerNorm_{i}"]["scale"], p[f"LayerNorm_{i}"]["bias"]
    elif "backbone" in pretrained:
        case = 3
        def _get_layer(i):
            bl = pretrained["backbone"][f"layer_{i}"]
            return bl["W"], bl["b"], bl["ln_scale"], bl["ln_bias"]
    else:
        raise ValueError(
            "pretrained for dense_tiny_lora must be None, a dense Flax checkpoint "
            "('params' key), or a TinyLoRA checkpoint ('backbone' key)."
        )

    proj_key = jax.random.key(projection_seed)
    frozen_layers = []
    backbone_params = {}
    adapter_params = {}

    for i in range(len(features)):
        W, b, ln_scale, ln_bias = _get_layer(i)
        layer = {"adapted": i in adapt_layers}

        if case == 1:
            backbone_params[f"layer_{i}"] = {"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias}
        else:
            layer.update({"W": W, "b": b, "ln_scale": ln_scale, "ln_bias": ln_bias})

        if i in adapt_layers:
            if case == 1:
                U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
                proj_key, pk = jax.random.split(proj_key)
                P = jax.random.normal(pk, (steering_dim, svd_rank, svd_rank)) / jnp.sqrt(steering_dim * svd_rank)
                adapter_params[f"U_{i}"] = U_full[:, :svd_rank]
                adapter_params[f"Sigma_{i}"] = S_full[:svd_rank]
                adapter_params[f"V_{i}"] = Vh_full[:svd_rank, :].T
                adapter_params[f"P_{i}"] = P
                adapter_params[f"v_{i}"] = jnp.zeros(steering_dim)
            elif case == 2:
                U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
                proj_key, pk = jax.random.split(proj_key)
                P = jax.random.normal(pk, (steering_dim, svd_rank, svd_rank)) / jnp.sqrt(steering_dim * svd_rank)
                layer["U"] = U_full[:, :svd_rank]
                layer["Sigma"] = S_full[:svd_rank]
                layer["V"] = Vh_full[:svd_rank, :].T
                layer["P"] = P
                adapter_params[f"v_{i}"] = jnp.zeros(steering_dim)
            else:
                layer["U"] = pretrained["adapter"][f"U_{i}"]
                layer["Sigma"] = pretrained["adapter"][f"Sigma_{i}"]
                layer["V"] = pretrained["adapter"][f"V_{i}"]
                layer["P"] = pretrained["adapter"][f"P_{i}"]
                adapter_params[f"v_{i}"] = pretrained["adapter"][f"v_{i}"]

        frozen_layers.append(layer)

    n_layers = len(features)

    def _forward(params: Any, x: jnp.ndarray) -> jnp.ndarray:
        for i, layer in enumerate(frozen_layers):
            if case == 1:
                bl = params["backbone"][f"layer_{i}"]
                W, b, ln_scale, ln_bias = bl["W"], bl["b"], bl["ln_scale"], bl["ln_bias"]
            else:
                W, b, ln_scale, ln_bias = layer["W"], layer["b"], layer["ln_scale"], layer["ln_bias"]

            if layer["adapted"]:
                if case == 1:
                    U = params["adapter"][f"U_{i}"]
                    Sigma = params["adapter"][f"Sigma_{i}"]
                    V = params["adapter"][f"V_{i}"]
                    P = params["adapter"][f"P_{i}"]
                else:
                    U, Sigma, V, P = layer["U"], layer["Sigma"], layer["V"], layer["P"]
                v = params["adapter"][f"v_{i}"]
                Delta = jnp.einsum("u,urk->rk", v, P)
                W_eff = W + U @ (Sigma[:, None] * Delta) @ V.T
                x = x @ W_eff + b
            else:
                x = x @ W + b

            x = ln_scale * jax.nn.standardize(x, axis=-1, epsilon=1e-6) + ln_bias
            x = mish(x) if i < n_layers - 1 else simnorm(x, simnorm_dim_v, simnorm_tau)
        return x

    def predict(mean_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        return _forward(mean_params, jnp.concatenate([z, action], axis=-1))

    dyn_params = {}
    if case == 1:
        dyn_params["backbone"] = backbone_params
    dyn_params["adapter"] = adapter_params

    return Dynamics(predict=predict), dyn_params
