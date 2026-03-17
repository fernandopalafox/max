# dynamics.py

import jax
import jax.numpy as jnp
import flax.linen as nn
import pickle
from typing import Sequence, NamedTuple, Callable, Any, Optional
from max.normalizers import Normalizer, init_normalizer


# --- Container for the final dynamics model ---
class DynamicsModel(NamedTuple):
    pred_one_step: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    pred_norm_delta: Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]
    pred_one_step_with_info: Optional[Callable[[Any, jnp.ndarray, jnp.ndarray], tuple]] = None
    encode: Optional[Callable[[Any, jnp.ndarray], jnp.ndarray]] = None
    infer_dynamics: Optional[Callable[[Any, jnp.ndarray, jnp.ndarray], jnp.ndarray]] = None
    decode: Optional[Callable[[Any, jnp.ndarray], jnp.ndarray]] = None


def _compute_svd_components(W: jnp.ndarray, svd_rank: int) -> dict:
    """
    Compute truncated SVD of weight matrix.

    Args:
        W: Weight matrix (in_features, out_features)
        svd_rank: Rank of truncation (r)

    Returns:
        Dict with U (in_features, r), Sigma (r,), V (out_features, r)
    """
    U_full, S_full, Vh_full = jnp.linalg.svd(W, full_matrices=False)
    U = U_full[:, :svd_rank]
    Sigma = S_full[:svd_rank]
    V = Vh_full[:svd_rank, :].T
    return {"U": U, "Sigma": Sigma, "V": V}


def _init_latent_lora_layers(
    key: jax.Array,
    dynamics_features: Sequence[int],
    input_dim: int,
    pretrained_dyn_params: dict,
    svd_rank: int,
    r_init_std: float = 1e-5,
) -> tuple[list, dict]:
    """
    Initialize LoRA-XS layers for the latent dynamics MLP (Mish + LayerNorm architecture).

    Args:
        key: JAX random key for R matrix initialization
        dynamics_features: Hidden/output layer sizes (e.g. [512, 512, 512])
        input_dim: Input dimension (latent_dim + action_dim)
        pretrained_dyn_params: Dict of pretrained dynamics params (the "params" sub-dict,
            containing Dense_i, LayerNorm_i keys)
        svd_rank: Rank r for all layers
        r_init_std: Std for Gaussian init of R matrices

    Returns:
        frozen_dyn_layers: List of dicts with W, b, U, Sigma, V, ln_scale, ln_bias
        trainable_R_init: Dict {"R_0": ..., "R_1": ..., ...}
    """
    frozen_dyn_layers = []
    trainable_R_init = {}

    for i in range(len(dynamics_features)):
        W = pretrained_dyn_params[f"Dense_{i}"]["kernel"]
        b = pretrained_dyn_params[f"Dense_{i}"]["bias"]
        ln_scale = pretrained_dyn_params[f"LayerNorm_{i}"]["scale"]
        ln_bias = pretrained_dyn_params[f"LayerNorm_{i}"]["bias"]

        svd_components = _compute_svd_components(W, svd_rank)

        frozen_dyn_layers.append({
            "W": W,
            "b": b,
            "U": svd_components["U"],
            "Sigma": svd_components["Sigma"],
            "V": svd_components["V"],
            "ln_scale": ln_scale,
            "ln_bias": ln_bias,
        })

        key, layer_key = jax.random.split(key)
        R = jax.random.normal(layer_key, (svd_rank, svd_rank)) * r_init_std
        trainable_R_init[f"R_{i}"] = R.astype(jnp.float32)

    return frozen_dyn_layers, trainable_R_init

def _latent_lora_dynamics_forward(
    x: jnp.ndarray,
    R_params: dict,
    frozen_dyn_layers: list,
    simnorm_fn,
) -> jnp.ndarray:
    """
    Forward pass through the latent dynamics LoRA-XS MLP (Mish + LayerNorm + SimNorm).

    Args:
        x: Input tensor (concatenated latent + action)
        R_params: Dict with R matrices {"R_0": ..., "R_1": ..., ...}
        frozen_dyn_layers: List of frozen layer dicts from _init_latent_lora_layers
        simnorm_fn: SimNorm function to apply on the final layer output

    Returns:
        Next latent state
    """
    n_layers = len(frozen_dyn_layers)

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    for i, layer in enumerate(frozen_dyn_layers):
        R = R_params[f"R_{i}"]

        # Compute effective weight: W + U @ diag(Sigma) @ R @ V.T
        SigmaR = layer["Sigma"][:, None] * R
        W_eff = layer["W"] + layer["U"] @ SigmaR @ layer["V"].T

        x = x @ W_eff + layer["b"]

        # LayerNorm (manual inline)
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / jnp.sqrt(var + 1e-6)
        x = layer["ln_scale"] * x_norm + layer["ln_bias"]

        if i < n_layers - 1:
            x = _mish(x)
        else:
            x = simnorm_fn(x)

    return x

def create_latent_dynamics(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    TD-MPC2-inspired latent dynamics model: encoder -> dynamics MLP -> decoder.

    Architecture:
      - Encoder:   NormedLinear blocks (Dense->LayerNorm->Mish), final layer uses SimNorm
      - Dynamics:  NormedLinear blocks (Dense->LayerNorm->Mish), final layer uses SimNorm
      - Decoder:   NormedLinear blocks (Dense->LayerNorm->Mish), final layer is plain Dense

    SimNorm projects the latent into L simplices of dimension V via softmax.
    NormedLinear = Dense -> LayerNorm -> Mish.

    pred_one_step: normalizes obs -> encodes -> dynamics -> decodes -> raw obs.

    config["dynamics_params"]:
        - encoder_features:   list[int], e.g. [256, 512]  (last entry = latent_dim)
        - dynamics_features:  list[int], e.g. [512, 512, 512] (last entry = latent_dim)
        - decoder_features:   list[int], e.g. [512, 256]
        - simnorm_dim_v:      int, simplex dimension V (latent_dim must be divisible by V)
        - simnorm_tau:        float, softmax temperature (default 1.0)
        - pretrained_params_path: optional path to load pretrained params
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    dyn_params = config["dynamics_params"]
    encoder_features = dyn_params["encoder_features"]
    dynamics_features = dyn_params["dynamics_features"]
    decoder_features = dyn_params["decoder_features"]
    simnorm_dim_v = dyn_params["simnorm_dim_v"]
    simnorm_tau = dyn_params.get("simnorm_tau", 1.0)

    latent_dim = encoder_features[-1]
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )
    assert dynamics_features[-1] == latent_dim, (
        f"dynamics output dim ({dynamics_features[-1]}) must equal latent_dim ({latent_dim})"
    )

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    def _simnorm(x):
        shape = x.shape
        L = shape[-1] // simnorm_dim_v
        x = x.reshape(*shape[:-1], L, simnorm_dim_v)
        x = jax.nn.softmax(x / simnorm_tau, axis=-1)
        return x.reshape(shape)

    class _Encoder(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in encoder_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(encoder_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _DynamicsNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in dynamics_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(dynamics_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _Decoder(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in decoder_features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            return nn.Dense(dim_state)(x)

    encoder_net = _Encoder()
    dynamics_net = _DynamicsNet()
    decoder_net = _Decoder()

    dummy_norm_state = jnp.ones((dim_state,))
    dummy_action = jnp.ones((dim_action,))
    dummy_z = jnp.ones((latent_dim,))

    pretrained_path = dyn_params.get("pretrained_params_path")
    if pretrained_path:
        with open(pretrained_path, "rb") as f:
            pretrained = pickle.load(f)
        model_params = pretrained["model"]
        print(f"📦 Loaded pretrained latent_dynamics weights from {pretrained_path}")
    else:
        key, k1, k2, k3 = jax.random.split(key, 4)
        encoder_params = encoder_net.init(k1, dummy_norm_state)
        dynamics_params = dynamics_net.init(
            k2, jnp.concatenate([dummy_z, dummy_action], axis=-1)
        )
        decoder_params = decoder_net.init(k3, dummy_z)
        model_params = {
            "encoder": encoder_params,
            "dynamics": dynamics_params,
            "decoder": decoder_params,
        }

    params = {"model": model_params, "normalizer": normalizer_params}

    @jax.jit
    def encode(params: Any, norm_state: jnp.ndarray) -> jnp.ndarray:
        return encoder_net.apply(params["model"]["encoder"], norm_state)

    @jax.jit
    def infer_dynamics(params: Any, z: jnp.ndarray, norm_action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_net.apply(
            params["model"]["dynamics"],
            jnp.concatenate([z, norm_action], axis=-1)
        )

    @jax.jit
    def decode(params: Any, z: jnp.ndarray) -> jnp.ndarray:
        return decoder_net.apply(params["model"]["decoder"], z)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns normalized observation-space delta (decoder output)."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        return decode(params, z_next)

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the absolute next state."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        norm_next_state = decode(params, z_next)
        return normalizer.unnormalize(norm_params["state"], norm_next_state)

    return DynamicsModel(
        pred_one_step=pred_one_step,
        pred_norm_delta=pred_norm_delta,
        encode=encode,
        infer_dynamics=infer_dynamics,
        decode=decode,
    ), params


def create_latent_dynamics_frozen_encdec(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Latent dynamics model with frozen encoder and decoder.

    Loads a pretrained latent_dynamics checkpoint and exposes only the dynamics
    MLP as trainable parameters. The encoder and decoder are frozen in the closure.

    Use case: Fine-tuning the latent transition function on new data while keeping
    the encoder/decoder fixed.

    config["dynamics_params"]:
        - encoder_features, dynamics_features, decoder_features, simnorm_dim_v,
          simnorm_tau: same as create_latent_dynamics
        - pretrained_params_path: REQUIRED path to pretrained latent_dynamics params

    Returns:
        (DynamicsModel, params) where params["model"] contains only {"dynamics": ...}
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    dyn_params = config["dynamics_params"]
    encoder_features = dyn_params["encoder_features"]
    dynamics_features = dyn_params["dynamics_features"]
    decoder_features = dyn_params["decoder_features"]
    simnorm_dim_v = dyn_params["simnorm_dim_v"]
    simnorm_tau = dyn_params.get("simnorm_tau", 1.0)

    pretrained_path = dyn_params.get("pretrained_params_path")
    assert pretrained_path is not None, (
        "create_latent_dynamics_frozen_encdec requires pretrained_params_path — "
        "there is no meaning to freezing a randomly initialized encoder/decoder."
    )

    latent_dim = encoder_features[-1]
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )
    assert dynamics_features[-1] == latent_dim, (
        f"dynamics output dim ({dynamics_features[-1]}) must equal latent_dim ({latent_dim})"
    )

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    def _simnorm(x):
        shape = x.shape
        L = shape[-1] // simnorm_dim_v
        x = x.reshape(*shape[:-1], L, simnorm_dim_v)
        x = jax.nn.softmax(x / simnorm_tau, axis=-1)
        return x.reshape(shape)

    class _Encoder(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in encoder_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(encoder_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _DynamicsNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in dynamics_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(dynamics_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _Decoder(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in decoder_features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            return nn.Dense(dim_state)(x)

    encoder_net = _Encoder()
    dynamics_net = _DynamicsNet()
    decoder_net = _Decoder()

    with open(pretrained_path, "rb") as f:
        pretrained = pickle.load(f)
    print(f"📦 Loaded pretrained latent_dynamics weights from {pretrained_path}")

    frozen_encoder_params = pretrained["model"]["encoder"]
    frozen_decoder_params = pretrained["model"]["decoder"]
    dynamics_params = pretrained["model"]["dynamics"]

    params = {"model": {"dynamics": dynamics_params}, "normalizer": normalizer_params}

    @jax.jit
    def encode(params: Any, norm_state: jnp.ndarray) -> jnp.ndarray:
        # Uses frozen encoder from closure; ignores params["model"]
        return encoder_net.apply(frozen_encoder_params, norm_state)

    @jax.jit
    def infer_dynamics(params: Any, z: jnp.ndarray, norm_action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_net.apply(
            params["model"]["dynamics"],
            jnp.concatenate([z, norm_action], axis=-1)
        )

    @jax.jit
    def decode(params: Any, z: jnp.ndarray) -> jnp.ndarray:
        # Uses frozen decoder from closure; ignores params["model"]
        return decoder_net.apply(frozen_decoder_params, z)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns normalized observation-space delta (decoder output)."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        return decode(params, z_next)

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the absolute next state."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        norm_next_state = decode(params, z_next)
        return normalizer.unnormalize(norm_params["state"], norm_next_state)

    return DynamicsModel(
        pred_one_step=pred_one_step,
        pred_norm_delta=pred_norm_delta,
        encode=encode,
        infer_dynamics=infer_dynamics,
        decode=decode,
    ), params


def create_latent_dynamics_frozen_except_last(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Latent dynamics model with only the last dynamics layer trainable.

    Loads a pretrained latent_dynamics checkpoint and exposes only the final
    Dense + LayerNorm of the dynamics MLP as trainable parameters. Encoder,
    decoder, and all earlier dynamics layers are frozen in the closure.

    config["dynamics_params"]:
        - encoder_features, dynamics_features, decoder_features, simnorm_dim_v,
          simnorm_tau: same as create_latent_dynamics
        - pretrained_params_path: REQUIRED path to pretrained latent_dynamics params

    Returns:
        (DynamicsModel, params) where params["model"] contains only {"dynamics_last": ...}
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    dyn_params = config["dynamics_params"]
    encoder_features = dyn_params["encoder_features"]
    dynamics_features = dyn_params["dynamics_features"]
    decoder_features = dyn_params["decoder_features"]
    simnorm_dim_v = dyn_params["simnorm_dim_v"]
    simnorm_tau = dyn_params.get("simnorm_tau", 1.0)

    pretrained_path = dyn_params.get("pretrained_params_path")
    assert pretrained_path is not None, (
        "create_latent_dynamics_frozen_except_last requires pretrained_params_path — "
        "there is no meaning to freezing a randomly initialized model."
    )

    latent_dim = encoder_features[-1]
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )
    assert dynamics_features[-1] == latent_dim, (
        f"dynamics output dim ({dynamics_features[-1]}) must equal latent_dim ({latent_dim})"
    )

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    def _simnorm(x):
        shape = x.shape
        L = shape[-1] // simnorm_dim_v
        x = x.reshape(*shape[:-1], L, simnorm_dim_v)
        x = jax.nn.softmax(x / simnorm_tau, axis=-1)
        return x.reshape(shape)

    class _Encoder(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in encoder_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(encoder_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _DynamicsNet(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in dynamics_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(dynamics_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _Decoder(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in decoder_features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            return nn.Dense(dim_state)(x)

    encoder_net = _Encoder()
    dynamics_net = _DynamicsNet()
    decoder_net = _Decoder()

    with open(pretrained_path, "rb") as f:
        pretrained = pickle.load(f)
    print(f"📦 Loaded pretrained latent_dynamics weights from {pretrained_path}")

    frozen_encoder_params = pretrained["model"]["encoder"]
    frozen_decoder_params = pretrained["model"]["decoder"]

    n = len(dynamics_features)
    last_layer_name = f"Dense_{n-1}"
    last_ln_name = f"LayerNorm_{n-1}"

    pretrained_dyn = pretrained["model"]["dynamics"]["params"]
    frozen_dynamics_params = {"params": {
        k: v for k, v in pretrained_dyn.items()
        if k not in (last_layer_name, last_ln_name)
    }}
    trainable_dynamics_params = {"params": {
        last_layer_name: pretrained_dyn[last_layer_name],
        last_ln_name: pretrained_dyn[last_ln_name],
    }}

    params = {"model": {"dynamics_last": trainable_dynamics_params}, "normalizer": normalizer_params}

    @jax.jit
    def encode(params: Any, norm_state: jnp.ndarray) -> jnp.ndarray:
        return encoder_net.apply(frozen_encoder_params, norm_state)

    @jax.jit
    def infer_dynamics(params: Any, z: jnp.ndarray, norm_action: jnp.ndarray) -> jnp.ndarray:
        full_dyn_params = {"params": {
            **frozen_dynamics_params["params"],
            **params["model"]["dynamics_last"]["params"]
        }}
        return dynamics_net.apply(full_dyn_params, jnp.concatenate([z, norm_action], axis=-1))

    @jax.jit
    def decode(params: Any, z: jnp.ndarray) -> jnp.ndarray:
        return decoder_net.apply(frozen_decoder_params, z)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns normalized observation-space delta (decoder output)."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        return decode(params, z_next)

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Predicts the absolute next state."""
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        norm_next_state = decode(params, z_next)
        return normalizer.unnormalize(norm_params["state"], norm_next_state)

    return DynamicsModel(
        pred_one_step=pred_one_step,
        pred_norm_delta=pred_norm_delta,
        encode=encode,
        infer_dynamics=infer_dynamics,
        decode=decode,
    ), params


def create_latent_dynamics_lora(
    key: jax.Array,
    config: dict,
    normalizer: Normalizer,
    normalizer_params: Optional[jnp.ndarray] = None,
) -> tuple[NamedTuple, dict]:
    """
    Latent dynamics model with LoRA-XS adaptation on the dynamics MLP only.

    Encoder and decoder are fully frozen in the closure. Only R matrices
    (one per dynamics Dense layer, shape r×r) are trainable.

    Param count: 3r² for 3-layer dynamics (e.g. r=32 → 3,072 params).

    config["dynamics_params"]:
        - encoder_features, dynamics_features, decoder_features, simnorm_dim_v,
          simnorm_tau: same as create_latent_dynamics
        - svd_rank: LoRA rank r (default 32)
        - r_init_std: Init std for R matrices (default 1e-5)
        - pretrained_params_path: REQUIRED path to pretrained latent_dynamics params
    """
    dim_state = config["dim_state"]
    dim_action = config["dim_action"]
    dyn_params = config["dynamics_params"]
    encoder_features = dyn_params["encoder_features"]
    dynamics_features = dyn_params["dynamics_features"]
    decoder_features = dyn_params["decoder_features"]
    simnorm_dim_v = dyn_params["simnorm_dim_v"]
    simnorm_tau = dyn_params.get("simnorm_tau", 1.0)
    svd_rank = dyn_params.get("svd_rank", 32)
    r_init_std = dyn_params.get("r_init_std", 1e-5)

    pretrained_path = dyn_params.get("pretrained_params_path")
    assert pretrained_path is not None, (
        "create_latent_dynamics_lora requires pretrained_params_path — "
        "there is no meaning to LoRA-adapting a randomly initialized model."
    )

    latent_dim = encoder_features[-1]
    assert latent_dim % simnorm_dim_v == 0, (
        f"latent_dim={latent_dim} must be divisible by simnorm_dim_v={simnorm_dim_v}"
    )
    assert dynamics_features[-1] == latent_dim, (
        f"dynamics output dim ({dynamics_features[-1]}) must equal latent_dim ({latent_dim})"
    )

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    def _simnorm(x):
        shape = x.shape
        L = shape[-1] // simnorm_dim_v
        x = x.reshape(*shape[:-1], L, simnorm_dim_v)
        x = jax.nn.softmax(x / simnorm_tau, axis=-1)
        return x.reshape(shape)

    class _Encoder(nn.Module):
        @nn.compact
        def __call__(self, x):
            for feat in encoder_features[:-1]:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            x = nn.Dense(encoder_features[-1])(x)
            x = nn.LayerNorm()(x)
            return _simnorm(x)

    class _Decoder(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in decoder_features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = _mish(x)
            return nn.Dense(dim_state)(x)

    encoder_net = _Encoder()
    decoder_net = _Decoder()

    with open(pretrained_path, "rb") as f:
        pretrained = pickle.load(f)
    print(f"📦 Loaded pretrained latent_dynamics weights from {pretrained_path}")

    frozen_encoder_params = pretrained["model"]["encoder"]
    frozen_decoder_params = pretrained["model"]["decoder"]
    pretrained_dyn_params = pretrained["model"]["dynamics"]["params"]

    input_dim = latent_dim + dim_action
    frozen_dyn_layers, trainable_R_init = _init_latent_lora_layers(
        key, dynamics_features, input_dim, pretrained_dyn_params, svd_rank, r_init_std
    )

    params = {
        "model": {"dynamics_lora": trainable_R_init},
        "normalizer": normalizer_params,
    }

    @jax.jit
    def encode(params: Any, norm_state: jnp.ndarray) -> jnp.ndarray:
        return encoder_net.apply(frozen_encoder_params, norm_state)

    @jax.jit
    def infer_dynamics(params: Any, z: jnp.ndarray, norm_action: jnp.ndarray) -> jnp.ndarray:
        x = jnp.concatenate([z, norm_action], axis=-1)
        return _latent_lora_dynamics_forward(
            x, params["model"]["dynamics_lora"], frozen_dyn_layers, _simnorm
        )

    @jax.jit
    def decode(params: Any, z: jnp.ndarray) -> jnp.ndarray:
        return decoder_net.apply(frozen_decoder_params, z)

    @jax.jit
    def pred_norm_delta(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        return decode(params, z_next)

    @jax.jit
    def pred_one_step(
        params: Any, state: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        norm_params = params["normalizer"]
        norm_state = normalizer.normalize(norm_params["state"], state)
        norm_action = normalizer.normalize(norm_params["action"], action)
        z = encode(params, norm_state)
        z_next = infer_dynamics(params, z, norm_action)
        norm_next_state = decode(params, z_next)
        return normalizer.unnormalize(norm_params["state"], norm_next_state)

    return DynamicsModel(
        pred_one_step=pred_one_step,
        pred_norm_delta=pred_norm_delta,
        encode=encode,
        infer_dynamics=infer_dynamics,
        decode=decode,
    ), params



def init_dynamics(
    key: jax.Array,
    config: Any,
    normalizer: Normalizer = None,
    normalizer_params=None,
    encoder=None,
) -> tuple[Any, Any]:
    """
    Initializes the appropriate dynamics model based on the configuration.

    When `encoder` is provided (TDMPC2 path), creates a dynamics model whose
    parameters have structure {"mean": ...} and whose infer_dynamics signature
    is (dyn_params, z, action) -> z_next  (action normalization baked in).

    When `encoder` is None (legacy path), creates the original DynamicsModel.
    """
    dynamics_type = config["dynamics"]
    print(f"🚀 Initializing dynamics model: {dynamics_type.upper()}")

    if normalizer is None:
        normalizer, normalizer_params = init_normalizer(config)

    if encoder is not None:
        # TDMPC2 path: separate encoder, dynamics-only params
        return _create_dynamics_tdmpc2(key, config, encoder, normalizer, normalizer_params)

    if dynamics_type == "latent_dynamics":
        model, params = create_latent_dynamics(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "latent_dynamics_frozen_encdec":
        model, params = create_latent_dynamics_frozen_encdec(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "latent_dynamics_frozen_except_last":
        model, params = create_latent_dynamics_frozen_except_last(
            key, config, normalizer, normalizer_params
        )

    elif dynamics_type == "latent_dynamics_lora":
        model, params = create_latent_dynamics_lora(
            key, config, normalizer, normalizer_params
        )

    else:
        raise ValueError(f"Unknown dynamics type: '{dynamics_type}'")

    return model, params


def _create_dynamics_tdmpc2(
    key: jax.Array,
    config: Any,
    encoder,
    normalizer: Normalizer,
    normalizer_params: dict,
) -> tuple[Any, dict]:
    """
    TDMPC2-compatible dynamics model.

    Supports both "latent_dynamics" (fresh dense MLP) and "latent_dynamics_lora"
    (LoRA-XS on pretrained). The encoder/decoder are owned by the Encoder object
    (not here). Only the dynamics MLP (or LoRA R matrices) are returned as params.

    Returns:
        (DynamicsModel, {"mean": dyn_params})

    DynamicsModel.infer_dynamics(dyn_params, z, action) -> z_next
        - dyn_params  = parameters["dynamics"]["mean"]
        - action      = raw (un-normalized); normalization baked in via closure
    DynamicsModel.pred_one_step(parameters, obs, action) -> obs_next
        - parameters  = full unified parameters dict (uses parameters["encoder"])
    """
    dynamics_type = config["dynamics"]
    dyn_cfg = config["dynamics_params"]
    encoder_features = dyn_cfg["encoder_features"]
    dynamics_features = dyn_cfg["dynamics_features"]
    simnorm_dim_v = dyn_cfg["simnorm_dim_v"]
    simnorm_tau = dyn_cfg.get("simnorm_tau", 1.0)

    latent_dim = encoder_features[-1]
    dim_action = config["dim_action"]
    action_norm_params = normalizer_params["action"]
    state_norm_params = normalizer_params["state"]

    def _mish(x):
        return x * jnp.tanh(jax.nn.softplus(x))

    def _simnorm(x):
        shape = x.shape
        L = shape[-1] // simnorm_dim_v
        x = x.reshape(*shape[:-1], L, simnorm_dim_v)
        x = jax.nn.softmax(x / simnorm_tau, axis=-1)
        return x.reshape(shape)

    if dynamics_type == "latent_dynamics":
        # Fresh dense dynamics MLP
        class _DynamicsNet(nn.Module):
            @nn.compact
            def __call__(self, x):
                for feat in dynamics_features[:-1]:
                    x = nn.Dense(feat)(x)
                    x = nn.LayerNorm()(x)
                    x = _mish(x)
                x = nn.Dense(dynamics_features[-1])(x)
                x = nn.LayerNorm()(x)
                return _simnorm(x)

        dynamics_net = _DynamicsNet()
        dummy_x = jnp.ones((latent_dim + dim_action,))
        key, k1 = jax.random.split(key)
        dense_params = dynamics_net.init(k1, dummy_x)
        dyn_params = {"mean": dense_params}

        def _infer_dynamics_inner(mean_params, z, norm_action):
            x = jnp.concatenate([z, norm_action], axis=-1)
            return dynamics_net.apply(mean_params, x)

    elif dynamics_type == "latent_dynamics_lora":
        # LoRA-XS on pretrained dynamics
        svd_rank = dyn_cfg.get("svd_rank", 32)
        r_init_std = dyn_cfg.get("r_init_std", 1e-5)
        pretrained_path = dyn_cfg.get("pretrained_params_path")
        assert pretrained_path is not None, (
            "_create_dynamics_tdmpc2 with latent_dynamics_lora requires pretrained_params_path"
        )
        with open(pretrained_path, "rb") as f:
            import pickle
            pretrained = pickle.load(f)
        pretrained_dyn_params = pretrained["model"]["dynamics"]["params"]
        input_dim = latent_dim + dim_action
        frozen_dyn_layers, trainable_R_init = _init_latent_lora_layers(
            key, dynamics_features, input_dim, pretrained_dyn_params, svd_rank, r_init_std
        )
        dyn_params = {"mean": trainable_R_init}

        def _infer_dynamics_inner(mean_params, z, norm_action):
            x = jnp.concatenate([z, norm_action], axis=-1)
            return _latent_lora_dynamics_forward(
                x, mean_params, frozen_dyn_layers, _simnorm
            )

    else:
        raise ValueError(
            f"_create_dynamics_tdmpc2 does not support dynamics type: {dynamics_type!r}"
        )

    def infer_dynamics(dyn_params: Any, z: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """
        Latent dynamics step.
        dyn_params: parameters["dynamics"]["mean"]
        z:          latent state
        action:     raw action (normalized internally)
        """
        norm_action = normalizer.normalize(action_norm_params, action)
        return _infer_dynamics_inner(dyn_params, z, norm_action)

    def pred_one_step(
        parameters: Any, obs: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Full obs-space prediction using encoder/decoder from parameters["encoder"]."""
        z = encoder.encode(parameters["encoder"], obs)
        z_next = infer_dynamics(parameters["dynamics"]["mean"], z, action)
        norm_next = encoder.decode(parameters["encoder"], z_next)
        return normalizer.unnormalize(state_norm_params, norm_next)

    def pred_norm_delta(
        parameters: Any, obs: jnp.ndarray, action: jnp.ndarray
    ) -> jnp.ndarray:
        """Returns normalized-space next observation (decoder output)."""
        z = encoder.encode(parameters["encoder"], obs)
        z_next = infer_dynamics(parameters["dynamics"]["mean"], z, action)
        return encoder.decode(parameters["encoder"], z_next)

    return DynamicsModel(
        pred_one_step=pred_one_step,
        pred_norm_delta=pred_norm_delta,
        infer_dynamics=infer_dynamics,
    ), dyn_params
