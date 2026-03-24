# policies.py

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import NamedTuple, Callable, Any

from max.utilities import mish


class Policy(NamedTuple):
    sample: Callable  # (policy_params, z, key) -> (tanh_action, log_prob)


def init_policy(key: jax.Array, config: dict, pretrained: dict = None) -> tuple["Policy", dict]:
    """
    Initialize policy.

    config["policy"]:
        type:         str, policy type (e.g. "squashed_gaussian")
        features:     list[int], MLP hidden sizes
        log_std_min:  float, minimum log std (default -10)
        log_std_max:  float, maximum log std (default 2)

    Returns:
        (Policy, policy_params) where policy_params = {"policy_net": flax_params}

    If pretrained is provided, it is used as the initial parameters (fully trainable).
    """
    policy_type = config["policy"]["type"]
    if policy_type == "squashed_gaussian":
        return _init_squashed_gaussian_policy(key, config, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown policy type: {policy_type!r}")


def _init_squashed_gaussian_policy(key: jax.Array, config: dict, pretrained: dict = None) -> tuple["Policy", dict]:
    policy_cfg = config["policy"]
    features = policy_cfg["features"]
    log_std_min: float = policy_cfg["log_std_min"]
    log_std_max: float = policy_cfg["log_std_max"]

    latent_dim: int = config["encoder"]["encoder_features"][-1]
    dim_a: int = config["dim_action"]

    class _PolicyNet(nn.Module):
        @nn.compact
        def __call__(self, z):
            x = z
            for feat in features:
                x = nn.Dense(feat)(x)
                x = nn.LayerNorm()(x)
                x = mish(x)
            mean = nn.Dense(dim_a)(x)
            log_std = nn.Dense(dim_a)(x)
            return mean, log_std

    policy_net = _PolicyNet()

    if config["policy"].get("frozen", False):
        def sample(
            params: Any, z: jnp.ndarray, key: jax.Array
        ) -> tuple[jnp.ndarray, jnp.ndarray]:
            mean, log_std_raw = policy_net.apply(pretrained["policy_net"], z)
            log_std_dif = log_std_max - log_std_min
            log_std = log_std_min + 0.5 * log_std_dif * (jnp.tanh(log_std_raw) + 1)
            std = jnp.exp(log_std)
            key, noise_key = jax.random.split(key)
            noise = jax.random.normal(noise_key, shape=mean.shape)
            x = mean + std * noise
            action = jnp.tanh(x)
            log_prob_gauss = jnp.sum(
                -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std, axis=-1
            )
            log_jacob = jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
            return action, log_prob_gauss - log_jacob

        return Policy(sample=sample), {}

    if pretrained is not None:
        policy_params = pretrained
    else:
        dummy_z = jnp.ones((latent_dim,))
        key, k1 = jax.random.split(key)
        policy_params = {"policy_net": policy_net.init(k1, dummy_z)}

    def sample(
        policy_params: Any, z: jnp.ndarray, key: jax.Array
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Sample action from squashed Gaussian policy.

        Returns:
            (tanh_action, log_prob) with Jacobian correction for tanh squashing.
        """
        mean, log_std_raw = policy_net.apply(policy_params["policy_net"], z)
        log_std_dif = log_std_max - log_std_min
        log_std = log_std_min + 0.5 * log_std_dif * (jnp.tanh(log_std_raw) + 1)
        std = jnp.exp(log_std)

        key, noise_key = jax.random.split(key)
        noise = jax.random.normal(noise_key, shape=mean.shape)
        x = mean + std * noise

        action = jnp.tanh(x)

        log_prob_gauss = jnp.sum(
            -0.5 * (noise ** 2 + jnp.log(2.0 * jnp.pi)) - log_std,
            axis=-1,
        )
        log_jacob = jnp.sum(jnp.log(1.0 - action ** 2 + 1e-6), axis=-1)
        log_prob = log_prob_gauss - log_jacob

        return action, log_prob

    return Policy(sample=sample), policy_params
