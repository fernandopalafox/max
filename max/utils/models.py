import jax
from jax import numpy as jnp
from flax import linen as nn


def uniform_spread(spread):
    def init(key, shape, dtype=jnp.float64):
        return jax.random.uniform(
            key, shape, dtype, minval=-spread, maxval=spread
        )

    return init


class KalmanFeatureMapping(nn.Module):
    n_phi: int  # Dimension of the feature mapping output
    weight_spread: float = 5.0

    @nn.compact
    def __call__(self, x):
        init = uniform_spread(self.weight_spread)  # kernel and bias init
        x = nn.Dense(features=20, kernel_init=init, bias_init=init)(x)
        x = nn.tanh(x)
        x = nn.Dense(features=self.n_phi, kernel_init=init, bias_init=init)(x)
        return x


class KalmanFeatureMappingLinearLast(nn.Module):
    n_phi: int  # Dimension of the feature mapping output
    weight_spread: float = 5.0

    @nn.compact
    def get_phi(self, x):
        init = uniform_spread(self.weight_spread)

        phi = nn.Dense(features=10, kernel_init=init, bias_init=init)(x)
        phi = nn.tanh(phi)
        phi = nn.Dense(features=10, kernel_init=init, bias_init=init)(
            phi
        )
        return phi

    @nn.compact
    def __call__(self, x):
        init = uniform_spread(self.weight_spread)
        phi = self.get_phi(x)
        y = nn.Dense(features=self.n_phi, kernel_init=init, use_bias=False)(
            phi
        )
        return y
