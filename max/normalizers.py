# normalizers.py

import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable, Dict, Tuple


class Normalizer(NamedTuple):
    normalize: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]
    unnormalize: Callable[[Dict[str, jnp.ndarray], jnp.ndarray], jnp.ndarray]


# --- Identity "None" Normalizer ---
def _identity(params: Dict, x: jnp.ndarray) -> jnp.ndarray:
    """Identity function that returns the input unchanged. Ignores params."""
    return x


NONE_NORMALIZER = Normalizer(normalize=_identity, unnormalize=_identity)


# --- Standard Normalizer ---
def _normalize(params: Dict[str, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    """Applies standard normalization using provided params."""
    return (x - params["mean"]) / (params["std"] + 1e-8)


def _unnormalize(
    params: Dict[str, jnp.ndarray], x: jnp.ndarray
) -> jnp.ndarray:
    """Reverses standard normalization using provided params."""
    return x * params["std"] + params["mean"]


STANDARD_NORMALIZER = Normalizer(
    normalize=_normalize, unnormalize=_unnormalize
)


def compute_data_driven_stats(data: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    mean = jnp.mean(data, axis=0)
    std = jnp.std(data, axis=0)
    return {"mean": mean, "std": std}


def compute_static_stats(
    min_vals: jnp.ndarray, max_vals: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    mean = (max_vals + min_vals) / 2.0
    std = (max_vals - min_vals) / 2.0
    return {"mean": mean, "std": std}


def _initialize_normalization_params(config: Dict) -> Dict:
    """Internal function to initialize normalization parameters."""
    method = config.get("normalization", {}).get("method", "none")

    if method == "static":
        static_params = config["normalization_params"]
        norm_params = {}
        state_min = jnp.array(static_params["state"]["min"])
        state_max = jnp.array(static_params["state"]["max"])
        norm_params["state"] = compute_static_stats(state_min, state_max)
        action_min = jnp.array(static_params["action"]["min"])
        action_max = jnp.array(static_params["action"]["max"])
        norm_params["action"] = compute_static_stats(action_min, action_max)
        if "delta" in static_params:
            delta_min = jnp.array(static_params["delta"]["min"])
            delta_max = jnp.array(static_params["delta"]["max"])
            norm_params["delta"] = compute_static_stats(delta_min, delta_max)
        else:
            norm_params["delta"] = norm_params["state"]
        return norm_params

    elif method == "none":
        return {"state": {}, "action": {}, "delta": {}}

    else:
        raise ValueError(f"Unknown normalization method: {method}")


def init_normalizer(config: Dict) -> Tuple[Normalizer, Dict]:
    """
    Initializes and returns the appropriate Normalizer object and its parameters.
    This is the main factory function for normalization.
    """
    method = config.get("normalization", {}).get("method", "none")

    if method == "none":
        normalizer = NONE_NORMALIZER
        print("-> Using NO normalization.")
    else:
        normalizer = STANDARD_NORMALIZER
        print(f"-> Using {method.upper()} normalization.")

    params = _initialize_normalization_params(config)
    return normalizer, params


# --- Rolling Return Normalizer ---
@jax.jit
def _normalize_rolling_return(
    params: Dict[str, jnp.ndarray], rewards: jnp.ndarray
) -> jnp.ndarray:
    """
    Normalizes rewards by dividing by the standard deviation of rolling returns.

    Args:
        params: Dict containing 'var' (variance) and 'clip_range' (clipping threshold)
        rewards: Raw rewards to normalize

    Returns:
        Normalized and clipped rewards
    """
    std = jnp.sqrt(params["var"] + 1e-8)
    normalized = rewards / std
    return jnp.clip(normalized, -params["clip_range"], params["clip_range"])


@jax.jit
def _unnormalize_rolling_return(
    params: Dict[str, jnp.ndarray], rewards: jnp.ndarray
) -> jnp.ndarray:
    """
    Reverses rolling return normalization by multiplying by the standard deviation.

    Args:
        params: Dict containing 'var' (variance)
        rewards: Normalized rewards

    Returns:
        Unnormalized rewards (note: clipping is not reversed)
    """
    std = jnp.sqrt(params["var"] + 1e-8)
    return rewards * std


ROLLING_RETURN_NORMALIZER = Normalizer(
    normalize=_normalize_rolling_return,
    unnormalize=_unnormalize_rolling_return,
)


@jax.jit
def update_rolling_return_stats(
    params: Dict[str, jnp.ndarray], rewards: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Updates rolling return statistics using Welford's online algorithm.

    This pure function:
    1. Updates the discounted rolling return: rolling_return = gamma * rolling_return + reward
    2. Updates running mean/variance using Welford's algorithm

    Args:
        params: Dict containing:
            - 'rolling_return': Current discounted cumulative return per agent
            - 'mean': Running mean of rolling returns
            - 'var': Running variance of rolling returns
            - 'count': Number of samples seen
            - 'gamma': Discount factor for rolling return
            - 'clip_range': Clipping threshold (passed through unchanged)
        rewards: New rewards to incorporate (one per agent)

    Returns:
        Updated params dict with new rolling_return, mean, var, and count
    """
    # Update rolling return with discount factor
    rolling_return = params["gamma"] * params["rolling_return"] + rewards

    # Welford's online algorithm (vectorized for multiple agents)
    # Treat each new rolling_return value as a "batch" of size 1
    batch_mean = rolling_return
    batch_var = jnp.zeros_like(batch_mean)  # Variance of single value is 0
    batch_count = 1.0

    # Combine old stats with new batch
    delta = batch_mean - params["mean"]
    tot_count = params["count"] + batch_count

    new_mean = params["mean"] + delta * batch_count / tot_count

    # Update variance using Welford's method
    m_a = params["var"] * params["count"]
    m_b = batch_var * batch_count
    M2 = (
        m_a
        + m_b
        + jnp.square(delta) * params["count"] * batch_count / tot_count
    )

    new_var = M2 / tot_count

    return {
        "rolling_return": rolling_return,
        "mean": new_mean,
        "var": new_var,
        "count": tot_count,
        "gamma": params["gamma"],
        "clip_range": params["clip_range"],
    }


def init_rolling_return_normalizer(
    num_agents: int, gamma: float = 0.99, clip_range: float = 100.0
) -> Tuple[Normalizer, Dict[str, jnp.ndarray]]:
    """
    Initializes a rolling return normalizer and its parameters.

    Args:
        num_agents: Number of agents (determines array sizes)
        gamma: Discount factor for computing rolling returns
        clip_range: Clipping threshold for normalized rewards

    Returns:
        Tuple of (normalizer, params) where params contains:
            - rolling_return: Discounted cumulative return (reset each episode)
            - mean: Running mean of rolling returns
            - var: Running variance of rolling returns
            - count: Number of samples seen
            - gamma: Discount factor
            - clip_range: Clipping threshold
    """
    params = {
        "rolling_return": jnp.zeros(num_agents, dtype=jnp.float32),
        "mean": jnp.zeros(num_agents, dtype=jnp.float32),
        "var": jnp.ones(num_agents, dtype=jnp.float32),
        "count": jnp.array(1e-8),
        "gamma": jnp.array(gamma),
        "clip_range": jnp.array(clip_range),
    }

    @jax.jit
    def reset_rolling_return(
        params: Dict[str, jnp.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """
        Resets the rolling return to zeros while preserving running statistics.

        This should be called at the end of each episode to reset the discounted
        cumulative return, while keeping the learned mean/variance/count for
        normalization purposes.

        Args:
            params: Current normalizer params
            num_agents: Number of agents (to create correctly sized zeros array)

        Returns:
            Updated params with rolling_return reset to zeros
        """
        return {
            **params,
            "rolling_return": jnp.zeros(num_agents, dtype=jnp.float32),
        }

    return ROLLING_RETURN_NORMALIZER, reset_rolling_return, params
