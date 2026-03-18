# environments.py

import jax
import jax.numpy as jnp
from typing import Dict, Any
from dataclasses import dataclass


@dataclass(frozen=True)
class EnvParams:
    """Environment parameters."""

    num_agents: int = 1
    max_episode_steps: int = 200
    dt: float = 0.1

    # Specific to cheetah
    cheetah_mass_scale: float = 1.0  # Multiplier for body masses (default 14kg total)


def make_cheetah_env(params: EnvParams):
    """
    Factory function that wraps mujoco_playground CheetahRun environment.

    Internal state: mjx.Data (full MuJoCo physics state)
    Observation: 17D = [qpos[1:] (8D), qvel (9D)] (matches playground)
    Action: 6D torques in [-1, 1]
    Forward velocity = data.qvel[0]
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco import mjx

    # Load environment and extract models (closed over)
    env = registry.load('CheetahRun')

    # Apply mass scaling if specified
    if params.cheetah_mass_scale != 1.0:
        import mujoco
        mj_model = env.mj_model
        # Scale both mass and inertia (inertia scales linearly with mass for uniform density)
        mj_model.body_mass[:] *= params.cheetah_mass_scale
        mj_model.body_inertia[:] *= params.cheetah_mass_scale
        # Recalculate dependent constants (invweight, actuator_acc0, etc.)
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        """Resets the cheetah environment and returns mjx.Data directly."""
        env_state = env.reset(key)
        return env_state.data

    @jax.jit
    def step_fn(
        data: mjx.Data,
        step_count: int,
        action: jnp.ndarray,
    ):
        """Steps the cheetah environment forward using mjx.Data directly."""
        # Two physics steps, reward sampled after each — matches TDMPC2's DMControl wrapper
        # which calls env.step(action) twice and sums rewards (dmcontrol.py:57-59).
        # Each call is one mujoco step at sim_dt=0.01s → 50Hz control frequency.
        a = action.squeeze()
        mid_data = mjx_env.step(mjx_model, data, a, 1)
        next_data = mjx_env.step(mjx_model, mid_data, a, 1)

        # Get observation
        obs = get_obs_fn(next_data)

        # Reward sampled at both substeps and summed (max = 2.0 per agent step, 1000 per episode)
        reward = (jnp.clip(mid_data.qvel[0] / 10.0, 0.0, 1.0)
                  + jnp.clip(next_data.qvel[0] / 10.0, 0.0, 1.0))
        rewards = jnp.array([reward])

        # Check termination (NaN in state)
        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        terminated = done

        # Truncation based on max steps
        truncated = step_count >= params.max_episode_steps

        info = {
            "forward_vel": next_data.qvel[0],
        }

        return next_data, obs, rewards, terminated, truncated, info

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        """Returns 17D observation: [qpos[1:], qvel] from mjx.Data."""
        obs = jnp.concatenate([data.qpos[1:], data.qvel])
        return obs[None, :]  # Add agent dimension

    return reset_fn, step_fn, get_obs_fn


def init_env(config: Dict[str, Any]):
    """
    Initialize environment functions based on configuration.
    Only supports cheetah.
    """
    env_name = config.get("env_name", None)
    env_params_dict = config.get("env_params", {})

    # Filter to only EnvParams fields
    valid_fields = {f for f in EnvParams.__dataclass_fields__}
    filtered = {k: v for k, v in env_params_dict.items() if k in valid_fields}

    params = EnvParams(**filtered)

    print(f"Initializing environment: {env_name}")

    if env_name == "cheetah":
        return make_cheetah_env(params)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
