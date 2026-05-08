# environments.py

import jax
import jax.numpy as jnp
from typing import Dict, Any


def init_env(config: Dict[str, Any]):
    """
    Initialize environment functions based on config["environment"]["type"].
    """
    env_type = config["environment"]["type"]
    if env_type == "cheetah":
        return _make_cheetah_env(config)
    elif env_type == "cartpole_balance":
        return _make_cartpole_balance_env(config)
    else:
        raise ValueError(f"Unknown environment: {env_type!r}")


def _make_cheetah_env(config: Dict[str, Any]):
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

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 200)
    cheetah_mass_scale = env_cfg.get("cheetah_mass_scale", 1.0)

    print(f"Initializing environment: cheetah")

    # Load environment and extract models (closed over)
    env = registry.load('CheetahRun')

    # Apply mass scaling if specified
    if cheetah_mass_scale != 1.0:
        import mujoco
        mj_model = env.mj_model
        # Scale both mass and inertia (inertia scales linearly with mass for uniform density)
        mj_model.body_mass[:] *= cheetah_mass_scale
        mj_model.body_inertia[:] *= cheetah_mass_scale
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
        truncated = step_count >= max_episode_steps

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


def _make_cartpole_balance_env(config: Dict[str, Any]):
    """
    Factory for CartpoleBalance from mujoco_playground.

    Observation: 5D = [cart_pos, pole_cos, pole_sin, cart_vel, pole_vel]
    Action: 1D cart force in [-1, 1]
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env, reward as reward_lib
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg["max_episode_steps"]
    cartpole_pole_mass_scale = env_cfg.get("cartpole_pole_mass_scale", 1.0)

    print("Initializing environment: cartpole_balance")

    env = registry.load('CartpoleBalance')

    if cartpole_pole_mass_scale != 1.0:
        import mujoco
        mj_model = env.mj_model
        pole_idx = mj_model.body('pole_1').id
        mj_model.body_mass[pole_idx] *= cartpole_pole_mass_scale
        mj_model.body_inertia[pole_idx] *= cartpole_pole_mass_scale
        mujoco.mj_setConst(mj_model, mujoco.MjData(mj_model))
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    slider_qposadr = env._slider_qposadr
    hinge_1_qposadr = env._hinge_1_qposadr

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        env_state = env.reset(key)
        return env_state.data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)  # (dim_action,) — mjx_env.step requires shape [nu]
        next_data = mjx_env.step(mjx_model, data, a, 1)

        obs = get_obs_fn(next_data)

        pole_angle_cos = next_data.xmat[2, 2, 2]
        upright = (pole_angle_cos + 1) / 2

        cart_position = next_data.qpos[slider_qposadr]
        centered = (1 + reward_lib.tolerance(cart_position, margin=2)) / 2

        small_control = (4 + reward_lib.tolerance(
            a[0], margin=1, value_at_margin=0, sigmoid="quadratic"
        )) / 5

        angular_vel = next_data.qvel[1:]
        small_velocity = (1 + reward_lib.tolerance(angular_vel, margin=5).min()) / 2

        r = upright * small_control * small_velocity * centered
        rewards = jnp.array([r])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        terminated = done
        truncated = step_count >= max_episode_steps

        return next_data, obs, rewards, terminated, truncated, {}

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        cart_position = data.qpos[slider_qposadr]
        pole_angle_cos = data.xmat[2:, 2, 2]
        pole_angle_sin = data.xmat[2:, 0, 2]
        obs = jnp.concatenate([
            cart_position.reshape(1),
            pole_angle_cos,
            pole_angle_sin,
            data.qvel,
        ])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn
