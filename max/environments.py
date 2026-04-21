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
    elif env_type == "humanoid":
        return _make_humanoid_env(config)
    elif env_type == "quadruped":
        return _make_quadruped_env(config)
    elif env_type == "walker":
        return _make_walker_env(config)
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


def _make_humanoid_env(config: Dict[str, Any]):
    """
    Factory function that wraps mujoco_playground HumanoidRun environment.

    Internal state: mjx.Data (full MuJoCo physics state)
    Observation: flattened vector [qpos[1:], qvel] (body pose and velocities, excluding root x)
    Action: nu-dimensional continuous control (HumanoidRun has nu=21)
    Forward velocity = data.qvel[0]
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 500)
    humanoid_mass_scale = env_cfg.get("humanoid_mass_scale", 1.0)

    print(f"Initializing environment: humanoid")

    # Load environment and extract models (closed over)
    env = registry.load('HumanoidRun')

    # Apply mass scaling if specified
    if humanoid_mass_scale != 1.0:
        import mujoco
        mj_model = env.mj_model
        # Scale both mass and inertia
        mj_model.body_mass[:] *= humanoid_mass_scale
        mj_model.body_inertia[:] *= humanoid_mass_scale
        # Recalculate dependent constants
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        """Resets the humanoid environment and returns mjx.Data directly."""
        env_state = env.reset(key)
        return env_state.data

    @jax.jit
    def step_fn(
        data: mjx.Data,
        step_count: int,
        action: jnp.ndarray,
    ):
        """Steps the humanoid environment forward using mjx.Data directly."""
        # Two physics steps, reward sampled after each
        a = action.squeeze()
        mid_data = mjx_env.step(mjx_model, data, a, 1)
        next_data = mjx_env.step(mjx_model, mid_data, a, 1)

        # Get observation
        obs = get_obs_fn(next_data)

        # Reward: forward velocity task (clipped to [0, 1] per step, max 2.0 per agent step)
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
        """Returns observation: [qpos[1:], qvel] from mjx.Data (excluding root x position)."""
        obs = jnp.concatenate([data.qpos[1:], data.qvel])
        return obs[None, :]  # Add agent dimension

    return reset_fn, step_fn, get_obs_fn


def _make_quadruped_env(config: Dict[str, Any]):
    """
    Factory function that wraps dm_control's dog-run environment.

    Note: This environment is stateful (dm_control maintains internal state).
    The state returned from step_fn and reset_fn is just the observation (JAX array),
    not the full timestep object. This allows compatibility with JAX's JIT compilation.
    
    Observation: flattened vector of proprioceptive observations
    Action: 10D continuous control
    Reward: forward velocity task from DeepMind Control Suite dog/run
    """
    import numpy as np
    import dm_control.suite as suite
    import dm_control

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)

    print(f"Initializing environment: quadruped (dm_control dog/run task)")

    # Load dog-run task from dm_control (closed over)
    env = suite.load(domain_name='dog', task_name='run')
    action_spec = env.action_spec()

    def reset_fn(key: jax.random.PRNGKey):
        """Resets the environment and returns the initial observation as a JAX array."""
        timestep = env.reset()
        obs = _extract_obs(timestep)
        return obs

    def step_fn(
        obs_state,
        step_count: int,
        action: jnp.ndarray,
    ):
        """Steps the environment forward.
        
        Note: obs_state parameter is the observation from the previous step, but is ignored.
        The dm_control environment maintains its own internal state between calls.
        """
        # Convert JAX array to numpy and clip to action bounds
        action_np = np.clip(
            np.asarray(action.squeeze()),
            action_spec.minimum,
            action_spec.maximum
        )
        
        # Step environment (stateful; ignores obs_state because dm_control manages state internally)
        timestep = env.step(action_np)

        # Extract observation as JAX array (this becomes the new state)
        obs = _extract_obs(timestep)

        # Reward from dm_control
        rewards = jnp.array([timestep.reward])

        # Check termination
        terminated = timestep.step_type == dm_control.dm_env.StepType.LAST
        truncated = step_count >= max_episode_steps

        info = {
            "reward": float(timestep.reward),
            "step_type": int(timestep.step_type),
        }

        # Return observation as state (JAX-compatible)
        return obs, obs, rewards, terminated, truncated, info

    def _extract_obs(timestep) -> jnp.ndarray:
        """Extracts and flattens observation from dm_control timestep."""
        obs_dict = timestep.observation
        obs_parts = []
        for key in sorted(obs_dict.keys()):
            val = np.asarray(obs_dict[key])
            obs_parts.append(val.flatten())
        obs_flat = np.concatenate(obs_parts)
        # Convert to JAX array and add agent batch dimension
        obs_jax = jnp.asarray(obs_flat, dtype=jnp.float32)[None, :]
        return obs_jax

    def get_obs_fn(obs_or_timestep) -> jnp.ndarray:
        """Extracts observation from either timestep (from reset_fn during init) or JAX array (from step_fn)."""
        # If it's already a JAX array (from step_fn), return it as-is
        if isinstance(obs_or_timestep, jnp.ndarray):
            return obs_or_timestep
        # Otherwise it's a timestep from reset_fn
        return _extract_obs(obs_or_timestep)

    return reset_fn, step_fn, get_obs_fn


def _make_walker_env(config: Dict[str, Any]):
    """
    Factory function that wraps mujoco_playground WalkerWalk environment.

    Internal state: mjx.Data (full MuJoCo physics state)
    Observation: 24D = [orientations (14D), height (1D), qvel (9D)]
        orientations: xmat[body, 0, 0] and xmat[body, 0, 2] for each of 7 bodies (world excluded)
        height: xmat[torso, 2, 2] (upright component)
        velocity: qvel (9D)
    Action: 6D torques in [-1, 1]
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)

    print(f"Initializing environment: walker")

    env = registry.load('WalkerWalk')
    mjx_model = env.mjx_model
    mj_model = env.mj_model

    torso_id = mj_model.body("torso").id
    sensor_id = mj_model.sensor("torso_subtreelinvel").id
    sensor_adr = int(mj_model.sensor_adr[sensor_id])

    _STAND_HEIGHT = 1.2
    _WALK_SPEED = 1.0

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        env_state = env.reset(key)
        return env_state.data

    @jax.jit
    def step_fn(
        data: mjx.Data,
        step_count: int,
        action: jnp.ndarray,
    ):
        a = action.squeeze()
        # ctrl_dt=0.025, sim_dt=0.0025 => 10 substeps per control step
        next_data = mjx_env.step(mjx_model, data, a, 10)

        obs = get_obs_fn(next_data)

        # Standing reward
        torso_height = next_data.xpos[torso_id, 2]
        standing = reward_fns.tolerance(
            torso_height,
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
        )
        torso_upright = next_data.xmat[torso_id, 2, 2]
        upright = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4

        # Move reward
        horizontal_velocity = next_data.sensordata[sensor_adr]
        move_reward = reward_fns.tolerance(
            horizontal_velocity,
            bounds=(_WALK_SPEED, float("inf")),
            margin=_WALK_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )

        reward = stand_reward * (5 * move_reward + 1) / 6
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        terminated = done
        truncated = step_count >= max_episode_steps

        info = {
            "horizontal_velocity": horizontal_velocity,
            "torso_height": torso_height,
        }

        return next_data, obs, rewards, terminated, truncated, info

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        """Returns 24D observation: [orientations (14D), height (1D), qvel (9D)]."""
        orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
        height = data.xmat[torso_id, 2, 2]
        obs = jnp.concatenate([orientations, height.reshape(1), data.qvel])
        return obs[None, :]  # Add agent dimension

    return reset_fn, step_fn, get_obs_fn
