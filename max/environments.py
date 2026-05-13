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
    elif env_type == "walker_run":
        return _make_walker_run_env(config)
    elif env_type == "ball_in_cup":
        return _make_ball_in_cup_env(config)
    elif env_type == "cartpole_balance":
        return _make_cartpole_balance_env(config)
    elif env_type == "cartpole_swingup":
        return _make_cartpole_env(config, "CartpoleSwingup")
    elif env_type == "finger_spin":
        return _make_finger_spin_env(config)
    elif env_type == "hopper_hop":
        return _make_hopper_hop_env(config)
    elif env_type == "reacher_easy":
        return _make_reacher_env(config, "ReacherEasy")
    elif env_type == "reacher_hard":
        return _make_reacher_env(config, "ReacherHard")
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
    cartpole_pole_length_scale = env_cfg.get("cartpole_pole_length_scale", 1.0)

    print("Initializing environment: cartpole_balance")

    env = registry.load('CartpoleBalance')

    if cartpole_pole_mass_scale != 1.0 or cartpole_pole_length_scale != 1.0:
        import mujoco
        mj_model = env.mj_model
        pole_idx = mj_model.body('pole_1').id

        if cartpole_pole_mass_scale != 1.0:
            mj_model.body_mass[pole_idx] *= cartpole_pole_mass_scale
            mj_model.body_inertia[pole_idx] *= cartpole_pole_mass_scale

        if cartpole_pole_length_scale != 1.0:
            s = cartpole_pole_length_scale
            pole_geom_idx = mj_model.geom('pole_1').id
            mj_model.geom_size[pole_geom_idx, 1] *= s       # capsule half-length
            mj_model.geom_pos[pole_geom_idx, 2] *= s        # capsule center
            mj_model.body_ipos[pole_idx, 2] *= s            # center of mass
            mj_model.body_inertia[pole_idx, 0] *= s ** 2    # Ixx
            mj_model.body_inertia[pole_idx, 1] *= s ** 2    # Iyy

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
    humanoid_friction_scale = env_cfg.get("humanoid_friction_scale", 1.0)
    humanoid_gravity_scale = env_cfg.get("humanoid_gravity_scale", 1.0)

    print(f"Initializing environment: humanoid")

    env = registry.load('HumanoidRun')
    mj_model = env.mj_model

    if humanoid_mass_scale != 1.0 or humanoid_friction_scale != 1.0 or humanoid_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= humanoid_mass_scale
        mj_model.body_inertia[:] *= humanoid_mass_scale
        mj_model.geom_friction[:] *= humanoid_friction_scale
        mj_model.opt.gravity[:] *= humanoid_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

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
        mid_data = mjx_env.step(mjx_model, data, a, 1)
        next_data = mjx_env.step(mjx_model, mid_data, a, 1)

        obs = get_obs_fn(next_data)

        reward = (jnp.clip(mid_data.qvel[0] / 10.0, 0.0, 1.0)
                  + jnp.clip(next_data.qvel[0] / 10.0, 0.0, 1.0))
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        terminated = done
        truncated = step_count >= max_episode_steps

        info = {"forward_vel": next_data.qvel[0]}

        return next_data, obs, rewards, terminated, truncated, info

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        obs = jnp.concatenate([data.qpos[1:], data.qvel])
        return obs[None, :]

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

    env = suite.load(domain_name='dog', task_name='run')
    action_spec = env.action_spec()

    def reset_fn(key: jax.random.PRNGKey):
        timestep = env.reset()
        obs = _extract_obs(timestep)
        return obs

    def step_fn(obs_state, step_count: int, action: jnp.ndarray):
        action_np = np.clip(
            np.asarray(action.squeeze()),
            action_spec.minimum,
            action_spec.maximum
        )
        timestep = env.step(action_np)
        obs = _extract_obs(timestep)
        rewards = jnp.array([timestep.reward])
        terminated = timestep.step_type == dm_control.dm_env.StepType.LAST
        truncated = step_count >= max_episode_steps
        info = {
            "reward": float(timestep.reward),
            "step_type": int(timestep.step_type),
        }
        return obs, obs, rewards, terminated, truncated, info

    def _extract_obs(timestep) -> jnp.ndarray:
        obs_dict = timestep.observation
        obs_parts = []
        for key in sorted(obs_dict.keys()):
            val = np.asarray(obs_dict[key])
            obs_parts.append(val.flatten())
        obs_flat = np.concatenate(obs_parts)
        obs_jax = jnp.asarray(obs_flat, dtype=jnp.float32)[None, :]
        return obs_jax

    def get_obs_fn(obs_or_timestep) -> jnp.ndarray:
        if isinstance(obs_or_timestep, jnp.ndarray):
            return obs_or_timestep
        return _extract_obs(obs_or_timestep)

    return reset_fn, step_fn, get_obs_fn


def _make_walker_env(config: Dict[str, Any]):
    """
    Factory function that wraps mujoco_playground WalkerWalk environment.

    Internal state: mjx.Data (full MuJoCo physics state)
    Observation: 24D = [orientations (14D), height (1D), qvel (9D)]
    Action: 6D torques in [-1, 1]
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    walker_mass_scale = env_cfg.get("walker_mass_scale", 1.0)
    walker_friction_scale = env_cfg.get("walker_friction_scale", 1.0)
    walker_gravity_scale = env_cfg.get("walker_gravity_scale", 1.0)

    print(f"Initializing environment: walker")

    env = registry.load('WalkerWalk')
    mj_model = env.mj_model

    if walker_mass_scale != 1.0 or walker_friction_scale != 1.0 or walker_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= walker_mass_scale
        mj_model.body_inertia[:] *= walker_mass_scale
        mj_model.geom_friction[:] *= walker_friction_scale
        mj_model.opt.gravity[:] *= walker_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

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
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.squeeze()
        next_data = mjx_env.step(mjx_model, data, a, 10)
        obs = get_obs_fn(next_data)

        torso_height = next_data.xpos[torso_id, 2]
        standing = reward_fns.tolerance(
            torso_height,
            bounds=(_STAND_HEIGHT, float("inf")),
            margin=_STAND_HEIGHT / 2,
        )
        torso_upright = next_data.xmat[torso_id, 2, 2]
        upright = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4

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
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "horizontal_velocity": horizontal_velocity,
            "torso_height": torso_height,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
        height = data.xmat[torso_id, 2, 2]
        obs = jnp.concatenate([orientations, height.reshape(1), data.qvel])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_walker_run_env(config: Dict[str, Any]):
    """
    WalkerRun: same as WalkerWalk but move_speed = 8 m/s.

    Observation: 24D = [orientations (14D), height (1D), qvel (9D)]
    Action: 6D
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    walker_mass_scale = env_cfg.get("walker_mass_scale", 1.0)
    walker_friction_scale = env_cfg.get("walker_friction_scale", 1.0)
    walker_gravity_scale = env_cfg.get("walker_gravity_scale", 1.0)

    print("Initializing environment: walker_run")

    env = registry.load("WalkerRun")
    mj_model = env.mj_model

    if walker_mass_scale != 1.0 or walker_friction_scale != 1.0 or walker_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= walker_mass_scale
        mj_model.body_inertia[:] *= walker_mass_scale
        mj_model.geom_friction[:] *= walker_friction_scale
        mj_model.opt.gravity[:] *= walker_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    torso_id = mj_model.body("torso").id
    sensor_id = mj_model.sensor("torso_subtreelinvel").id
    sensor_adr = int(mj_model.sensor_adr[sensor_id])

    _STAND_HEIGHT = 1.2
    _RUN_SPEED = 8.0

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 10)
        obs = get_obs_fn(next_data)

        torso_height = next_data.xpos[torso_id, 2]
        standing = reward_fns.tolerance(
            torso_height, bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 2,
        )
        torso_upright = next_data.xmat[torso_id, 2, 2]
        upright = (1 + torso_upright) / 2
        stand_reward = (3 * standing + upright) / 4

        horizontal_velocity = next_data.sensordata[sensor_adr]
        move_reward = reward_fns.tolerance(
            horizontal_velocity,
            bounds=(_RUN_SPEED, float("inf")),
            margin=_RUN_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        reward = stand_reward * (5 * move_reward + 1) / 6
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "horizontal_velocity": horizontal_velocity,
            "torso_height": torso_height,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        orientations = data.xmat[1:, [0, 0], [0, 2]].ravel()
        height = data.xmat[torso_id, 2, 2]
        return jnp.concatenate([orientations, height.reshape(1), data.qvel])[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_ball_in_cup_env(config: Dict[str, Any]):
    """
    BallInCup: cup swings to catch a ball on a string.

    Observation: 8D = [qpos (4D: cup_x, cup_z, ball_x, ball_z), qvel (4D)]
    Action: 2D (cup x/z force)
    Reward: 1 if ball inside cup target site in both x and z, else 0
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    ball_in_cup_mass_scale = env_cfg.get("ball_in_cup_mass_scale", 1.0)
    ball_in_cup_friction_scale = env_cfg.get("ball_in_cup_friction_scale", 1.0)
    ball_in_cup_gravity_scale = env_cfg.get("ball_in_cup_gravity_scale", 1.0)

    print("Initializing environment: ball_in_cup")

    env = registry.load("BallInCup")
    mj_model = env.mj_model

    if ball_in_cup_mass_scale != 1.0 or ball_in_cup_friction_scale != 1.0 or ball_in_cup_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= ball_in_cup_mass_scale
        mj_model.body_inertia[:] *= ball_in_cup_mass_scale
        mj_model.geom_friction[:] *= ball_in_cup_friction_scale
        mj_model.opt.gravity[:] *= ball_in_cup_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    target_site_id = int(mj_model.site("target").id)
    ball_body_id = int(mj_model.body("ball").id)
    ball_geom_id = int(mj_model.geom("ball").id)
    target_sz = jnp.array([
        float(mj_model.site_size[target_site_id, 0]),
        float(mj_model.site_size[target_site_id, 2]),
    ])
    ball_sz = float(mj_model.geom_size[ball_geom_id, 0])

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 10)
        obs = get_obs_fn(next_data)

        target_xz = jnp.stack([next_data.site_xpos[target_site_id, 0],
                               next_data.site_xpos[target_site_id, 2]])
        ball_xz = jnp.stack([next_data.xpos[ball_body_id, 0],
                             next_data.xpos[ball_body_id, 2]])
        ball_to_target = jnp.abs(target_xz - ball_xz)
        inside = jnp.where(ball_to_target < target_sz - ball_sz, 1.0, 0.0)
        reward = jnp.prod(inside)
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {}

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        return jnp.concatenate([data.qpos, data.qvel])[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_finger_spin_env(config: Dict[str, Any]):
    """
    FingerSpin: two-link finger spins a cylinder.

    Observation: 9D = [proximal (1D), distal (1D), tip_xz rel spinner (2D), qvel (3D), touch (2D)]
    Action: 2D (proximal/distal torques)
    Reward: 1 if hinge_velocity <= -15 rad/s, else 0
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    finger_spin_mass_scale = env_cfg.get("finger_spin_mass_scale", 1.0)
    finger_spin_friction_scale = env_cfg.get("finger_spin_friction_scale", 1.0)
    finger_spin_gravity_scale = env_cfg.get("finger_spin_gravity_scale", 1.0)

    print("Initializing environment: finger_spin")

    env = registry.load("FingerSpin")
    mj_model = env.mj_model

    if finger_spin_mass_scale != 1.0 or finger_spin_friction_scale != 1.0 or finger_spin_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= finger_spin_mass_scale
        mj_model.body_inertia[:] *= finger_spin_mass_scale
        mj_model.geom_friction[:] *= finger_spin_friction_scale
        mj_model.opt.gravity[:] *= finger_spin_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    def _sadr(name):
        return int(mj_model.sensor_adr[mj_model.sensor(name).id])

    proximal_adr    = _sadr("proximal")
    distal_adr      = _sadr("distal")
    tip_adr         = _sadr("tip")
    spinner_adr     = _sadr("spinner")
    hinge_vel_adr   = _sadr("hinge_velocity")
    touchtop_adr    = _sadr("touchtop")
    touchbottom_adr = _sadr("touchbottom")

    _SPIN_VELOCITY = 15.0

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 4)
        obs = get_obs_fn(next_data)

        hinge_vel = next_data.sensordata[hinge_vel_adr]
        reward = (hinge_vel <= -_SPIN_VELOCITY).astype(jnp.float32)
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "hinge_velocity": hinge_vel,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        proximal = data.sensordata[proximal_adr : proximal_adr + 1]
        distal   = data.sensordata[distal_adr   : distal_adr   + 1]
        tip_x    = data.sensordata[tip_adr]
        tip_z    = data.sensordata[tip_adr + 2]
        spin_x   = data.sensordata[spinner_adr]
        spin_z   = data.sensordata[spinner_adr + 2]
        tip_pos  = jnp.array([tip_x - spin_x, tip_z - spin_z])
        top      = data.sensordata[touchtop_adr    : touchtop_adr    + 1]
        bottom   = data.sensordata[touchbottom_adr : touchbottom_adr + 1]
        touch    = jnp.log1p(jnp.concatenate([top, bottom]))
        bounded_pos = jnp.concatenate([proximal, distal, tip_pos])
        obs = jnp.concatenate([bounded_pos, data.qvel, touch])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_hopper_hop_env(config: Dict[str, Any]):
    """
    HopperHop: one-legged hopper that hops forward.

    Observation: 15D = [qpos[1:] (6D), qvel (7D), touch (2D: log1p toe+heel)]
    Action: 4D
    Reward: standing * hopping
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    hopper_hop_mass_scale = env_cfg.get("hopper_hop_mass_scale", 1.0)
    hopper_hop_friction_scale = env_cfg.get("hopper_hop_friction_scale", 1.0)
    hopper_hop_gravity_scale = env_cfg.get("hopper_hop_gravity_scale", 1.0)

    print("Initializing environment: hopper_hop")

    env = registry.load("HopperHop")
    mj_model = env.mj_model

    if hopper_hop_mass_scale != 1.0 or hopper_hop_friction_scale != 1.0 or hopper_hop_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= hopper_hop_mass_scale
        mj_model.body_inertia[:] *= hopper_hop_mass_scale
        mj_model.geom_friction[:] *= hopper_hop_friction_scale
        mj_model.opt.gravity[:] *= hopper_hop_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    torso_id = int(mj_model.body("torso").id)
    foot_id  = int(mj_model.body("foot").id)

    def _sadr(name):
        return int(mj_model.sensor_adr[mj_model.sensor(name).id])

    linvel_adr  = _sadr("torso_subtreelinvel")
    toe_adr     = _sadr("touch_toe")
    heel_adr    = _sadr("touch_heel")

    _STAND_HEIGHT = 0.6
    _HOP_SPEED    = 2.0

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 4)
        obs = get_obs_fn(next_data)

        height = next_data.xipos[torso_id, -1] - next_data.xipos[foot_id, -1]
        speed  = next_data.sensordata[linvel_adr]

        standing = reward_fns.tolerance(height, (_STAND_HEIGHT, 2.0))
        hopping  = reward_fns.tolerance(
            speed,
            bounds=(_HOP_SPEED, float("inf")),
            margin=_HOP_SPEED / 2,
            value_at_margin=0.5,
            sigmoid="linear",
        )
        reward  = standing * hopping
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "height": height,
            "speed": speed,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        toe    = data.sensordata[toe_adr  : toe_adr  + 1]
        heel   = data.sensordata[heel_adr : heel_adr + 1]
        touch  = jnp.log1p(jnp.concatenate([toe, heel]))
        obs    = jnp.concatenate([data.qpos[1:], data.qvel, touch])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_reacher_env(config: Dict[str, Any], registry_name: str):
    """
    Shared factory for ReacherEasy and ReacherHard.

    Observation: 6D = [qpos (2D), finger_to_target (2D), qvel (2D)]
    Action: 2D (shoulder/wrist torques)
    Reward: tolerance(finger_to_target_dist, (0, radii))
    """
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    reacher_mass_scale = env_cfg.get("reacher_mass_scale", 1.0)
    reacher_friction_scale = env_cfg.get("reacher_friction_scale", 1.0)
    reacher_gravity_scale = env_cfg.get("reacher_gravity_scale", 1.0)

    print(f"Initializing environment: {registry_name.lower()}")

    env = registry.load(registry_name)
    mj_model = env.mj_model

    if reacher_mass_scale != 1.0 or reacher_friction_scale != 1.0 or reacher_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= reacher_mass_scale
        mj_model.body_inertia[:] *= reacher_mass_scale
        mj_model.geom_friction[:] *= reacher_friction_scale
        mj_model.opt.gravity[:] *= reacher_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    finger_geom_id = int(mj_model.geom("finger").id)
    target_geom_id = int(mj_model.geom("target").id)
    radii = float(mj_model.geom_size[[target_geom_id, finger_geom_id], 0].sum())

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 4)
        obs = get_obs_fn(next_data)

        finger_pos = next_data.geom_xpos[finger_geom_id, :2]
        target_pos = next_data.geom_xpos[target_geom_id, :2]
        dist   = jnp.linalg.norm(target_pos - finger_pos)
        reward = reward_fns.tolerance(dist, (0.0, radii))
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "finger_to_target_dist": dist,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        finger_pos       = data.geom_xpos[finger_geom_id, :2]
        target_pos       = data.geom_xpos[target_geom_id, :2]
        finger_to_target = target_pos - finger_pos
        obs = jnp.concatenate([data.qpos, finger_to_target, data.qvel])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn


def _make_cartpole_env(config: Dict[str, Any], registry_name: str):
    """Stub used by cartpole_swingup dispatch."""
    from mujoco_playground import registry
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src import reward as reward_fns
    from mujoco import mjx

    env_cfg = config["environment"]
    max_episode_steps = env_cfg.get("max_episode_steps", 1000)
    cartpole_mass_scale = env_cfg.get("cartpole_mass_scale", 1.0)
    cartpole_friction_scale = env_cfg.get("cartpole_friction_scale", 1.0)
    cartpole_gravity_scale = env_cfg.get("cartpole_gravity_scale", 1.0)

    print(f"Initializing environment: {registry_name.lower()}")

    env = registry.load(registry_name)
    mj_model = env.mj_model

    if cartpole_mass_scale != 1.0 or cartpole_friction_scale != 1.0 or cartpole_gravity_scale != 1.0:
        import mujoco
        mj_model.body_mass[:] *= cartpole_mass_scale
        mj_model.body_inertia[:] *= cartpole_mass_scale
        mj_model.geom_friction[:] *= cartpole_friction_scale
        mj_model.opt.gravity[:] *= cartpole_gravity_scale
        mj_data = mujoco.MjData(mj_model)
        mujoco.mj_setConst(mj_model, mj_data)
        mjx_model = mjx.put_model(mj_model)
    else:
        mjx_model = env.mjx_model

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey) -> mjx.Data:
        return env.reset(key).data

    @jax.jit
    def step_fn(data: mjx.Data, step_count: int, action: jnp.ndarray):
        a = action.reshape(-1)
        next_data = mjx_env.step(mjx_model, data, a, 1)
        obs = get_obs_fn(next_data)

        pole_cos = next_data.xmat[2, 2, 2]
        upright = (pole_cos + 1) / 2
        cart_pos = next_data.qpos[0]
        centered = (1 + reward_fns.tolerance(cart_pos, margin=2.0)) / 2
        small_control = (
            4 + reward_fns.tolerance(a[0], margin=1.0, value_at_margin=0.0, sigmoid="quadratic")
        ) / 5
        angular_vel = next_data.qvel[1:]
        small_velocity = (1 + reward_fns.tolerance(angular_vel, margin=5.0).min()) / 2
        reward = upright * centered * small_control * small_velocity
        rewards = jnp.array([reward])

        done = jnp.isnan(next_data.qpos).any() | jnp.isnan(next_data.qvel).any()
        return next_data, obs, rewards, done, step_count >= max_episode_steps, {
            "upright": upright,
            "cart_pos": cart_pos,
        }

    @jax.jit
    def get_obs_fn(data: mjx.Data) -> jnp.ndarray:
        cart_pos = data.qpos[0:1]
        pole_cos = data.xmat[2:, 2, 2]
        pole_sin = data.xmat[2:, 0, 2]
        obs = jnp.concatenate([cart_pos, pole_cos, pole_sin, data.qvel])
        return obs[None, :]

    return reset_fn, step_fn, get_obs_fn
