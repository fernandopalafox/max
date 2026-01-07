# environments.py

import jax
import jax.numpy as jnp
from typing import Dict, Any
from dataclasses import dataclass
from max.dynamics import create_pursuit_evader_dynamics, create_unicycle_mpc_dynamics


@dataclass(frozen=True)
class EnvParams:
    """Environment parameters."""

    num_agents: int = 2
    box_half_width: float = 10.0
    max_episode_steps: int = 200
    dt: float = 0.1

    # Default for cooperative task
    max_accel: float = 2.0

    # Specific to pursuit-evasion
    pursuer_size: float = 0.075
    evader_size: float = 0.05
    pursuer_max_accel: float = 3.0
    evader_max_accel: float = 4.0
    pursuer_max_speed: float = 1.0
    evader_max_speed: float = 1.3

    # Specific to blocker-goal-seeker
    blocker_max_accel: float = 3.0
    seeker_max_accel: float = 4.0
    blocker_max_speed: float = 1.0
    seeker_max_speed: float = 1.3
    epsilon_goal: float = 0.1
    epsilon_collide: float = 0.125
    reward_win: float = 10.0
    reward_collision_penalty: float = 1.0  # C_collide
    reward_shaping_k1: float = 0.01  # k_1 for goal-seeker potential
    reward_shaping_k2: float = 0.01  # k_2 for blocker area denial

    # Specific to LQR pursuit-evasion
    


def make_env(params: EnvParams):
    """
    Factory function that creates cooperative tracking environment functions.
    """

    # --- Close over physics and env parameters ---
    num_agents = params.num_agents
    dt = params.dt
    max_accel = params.max_accel
    max_dist = params.box_half_width * 2.0 * jnp.sqrt(2.0)
    coop_max_speeds = jnp.full((num_agents,), jnp.inf)

    # --- Core env logic (bound to params) ---

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Double integrator dynamics
        """
        goal = state[-2:]
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        velocities = agent_states[:, 2:]  # (num_agents, 2)
        next_velocities = velocities + actions * dt
        speeds = jnp.linalg.norm(next_velocities, axis=1, keepdims=True)
        max_speeds_expanded = coop_max_speeds.reshape(-1, 1)  # (num_agents, 1)
        scales = jnp.minimum(1.0, max_speeds_expanded / (speeds + 1e-6))
        clipped_velocities = next_velocities * scales
        next_positions = positions + velocities * dt + 0.5 * actions * dt**2
        next_agent_states = jnp.concatenate(
            [next_positions, clipped_velocities], axis=1
        )  # (num_agents, 4)
        next_state = jnp.concatenate([next_agent_states.flatten(), goal])
        return next_state

    @jax.jit
    def compute_potential_fn(state: jnp.ndarray) -> jnp.ndarray:
        """Computes the potential for each agent based on distance to goal."""
        goal = state[-2:]
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        dist_to_goal = jnp.linalg.norm(positions - goal, axis=1)  # (num_agents,)
        normalized_dist = dist_to_goal / max_dist
        potential = 1.0 - normalized_dist
        return potential

    @jax.jit
    def compute_rewards_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """Reward = Delta Potential + OOB_Penalty"""
        potential_prev = compute_potential_fn(state)
        potential_next = compute_potential_fn(next_state)
        delta_potential_reward = potential_next - potential_prev

        agent_states = next_state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]

        is_oob = (jnp.abs(positions[:, 0]) > params.box_half_width) | (
            jnp.abs(positions[:, 1]) > params.box_half_width
        )
        oob_penalties = jnp.where(is_oob, -1.0, 0.0)

        rewards = delta_potential_reward + oob_penalties
        return rewards

    @jax.jit
    def check_termination_fn(state: jnp.ndarray) -> bool:
        """Terminates if any agent is OOB."""
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]
        is_oob = (jnp.abs(positions[:, 0]) > params.box_half_width) | (
            jnp.abs(positions[:, 1]) > params.box_half_width
        )
        return jnp.any(is_oob)

    # --- Public-facing API functions ---

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Resets environment."""
        key, goal_key = jax.random.split(key)
        goal_pos = jax.random.uniform(
            goal_key,
            shape=(2,),
            minval=-0.8 * params.box_half_width,
            maxval=0.8 * params.box_half_width,
            dtype=jnp.float32,
        )
        agent_positions = jax.random.uniform(
            key,
            shape=(num_agents, 2),
            minval=-params.box_half_width,
            maxval=params.box_half_width,
            dtype=jnp.float32,
        )
        agent_velocities = jnp.zeros((num_agents, 2), dtype=jnp.float32)
        agent_states = jnp.concatenate([agent_positions, agent_velocities], axis=1)
        state = jnp.concatenate([agent_states.flatten(), goal_pos])
        return state

    @jax.jit
    def step_fn(
        state: jnp.ndarray,
        step_count: int,
        actions: jnp.ndarray,
    ):
        """Steps the environment forward."""
        clipped_actions = jnp.clip(actions, -max_accel, max_accel)
        next_state = _step_dynamics(state, clipped_actions)

        rewards = compute_rewards_fn(state, next_state)
        next_step_count = step_count + 1
        terminated = check_termination_fn(next_state)
        truncated = next_step_count >= params.max_episode_steps
        observations = get_obs_fn(next_state)
        info = {"avg_reward": jnp.mean(rewards), "collision": False}

        return (
            next_state,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """Replicates the global state for each agent."""
        return jnp.stack([state] * num_agents)

    return reset_fn, step_fn, get_obs_fn


def make_pursuit_evasion_env(params: EnvParams):
    """
    Factory function that creates pursuit-evasion environment functions.
    - Agents 0..N-2 (Pursuers) try to reach the Evader.
    - Agent N-1 (Evader) tries to avoid the pursuers.
    """

    num_agents = params.num_agents
    if num_agents < 2:
        raise ValueError("Pursuit-Evasion environment requires at least 2 agents.")

    dt = params.dt

    # Generalized speed/accel configs
    # The first N-1 agents are pursuers, the last one is the evader.
    pursuer_accels = jnp.full((num_agents - 1,), params.pursuer_max_accel)
    evader_accel = jnp.array([params.evader_max_accel])
    max_accels_config = jnp.concatenate([pursuer_accels, evader_accel])

    pursuer_speeds = jnp.full((num_agents - 1,), params.pursuer_max_speed)
    evader_speed = jnp.array([params.evader_max_speed])
    max_speeds_config = jnp.concatenate([pursuer_speeds, evader_speed])

    # --- Core env logic (bound to params) ---

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Double integrator dynamics
        """
        goal = state[-2:]  # (not used in dynamics)
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        velocities = agent_states[:, 2:]  # (num_agents, 2)

        next_velocities = velocities + actions * dt
        speeds = jnp.linalg.norm(next_velocities, axis=1, keepdims=True)
        max_speeds_expanded = max_speeds_config.reshape(-1, 1)  # (num_agents, 1)
        scales = jnp.minimum(1.0, max_speeds_expanded / (speeds + 1e-6))
        clipped_velocities = next_velocities * scales
        next_positions = positions + velocities * dt + 0.5 * actions * dt**2
        next_agent_states = jnp.concatenate(
            [next_positions, clipped_velocities], axis=1
        )  # (num_agents, 4)
        next_state = jnp.concatenate([next_agent_states.flatten(), goal])
        return next_state

    @jax.jit
    def is_collision_fn(state: jnp.ndarray) -> bool:
        """Checks if any pursuer and the evader have collided."""
        agent_states = state[:-2].reshape(num_agents, 4)
        all_pos = agent_states[:, :2]
        pursuer_pos = all_pos[:-1]  # All agents except the last
        evader_pos = all_pos[-1]  # The last agent

        dist = jnp.linalg.norm(pursuer_pos - evader_pos, axis=1)
        dist_min = params.pursuer_size + params.evader_size
        return jnp.any(dist < dist_min)

    @jax.jit
    def _bound_scalar(x_scaled: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible helper for the OpenAI 'bound' penalty."""
        return jnp.where(
            x_scaled < 0.9,
            0.0,
            jnp.where(
                x_scaled < 1.0,
                (x_scaled - 0.9) * 10.0,
                jnp.minimum(jnp.exp(2.0 * x_scaled - 2.0), 10.0),
            ),
        )

    @jax.jit
    def compute_oob_penalty(agent_pos: jnp.ndarray) -> jnp.ndarray:
        """Computes the OpenAI-style OOB penalty for a single agent."""
        scaled_pos_abs = jnp.abs(agent_pos) / params.box_half_width  # shape (2,)
        penalties = jax.vmap(_bound_scalar)(scaled_pos_abs)  # shape (2,)

        return -jnp.sum(penalties)

    @jax.jit
    def compute_rewards_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute cumulative, non-terminal rewards (OpenAI-style).
        OOB penalty is ONLY applied to the evader (last agent).
        """
        # This flag enables the dense, shaped rewards.
        # Set to False to revert to sparse collision-only rewards.
        use_dense_reward = True

        collided = is_collision_fn(next_state)

        # --- Get positions from next_state ---
        agent_states_next = next_state[:-2].reshape(num_agents, 4)
        positions = agent_states_next[:, :2]  # (num_agents, 2)
        pursuer_pos = positions[:-1]  # (num_agents-1, 2)
        evader_pos = positions[-1]  # (2,)

        # --- 1. Dense/Shaped Rewards (MPE2-style) ---
        # Distances from each pursuer to the evader
        distances_to_evader = jnp.linalg.norm(
            pursuer_pos - evader_pos, axis=1
        )  # (num_agents-1,)

        # Pursuers: rewarded for minimizing distance to evader
        # -0.1 * distance (for each pursuer)
        pursuer_dense_rewards = -0.1 * distances_to_evader  # (num_agents-1,)

        # Evader: rewarded for maximizing sum of distances to all pursuers
        # +0.1 * sum(distances)
        evader_dense_reward = 0.1 * jnp.sum(distances_to_evader)  # scalar

        # Combine into one array
        dense_rewards_arr = jnp.concatenate(
            [pursuer_dense_rewards, jnp.array([evader_dense_reward])]
        )

        # Toggle dense rewards
        # If not use_dense_reward, this becomes an array of zeros
        dense_rewards = jnp.where(
            use_dense_reward, dense_rewards_arr, jnp.zeros(num_agents)
        )

        # --- 2. Collision Rewards ---
        # This is the "terminal" reward given at the step of collision
        pursuer_terminal_reward = jnp.where(collided, 10.0, 0.0)
        evader_terminal_reward = jnp.where(collided, -10.0, 0.0)

        pursuer_rewards_arr = jnp.full((num_agents - 1,), pursuer_terminal_reward)
        evader_reward_arr = jnp.array([evader_terminal_reward])
        collision_rewards = jnp.concatenate([pursuer_rewards_arr, evader_reward_arr])

        # --- 3. OOB Penalties (Evader only) ---
        # 1. Calculate penalty ONLY for the evader (last agent)
        evader_oob_penalty = compute_oob_penalty(evader_pos)  # scalar

        # 2. Create OOB penalty array, applying penalty only to the evader
        pursuer_oob_penalties = jnp.zeros(num_agents - 1)
        evader_oob_penalty_arr = jnp.array([evader_oob_penalty])

        oob_penalties = jnp.concatenate([pursuer_oob_penalties, evader_oob_penalty_arr])

        # --- 4. Total rewards ---
        # Total reward is the sum of all three components
        rewards = dense_rewards + collision_rewards + oob_penalties

        return rewards

    @jax.jit
    def check_termination_fn(state: jnp.ndarray) -> bool:
        """
        Termination only occurs via truncation (max_episode_steps).
        Collision is non-terminal.
        """
        return jnp.array(False)

    # --- Public-facing API functions ---

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Resets environment."""
        key, goal_key = jax.random.split(key)
        goal_pos = jax.random.uniform(
            goal_key,
            shape=(2,),
            minval=-0.8 * params.box_half_width,
            maxval=0.8 * params.box_half_width,
            dtype=jnp.float32,
        )
        agent_positions = jax.random.uniform(
            key,
            shape=(num_agents, 2),
            minval=-params.box_half_width,
            maxval=params.box_half_width,
            dtype=jnp.float32,
        )
        agent_velocities = jnp.zeros((num_agents, 2), dtype=jnp.float32)
        agent_states = jnp.concatenate(
            [agent_positions, agent_velocities], axis=1
        )  # (num_agents, 4)
        state = jnp.concatenate([agent_states.flatten(), goal_pos])
        return state

    @jax.jit
    def step_fn(
        state: jnp.ndarray,
        step_count: int,
        actions: jnp.ndarray,
    ):
        """Steps the environment forward."""
        clipped_actions = jnp.clip(
            actions, -max_accels_config[:, None], max_accels_config[:, None]
        )
        next_state = _step_dynamics(state, clipped_actions)
        rewards = compute_rewards_fn(state, next_state)
        next_step_count = step_count + 1
        terminated = check_termination_fn(next_state)
        truncated = next_step_count >= params.max_episode_steps
        observations = get_obs_fn(next_state)
        info = {
            "reward_pursuer_avg": jnp.mean(rewards[:-1]),
            "reward_evader": rewards[-1],
            "collision": is_collision_fn(next_state),
        }

        return (
            next_state,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """Replicates the global state for each agent."""
        return jnp.stack([state] * num_agents)

    return reset_fn, step_fn, get_obs_fn


def make_pursuit_evasion_lqr_env(params: EnvParams, true_q_diag, true_r_diag):
    num_agents = params.num_agents
    dt = params.dt

    config_for_dynamics = {
        "dynamics_params": {
            "dt": dt,
            "init_q_diag": true_q_diag,
            "init_r_diag": true_r_diag,
        }
    }

    dynamics_model, _ = create_pursuit_evader_dynamics(config_for_dynamics, None, None)

    true_q_cholesky = jnp.diag(jnp.sqrt(jnp.array(true_q_diag)))
    true_r_cholesky = jnp.diag(jnp.sqrt(jnp.array(true_r_diag)))
    true_params = {
        "model": {
            "q_cholesky": true_q_cholesky,
            "r_cholesky": true_r_cholesky,
        },
        "normalizer": None,
    }

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, evader_action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_model.pred_one_step(true_params, state, evader_action)

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        agent_positions = jax.random.uniform(
            key,
            shape=(1, 4),
            minval=-params.box_half_width,
            maxval=params.box_half_width,
        )
        # FIX: State layout must be [evader_x, evader_y, evader_vx, evader_vy, pursuer_x, pursuer_y, pursuer_vx, pursuer_vy]
        # Previous code incorrectly concatenated as [positions, velocities] instead of interleaving per agent.
        evader_state = jnp.concatenate([agent_positions[0, 0:2], jnp.zeros(2)])
        pursuer_state = jnp.concatenate([agent_positions[0, 2:4], jnp.zeros(2)])
        return jnp.concatenate([evader_state, pursuer_state])
    
    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        return state[None, :]

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        next_state = _step_dynamics(state, action)

        evader_pos = next_state[0:2]
        pursuer_pos = next_state[4:6]
        dist_sq = jnp.sum((evader_pos - pursuer_pos) ** 2)

        pursuer_reward = -dist_sq
        evader_reward = dist_sq
        rewards = jnp.array([evader_reward])

        terminated = False
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {"reward": pursuer_reward}

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


def make_blocker_goal_seeker_env(params: EnvParams):
    """
    Factory function that creates the blocker-goal-seeker environment.
    - Agents 0..N-2 (Blockers) try to keep the Goal-Seeker from the goal.
    - Agent N-1 (Goal-Seeker) tries to reach the goal.
    - Based on new reward structure (general-sum, mutual collision penalty).
    """

    num_agents = params.num_agents
    dt = params.dt

    # Generalized speed/accel configs
    # The first N-1 agents are blockers, the last one is the seeker.
    blocker_accels = jnp.full((num_agents - 1,), params.blocker_max_accel)
    seeker_accel = jnp.array([params.seeker_max_accel])
    max_accels_config = jnp.concatenate([blocker_accels, seeker_accel])

    blocker_speeds = jnp.full((num_agents - 1,), params.blocker_max_speed)
    seeker_speed = jnp.array([params.seeker_max_speed])
    max_speeds_config = jnp.concatenate([blocker_speeds, seeker_speed])

    # --- Core env logic (bound to params) ---

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Double integrator dynamics with agent-specific speed/accel limits.
        """
        goal = state[-2:]  # (not used in dynamics)
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        velocities = agent_states[:, 2:]  # (num_agents, 2)

        next_velocities = velocities + actions * dt
        speeds = jnp.linalg.norm(next_velocities, axis=1, keepdims=True)
        max_speeds_expanded = max_speeds_config.reshape(-1, 1)  # (num_agents, 1)
        scales = jnp.minimum(1.0, max_speeds_expanded / (speeds + 1e-6))
        clipped_velocities = next_velocities * scales
        next_positions = positions + velocities * dt + 0.5 * actions * dt**2
        next_agent_states = jnp.concatenate(
            [next_positions, clipped_velocities], axis=1
        )  # (num_agents, 4)
        next_state = jnp.concatenate([next_agent_states.flatten(), goal])
        return next_state

    @jax.jit
    def is_collision_fn(state: jnp.ndarray) -> bool:
        """
        Checks if *any* blocker has collided with the seeker.
        """
        agent_states = state[:-2].reshape(num_agents, 4)
        all_pos = agent_states[:, :2]
        blocker_pos = all_pos[:-1]  # All agents except the last
        seeker_pos = all_pos[-1]  # The last agent

        # Calculate distance from all blockers to the seeker
        dist = jnp.linalg.norm(blocker_pos - seeker_pos, axis=1)

        # Return True if any distance is less than the threshold
        return jnp.any(dist < params.epsilon_collide)

    @jax.jit
    def is_goal_reached_fn(state: jnp.ndarray) -> bool:
        """
        Checks if the seeker (last agent) has reached the goal.
        """
        agent_states = state[:-2].reshape(num_agents, 4)
        seeker_pos = agent_states[-1, :2]  # Use last agent
        goal_pos = state[-2:]
        dist = jnp.linalg.norm(seeker_pos - goal_pos)
        return dist < params.epsilon_goal

    @jax.jit
    def dist_seeker_goal_fn(state: jnp.ndarray) -> float:
        """
        Computes distance from seeker (last agent) to goal.
        """
        agent_states = state[:-2].reshape(num_agents, 4)
        seeker_pos = agent_states[-1, :2]  # Use last agent
        goal_pos = state[-2:]
        return jnp.linalg.norm(seeker_pos - goal_pos)

    @jax.jit
    def _bound_scalar(x_scaled: jnp.ndarray) -> jnp.ndarray:
        """JAX-compatible helper for the OpenAI 'bound' penalty."""
        return jnp.where(
            x_scaled < 0.9,
            0.0,
            jnp.where(
                x_scaled < 1.0,
                (x_scaled - 0.9) * 10.0,
                jnp.minimum(jnp.exp(2.0 * x_scaled - 2.0), 10.0),
            ),
        )

    @jax.jit
    def compute_oob_penalty(agent_pos: jnp.ndarray) -> jnp.ndarray:
        """Computes the OpenAI-style OOB penalty for a single agent."""
        scaled_pos_abs = jnp.abs(agent_pos) / params.box_half_width  # shape (2,)
        penalties = jax.vmap(_bound_scalar)(scaled_pos_abs)  # shape (2,)
        return -jnp.sum(penalties)

    @jax.jit
    def check_termination_fn(state: jnp.ndarray) -> bool:
        """
        Termination only occurs if the goal is reached.
        Time-limit truncation is handled in step_fn.
        """
        return is_goal_reached_fn(state)

    # --- Public-facing API functions ---

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Resets environment."""
        key, goal_key = jax.random.split(key)
        # Goal position
        goal_pos = jax.random.uniform(
            goal_key,
            shape=(2,),
            minval=-0.8 * params.box_half_width,
            maxval=0.8 * params.box_half_width,
            dtype=jnp.float32,
        )
        # Agent positions
        agent_positions = jax.random.uniform(
            key,
            shape=(num_agents, 2),
            minval=-0.9 * params.box_half_width,
            maxval=0.9 * params.box_half_width,
            dtype=jnp.float32,
        )
        # Agent velocities
        agent_velocities = jnp.zeros((num_agents, 2), dtype=jnp.float32)
        agent_states = jnp.concatenate(
            [agent_positions, agent_velocities], axis=1
        )  # (num_agents, 4)
        state = jnp.concatenate([agent_states.flatten(), goal_pos])
        return state

    @jax.jit
    def step_fn(
        state: jnp.ndarray,
        step_count: int,
        actions: jnp.ndarray,
    ):
        """Steps the environment forward."""
        # 1. Step physics
        clipped_actions = jnp.clip(
            actions, -max_accels_config[:, None], max_accels_config[:, None]
        )
        next_state = _step_dynamics(state, clipped_actions)

        # 2. Check terminations
        next_step_count = step_count + 1
        # terminated = Goal Reached (Win for Seeker, Loss for Blockers)
        terminated = check_termination_fn(next_state)
        # truncated = Time-Limit (Win for Blockers, Loss for Seeker)
        truncated = next_step_count >= params.max_episode_steps

        # 3. Compute rewards

        # Potential-based shaping rewards
        d_p2_g_prev = dist_seeker_goal_fn(state)
        d_p2_g_next = dist_seeker_goal_fn(next_state)

        potential_1_prev = +params.reward_shaping_k2 * d_p2_g_prev
        potential_1_next = +params.reward_shaping_k2 * d_p2_g_next

        potential_2_prev = -params.reward_shaping_k1 * d_p2_g_prev
        potential_2_next = -params.reward_shaping_k1 * d_p2_g_next

        shaping_r1 = potential_1_next - potential_1_prev  # Area Denial
        shaping_r2 = potential_2_next - potential_2_prev  # Potential

        # Build shaping reward array
        blocker_shaping_rewards = jnp.full((num_agents - 1,), shaping_r1)
        seeker_shaping_reward = jnp.array([shaping_r2])
        shaping_rewards = jnp.concatenate(
            [blocker_shaping_rewards, seeker_shaping_reward]
        )

        # Other rewards
        collision = is_collision_fn(next_state)
        r_win = params.reward_win
        c_collide = params.reward_collision_penalty

        # Collision penalty is applied to all agents
        collision_penalty = jnp.where(collision, -c_collide, 0.0)

        # terminal rewards
        terminal_r1 = jnp.where(
            terminated, -r_win, jnp.where(truncated, +r_win, 0.0)
        )  # Blocker terminal
        terminal_r2 = jnp.where(
            terminated, +r_win, jnp.where(truncated, -r_win, 0.0)
        )  # Seeker terminal

        # Build terminal reward array
        blocker_terminal_rewards = jnp.full((num_agents - 1,), terminal_r1)
        seeker_terminal_reward = jnp.array([terminal_r2])
        terminal_rewards = jnp.concatenate(
            [blocker_terminal_rewards, seeker_terminal_reward]
        )

        # out-of-bounds rewards
        agent_states_next = next_state[:-2].reshape(num_agents, 4)
        positions = agent_states_next[:, :2]  # (num_agents, 2)
        oob_penalties = jax.vmap(compute_oob_penalty)(positions)  # (num_agents,)

        rewards = shaping_rewards + collision_penalty + terminal_rewards + oob_penalties

        # 4. Get observations and info
        observations = get_obs_fn(next_state)

        # 5. Construct info dict
        info = {
            "reward_blocker_avg": jnp.mean(rewards[:-1]),
            "reward_seeker": rewards[-1],
            "collision": collision,
            "shaping_reward_blocker": shaping_r1,
            "shaping_reward_seeker": shaping_r2,
            "terminal_reward_blocker": terminal_r1,
            "terminal_reward_seeker": terminal_r2,
            "oob_penalty_blocker_avg": jnp.mean(oob_penalties[:-1]),
            "oob_penalty_seeker": oob_penalties[-1],
        }

        return (
            next_state,
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """
        Replicates the global state for each agent.
        """
        return jnp.stack([state] * num_agents)

    return reset_fn, step_fn, get_obs_fn


def make_unicycle_mpc_env(params: EnvParams, true_theta1, true_theta2, dynamics_config):
    """
    Factory function that creates unicycle MPC environment.

    Player 1 (evader): double integrator, controlled by the planner
    Player 2 (opponent): unicycle with MPC, uses true_theta1/theta2

    State: 8D [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
    Action: 2D [a1x, a1y] for evader's acceleration
    """
    dt = params.dt

    # Create dynamics with TRUE parameters (for environment simulation)
    config_for_dynamics = {
        "dynamics_params": {
            "dt": dt,
            "newton_iters": dynamics_config.get("newton_iters", 10),
            "init_theta1": true_theta1,
            "init_theta2": true_theta2,
            "weight_w": dynamics_config.get("weight_w", 0.1),
            "weight_a": dynamics_config.get("weight_a", 1.0),
            "weight_speed": dynamics_config.get("weight_speed", 0.0),
            "target_speed": dynamics_config.get("target_speed", 1.0),
        }
    }

    dynamics_model, _ = create_unicycle_mpc_dynamics(config_for_dynamics, None, None)

    true_params = {
        "model": {
            "theta1": jnp.array(true_theta1),
            "theta2": jnp.array(true_theta2),
        },
        "normalizer": None,
    }

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, evader_action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_model.pred_one_step(true_params, state, evader_action)

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Reset environment with random initial positions."""
        key1, key2, key3 = jax.random.split(key, 3)

        # Evader (P1): random position, zero velocity
        p1 = jax.random.uniform(
            key1, shape=(2,),
            minval=-0.5 * params.box_half_width,
            maxval=0.5 * params.box_half_width,
        )
        v1 = jnp.zeros(2)

        # Opponent (P2): random position, random heading, some initial speed
        p2 = jax.random.uniform(
            key2, shape=(2,),
            minval=-0.5 * params.box_half_width,
            maxval=0.5 * params.box_half_width,
        )
        alpha2 = jax.random.uniform(key3, minval=-jnp.pi, maxval=jnp.pi)
        v2 = 0.5  # initial speed

        # State: [p1x, p1y, v1x, v1y, p2x, p2y, alpha2, v2]
        return jnp.array([p1[0], p1[1], v1[0], v1[1], p2[0], p2[1], alpha2, v2])

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        return state[None, :]

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        # Clip evader action
        clipped_action = jnp.clip(action, -params.evader_max_accel, params.evader_max_accel)

        next_state = _step_dynamics(state, clipped_action)

        # Reward: evader wants to maximize distance
        evader_pos = next_state[0:2]
        opponent_pos = next_state[4:6]
        dist_sq = jnp.sum((evader_pos - opponent_pos) ** 2)

        evader_reward = dist_sq
        rewards = jnp.array([evader_reward])

        terminated = False
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {"distance": jnp.sqrt(dist_sq)}

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


def init_env(config: Dict[str, Any]):
    """
    Initialize environment functions based on configuration.
    """
    env_name = config.get("env_name", None)
    env_params_dict = config.get("env_params", {})

    # Hacky fix for LQR-specific params
    if env_name == "pursuit_evasion_lqr":
        true_q_diag = env_params_dict.pop("true_q_diag", None)
        true_r_diag = env_params_dict.pop("true_r_diag", None)

    # Hacky fix for unicycle-specific params
    if env_name == "unicycle_mpc":
        true_theta1 = env_params_dict.pop("true_theta1", None)
        true_theta2 = env_params_dict.pop("true_theta2", None)
        dynamics_config = config.get("dynamics_params", {})

    params = EnvParams(**env_params_dict)

    print(f"Initializing environment: {env_name}")

    if env_name == "multi_agent_tracking":
        return make_env(params)
    elif env_name == "pursuit_evasion":
        return make_pursuit_evasion_env(params)
    elif env_name == "pursuit_evasion_lqr":
        return make_pursuit_evasion_lqr_env(params, true_q_diag, true_r_diag)
    elif env_name == "blocker_goal_seeker":
        return make_blocker_goal_seeker_env(params)
    elif env_name == "unicycle_mpc":
        return make_unicycle_mpc_env(params, true_theta1, true_theta2, dynamics_config)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
