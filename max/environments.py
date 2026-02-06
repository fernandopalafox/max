# environments.py

import jax
import jax.numpy as jnp
from typing import Dict, Any
from dataclasses import dataclass
from max.dynamics import create_pursuit_evader_dynamics, create_pursuit_evader_dynamics_unicycle


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

    # Specific to unicycle-double-integrator pursuit-evasion
    true_tracking_weight: float = jnp.inf
    mpc_horizon: int = -1
    learning_rate: float = jnp.inf
    max_gd_iters: int = 100
    
    


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


def make_pursuit_evasion_unicycle_double_integrator_env(params: EnvParams, true_tracking_weight):
    num_agents = params.num_agents
    dt = params.dt

    config_for_dynamics = {
        "dynamics_params": {
            "dt": dt,
            "init_tracking_weight": true_tracking_weight,
            "mpc_horizon": params.mpc_horizon,
            "learning_rate": params.learning_rate,
            "max_gd_iters": params.max_gd_iters,
        }
    }

    dynamics_model, _ = create_pursuit_evader_dynamics_unicycle(config_for_dynamics, None, None)

    true_params = {
        "model": {
            "tracking_weight": true_tracking_weight
        },
        "normalizer": None,
    }

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, evader_action: jnp.ndarray) -> jnp.ndarray:
        return dynamics_model.pred_one_step(true_params, state, evader_action)

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        # TODO: Hardcoded right now
        return jnp.array([-5.0, -5.0, 0.0, 0.0, 5.0, 5.0, 0.0, 0.0])
    
    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        return state[None, :]

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        next_state = _step_dynamics(state, action)

        terminated = False
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)

        # Dummy reward
        evader_reward = 0.0
        rewards = jnp.array([evader_reward])
        info = {"reward": evader_reward}

        return next_state, observations, rewards, terminated, truncated, info

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
        agent_velocities = jnp.zeros((1, 4))
        agent_states = jnp.concatenate([agent_positions, agent_velocities], axis=1)
        return agent_states.flatten()
    
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


def make_linear_tracking_env(params: EnvParams, true_A, true_B, target_point):
    """
    Factory function that creates a single-agent linear tracking environment.

    - State: A 4D vector [px, py, vx, vy].
    - Action: A 2D vector [ax, ay] for acceleration.
    - Dynamics: x_{t+1} = A @ x_t + B @ u_t (using ground truth A and B).
    - Reward: Negative squared distance to target point.

    Args:
        params: EnvParams dataclass with common environment parameters.
        true_A: Ground truth state transition matrix (4x4).
        true_B: Ground truth control matrix (4x2).
        target_point: Fixed target state [px, py, vx, vy] (4,).

    Returns:
        Tuple of (reset_fn, step_fn, get_obs_fn).
    """
    true_A = jnp.array(true_A)
    true_B = jnp.array(true_B)
    target_point = jnp.array(target_point)

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Linear dynamics: x_{t+1} = A @ x_t + B @ u_t"""
        return true_A @ state + true_B @ action

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Reset to random position with zero velocity."""
        pos = jax.random.uniform(
            key,
            shape=(2,),
            minval=-0.1 * params.box_half_width,
            maxval=0.1 * params.box_half_width,
            dtype=jnp.float32,
        )
        vel = jnp.zeros(2, dtype=jnp.float32)
        return jnp.concatenate([pos, vel])

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """Returns state with agent dimension."""
        return state[None, :]  # Shape: (1, 4)

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        """Steps the environment forward."""
        # Clip action to max acceleration
        clipped_action = jnp.clip(action.squeeze(), -params.max_accel, params.max_accel)

        # Step dynamics
        next_state = _step_dynamics(state, clipped_action)

        # Compute reward: negative squared distance to target
        dist_sq = jnp.sum((next_state - target_point) ** 2)
        reward = -dist_sq
        rewards = jnp.array([reward])

        # Check out-of-bounds
        pos = next_state[:2]
        is_oob = jnp.any(jnp.abs(pos) > params.box_half_width)
        terminated = is_oob

        # Check truncation
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {
            "reward": reward,
            "dist_to_target": jnp.sqrt(dist_sq),
        }

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


<<<<<<< Updated upstream
=======
def make_damped_pendulum_env(params: EnvParams, true_b, true_J):
    """
    Factory function that creates a damped pendulum environment.

    State: [phi, phi_dot] (angle from downward vertical, angular velocity)
    Action: [tau] (applied torque)
    Dynamics: Uses ground truth b (damping) and J (moment of inertia).

    Args:
        params: EnvParams dataclass with common environment parameters.
        true_b: Ground truth damping coefficient.
        true_J: Ground truth moment of inertia.

    Returns:
        Tuple of (reset_fn, step_fn, get_obs_fn).
    """
    # Known constants
    m = 1.0
    g = 9.81
    l = 1.0
    dt = params.dt

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Damped pendulum dynamics with ground truth parameters."""
        phi, phi_dot = state[0], state[1]
        tau = action[0]
        phi_ddot = (tau - true_b * phi_dot - m * g * l * jnp.sin(phi)) / true_J
        phi_next = phi + phi_dot * dt
        phi_dot_next = phi_dot + phi_ddot * dt
        return jnp.array([phi_next, phi_dot_next])

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Reset to initial state: small angle from downward vertical, at rest."""
        return jnp.array([0.1, 0.0], dtype=jnp.float32)

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """Returns state with agent dimension."""
        return state[None, :]  # Shape: (1, 2)

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        """Steps the environment forward."""
        # Clip action to torque bounds
        clipped_action = jnp.clip(action.squeeze(), -params.max_accel, params.max_accel)

        # Step dynamics
        next_state = _step_dynamics(state, jnp.atleast_1d(clipped_action))

        # Compute reward: negative swing-up cost
        phi, phi_dot = next_state[0], next_state[1]
        cost = (phi - jnp.pi) ** 2 + 0.1 * phi_dot ** 2 + 0.01 * clipped_action ** 2
        reward = -cost
        rewards = jnp.array([reward])

        # No early termination for pendulum
        terminated = False

        # Check truncation
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {
            "reward": reward,
            "angle_error": jnp.abs(phi - jnp.pi),
        }

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


def make_merging_idm_env(params: EnvParams, true_T_vec, true_b_vec, idm_params):
    """
    Factory function for the highway merging IDM environment.

    State: [ego_px, ego_py, ego_vx, ego_vy, v2_px, v2_vx, v3_px, v3_vx, v4_px, v4_vx] (10D)
    Action: [ax, ay] (2D ego acceleration)
    Dynamics: Ego double-integrator + 3 IDM vehicles with sigmoid-blended lead selection.

    Args:
        params: EnvParams dataclass with dt, max_episode_steps, max_accel.
        true_T_vec: jnp.ndarray (3,) -- ground truth time headway per IDM vehicle.
        true_b_vec: jnp.ndarray (3,) -- ground truth comfortable deceleration per IDM vehicle.
        idm_params: dict with IDM constants (v0, s0, a_max, delta, L, k_lat, d0, k_lon, s_min, p_y_target).

    Returns:
        Tuple of (reset_fn, step_fn, get_obs_fn).
    """
    dt = params.dt
    v0 = idm_params["v0"]
    s0 = idm_params["s0"]
    a_max_idm = idm_params["a_max"]
    delta = idm_params["delta"]
    L = idm_params["L"]
    k_lat = idm_params["k_lat"]
    d0 = idm_params["d0"]
    k_lon = idm_params["k_lon"]
    s_min = idm_params["s_min"]
    p_y_target = idm_params["p_y_target"]

    @jax.jit
    def _step_dynamics(state: jnp.ndarray, action: jnp.ndarray) -> jnp.ndarray:
        """Merging IDM dynamics with ground truth parameters."""
        ego_px, ego_py, ego_vx, ego_vy = state[0], state[1], state[2], state[3]
        ax, ay = action[0], action[1]

        # Ego double integrator
        next_ego_px = ego_px + ego_vx * dt + 0.5 * ax * dt**2
        next_ego_py = ego_py + ego_vy * dt + 0.5 * ay * dt**2
        next_ego_vx = jnp.maximum(ego_vx + ax * dt, 0.0)
        next_ego_vy = ego_vy + ay * dt

        # Lateral proximity sigmoid (shared)
        sigma_lat = 1.0 / (1.0 + jnp.exp(k_lat * (jnp.abs(ego_py - p_y_target) - d0)))

        # IDM vehicle states
        v2_px, v2_vx = state[4], state[5]
        v3_px, v3_vx = state[6], state[7]
        v4_px, v4_vx = state[8], state[9]

        veh_px = jnp.array([v2_px, v3_px, v4_px])
        veh_vx = jnp.array([v2_vx, v3_vx, v4_vx])

        # In-lane gaps and approach rates
        s_lane = jnp.array([1000.0, v2_px - v3_px - L, v3_px - v4_px - L])
        dv_lane = jnp.array([0.0, v3_vx - v2_vx, v4_vx - v3_vx])

        # Gaps and approach rates w.r.t. ego
        s_ego = ego_px - veh_px - L
        dv_ego = veh_vx - ego_vx

        # Blending weights
        sigma_lon = 1.0 / (1.0 + jnp.exp(-k_lon * (ego_px - veh_px)))
        alpha = sigma_lat * sigma_lon

        # Blended effective quantities
        s_eff = alpha * s_ego + (1.0 - alpha) * s_lane
        dv_eff = alpha * dv_ego + (1.0 - alpha) * dv_lane

        # Safety clamp
        s_eff = jax.nn.softplus(s_eff - s_min) + s_min

        # IDM acceleration for each vehicle
        s_star = s0 + veh_vx * true_T_vec + veh_vx * dv_eff / (2.0 * jnp.sqrt(a_max_idm * true_b_vec + 1e-8))
        a_idm = a_max_idm * (1.0 - (veh_vx / v0) ** delta - (s_star / s_eff) ** 2)

        # Update IDM vehicles
        next_veh_px = veh_px + veh_vx * dt + 0.5 * a_idm * dt**2
        next_veh_vx = jnp.maximum(veh_vx + a_idm * dt, 0.0)

        return jnp.array([
            next_ego_px, next_ego_py, next_ego_vx, next_ego_vy,
            next_veh_px[0], next_veh_vx[0],
            next_veh_px[1], next_veh_vx[1],
            next_veh_px[2], next_veh_vx[2],
        ])

    @jax.jit
    def reset_fn(key: jax.random.PRNGKey):
        """Reset: ego in merge lane alongside gap between v3 and v4."""
        return jnp.array([
            12.0, p_y_target - 3.5, 10.0, 0.0,  # ego
            24.0, 10.0,  # v2
            16.0, 10.0,  # v3
            8.0, 10.0,   # v4
        ], dtype=jnp.float32)

    @jax.jit
    def get_obs_fn(state: jnp.ndarray):
        """Returns state with agent dimension."""
        return state[None, :]  # Shape: (1, 10)

    @jax.jit
    def step_fn(state: jnp.ndarray, step_count: int, action: jnp.ndarray):
        """Steps the merging IDM environment forward."""
        clipped_action = jnp.clip(action.squeeze(), -params.max_accel, params.max_accel)

        next_state = _step_dynamics(state, jnp.atleast_1d(clipped_action))

        # Reward: penalize deviation from target speed and lane
        ego_vx_err = (next_state[2] - 10.0) ** 2
        ego_py_err = (next_state[1] - p_y_target) ** 2
        reward = -(ego_vx_err + ego_py_err)
        rewards = jnp.array([reward])

        terminated = False
        truncated = step_count >= params.max_episode_steps

        observations = get_obs_fn(next_state)
        info = {
            "reward": reward,
            "ego_py": next_state[1],
            "ego_vx": next_state[2],
        }

        return next_state, observations, rewards, terminated, truncated, info

    return reset_fn, step_fn, get_obs_fn


>>>>>>> Stashed changes
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

    # Hacky fix for linear_tracking-specific params
    if env_name == "linear_tracking":
        true_A = env_params_dict.pop("true_A", None)
        true_B = env_params_dict.pop("true_B", None)
        target_point = env_params_dict.pop("target_point", None)

<<<<<<< Updated upstream
=======
    # Hacky fix for damped_pendulum-specific params
    if env_name == "damped_pendulum":
        true_b = env_params_dict.pop("true_b", None)
        true_J = env_params_dict.pop("true_J", None)

    # Hacky fix for merging_idm-specific params
    if env_name == "merging_idm":
        true_T_vec = jnp.array(env_params_dict.pop("true_T_vec"))
        true_b_vec = jnp.array(env_params_dict.pop("true_b_vec"))
        merging_idm_params = env_params_dict.pop("idm_params")

>>>>>>> Stashed changes
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
    elif env_name == "linear_tracking":
        return make_linear_tracking_env(params, true_A, true_B, target_point)
<<<<<<< Updated upstream
=======
    elif env_name == "damped_pendulum":
        return make_damped_pendulum_env(params, true_b, true_J)
    elif env_name == "merging_idm":
        return make_merging_idm_env(params, true_T_vec, true_b_vec, merging_idm_params)
    elif env_name == "unicycle":
        true_tracking_weight = env_params_dict.pop("true_tracking_weight", None)
        return make_pursuit_evasion_unicycle_double_integrator_env(params, true_tracking_weight)
>>>>>>> Stashed changes
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")
