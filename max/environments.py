"""
2D Multi-Agent Tracking Environment

A simple cooperative task where N agents try to navigate
to a static goal position within a square bounding box.

Also includes a pursuit-evasion task and a blocker-goal-seeker task.
"""

import jax
import jax.numpy as jnp
from typing import Dict, Any
from dataclasses import dataclass


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
    def _step_dynamics(
        state: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
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
        dist_to_goal = jnp.linalg.norm(
            positions - goal, axis=1
        )  # (num_agents,)
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
            dtype=jnp.float33,
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
        )
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
        info = {}

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
    - Agent 0 (Pursuer) tries to reach the Evader (Agent 1).
    - Agent 1 (Evader) tries to avoid the pursuer.
    """

    if params.num_agents != 2:
        raise ValueError(
            "Pursuit-Evasion environment requires exactly 2 agents."
        )
    num_agents = params.num_agents  # (will be 2)
    dt = params.dt
    max_accels_config = jnp.array(
        [params.pursuer_max_accel, params.evader_max_accel]
    )
    max_speeds_config = jnp.array(
        [params.pursuer_max_speed, params.evader_max_speed]
    )

    # --- Core env logic (bound to params) ---

    @jax.jit
    def _step_dynamics(
        state: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Double integrator dynamics
        """
        goal = state[-2:]  # (not used in dynamics)
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        velocities = agent_states[:, 2:]  # (num_agents, 2)

        next_velocities = velocities + actions * dt
        speeds = jnp.linalg.norm(next_velocities, axis=1, keepdims=True)
        max_speeds_expanded = max_speeds_config.reshape(
            -1, 1
        )  # (num_agents, 1)
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
        """Checks if the pursuer and evader have collided."""
        agent_states = state[:-2].reshape(2, 4)
        pursuer_pos = agent_states[0, :2]
        evader_pos = agent_states[1, :2]

        dist = jnp.linalg.norm(pursuer_pos - evader_pos)
        dist_min = params.pursuer_size + params.evader_size
        return dist < dist_min

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
        scaled_pos_abs = (
            jnp.abs(agent_pos) / params.box_half_width
        )  # shape (2,)
        penalties = jax.vmap(_bound_scalar)(scaled_pos_abs)  # shape (2,)

        return -jnp.sum(penalties)

    @jax.jit
    def compute_rewards_fn(
        state: jnp.ndarray,
        next_state: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Compute cumulative, non-terminal rewards (OpenAI-style).
        """
        collided = is_collision_fn(next_state)

        reward_pursuer = jnp.where(collided, 10.0, 0.0)
        reward_evader = jnp.where(collided, -10.0, 0.0)

        agent_states_next = next_state[:-2].reshape(2, 4)
        positions = agent_states_next[:, :2]  # (2, 2)
        oob_penalties = jax.vmap(compute_oob_penalty)(positions)  # (2,)

        rewards = jnp.array([reward_pursuer, reward_evader]) + oob_penalties

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
            minval=-0.9 * params.box_half_width,
            maxval=0.9 * params.box_half_width,
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
        info = {}

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


def make_blocker_goal_seeker_env(params: EnvParams):
    """
    Factory function that creates the blocker-goal-seeker environment.
    - Agent 0 (Blocker) tries to keep Agent 1 away from the goal.
    - Agent 1 (Goal-Seeker) tries to reach the goal.
    - Based on new reward structure (general-sum, mutual collision penalty).
    """

    if params.num_agents != 2:
        raise ValueError(
            "Blocker-Goal-Seeker environment requires exactly 2 agents."
        )
    num_agents = params.num_agents  # (will be 2)
    dt = params.dt
    max_accels_config = jnp.array(
        [params.blocker_max_accel, params.seeker_max_accel]
    )
    max_speeds_config = jnp.array(
        [params.blocker_max_speed, params.seeker_max_speed]
    )

    # --- Core env logic (bound to params) ---

    @jax.jit
    def _step_dynamics(
        state: jnp.ndarray, actions: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Double integrator dynamics with agent-specific speed/accel limits.
        """
        goal = state[-2:]  # (not used in dynamics)
        agent_states = state[:-2].reshape(num_agents, 4)
        positions = agent_states[:, :2]  # (num_agents, 2)
        velocities = agent_states[:, 2:]  # (num_agents, 2)

        next_velocities = velocities + actions * dt
        speeds = jnp.linalg.norm(next_velocities, axis=1, keepdims=True)
        max_speeds_expanded = max_speeds_config.reshape(
            -1, 1
        )  # (num_agents, 1)
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
        """Checks if the agents (p1, p2) have collided."""
        agent_states = state[:-2].reshape(2, 4)
        p1_pos = agent_states[0, :2]
        p2_pos = agent_states[1, :2]
        dist = jnp.linalg.norm(p1_pos - p2_pos)
        return dist < params.epsilon_collide

    @jax.jit
    def is_goal_reached_fn(state: jnp.ndarray) -> bool:
        """Checks if the seeker (p2) has reached the goal (g)."""
        agent_states = state[:-2].reshape(2, 4)
        seeker_pos = agent_states[1, :2]
        goal_pos = state[-2:]
        dist = jnp.linalg.norm(seeker_pos - goal_pos)
        return dist < params.epsilon_goal

    @jax.jit
    def dist_seeker_goal_fn(state: jnp.ndarray) -> float:
        """Computes distance from seeker (p2) to goal (g)."""
        agent_states = state[:-2].reshape(2, 4)
        seeker_pos = agent_states[1, :2]
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
        scaled_pos_abs = (
            jnp.abs(agent_pos) / params.box_half_width
        )  # shape (2,)
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
        # terminated = Goal Reached (Win for A2, Loss for A1)
        terminated = check_termination_fn(next_state)
        # truncated = Time-Limit (Win for A1, Loss for A2)
        truncated = next_step_count >= params.max_episode_steps

        # 3. Compute rewards
        d_p2_g = dist_seeker_goal_fn(next_state)
        collision = is_collision_fn(next_state)
        r_win = params.reward_win
        c_collide = params.reward_collision_penalty

        # 3a. Shaping + Non-Terminal Collision Penalty
        shaping_r1 = +params.reward_shaping_k2 * d_p2_g  # A1: Area Denial
        shaping_r2 = -params.reward_shaping_k1 * d_p2_g  # A2: Potential
        # Mutual collision penalty
        collision_penalty = jnp.where(collision, -c_collide, 0.0)
        
        shaping_rewards = jnp.array([
            shaping_r1 + collision_penalty,
            shaping_r2 + collision_penalty
        ])

        # 3b. Terminal Rewards
        # A1 (Blocker) terminal reward:
        terminal_r1 = jnp.where(
            terminated, -r_win, jnp.where(truncated, +r_win, 0.0)
        )
        # A2 (Seeker) terminal reward:
        terminal_r2 = jnp.where(
            terminated, +r_win, jnp.where(truncated, -r_win, 0.0)
        )
        
        terminal_rewards = jnp.array([terminal_r1, terminal_r2])

        # 3c. OOB Penalties (from pursuit-evasion)
        agent_states_next = next_state[:-2].reshape(2, 4)
        positions = agent_states_next[:, :2]  # (2, 2)
        oob_penalties = jax.vmap(compute_oob_penalty)(positions)  # (2,)
        
        # 3d. Total Rewards
        rewards = shaping_rewards + terminal_rewards + oob_penalties

        # 4. Get observations and info
        observations = get_obs_fn(next_state)
        info = {}

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


def init_env(config: Dict[str, Any]):
    """
    Initialize environment functions based on configuration.
    """
    env_name = config.get("env_name", "multi_agent_tracking")
    env_params_dict = config.get("env_params", {})
    params = EnvParams(**env_params_dict)

    print(f"Initializing environment: {env_name}")

    if env_name == "multi_agent_tracking":
        return make_env(params)
    elif env_name == "pursuit_evasion":
        return make_pursuit_evasion_env(params)
    elif env_name == "blocker_goal_seeker":
        return make_blocker_goal_seeker_env(params)
    else:
        raise ValueError(f"Unknown environment name: '{env_name}'")