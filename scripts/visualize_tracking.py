"""
Simple script to visualize trained multi-agent tracking policies
with a goal that switches at a fixed interval.

Usage:
    python visualize_tracking.py --policy-path [PATH_TO_POLICY] --goal-switch-interval 25
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Circle, Rectangle
import pickle
import argparse
import sys
import os

from max.normalizers import init_normalizer
from max.policies import init_policy
from max.environments import init_env


# ============================================================================
# Configuration - MUST MATCH TRAINING CONFIG
# ============================================================================

# This config is for the "multi_agent_tracking" environment
CONFIG = {
    "env_name": "multi_agent_tracking",
    "env_params": {
        "num_agents": 3,
        "box_half_width": 1.0,
        "max_episode_steps": 100,
        "dt": 0.1,
        "max_accel": 2.0,
        "pursuer_max_accel": 3.0,
        "evader_max_accel": 4.0,
        "pursuer_max_speed": 1.0,
        "evader_max_speed": 1.3,
        "pursuer_size": 0.075,
        "evader_size": 0.05,
        "reward_shaping_k1": 1.0,
        "reward_shaping_k2": 1.0,
        "reward_collision_penalty": 1.0,
    },
    "total_steps": 100_000,
    "num_agents": 3,
    "dim_state": 14,
    "dim_action": 2,
    "train_freq": 1,
    "train_policy_freq": 2048,
    "normalize_freq": 1000000,
    "eval_freq": 100,
    "eval_traj_horizon": 100,
    "normalization": {"method": "static"},
    "normalization_params": {
        "state": {
            "min": [
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
            ],
            "max": [
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
            ],
        },
        "action": {
            "min": [-2.0, -2.0],
            "max": [2.0, 2.0],
        },
    },
    "policy": "actor-critic",
    "policy_params": {
        "hidden_layers": [64, 64],
    },
    "policy_trainer": "ippo",
    "policy_trainer_params": {
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "ppo_lambda": 0.95,
        "ppo_gamma": 0.99,
        "clip_epsilon": 0.2,
        "n_epochs": 4,
        "mini_batch_size": 64,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "policy_evaluator_params": {
        "n_episodes": 10,
    },
    "reward_scaling_discount_factor": 0.99,
    "reward_clip": 100.0,
}


# ============================================================================
# CUSTOMIZE YOUR INITIAL CONDITIONS HERE!
# ============================================================================

# Option 1: Use random initialization (set to None)
CUSTOM_INITIAL_STATES = None  # Will use random initialization
CUSTOM_GOAL = None  # Will use random initialization

# Option 2: Use custom states (for 3-agent tracking)
# CUSTOM_INITIAL_STATES = [
#     jnp.array([-0.5, 0.5, 0.0, 0.0]),   # Agent 0
#     jnp.array([0.5, 0.5, 0.0, 0.0]),  # Agent 1
#     jnp.array([0.0, -0.5, 0.0, 0.0]),  # Agent 2
# ]
# CUSTOM_GOAL = jnp.array([0.0, 0.0]) # Initial goal


# ============================================================================


def load_policy(policy_path, config):
    """Load trained policy parameters."""
    print(f"Loading policy from {policy_path}...")

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize normalizer
    normalizer, norm_params = init_normalizer(config)

    # Initialize policy structure
    key = jax.random.key(0)
    policy, policy_state = init_policy(key, config, None, None, normalizer, norm_params)

    # Load saved parameters
    with open(policy_path, "rb") as f:
        trained_params = pickle.load(f)

    print("Policy loaded successfully!")
    return policy, trained_params, reset_fn, step_fn, get_obs_fn


def run_episode(
    policy,
    params,
    reset_fn,
    step_fn,
    get_obs_fn,
    config,
    goal_switch_interval,
    seed=0,
):
    """Run a single episode with the policy, switching goals."""
    key = jax.random.key(seed)
    key, reset_key, goal_key_init = jax.random.split(key, 3)

    # Reset environment - returns concatenated state: [agent_0(4D), ..., goal(2D)]
    state = reset_fn(reset_key)

    # Extract number of agents and env params
    num_agents = config["num_agents"]
    env_params = config["env_params"]
    box_half_width = env_params["box_half_width"]
    env_name = config["env_name"]

    # Extract individual agent states and goal from concatenated state
    agent_states = state[:-2].reshape(num_agents, 4)  # Shape: (num_agents, 4)
    goal = state[-2:]  # Last 2 elements are goal position

    # Apply custom initial conditions if specified
    if CUSTOM_INITIAL_STATES is not None:
        for i, custom_state in enumerate(CUSTOM_INITIAL_STATES):
            if i < num_agents:
                agent_states = agent_states.at[i].set(custom_state)
    if CUSTOM_GOAL is not None:
        goal = CUSTOM_GOAL

    # Reconstruct state with custom initial conditions
    state = jnp.concatenate([agent_states.flatten(), goal])

    print(f"\nInitial state (Env: {env_name}):")
    for i in range(num_agents):
        label = f"Agent {i}"
        print(
            f"  {label} ({i}): pos=({agent_states[i, 0]:.2f}, {agent_states[i, 1]:.2f}), "
            f"vel=({agent_states[i, 2]:.2f}, {agent_states[i, 3]:.2f})"
        )
    print(f"  Initial Goal:    pos=({goal[0]:.2f}, {goal[1]:.2f})")
    print(f"  Goal switching every {goal_switch_interval} steps.")

    # Get observations (replicated for all agents)
    current_obs = get_obs_fn(state)

    # Storage for trajectory (support dynamic number of agents)
    traj_agent_states = [[] for _ in range(num_agents)]
    traj_goals = []
    traj_actions = [[] for _ in range(num_agents)]
    traj_rewards = []
    traj_info = []

    # Store initial states
    for i in range(num_agents):
        traj_agent_states[i].append(np.array(agent_states[i]))
    traj_goals.append(np.array(goal))  # Store initial goal

    # Run episode
    done = False
    max_steps = config["env_params"]["max_episode_steps"]
    env_step_count = 0

    # Key for goal switching
    goal_key = goal_key_init

    while not done and env_step_count < max_steps:

        # --- NEW: GOAL SWITCHING LOGIC ---
        if env_step_count > 0 and env_step_count % goal_switch_interval == 0:
            print(f"Step {env_step_count}: Switching goal...")
            key, goal_key = jax.random.split(key)
            new_goal = jax.random.uniform(
                goal_key,
                shape=(2,),
                minval=-0.8 * box_half_width,
                maxval=0.8 * box_half_width,
                dtype=jnp.float32,
            )

            # Update state and observations with the new goal
            state = state.at[-2:].set(new_goal)
            current_obs = get_obs_fn(state)
        # --- END GOAL SWITCHING LOGIC ---

        # Select actions deterministically (vmap over agents)
        actions = jax.vmap(policy.select_action_deterministic, in_axes=(0, 0, None))(
            params, current_obs, None
        )

        # Step environment
        state, current_obs, rewards, terminated, truncated, info = step_fn(
            state, env_step_count, np.array(actions)
        )

        env_step_count += 1
        done = terminated or truncated

        # Extract agent states and goal from new state for storage
        agent_states = state[:-2].reshape(num_agents, 4)
        goal = state[-2:]  # This goal may or may not have changed

        # Store trajectory
        for i in range(num_agents):
            traj_agent_states[i].append(np.array(agent_states[i]))
            traj_actions[i].append(np.array(actions[i]))
        traj_goals.append(np.array(goal))  # Store goal at each step
        traj_rewards.append(np.array(rewards))
        traj_info.append(info)

    # Convert to arrays
    traj_agent_states = [np.array(traj) for traj in traj_agent_states]
    traj_goals = np.array(traj_goals)
    traj_actions = [np.array(traj) for traj in traj_actions]
    traj_rewards = np.array(traj_rewards)
    # traj_info is a list of dicts

    print(f"\nEpisode finished after {env_step_count} steps")
    for i in range(num_agents):
        print(f"Total rewards: Agent {i}={traj_rewards[:, i].sum():.2f}")

    print(
        f"Termination: {'Terminated' if terminated else ('Max steps' if truncated else 'Other')}"
    )

    # Return in format compatible with animation
    return {
        "agent_states": traj_agent_states,  # List of arrays, one per agent
        "goals": traj_goals,  # Array of goals (T+1, 2)
        "actions": traj_actions,  # List of arrays, one per agent
        "rewards": traj_rewards,
        "info": traj_info,
        "num_agents": num_agents,
    }


def animate_trajectory(
    trajectory, config, save_path="tracking_animation.gif", fps=10, seed=None
):
    """Create an animation of the multi-agent tracking trajectory."""
    agent_states = trajectory["agent_states"]  # List of arrays, one per agent
    goals = trajectory["goals"]  # NEW: Array (T+1, 2) of goals
    num_agents = trajectory["num_agents"]
    rewards = trajectory["rewards"]
    info_list = trajectory["info"]

    env_name = config["env_name"]
    dt = config["env_params"]["dt"]
    box_half_width = config["env_params"]["box_half_width"]

    # --- Define Agent-Specific Visuals ---

    agent_labels = []
    colors = []
    dark_colors = []
    agent_sizes = []

    if env_name == "pursuit_evasion":
        # ... (This logic remains, in case env_name is set to it)
        num_pursuers = num_agents - 1
        agent_labels = [f"Pursuer {i}" for i in range(num_pursuers)] + ["Evader"]
        colors = ["red"] * num_pursuers + ["blue"]
        dark_colors = ["darkred"] * num_pursuers + ["darkblue"]
        agent_sizes = [config["env_params"]["pursuer_size"]] * num_pursuers + [
            config["env_params"]["evader_size"]
        ]

    elif env_name == "blocker_goal_seeker":
        # ... (This logic remains)
        num_blockers = num_agents - 1
        agent_labels = [f"Blocker {i}" for i in range(num_blockers)] + ["Seeker"]
        colors = ["red"] * num_blockers + ["blue"]
        dark_colors = ["darkred"] * num_blockers + ["darkblue"]
        agent_sizes = [config["env_params"].get("epsilon_collide", 0.05)] * num_agents

    else:  # Default/Cooperative (multi_agent_tracking will use this)
        color_cycle = ["red", "blue", "green", "orange", "purple"]
        dark_color_cycle = [
            "darkred",
            "darkblue",
            "darkgreen",
            "darkorange",
            "darkviolet",
        ]

        agent_labels = [f"Agent {i}" for i in range(num_agents)]
        colors = [color_cycle[i % len(color_cycle)] for i in range(num_agents)]
        dark_colors = [
            dark_color_cycle[i % len(dark_color_cycle)] for i in range(num_agents)
        ]
        agent_sizes = [0.05] * num_agents  # Default size

    # Create figure
    fig, (ax_main, ax_reward) = plt.subplots(
        2, 1, figsize=(10, 12), gridspec_kw={"height_ratios": [3, 1]}
    )

    # Set up main plot
    ax_main.set_aspect("equal")
    margin = 0.25
    ax_main.set_xlim(-box_half_width - margin, box_half_width + margin)
    ax_main.set_ylim(-box_half_width - margin, box_half_width + margin)
    ax_main.grid(True, alpha=0.3)
    ax_main.set_xlabel("X Position (m)")
    ax_main.set_ylabel("Y Position (m)")
    ax_main.set_title(f"Environment: '{env_name}' (Seed: {seed})")

    # Draw boundary
    boundary = Rectangle(
        (-box_half_width, -box_half_width),
        2 * box_half_width,
        2 * box_half_width,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        alpha=0.5,
    )
    ax_main.add_patch(boundary)

    # Draw goal (if not pursuit-evasion)
    if env_name != "pursuit_evasion":
        (goal_marker,) = ax_main.plot(
            goals[0, 0], goals[0, 1], "g*", markersize=20, zorder=5, label="Goal"
        )
    else:
        goal_marker = None  # No goal marker for PE

    # Initialize trajectory lines
    agent_lines = []
    for i in range(num_agents):
        (line,) = ax_main.plot(
            [],
            [],
            "-",
            color=colors[i],
            alpha=0.4,
            linewidth=2,
            label=f"{agent_labels[i]} path",
        )
        agent_lines.append(line)

    # Initialize agent markers
    agent_markers = []
    for i in range(num_agents):
        marker = Circle(
            (0, 0),
            agent_sizes[i],
            facecolor=colors[i],
            edgecolor=dark_colors[i],
            linewidth=2,
            zorder=10,
        )
        ax_main.add_patch(marker)
        agent_markers.append(marker)

    # Initialize velocity arrows
    agent_vel_arrows = []
    for i in range(num_agents):
        arrow = ax_main.quiver(
            [0],
            [0],
            [0],
            [0],
            color=colors[i],
            scale=10,
            alpha=0.7,
            zorder=9,
            width=0.01,
        )
        agent_vel_arrows.append(arrow)

    ax_main.legend(loc="upper right")

    # Time and info text
    time_text = ax_main.text(
        0.02,
        0.98,
        "",
        transform=ax_main.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Set up reward plot
    ax_reward.set_xlim(0, len(rewards))
    if len(rewards) > 0:
        reward_min = rewards.min() - 0.1
        reward_max = rewards.max() + 0.1
    else:
        reward_min, reward_max = -1, 1
    ax_reward.set_ylim(reward_min, reward_max)
    ax_reward.set_xlabel("Time Step")
    ax_reward.set_ylabel("Reward")
    ax_reward.grid(True, alpha=0.3)
    ax_reward.axhline(y=0, color="k", linestyle="-", linewidth=0.5, alpha=0.3)

    # Initialize reward lines
    agent_reward_lines = []
    for i in range(num_agents):
        (line,) = ax_reward.plot(
            [], [], "-", color=colors[i], label=f"{agent_labels[i]}", linewidth=2
        )
        agent_reward_lines.append(line)
    ax_reward.legend(loc="upper right")

    def init():
        for line in agent_lines:
            line.set_data([], [])
        for line in agent_reward_lines:
            line.set_data([], [])
        time_text.set_text("")
        return []

    def animate(frame):
        # Update trajectories
        for i in range(num_agents):
            agent_lines[i].set_data(
                agent_states[i][: frame + 1, 0],
                agent_states[i][: frame + 1, 1],
            )

        # Update agent positions
        for i in range(num_agents):
            agent_markers[i].center = (
                agent_states[i][frame, 0],
                agent_states[i][frame, 1],
            )

        # Update velocities
        for i in range(num_agents):
            agent_vel_arrows[i].set_offsets(
                [[agent_states[i][frame, 0], agent_states[i][frame, 1]]]
            )
            agent_vel_arrows[i].set_UVC(
                [agent_states[i][frame, 2]], [agent_states[i][frame, 3]]
            )

        # --- NEW: Update Goal Marker Position ---
        current_goal = goals[frame]
        if goal_marker:
            goal_marker.set_data([current_goal[0]], [current_goal[1]])

        # Calculate distances and build text
        text_lines = [f"Time: {frame*dt:.1f}s (step {frame})"]

        if env_name == "pursuit_evasion":
            # ... (text logic)
            pass
        elif env_name == "blocker_goal_seeker":
            # ... (text logic)
            pass
        else:  # Default (multi_agent_tracking)
            for i in range(num_agents):
                # --- NEW: Use current_goal for distance ---
                dist_to_goal = np.linalg.norm(agent_states[i][frame, :2] - current_goal)
                text_lines.append(f"Dist A{i}-Goal: {dist_to_goal:.2f}m")

        text_lines.append("Cumulative Rewards:")
        for i in range(num_agents):
            text_lines.append(f"  {agent_labels[i]}: {rewards[:frame, i].sum():.1f}")

        # Update text
        time_text.set_text("\n".join(text_lines))

        # Update reward plot
        for i in range(num_agents):
            agent_reward_lines[i].set_data(range(frame), rewards[:frame, i])

        # Return all artists
        artists = (
            agent_lines
            + agent_markers
            + agent_vel_arrows
            + [time_text]
            + agent_reward_lines
        )
        if goal_marker:
            artists.append(goal_marker)

        return artists

    print(f"Creating animation with {len(agent_states[0])} frames...")
    ani = animation.FuncAnimation(
        fig,
        animate,
        init_func=init,
        frames=len(agent_states[0]),  # Number of steps + 1
        interval=1000 / fps,
        blit=True,
    )

    print(f"Saving animation to {save_path}...")
    ani.save(save_path, writer="pillow", fps=fps, dpi=100)
    plt.close(fig)
    print(f"Animation saved to {save_path}")

    return ani


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained multi-agent tracking policy"
    )
    parser.add_argument(
        "--policy-path",
        type=str,
        default="trained_policies/ippo/policy_params.pkl",
        help="Path to trained policy parameters (.pkl file).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=5,
        help="Random seed for episode initialization",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="figures/tracking_animation.gif",
        help="Output path for animation",
    )
    parser.add_argument(
        "--fps", type=int, default=40, help="Frames per second for animation"
    )
    parser.add_argument(  # --- NEW ARGUMENT ---
        "--goal-switch-interval",
        type=int,
        default=25,
        help="Number of steps before switching to a new random goal.",
    )
    args = parser.parse_args()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Load policy
    # Note: We use the CONFIG from this file, which MUST match the one
    # used for training the policy specified in --policy-path.
    policy, params, reset_fn, step_fn, get_obs_fn = load_policy(
        args.policy_path, CONFIG
    )

    # Run episode
    print(f"\nRunning episode with seed {args.seed}...")
    trajectory = run_episode(
        policy,
        params,
        reset_fn,
        step_fn,
        get_obs_fn,
        CONFIG,
        goal_switch_interval=args.goal_switch_interval,  # Pass new arg
        seed=args.seed,
    )

    # Animate
    print(f"\nCreating animation...")
    animate_trajectory(
        trajectory, CONFIG, save_path=args.output, fps=args.fps, seed=args.seed
    )

    print(f"\nDone! Open {args.output} to view the result.")


if __name__ == "__main__":
    main()
