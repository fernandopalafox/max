# collect_gridworld_data.py
"""
Collect exhaustive gridworld transition data for pretraining.

Enumerates ALL (state, action, next_state) transitions in the maze.
For each navigable cell and each of the 4 actions, steps the environment
once to record the exact transition. This guarantees perfect uniform
coverage over all possible transitions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from max.environments import init_env
import argparse
import os
import pickle
import json


def get_navigable_cells(maze_layout):
    """Extract all navigable (x, y) positions from the maze."""
    navigable = []
    for y in range(len(maze_layout)):
        for x in range(len(maze_layout[0])):
            if maze_layout[y][x] > 0:  # Non-zero bitmask = navigable
                navigable.append((x, y))
    return navigable


def collect_all_transitions(step_fn, get_obs_fn, navigable_cells):
    """
    Enumerate all (state, action, next_state) transitions.

    For each navigable cell and each action (0-3), step the environment
    and record the result.

    Returns:
        states: (N, 2) array of current states
        actions: (N, 1) array of actions taken
        next_states: (N, 2) array of resulting states
    """
    all_states = []
    all_actions = []
    all_next_states = []

    for (x, y) in navigable_cells:
        state = jnp.array([float(x), float(y)])

        for action_idx in range(4):
            action = jnp.array([float(action_idx)])

            # Step the environment
            next_state, next_obs, _, _, _, _ = step_fn(state, 0, action)

            obs = get_obs_fn(state)
            current_state = obs[0]  # Remove agent dim
            next_state_obs = next_obs[0]

            all_states.append(current_state)
            all_actions.append(action)
            all_next_states.append(next_state_obs)

    states = jnp.stack(all_states)       # (N, 2)
    actions = jnp.stack(all_actions)     # (N, 1)
    next_states = jnp.stack(all_next_states)  # (N, 2)

    return states, actions, next_states


def save_exhaustive_buffer(states, actions, next_states, save_path, env_name,
                           num_repeats=1):
    """
    Save exhaustive transition data in the buffer format expected by pretrain.py.

    pretrain.py expects:
      - raw_data["states"][0]: (N, dim_state) sequential states
      - raw_data["actions"][0]: (N, dim_action) actions
      - raw_data["dones"]: (N,) done flags

    It builds transitions as (states[t], actions[t], states[t+1]) where dones[t]==0.

    We lay out pairs: [state, next_state, state, next_state, ...]
    with dones =      [  0,      1,       0,      1,      ...]
    and actions =     [action,  dummy,   action,  dummy,  ...]

    The full transition set is repeated num_repeats times to provide
    sufficient training volume.
    """
    # Tile the transitions to get enough volume
    states = np.tile(np.array(states), (num_repeats, 1))
    actions = np.tile(np.array(actions), (num_repeats, 1))
    next_states = np.tile(np.array(next_states), (num_repeats, 1))

    n_transitions = len(states)
    n_entries = n_transitions * 2
    dim_state = states.shape[1]
    dim_action = actions.shape[1]

    # Interleave: state, next_state, state, next_state, ...
    buf_states = np.zeros((n_entries, dim_state), dtype=np.float32)
    buf_actions = np.zeros((n_entries, dim_action), dtype=np.float32)
    buf_rewards = np.zeros(n_entries, dtype=np.float32)
    buf_dones = np.zeros(n_entries, dtype=np.float32)

    for i in range(n_transitions):
        buf_states[2 * i] = np.array(states[i])
        buf_states[2 * i + 1] = np.array(next_states[i])
        buf_actions[2 * i] = np.array(actions[i])
        buf_actions[2 * i + 1] = 0.0  # dummy
        buf_dones[2 * i] = 0.0        # valid transition
        buf_dones[2 * i + 1] = 1.0    # boundary (no transition to next pair)

    # Wrap in (1, N, dim) format to match buffer convention
    data = {
        "states": buf_states[np.newaxis],     # (1, 2*N, dim_state)
        "actions": buf_actions[np.newaxis],   # (1, 2*N, dim_action)
        "rewards": buf_rewards[np.newaxis],   # (1, 2*N)
        "dones": buf_dones,                   # (2*N,)
        "num_transitions": n_transitions,
    }

    os.makedirs(save_path, exist_ok=True)
    filename = f"{env_name}_buffer.pkl"
    filepath = os.path.join(save_path, filename)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)

    print(f"\nData saved to {filepath}")
    print(f"Unique transitions: {n_transitions}")
    print(f"Buffer entries: {n_entries} ({num_repeats} repeats)")
    print(f"States shape: {data['states'].shape}")
    print(f"Actions shape: {data['actions'].shape}")
    return filepath


def plot_transition_coverage(states, actions, next_states, maze_layout):
    """Visualize which transitions are covered."""
    maze_arr = np.array(maze_layout)
    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw maze
    for y in range(10):
        for x in range(10):
            bitmask = maze_arr[y, x]
            if bitmask == 0:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='black', alpha=0.8))
            else:
                ax.add_patch(plt.Rectangle((x-0.5, y-0.5), 1, 1, color='white', edgecolor='gray', linewidth=0.5))

    # Draw transitions as arrows
    states_np = np.array(states)
    next_np = np.array(next_states)

    for i in range(len(states_np)):
        dx = next_np[i, 0] - states_np[i, 0]
        dy = next_np[i, 1] - states_np[i, 1]
        if abs(dx) > 0.1 or abs(dy) > 0.1:  # Movement occurred
            ax.annotate("",
                xy=(next_np[i, 0], next_np[i, 1]),
                xytext=(states_np[i, 0], states_np[i, 1]),
                arrowprops=dict(arrowstyle="->", color="blue", alpha=0.3, lw=1.5),
            )
        else:  # Blocked by wall - red dot
            ax.scatter(states_np[i, 0], states_np[i, 1], color='red', s=5, alpha=0.3)

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(-0.5, 9.5)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'All Transitions ({len(states)} total)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    return fig


def main(config):
    wandb.init(
        project=config.get("wandb_project", "gridworld_data_collection"),
        config=config,
        name=f"collect_{config['env_name']}_exhaustive",
        reinit=True,
    )

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Get all navigable cells
    maze_layout = config["env_params"]["maze_layout"]
    navigable_cells = get_navigable_cells(maze_layout)
    num_repeats = config.get("num_repeats", 100)
    n_unique = len(navigable_cells) * 4
    print(f"Found {len(navigable_cells)} navigable cells")
    print(f"Will enumerate {n_unique} unique transitions (4 actions per cell)")
    print(f"Repeating {num_repeats}x for {n_unique * num_repeats} total training pairs")

    # Collect all transitions
    states, actions, next_states = collect_all_transitions(
        step_fn, get_obs_fn, navigable_cells
    )

    # Count movement vs blocked
    deltas = np.array(next_states - states)
    n_moved = np.sum(np.any(np.abs(deltas) > 0.1, axis=1))
    n_blocked = len(states) - n_moved
    print(f"\nTransitions: {n_moved} movement + {n_blocked} blocked = {len(states)} total")

    # Plot coverage
    fig = plot_transition_coverage(states, actions, next_states, maze_layout)
    wandb.log({"transition_coverage": wandb.Image(fig)})
    plt.close(fig)

    # Save in buffer format
    save_exhaustive_buffer(
        states, actions, next_states,
        config["save_path"], config["env_name"],
        num_repeats=num_repeats,
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect exhaustive gridworld transition data"
    )
    parser.add_argument("--config", type=str, default="gridworld.json")
    parser.add_argument("--save-dir", type=str, default=None)
    parser.add_argument("--wandb-project", type=str, default=None)
    args = parser.parse_args()

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    config = full_config["data_collection"]

    if args.save_dir:
        config["save_path"] = args.save_dir
    if args.wandb_project:
        config["wandb_project"] = args.wandb_project

    main(config)
