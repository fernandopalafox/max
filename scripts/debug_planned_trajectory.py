# debug_planned_trajectory.py
"""
Debug script to visualize what the learned model + iCEM planner thinks
the open-loop trajectory will look like.

This script:
1. Initializes the learned dynamics model and iCEM planner
2. Gets an initial state from the environment
3. Computes a full action trajectory using iCEM
4. Rolls out the trajectory using the LEARNED MODEL (not the simulator)
5. Animates the predicted state trajectory

Optionally, can load a TDMPC2 episode and compare ground truth vs predicted.
"""

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import pickle
import wandb
from scipy.stats import wasserstein_distance

from max.normalizers import init_normalizer
from max.environments import make_cheetah_env, EnvParams
from max.dynamics import init_dynamics, create_cheetah_ground_truth
from max.dynamics_trainers import init_trainer
from max.planners import init_planner
from max.costs import init_cost

from collect_data_cheetah import create_cheetah_xy_animation


def get_num_episodes(data_path):
    """Get the number of episodes in the dataset."""
    with open(data_path, "rb") as f:
        data = pickle.load(f)
    dones = data["dones"]
    return int(np.sum(dones))


def load_tdmpc2_episode(data_path, episode_idx, max_steps=None, verbose=True):
    """
    Load a single episode from TDMPC2 data.

    Returns:
        states: (T, 17) ground truth states
        actions: (T-1, 6) actions (one less than states)
    """
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    states_all = data["states"][0]  # (N, 17)
    actions_all = data["actions"][0]  # (N, 6)
    dones = data["dones"]  # (N,)

    # Find episode boundaries
    done_indices = np.where(dones == 1.0)[0]
    ep_starts = np.concatenate([[0], done_indices[:-1] + 1])
    ep_ends = done_indices

    n_episodes = len(ep_starts)
    if episode_idx >= n_episodes:
        if verbose:
            print(f"Episode {episode_idx} out of range, using episode 0")
        episode_idx = 0

    start = ep_starts[episode_idx]
    end = ep_ends[episode_idx]

    ep_states = states_all[start:end + 1]
    ep_actions = actions_all[start:end]  # actions are one less than states

    if max_steps is not None and len(ep_actions) > max_steps:
        ep_states = ep_states[:max_steps + 1]
        ep_actions = ep_actions[:max_steps]

    if verbose:
        print(f"Loaded episode {episode_idx}: {len(ep_states)} states, "
              f"{len(ep_actions)} actions")

    return ep_states, ep_actions


def rollout_learned_model(init_state, actions, dynamics_model, dyn_params):
    """
    Roll out the learned dynamics model autoregressively.

    Args:
        init_state: Initial 17D state (without rootx)
        actions: Action sequence of shape (horizon, dim_action)
        dynamics_model: Dynamics model with pred_one_step method
        dyn_params: Parameters for the dynamics model

    Returns:
        states: Predicted states of shape (horizon+1, 17)
    """
    def step(state, action):
        next_state = dynamics_model.pred_one_step(dyn_params, state, action)
        return next_state, next_state

    _, predicted_states = jax.lax.scan(step, init_state, actions)

    # Concatenate initial state
    all_states = jnp.concatenate([init_state[None, :], predicted_states], axis=0)
    return all_states


def obs_17d_to_state_18d(obs_17d_sequence, dt=0.01):
    """
    Convert 17D observations to 18D states by integrating rootx from vel_x.

    17D obs layout: [rootz, rooty, joints..., vel_x, vel_z, vel_y, joint_vels...]
    18D state layout: [rootx, rootz, rooty, joints..., vel_x, vel_z, vel_y, joint_vels...]

    Args:
        obs_17d_sequence: Array of shape (T, 17)
        dt: Timestep for integration

    Returns:
        states_18d: Array of shape (T, 18)
    """
    obs_17d_sequence = np.array(obs_17d_sequence)
    T = len(obs_17d_sequence)

    # Extract vel_x (index 8 in 17D)
    vel_x = obs_17d_sequence[:, 8]

    # Integrate to get rootx (starting from 0)
    rootx = np.zeros(T)
    for t in range(1, T):
        rootx[t] = rootx[t-1] + vel_x[t-1] * dt

    # Build 18D states
    states_18d = np.zeros((T, 18))
    states_18d[:, 0] = rootx  # rootx
    states_18d[:, 1:] = obs_17d_sequence  # rest is the same

    return states_18d


def run_mpc_rollout(init_obs, init_mjx_data, planner, planner_state, cost_params,
                    step_fn, get_obs_fn, num_steps, seed_actions=None, use_gt_dynamics=False):
    """
    MPC rollout: replan at each step, execute first action, step MJX simulator.

    Args:
        init_obs: Initial 17D observation
        init_mjx_data: Initial MJX data state
        planner: iCEM planner
        planner_state: Initial planner state
        cost_params: Cost function parameters
        step_fn: MJX step function
        get_obs_fn: Function to extract 17D obs from mjx.Data
        num_steps: Number of MPC steps
        seed_actions: If provided, seed first planning step with these actions
        use_gt_dynamics: If True, pass mjx_data to planner (for GT dynamics cost fn)

    Returns:
        states: (num_steps+1, 17) trajectory
        actions: (num_steps, 6) executed actions
    """
    states = [init_obs]
    actions = []
    state = init_obs
    mjx_data = init_mjx_data

    for t in range(num_steps):
        # Seed only on first step if seed_actions provided
        if t == 0 and seed_actions is not None:
            tdmpc2_flat = seed_actions.flatten()
            num_elites = planner_state.elites.shape[0]
            seeded_elites = jnp.tile(tdmpc2_flat[None, :], (num_elites, 1))
            planner_state = planner_state.replace(mean=tdmpc2_flat, elites=seeded_elites)

        # Plan - pass mjx_data if using GT dynamics, else 17D obs
        planner_input = mjx_data if use_gt_dynamics else state
        action_seq, planner_state = planner.solve(planner_state, planner_input, cost_params)

        # Execute first action
        action = action_seq[0]
        actions.append(action)

        # Step MJX simulator (not learned model)
        mjx_data, next_obs, _, _, _, _ = step_fn(mjx_data, t, action[None, :])
        state = get_obs_fn(mjx_data).squeeze()
        states.append(state)

        if (t + 1) % 10 == 0:
            print(f"  MPC step {t + 1}/{num_steps}")

    return jnp.stack(states), jnp.stack(actions)


def run_mpc_comparison(config, data_path, episode_idx, seed_icem=False, use_gt_dynamics=False):
    """Run MPC rollout comparison with optional TDMPC2 warm-start.

    Args:
        use_gt_dynamics: If True, iCEM plans using ground truth MJX dynamics instead of learned model.
    """
    action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    suffix = "_gt" if use_gt_dynamics else ""
    run_name = f"mpc_rollout_ep{episode_idx}" + ("_seeded" if seed_icem else "") + suffix
    wandb.init(project="cheetah_debug", config=config, name=run_name, reinit=True)

    key = jax.random.key(config["seed"])

    # Initialize MJX environment
    print("Initializing MJX environment...")
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)

    if use_gt_dynamics:
        # Use ground truth MJX dynamics for planning
        print("Using GROUND TRUTH dynamics for iCEM planning...")
        dynamics_model, init_params = create_cheetah_ground_truth(config)
        # Override cost type to work with mjx.Data
        config = config.copy()
        config["cost_type"] = "cheetah_velocity_tracking"
        # Dummy train_state params (GT dynamics ignores params)
        class DummyTrainState:
            params = {"model": None, "normalizer": None}
            covariance = None
        train_state = DummyTrainState()
    else:
        # Use learned dynamics model
        print("Initializing learned dynamics model...")
        normalizer, norm_params = init_normalizer(config)
        key, model_key = jax.random.split(key)
        dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

        print("Initializing trainer...")
        key, trainer_key = jax.random.split(key)
        _, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Load TDMPC2 episode for warm-start actions (if needed)
    horizon = config["planner_params"]["horizon"]
    gt_states, tdmpc2_actions = load_tdmpc2_episode(data_path, episode_idx)
    tdmpc2_actions = jnp.array(tdmpc2_actions)
    num_steps = config.get("total_steps", 100)
    print(f"Running MPC for {num_steps} steps (horizon={horizon})")

    # Reset MJX to get initial state (fresh reset, not TDMPC2's initial state)
    print("Resetting MJX environment...")
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    init_obs = get_obs_fn(mjx_data).squeeze()

    # Initialize planner
    print("Initializing planner...")
    cost_fn = init_cost(config, dynamics_model)
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    target_velocity = config["cost_fn_params"]["target_velocity"]
    cost_params = {
        "dyn_params": train_state.params,
        "params_cov_model": train_state.covariance,
        "target_velocity": target_velocity,
    }

    # Run MPC rollout with MJX stepping
    seed_actions = tdmpc2_actions[:horizon] if seed_icem else None
    dynamics_label = "GT dynamics" if use_gt_dynamics else "learned dynamics"
    seed_label = "TDMPC2 warm-start" if seed_icem else "cold start"
    label = f"iCEM ({seed_label}, {dynamics_label})"
    print(f"\nRunning MPC rollout: {label}...")

    key, planner_key = jax.random.split(key)
    planner_state = planner_state.replace(key=planner_key)
    mpc_states, mpc_actions = run_mpc_rollout(
        init_obs, mjx_data, planner, planner_state, cost_params,
        step_fn, get_obs_fn, num_steps,
        seed_actions=seed_actions,
        use_gt_dynamics=use_gt_dynamics
    )

    # Convert to numpy for plotting
    mpc_states_np = np.array(mpc_states)
    mpc_actions_np = np.array(mpc_actions)
    gt_states_np = np.array(gt_states[:num_steps + 1])
    tdmpc2_actions_np = np.array(tdmpc2_actions[:num_steps])

    # Print stats
    vel_x_mpc = mpc_states_np[:, 8]
    vel_x_gt = gt_states_np[:, 8]
    print(f"\n=== Velocity Stats ===")
    print(f"  TDMPC2 GT mean vel_x: {np.mean(vel_x_gt):.2f} m/s")
    print(f"  MPC mean vel_x: {np.mean(vel_x_mpc):.2f} m/s")
    print(f"  Target vel_x: {target_velocity:.2f} m/s")

    # Plot velocity comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    timesteps = np.arange(len(vel_x_gt)) * 0.01
    ax.plot(timesteps, vel_x_gt, label="TDMPC2 Ground Truth", linewidth=2)
    ax.plot(timesteps[:len(vel_x_mpc)], vel_x_mpc, label=label, linewidth=2, linestyle="--")
    ax.axhline(target_velocity, color="red", linestyle=":", label=f"Target ({target_velocity:.1f} m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Forward Velocity: TDMPC2 vs iCEM MPC")
    ax.legend()
    ax.grid(True, alpha=0.3)
    wandb.log({"mpc/velocity_plot": wandb.Image(fig)})
    plt.close(fig)

    # Plot action comparison
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    action_timesteps = np.arange(num_steps) * 0.01

    for i, (ax, alabel) in enumerate(zip(axes, action_labels)):
        ax.plot(action_timesteps, tdmpc2_actions_np[:, i], label="TDMPC2", linewidth=2, color="blue")
        ax.plot(action_timesteps, mpc_actions_np[:, i], label=label, linewidth=2, color="orange", linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Action")
        ax.set_title(f"{alabel}")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Action Comparison: TDMPC2 vs {label}", fontsize=14)
    plt.tight_layout()
    wandb.log({"mpc/actions_comparison": wandb.Image(fig)})
    plt.close(fig)

    # Action smoothness
    print(f"\n=== Action Smoothness (mean |Δa|) ===")
    print(f"{'':12} {'TDMPC2':>8} {'MPC':>8}")
    for i, alabel in enumerate(action_labels):
        t_delta = np.abs(np.diff(tdmpc2_actions_np[:, i])).mean()
        m_delta = np.abs(np.diff(mpc_actions_np[:, i])).mean()
        print(f"{alabel:12} {t_delta:8.3f} {m_delta:8.3f}")

    # Animation
    print("\nGenerating animation...")
    mpc_states_18d = obs_17d_to_state_18d(mpc_states_np)
    gt_states_18d = obs_17d_to_state_18d(gt_states_np)
    combined_states = np.stack([gt_states_18d, mpc_states_18d], axis=0)
    save_path = config.get("save_path", "mpc_comparison.gif")
    gif_path = create_cheetah_xy_animation(combined_states, save_path=save_path)
    print(f"Animation saved to: {gif_path}")
    wandb.log({"mpc/animation": wandb.Video(gif_path, fps=20, format="gif")})

    wandb.finish()
    print("Done.")


def run_prediction_drift_test(config):
    """
    The "smoking gun" OOD test:
    1. iCEM picks an action sequence for the current state
    2. Roll out those actions through the LEARNED MODEL (what iCEM optimized for)
    3. Roll out same actions through MJX SIMULATOR (ground truth)
    4. Compare trajectories - large divergence = iCEM steering into "fantasy land"
    """
    wandb.init(project="cheetah_debug", config=config, name="prediction_drift_test", reinit=True)

    key = jax.random.key(config["seed"])

    # Initialize MJX environment
    print("Initializing MJX environment...")
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)

    # Initialize learned dynamics model
    print("Initializing learned dynamics model...")
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(model_key, config, normalizer, norm_params)

    print("Initializing trainer...")
    key, trainer_key = jax.random.split(key)
    _, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Initialize planner
    print("Initializing planner...")
    cost_fn = init_cost(config, dynamics_model)
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    target_velocity = config["cost_fn_params"]["target_velocity"]
    cost_params = {
        "dyn_params": train_state.params,
        "params_cov_model": train_state.covariance,
        "target_velocity": target_velocity,
    }

    # Reset MJX to get initial state
    print("Resetting MJX environment...")
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    init_obs = get_obs_fn(mjx_data).squeeze()

    # iCEM plans action sequence
    horizon = config["planner_params"]["horizon"]
    print(f"Running iCEM to get {horizon}-step action plan...")
    key, planner_key = jax.random.split(key)
    planner_state = planner_state.replace(key=planner_key)
    planned_actions, _ = planner.solve(planner_state, init_obs, cost_params)
    planned_actions *= 0.0

    # === Rollout 1: Learned model (what iCEM "thought" would happen) ===
    print("Rolling out actions through LEARNED MODEL...")
    predicted_states = rollout_learned_model(
        init_obs, planned_actions, dynamics_model, train_state.params
    )

    # === Rollout 2: MJX simulator (ground truth) ===
    print("Rolling out actions through MJX SIMULATOR...")
    actual_states = [init_obs]
    data = mjx_data
    for t in range(horizon):
        action = planned_actions[t]
        data, _, _, _, _, _ = step_fn(data, t, action[None, :])
        actual_states.append(get_obs_fn(data).squeeze())
    actual_states = jnp.stack(actual_states)

    # Convert to numpy
    predicted_states_np = np.array(predicted_states)
    actual_states_np = np.array(actual_states)

    # Per-step prediction error
    per_step_error = np.linalg.norm(predicted_states_np - actual_states_np, axis=1)
    cumulative_error = per_step_error  # already cumulative since states diverge

    # State dimension labels
    state_labels = ["rootz", "rooty", "bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot",
                    "vel_x", "vel_z", "vel_y", "vel_bthigh", "vel_bshin", "vel_bfoot",
                    "vel_fthigh", "vel_fshin", "vel_ffoot"]

    # Per-dimension error at final step
    final_error = np.abs(predicted_states_np[-1] - actual_states_np[-1])

    print(f"\n=== Prediction Drift Results (horizon={horizon}) ===")
    print(f"  Initial error: {per_step_error[0]:.4f}")
    print(f"  Final error:   {per_step_error[-1]:.4f}")
    print(f"  Mean error:    {per_step_error.mean():.4f}")
    print(f"\n  Per-dimension final error:")
    for label, err in zip(state_labels, final_error):
        print(f"    {label:15s}: {err:.4f}")

    # Velocity comparison
    vel_x_pred = predicted_states_np[:, 8]
    vel_x_actual = actual_states_np[:, 8]
    print(f"\n  vel_x comparison:")
    print(f"    Predicted mean: {vel_x_pred.mean():.2f} m/s")
    print(f"    Actual mean:    {vel_x_actual.mean():.2f} m/s")
    print(f"    Target:         {target_velocity:.2f} m/s")

    # === Plotting ===
    timesteps = np.arange(horizon + 1) * 0.02  # dt = 0.02s

    # Plot 1: Velocity trajectories
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(timesteps, vel_x_pred, label="Learned Model (iCEM's belief)", linewidth=2)
    ax.plot(timesteps, vel_x_actual, label="MJX Simulator (reality)", linewidth=2, linestyle="--")
    ax.axhline(target_velocity, color="red", linestyle=":", label=f"Target ({target_velocity:.1f} m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("vel_x (m/s)")
    ax.set_title("Forward Velocity: Predicted vs Actual")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Per-step prediction error
    ax = axes[0, 1]
    ax.plot(timesteps, per_step_error, linewidth=2, color="red")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("||predicted - actual||")
    ax.set_title("Prediction Error Over Horizon")
    ax.grid(True, alpha=0.3)

    # Plot 3: Position dimensions
    ax = axes[1, 0]
    for i, label in enumerate(state_labels[:8]):
        ax.plot(timesteps, predicted_states_np[:, i] - actual_states_np[:, i],
                label=label, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted - Actual")
    ax.set_title("Position Prediction Error by Dimension")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    # Plot 4: Velocity dimensions
    ax = axes[1, 1]
    for i, label in enumerate(state_labels[8:], start=8):
        ax.plot(timesteps, predicted_states_np[:, i] - actual_states_np[:, i],
                label=label, alpha=0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Predicted - Actual")
    ax.set_title("Velocity Prediction Error by Dimension")
    ax.legend(loc="upper left", fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Prediction Drift Test: iCEM Actions on Learned Model vs MJX", fontsize=14)
    plt.tight_layout()
    wandb.log({"drift/error_plots": wandb.Image(fig)})
    fig.savefig("/tmp/prediction_drift.png", dpi=100)
    plt.close(fig)
    print(f"\nPlot saved: /tmp/prediction_drift.png")

    # Animation: predicted (main) vs actual (ghost)
    print("Generating animation...")
    predicted_18d = obs_17d_to_state_18d(predicted_states_np)
    actual_18d = obs_17d_to_state_18d(actual_states_np)
    combined = np.stack([predicted_18d, actual_18d], axis=0)
    gif_path = create_cheetah_xy_animation(combined, save_path="/tmp/prediction_drift.gif")
    print(f"Animation saved: {gif_path}")
    print("  Solid = learned model prediction, ghost = MJX reality")
    wandb.log({"drift/animation": wandb.Video(gif_path, fps=20, format="gif")})

    wandb.finish()
    return predicted_states_np, actual_states_np, per_step_error


def run_multi_episode_analysis(config, data_path, num_episodes):
    """Run distribution analysis across multiple random episodes."""
    action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]

    key = jax.random.key(config["seed"])

    # Initialize dynamics model
    print("Initializing dynamics model...")
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    print("Initializing trainer...")
    key, trainer_key = jax.random.split(key)
    _, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    # Initialize planner
    print("Initializing planner...")
    cost_fn = init_cost(config, dynamics_model)
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    target_velocity = config["cost_fn_params"]["target_velocity"]
    cost_params = {
        "dyn_params": train_state.params,
        "params_cov_model": train_state.covariance,
        "target_velocity": target_velocity,
    }

    horizon = config["planner_params"]["horizon"]

    # Get total episodes and sample
    total_episodes = get_num_episodes(data_path)
    num_episodes = min(num_episodes, total_episodes)
    key, sample_key = jax.random.split(key)
    episode_indices = jax.random.choice(
        sample_key, total_episodes, shape=(num_episodes,), replace=False
    )
    episode_indices = np.array(episode_indices)

    print(f"\nAnalyzing {num_episodes} random episodes (out of {total_episodes})...")

    # Collect all actions
    all_tdmpc2_actions = []
    all_icem_actions = []
    all_tdmpc2_deltas = []
    all_icem_deltas = []

    for i, ep_idx in enumerate(episode_indices):
        gt_states, tdmpc2_actions = load_tdmpc2_episode(
            data_path, int(ep_idx), max_steps=horizon, verbose=False
        )
        init_obs = jnp.array(gt_states[0])
        tdmpc2_actions = jnp.array(tdmpc2_actions)
        ep_horizon = len(tdmpc2_actions)

        # Run iCEM
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)
        icem_actions, planner_state = planner.solve(
            planner_state, init_obs, cost_params
        )
        icem_actions = icem_actions[:ep_horizon]

        all_tdmpc2_actions.append(np.array(tdmpc2_actions))
        all_icem_actions.append(np.array(icem_actions))
        all_tdmpc2_deltas.append(np.diff(np.array(tdmpc2_actions), axis=0))
        all_icem_deltas.append(np.diff(np.array(icem_actions), axis=0))

        print(f"  Episode {ep_idx}: {ep_horizon} steps")

    # Concatenate all actions
    all_tdmpc2 = np.concatenate(all_tdmpc2_actions, axis=0)
    all_icem = np.concatenate(all_icem_actions, axis=0)
    all_tdmpc2_d = np.concatenate(all_tdmpc2_deltas, axis=0)
    all_icem_d = np.concatenate(all_icem_deltas, axis=0)

    # Print aggregate stats
    print(f"\n=== Aggregate Action Distribution ({num_episodes} episodes) ===")
    print(f"{'':12} {'TDMPC2':^16} {'iCEM':^16} {'Wasserstein':>12}")
    print(f"{'':12} {'mean':>8} {'std':>8} {'mean':>8} {'std':>8} {'distance':>12}")
    for i, label in enumerate(action_labels):
        t_mean, t_std = all_tdmpc2[:, i].mean(), all_tdmpc2[:, i].std()
        c_mean, c_std = all_icem[:, i].mean(), all_icem[:, i].std()
        wd = wasserstein_distance(all_tdmpc2[:, i], all_icem[:, i])
        print(f"{label:12} {t_mean:8.3f} {t_std:8.3f} {c_mean:8.3f} {c_std:8.3f} {wd:12.3f}")

    print(f"\n=== Aggregate Action Smoothness (mean |Δa|) ===")
    print(f"{'':12} {'TDMPC2':>8} {'iCEM':>8}")
    for i, label in enumerate(action_labels):
        t_delta = np.abs(all_tdmpc2_d[:, i]).mean()
        c_delta = np.abs(all_icem_d[:, i]).mean()
        print(f"{label:12} {t_delta:8.3f} {c_delta:8.3f}")

    print(f"\nTotal samples: {len(all_tdmpc2)}")


def main(config, tdmpc2_episode=None, data_path=None, compare_icem=False, seed_icem=False):
    use_tdmpc2 = tdmpc2_episode is not None

    run_name = "tdmpc2_comparison" if use_tdmpc2 else "planned_trajectory_debug"
    wandb.init(
        project="cheetah_debug",
        config=config,
        name=run_name,
        reinit=True,
    )

    key = jax.random.key(config["seed"])

    # Initialize learned dynamics model
    print("Initializing dynamics model...")
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    # Initialize trainer to get train_state (which holds the params)
    print("Initializing trainer...")
    key, trainer_key = jax.random.split(key)
    _, train_state = init_trainer(config, dynamics_model, init_params, trainer_key)

    if use_tdmpc2:
        # Load TDMPC2 episode
        print(f"\nLoading TDMPC2 episode {tdmpc2_episode}...")
        horizon = config["planner_params"]["horizon"]
        gt_states, tdmpc2_actions = load_tdmpc2_episode(
            data_path, tdmpc2_episode, max_steps=horizon
        )
        init_obs = jnp.array(gt_states[0])
        tdmpc2_actions = jnp.array(tdmpc2_actions)
        horizon = len(tdmpc2_actions)
        print(f"Using {horizon} actions from TDMPC2 episode")

        # If compare_icem, run iCEM on the same initial state
        icem_actions = None
        if compare_icem:
            print("\nRunning iCEM on initial state for comparison...")
            cost_fn = init_cost(config, dynamics_model)
            key, planner_key = jax.random.split(key)
            planner, planner_state = init_planner(config, cost_fn, planner_key)

            target_velocity = config["cost_fn_params"]["target_velocity"]
            cost_params = {
                "dyn_params": train_state.params,
                "params_cov_model": train_state.covariance,
                "target_velocity": target_velocity,
            }

            key, planner_key = jax.random.split(key)
            planner_state = planner_state.replace(key=planner_key)

            # Seed iCEM elites with TDMPC2 trajectory if requested
            if seed_icem:
                print("Seeding iCEM elites with TDMPC2 trajectory...")
                tdmpc2_flat = tdmpc2_actions.flatten()
                num_elites = int(
                    config["planner_params"]["elit_frac"]
                    * config["planner_params"]["batch_size"]
                )
                seeded_elites = jnp.tile(tdmpc2_flat[None, :], (num_elites, 1))
                planner_state = planner_state.replace(
                    mean=tdmpc2_flat, elites=seeded_elites
                )

            icem_actions, planner_state = planner.solve(
                planner_state, init_obs, cost_params
            )
            # Trim to same length as TDMPC2 actions
            icem_actions = icem_actions[:horizon]
            print(f"iCEM actions shape: {icem_actions.shape}")

        # Use TDMPC2 actions for rollout
        actions = tdmpc2_actions
    else:
        # Initialize environment to get initial state
        print("Initializing environment...")
        env_params = EnvParams(**config["env_params"])
        reset_fn, _, get_obs_fn = make_cheetah_env(env_params)

        print("Getting initial state...")
        key, reset_key = jax.random.split(key)
        mjx_data = reset_fn(reset_key)
        init_obs = get_obs_fn(mjx_data).squeeze()

        # Initialize cost function and planner
        print("Initializing cost function...")
        cost_fn = init_cost(config, dynamics_model)

        print("Initializing planner...")
        key, planner_key = jax.random.split(key)
        planner, planner_state = init_planner(config, cost_fn, planner_key)

        # Set up cost params
        target_velocity = config["cost_fn_params"]["target_velocity"]
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
            "target_velocity": target_velocity,
        }

        # Compute full action trajectory using iCEM
        print(f"Planning trajectory (target velocity: {target_velocity} m/s)...")
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)
        actions, planner_state = planner.solve(planner_state, init_obs, cost_params)
        horizon = actions.shape[0]

    print(f"Actions shape: {actions.shape}")

    # Roll out using learned model (NOT the simulator)
    print("Rolling out predicted trajectory using learned model...")
    predicted_states_17d = rollout_learned_model(
        init_obs, actions, dynamics_model, train_state.params
    )

    # Convert to 18D states for animation
    print("Converting to 18D states for animation...")
    predicted_states_18d = obs_17d_to_state_18d(predicted_states_17d)

    # Compute predicted velocity
    vel_x_pred = np.array(predicted_states_17d[:, 8])

    if use_tdmpc2:
        # Ground truth velocity from TDMPC2 data
        vel_x_gt = gt_states[:, 8]
        gt_states_18d = obs_17d_to_state_18d(gt_states)

        print("\nTrajectory stats:")
        print(f"  GT mean vel_x: {np.mean(vel_x_gt):.2f} m/s")
        print(f"  Predicted mean vel_x: {np.mean(vel_x_pred):.2f} m/s")

        # Plot velocity comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        timesteps = np.arange(len(vel_x_gt)) * 0.01
        ax.plot(timesteps, vel_x_gt, label="Ground Truth", linewidth=2)
        ax.plot(timesteps[:len(vel_x_pred)], vel_x_pred,
                label="Predicted (Learned Model)", linewidth=2, linestyle="--")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Forward Velocity: Ground Truth vs Predicted")
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"comparison/velocity_plot": wandb.Image(fig)})
        plt.close(fig)

        # Plot actions
        actions_np = np.array(tdmpc2_actions)
        fig, ax = plt.subplots(figsize=(10, 5))
        action_timesteps = np.arange(horizon) * 0.01
        action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
        for i, label in enumerate(action_labels):
            ax.plot(action_timesteps, actions_np[:, i], label=label, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Action")
        ax.set_title("TDMPC2 Actions")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        wandb.log({"comparison/actions_plot": wandb.Image(fig)})
        plt.close(fig)

        # Plot iCEM vs TDMPC2 action comparison (one subplot per action)
        if icem_actions is not None:
            icem_actions_np = np.array(icem_actions)
            fig, axes = plt.subplots(2, 3, figsize=(15, 8))
            axes = axes.flatten()

            for i, (ax, label) in enumerate(zip(axes, action_labels)):
                ax.plot(action_timesteps, actions_np[:, i],
                        label="TDMPC2", linewidth=2, color="blue")
                ax.plot(action_timesteps, icem_actions_np[:, i],
                        label="iCEM", linewidth=2, color="orange", linestyle="--")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Action")
                ax.set_title(f"{label}")
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)

            fig.suptitle("Action Comparison: TDMPC2 vs iCEM", fontsize=14)
            plt.tight_layout()
            wandb.log({"comparison/actions_icem_vs_tdmpc2": wandb.Image(fig)})
            plt.close(fig)

            # Distribution analysis
            print("\n=== Action Distribution Analysis ===")
            print(f"{'':12} {'TDMPC2':^16} {'iCEM':^16} {'Wasserstein':>12}")
            print(f"{'':12} {'mean':>8} {'std':>8} {'mean':>8} {'std':>8} {'distance':>12}")
            for i, label in enumerate(action_labels):
                t_mean, t_std = actions_np[:, i].mean(), actions_np[:, i].std()
                c_mean, c_std = icem_actions_np[:, i].mean(), icem_actions_np[:, i].std()
                wd = wasserstein_distance(actions_np[:, i], icem_actions_np[:, i])
                print(f"{label:12} {t_mean:8.3f} {t_std:8.3f} {c_mean:8.3f} {c_std:8.3f} {wd:12.3f}")

            # Action smoothness
            print("\n=== Action Smoothness (mean |Δa|) ===")
            print(f"{'':12} {'TDMPC2':>8} {'iCEM':>8}")
            for i, label in enumerate(action_labels):
                t_delta = np.abs(np.diff(actions_np[:, i])).mean()
                c_delta = np.abs(np.diff(icem_actions_np[:, i])).mean()
                print(f"{label:12} {t_delta:8.3f} {c_delta:8.3f}")

        # Animate both trajectories (GT as main, predicted as ghost)
        print("\nGenerating comparison animation...")
        # Stack: (2, T, 18) - GT first (main), predicted second (ghost)
        min_len = min(len(gt_states_18d), len(predicted_states_18d))
        combined_states = np.stack([
            gt_states_18d[:min_len],
            predicted_states_18d[:min_len]
        ], axis=0)
        save_path = config.get("save_path", "comparison_trajectory.gif")
        gif_path = create_cheetah_xy_animation(combined_states, save_path=save_path)
        print(f"Animation saved to: {gif_path}")

        wandb.log({
            "comparison/animation": wandb.Video(gif_path, fps=20, format="gif")
        })
    else:
        # Original iCEM mode
        target_velocity = config["cost_fn_params"]["target_velocity"]
        print("\nPredicted trajectory stats:")
        print(f"  Initial vel_x: {vel_x_pred[0]:.2f} m/s")
        print(f"  Final vel_x: {vel_x_pred[-1]:.2f} m/s")
        print(f"  Mean vel_x: {np.mean(vel_x_pred):.2f} m/s")
        print(f"  Target vel: {target_velocity:.2f} m/s")

        # Plot velocity
        fig, ax = plt.subplots(figsize=(10, 5))
        timesteps = np.arange(len(vel_x_pred)) * 0.01
        ax.plot(timesteps, vel_x_pred, label="Predicted vel_x", linewidth=2)
        ax.axhline(target_velocity, color="red", linestyle="--",
                   label=f"Target ({target_velocity:.1f} m/s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title("Predicted Forward Velocity (Learned Model)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"predicted/velocity_plot": wandb.Image(fig)})
        plt.close(fig)

        # Plot actions
        actions_np = np.array(actions)
        fig, ax = plt.subplots(figsize=(10, 5))
        action_timesteps = np.arange(horizon) * 0.01
        action_labels = ["bthigh", "bshin", "bfoot", "fthigh", "fshin", "ffoot"]
        for i, label in enumerate(action_labels):
            ax.plot(action_timesteps, actions_np[:, i], label=label, alpha=0.8)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Action")
        ax.set_title("Planned Actions (iCEM)")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        wandb.log({"predicted/actions_plot": wandb.Image(fig)})
        plt.close(fig)

        # Animate
        print("\nGenerating animation...")
        save_path = config.get("save_path", "predicted_trajectory.gif")
        gif_path = create_cheetah_xy_animation(
            predicted_states_18d, save_path=save_path
        )
        print(f"Animation saved to: {gif_path}")

        wandb.log({
            "predicted/animation": wandb.Video(gif_path, fps=20, format="gif")
        })

    wandb.finish()
    print("Logged to wandb.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize planned trajectory using learned model"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="cheetah.json",
        help="Config filename in configs folder",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="predicted_trajectory.gif",
        help="Path to save the animation GIF",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--tdmpc2-episode",
        type=int,
        default=None,
        help="If specified, load this episode from TDMPC2 data and compare "
             "ground truth vs learned model prediction",
    )
    parser.add_argument(
        "--compare-icem",
        action="store_true",
        help="When using --tdmpc2-episode, also run iCEM on the initial state "
             "and compare action trajectories",
    )
    parser.add_argument(
        "--seed-icem",
        action="store_true",
        help="When using --compare-icem or --mpc-rollout, seed iCEM with TDMPC2 trajectory",
    )
    parser.add_argument(
        "--mpc-rollout",
        action="store_true",
        help="Run MPC-style rollout (replan at each step). Use with --tdmpc2-episode",
    )
    parser.add_argument(
        "--use-gt-dynamics",
        action="store_true",
        help="Use ground truth MJX dynamics for iCEM planning (instead of learned model)",
    )
    parser.add_argument(
        "--num-analysis",
        type=int,
        default=None,
        help="Run distribution analysis on N random episodes (no plots)",
    )
    parser.add_argument(
        "--drift-test",
        action="store_true",
        help="Run prediction drift test: compare learned model vs MJX on iCEM-chosen actions",
    )
    args = parser.parse_args()

    # Load config
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "..", "configs", args.config)
    with open(config_path, "r") as f:
        full_config = json.load(f)

    # Use finetuning config (has the learned model and planner setup)
    config = full_config["finetuning"]

    # Get data path from pretraining config
    data_path = full_config["pretraining"]["data_path"]

    # Apply CLI overrides
    if args.seed is not None:
        config["seed"] = args.seed
    config["save_path"] = args.save_path

    if args.drift_test:
        run_prediction_drift_test(config)
    elif args.num_analysis is not None:
        run_multi_episode_analysis(config, data_path, args.num_analysis)
    elif args.mpc_rollout:
        if args.tdmpc2_episode is None:
            print("Error: --mpc-rollout requires --tdmpc2-episode")
            exit(1)
        run_mpc_comparison(config, data_path, args.tdmpc2_episode,
                          seed_icem=args.seed_icem, use_gt_dynamics=args.use_gt_dynamics)
    else:
        main(config, tdmpc2_episode=args.tdmpc2_episode, data_path=data_path,
             compare_icem=args.compare_icem, seed_icem=args.seed_icem)
