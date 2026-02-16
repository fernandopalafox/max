# run_drone.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch, Circle
import tempfile
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import init_env
from max.dynamics import init_dynamics
from max.dynamics_trainers import init_trainer
from max.dynamics_evaluators import init_evaluator
from max.planners import init_planner
from max.costs import init_cost
import argparse
import copy
import os
import pickle
import json


def plot_drone_trajectory(buffers, buffer_idx, config):
    # Extract XY and Goal
    pts = buffers["states"][0, :buffer_idx, :2]
    goal = config["cost_fn_params"]["goal_state"][:2]

    # Get normalization bounds for positions
    norm_params = config["normalization_params"]["state"]
    x_min, x_max = norm_params["min"][0], norm_params["max"][0]
    y_min, y_max = norm_params["min"][1], norm_params["max"][1]

    fig, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], label="Path")
    ax.scatter(*goal, color='red', marker='*', label="Goal", zorder=5)

    # Use normalization bounds for axes
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()

    return fig


def covariance_trace(cov):
    """Compute trace of covariance when LOFI stores a structured approximation.
    Mirrors the implementation used in run_pendulum.py so LOFI outputs
    (dict with 'Upsilon' and 'W') are handled correctly.
    """
    if cov is None:
        return 0.0
    # LOFI stores an approximation of the *precision* matrix as
    # Lambda ≈ diag(Upsilon) + W @ W.T. We want trace(Sigma) where
    # Sigma ≈ Lambda^{-1}. Use Woodbury identity to compute
    # trace(Lambda^{-1}) = trace(D^{-1}) - trace((I + W^T D^{-1} W)^{-1} W^T D^{-2} W)
    if isinstance(cov, dict):
        U = cov["Upsilon"]
        W = cov["W"]
        eps = 1e-12
        Dinv = jnp.where(U != 0.0, 1.0 / U, 1.0 / (U + eps))

        trace_Dinv = jnp.sum(Dinv)

        A = W.T @ (Dinv[:, None] * W)
        L_rank = A.shape[0]
        inv_term = jnp.linalg.inv(jnp.eye(L_rank) + A)

        B = W.T @ ((Dinv ** 2)[:, None] * W)

        correction = jnp.trace(inv_term @ B)
        return trace_Dinv - correction

    return jnp.trace(cov)


def make_drone_animation(buffers, buffer_idx, config, fps=50):
    """
    Create a GIF animation showing the drone flying with a wind vector arrow.
    Adjusted to ensure the wind arrow UI fits within the frame.
    """
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]

    # Get wind parameters for visualization (constant wind)
    wind_x = config["env_params"]["wind_x"]
    wind_y = config["env_params"]["wind_y"]

    # Get goal state
    goal_state = config["cost_fn_params"]["goal_state"]
    goal_pos = goal_state[:2]

    # Drone dimensions
    arm_length = config["env_params"]["arm_length"]
    rotor_radius = 0.08

    # Subsample frames
    skip = max(1, len(states) // 400)
    frames = states[::skip]

    # Use normalization bounds for axis limits
    norm_params = config["normalization_params"]["state"]
    x_min, x_max = norm_params["min"][0], norm_params["max"][0]
    y_min, y_max = norm_params["min"][1], norm_params["max"][1]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Drone Flight with Constant Wind")
    ax.grid(True, alpha=0.3)

    # --- WIND ARROW SETUP ---
    # Position the arrow in the top-left corner (in axes coordinates)
    # --- MODIFICATION 2 & 3: Smaller arrow, shifted position ---
    # Moved anchor slightly inward from (0.1, 0.9) to (0.12, 0.88)
    # Increased scale from 5 to 10 (larger scale = smaller visual arrow)
    wind_arrow = ax.quiver(
        0.12, 0.88, 0, 0,
        transform=ax.transAxes,
        color='teal',
        scale=20,
        width=0.008,
        zorder=20 # Ensure it's on top
    )
    # Adjusted text position to align with new arrow position
    wind_text = ax.text(0.12, 0.82, "Wind Dir.", transform=ax.transAxes, 
                        color='teal', fontweight='bold', ha='center', fontsize=9, zorder=20)

    # Goal marker
    ax.scatter([goal_pos[0]], [goal_pos[1]], marker="*", s=300, color="gold", edgecolors="black", label="Goal", zorder=10)

    traj_line, = ax.plot([], [], color="blue", linewidth=1.5, alpha=0.5)
    drone_body, = ax.plot([], [], color="black", linewidth=3, solid_capstyle="round", zorder=15)
    rotor_left = Circle((0, 0), rotor_radius, fc="red", ec="darkred", lw=1, zorder=15)
    rotor_right = Circle((0, 0), rotor_radius, fc="red", ec="darkred", lw=1, zorder=15)
    ax.add_patch(rotor_left)
    ax.add_patch(rotor_right)

    # Moved time text slightly to avoid crowding the wind arrow
    time_text = ax.text(0.02, 0.96, "", transform=ax.transAxes, fontsize=10, fontweight="bold",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, pad=0.3), zorder=20)

    ax.legend(loc="upper right")

    def update(i):
        t = i * skip * dt
        s = frames[i]
        p_x, p_y, phi = s[0], s[1], s[2]

        # 1. Update Wind Arrow (constant wind)
        wind_arrow.set_UVC(wind_x, wind_y)

        # 2. Update Drone
        traj_line.set_data(frames[:i + 1, 0], frames[:i + 1, 1])
        dx, dy = arm_length * np.cos(phi), arm_length * np.sin(phi)
        left_x, left_y = p_x - dx, p_y - dy
        right_x, right_y = p_x + dx, p_y + dy

        drone_body.set_data([left_x, right_x], [left_y, right_y])
        rotor_left.set_center((left_x, left_y))
        rotor_right.set_center((right_x, right_y))
        time_text.set_text(f"t = {t:.2f}s")

        return [traj_line, drone_body, rotor_left, rotor_right, time_text, wind_arrow, wind_text]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 // fps, blit=True)
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    anim.save(tmp.name, writer="pillow", fps=fps)
    plt.close(fig)
    return tmp.name


def plot_state_components(buffers, buffer_idx, config):
    """Plot velocities, angle, and angular velocity with normalization bounds."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    time = np.arange(buffer_idx) * dt

    norm_params = config["normalization_params"]["state"]
    state_min = norm_params["min"]
    state_max = norm_params["max"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    # Velocities subplot (v_x, v_y)
    ax = axes[0]
    ax.plot(time, states[:, 3], label="v_x")
    ax.plot(time, states[:, 4], label="v_y")
    ax.axhline(state_min[3], color='r', linestyle='--', alpha=0.5, label="bounds")
    ax.axhline(state_max[3], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Velocity (m/s)")
    ax.set_ylim(state_min[3], state_max[3])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Angle subplot (phi)
    ax = axes[1]
    ax.plot(time, states[:, 2], label="phi")
    ax.axhline(state_min[2], color='r', linestyle='--', alpha=0.5, label="bounds")
    ax.axhline(state_max[2], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Angle (rad)")
    ax.set_ylim(state_min[2], state_max[2])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    # Angular velocity subplot (phi_dot)
    ax = axes[2]
    ax.plot(time, states[:, 5], label="phi_dot")
    ax.axhline(state_min[5], color='r', linestyle='--', alpha=0.5, label="bounds")
    ax.axhline(state_max[5], color='r', linestyle='--', alpha=0.5)
    ax.set_ylabel("Angular Vel (rad/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylim(state_min[5], state_max[5])
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def main(config, save_dir):
    wandb.init(
        project="planar_drone",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    # Initialize environment
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize dynamics
    normalizer, norm_params = init_normalizer(config)
    key, model_key = jax.random.split(key)
    dynamics_model, init_params = init_dynamics(
        model_key, config, normalizer, norm_params
    )

    # Initialize dynamics trainer
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        config, dynamics_model, init_params, trainer_key
    )

    # Initialize evaluator
    evaluator = init_evaluator(config)

    # Initial evaluation before training
    eval_results = evaluator.evaluate(train_state.params)
    init_cov_trace = (
        jnp.trace(train_state.covariance) / train_state.covariance.shape[0]
        if train_state.covariance is not None
        else 0.0
    )
    wandb.log({**eval_results, "eval/cov_trace": float(init_cov_trace)}, step=0)

    # Initialize cost function
    cost_fn = init_cost(config, dynamics_model)

    # Initialize planner
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(config, cost_fn, planner_key)

    # Initialize buffer
    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    print(
        f"Starting simulation for {config['total_steps']} steps "
    )

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["reward", "position_error"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)
    goal_state = np.array(config["cost_fn_params"]["goal_state"])

    for step in range(1, config["total_steps"] + 1):

        # Compute actions
        cov_struct = train_state.covariance
        if cov_struct is None:
            cov_matrix = None
        elif isinstance(cov_struct, dict):
            # LOFI stores a structured approximation over the *flattened full params*
            # (model + normalizer). The cost/info-term flattens only the `model`
            # parameters, so we must extract the corresponding slice of Upsilon/W
            # before reconstructing the dense covariance for the model params.
            flat_model_params, _ = jax.flatten_util.ravel_pytree(
                train_state.params["model"]
            )
            p_model = int(flat_model_params.shape[0])

            U_full = cov_struct["Upsilon"]
            W_full = cov_struct["W"]

            U = U_full[:p_model]
            W = W_full[:p_model, :]

            # D_inv is diagonal of 1/U (with small epsilon)
            D_inv = jnp.diag(1.0 / (U + 1e-8))

            # Reconstruct dense covariance for model parameters only
            cov_matrix = D_inv - D_inv @ W @ jnp.linalg.inv(jnp.eye(W.shape[1]) + W.T @ D_inv @ W) @ W.T @ D_inv
        else:
            cov_matrix = cov_struct
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": cov_matrix,
            "goal_state": goal_state,
        }
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)

        actions, planner_state = planner.solve(planner_state, state, cost_params)
        action = actions[0][None, :]  # hacky add agent dim

        # Step environment
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, action
        )
        done = terminated or truncated
        episode_length += 1
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]

        # Update buffer
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs,
            action,
            rewards,
            jnp.zeros_like(rewards),  # dummy value
            jnp.zeros_like(rewards),  # dummy log_pi
            float(done),
        )
        buffer_idx += 1

        current_obs = next_obs

        # Reset environment if done
        if done:
            state = reset_fn(reset_key)
            current_obs = get_obs_fn(state)
            print(f"Episode finished at step {step}.")

            # Log and reset episode stats
            episode_log = {"episode/length": episode_length}
            if episode_length > 0:
                for info_key in reward_component_keys_to_avg:
                    avg_val = episode_reward_components[info_key] / episode_length
                    episode_log[f"rewards/{info_key}"] = float(avg_val)
            wandb.log(episode_log, step=step)
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train policy
        if step % config["train_policy_freq"] == 0:
            # Unused for policies like iCEM
            pass

        # Evaluate model
        if step % config["eval_freq"] == 0:
            # Run rollout evaluation
            eval_results = evaluator.evaluate(train_state.params)

            # Track covariance trace if available
            cov_trace = (
                jnp.trace(train_state.covariance)
                if train_state.covariance is not None
                else 0.0
            )
            cov_trace_per_param = cov_trace / train_state.covariance.shape[0] if train_state.covariance is not None else 0.0

            wandb.log(
                {
                    **eval_results,
                    "eval/cov_trace": float(cov_trace_per_param),
                },
                step=step,
            )

        # Train model
        # buffer_idx >= 2 to ensure we have at least one full transition
        if step % config["train_model_freq"] == 0 and buffer_idx >= 2:
            train_data = {
                "states": buffers["states"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "actions": buffers["actions"][0, buffer_idx - 2 : buffer_idx - 1, :],
                "next_states": buffers["states"][0, buffer_idx - 1 : buffer_idx, :],
            }
            train_state, loss = trainer.train(train_state, train_data, step=step)
            wandb.log({"train/model_loss": float(loss)}, step=step)

            if step % config["eval_freq"] == 0:
                multi_step_loss = evaluator.compute_multi_step_loss(
                    train_state.params, eval_trajectory_data
                )
                one_step_loss = evaluator.compute_one_step_loss(
                    train_state.params, eval_trajectory_data
                )

                # Track covariance trace if available
                cov_trace = covariance_trace(train_state.covariance)

                # Optional parameter difference vs ground-truth if provided
                # param_diff = 0.0
                # if config.get("true_params") is not None:
                #     try:
                #         diff_tree = jax.tree_map(
                #             lambda x, y: x - y, train_state.params, config["true_params"]
                #         )
                #         param_diff = sum(
                #             jnp.linalg.norm(leaf)
                #             for leaf in jax.tree_util.tree_leaves(diff_tree)
                #         )
                #     except Exception:
                #         param_diff = 0.0

                wandb.log(
                    {
                        "eval/multi_step_loss": multi_step_loss,
                        "eval/one_step_loss": one_step_loss,
                        "eval/cov_trace": cov_trace,
                    },
                    step=step,
                )

        # Handle Buffer Overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"],
                config["buffer_size"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # Save model parameters
    if save_dir:
        run_name = config.get("wandb_run_name", f"drone_model_{config['seed']}")
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving final model parameters to {save_path}...")
        dynamics_params_np = jax.device_get(train_state.params)
        file_path = os.path.join(save_path, "dynamics_params.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(dynamics_params_np, f)
        print(f"Dynamics parameters saved to {file_path}")
        if train_state.covariance is not None:
            cov_path = os.path.join(save_path, "param_covariance.pkl")
            cov_np = jax.device_get(train_state.covariance)
            with open(cov_path, "wb") as f:
                pickle.dump(cov_np, f)
            print(f"Parameter covariance saved to {cov_path}")

    # Plot and log trajectory
    # if buffer_idx > 0:
        print("\nGenerating trajectory plot...")
        fig = plot_drone_trajectory(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/drone_plot": wandb.Image(fig)}, step=config["total_steps"]
        )
        plt.close(fig)
        print("Trajectory plot logged to wandb.")

        print("Generating state components plot...")
        fig = plot_state_components(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/state_components": wandb.Image(fig)}, step=config["total_steps"]
        )
        plt.close(fig)
        print("State components plot logged to wandb.")

        # Animation
        print("Generating drone animation...")
        gif_path = make_drone_animation(buffers, buffer_idx, config)
        wandb.log(
            {"trajectory/animation": wandb.Video(gif_path, format="gif")},
            step=config["total_steps"],
        )
        print("Animation logged to wandb.")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run planar drone experiments.")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the W&B run.",
    )
    parser.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=None,
        help="List of weight_info values to sweep.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Starting random seed.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of seeds to run for each lambda value.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_models",
        help="Directory to save the learned dynamics model parameters.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "drone.json"
    )
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

    run_name_base = args.run_name or "drone"

    if args.lambdas is None:
        lambdas = [
            CONFIG["cost_fn_params"]["weight_info"]
        ]
    else:
        lambdas = args.lambdas

    for lam_idx, lam in enumerate(lambdas, start=1):
        for seed_idx in range(1, args.num_seeds + 1):
            seed = args.seed + seed_idx - 1
            print(f"--- Starting run for lam{lam_idx} (lambda={lam}), run {seed_idx}/{args.num_seeds} ---")
            run_config = copy.deepcopy(CONFIG)
            run_config["seed"] = seed
            run_config["wandb_group"] = "planar_drone_wind"
            run_config["cost_fn_params"]["weight_info"] = lam

            # Build run name: base_lam{idx}_seed{idx}
            run_name = run_name_base
            if len(lambdas) > 1:
                run_name = f"{run_name}_lam{lam}"
            if args.num_seeds > 1:
                run_name = f"{run_name}_{seed_idx}"
            run_config["wandb_run_name"] = run_name

            main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
