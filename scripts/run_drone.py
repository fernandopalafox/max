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
from max.dynamics_evaluators import DynamicsEvaluator
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

    fig, ax = plt.subplots()
    ax.plot(pts[:, 0], pts[:, 1], label="Path")
    ax.scatter(*goal, color='red', marker='*', label="Goal", zorder=5)
    
    # Combine data to find overall bounds
    all_x = np.concatenate([pts[:, 0], [goal[0]]])
    all_y = np.concatenate([pts[:, 1], [goal[1]]])
    
    # Calculate the range and the midpoint
    max_range = max(all_x.max() - all_x.min(), all_y.max() - all_y.min())
    mid_x = (all_x.max() + all_x.min()) / 2
    mid_y = (all_y.max() + all_y.min()) / 2
    
    # Add padding (e.g., 10% extra space)
    padding = 1.1 
    half_span = (max_range * padding) / 2
    
    # Set symmetric limits
    ax.set_xlim(mid_x - half_span, mid_x + half_span)
    ax.set_ylim(mid_y - half_span, mid_y + half_span)
    
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    return fig


def make_drone_animation(buffers, buffer_idx, config, fps=50):
    """
    Create a GIF animation showing the drone flying with a wind vector arrow.
    Adjusted to ensure the wind arrow UI fits within the frame.
    """
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    
    # Get wind parameters for visualization
    A_w = config["env_params"]["true_wind_amplitude"]
    omega = config["env_params"]["true_wind_frequency"]

    # Get goal state
    goal_state = config["cost_fn_params"]["goal_state"]
    goal_pos = goal_state[:2]

    # Drone dimensions
    arm_length = config["env_params"]["arm_length"]
    rotor_radius = 0.08

    # Subsample frames
    skip = max(1, len(states) // 400)
    frames = states[::skip]

    # Compute axis limits
    p_x_all, p_y_all = states[:, 0], states[:, 1]
    x_min, x_max = min(p_x_all.min(), goal_pos[0]), max(p_x_all.max(), goal_pos[0])
    y_min, y_max = min(p_y_all.min(), goal_pos[1]), max(p_y_all.max(), goal_pos[1])

    # --- MODIFICATION 1: Increase padding ---
    # Increased padding to give more breathing room around the trajectory
    padding = 1.0
    max_range = max(x_max - x_min, y_max - y_min) + 2 * padding
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(x_center - max_range / 2, x_center + max_range / 2)
    ax.set_ylim(y_center - max_range / 2, y_center + max_range / 2)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Drone Flight with Time-Varying Wind")
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
        scale=10,
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

        # 1. Update Wind Arrow based on dynamics equation
        w_x = A_w * np.cos(omega * t)
        w_y = A_w * np.sin(omega * t)
        wind_arrow.set_UVC(w_x, w_y)

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

    # Initialize dynamics evaluator
    evaluator = DynamicsEvaluator(dynamics_model.pred_one_step)

    key, action_key = jax.random.split(key)
    eval_actions = jax.random.uniform(
        action_key,
        shape=(config["eval_traj_horizon"], config["dim_action"]),
        minval=jnp.array(config["normalization_params"]["action"]["min"]),
        maxval=jnp.array(config["normalization_params"]["action"]["max"]),
    )

    key, reset_key = jax.random.split(key)
    eval_trajectory = [reset_fn(reset_key)]
    current_state = eval_trajectory[0]
    for action in eval_actions:
        next_state, _, _, _, _, _ = step_fn(current_state, 0, action)
        eval_trajectory.append(next_state)
        current_state = next_state

    eval_trajectory_data = {
        "trajectory": jnp.array(eval_trajectory),
        "actions": eval_actions,
    }

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

    initial_multi_step_loss = evaluator.compute_multi_step_loss(
        train_state.params, eval_trajectory_data
    )
    initial_one_step_loss = evaluator.compute_one_step_loss(
        train_state.params, eval_trajectory_data
    )
    wandb.log(
        {
            "eval/multi_step_loss": initial_multi_step_loss,
            "eval/one_step_loss": initial_one_step_loss,
        },
        step=0,
    )

    episode_length = 0

    # Reward component accumulators
    reward_component_keys_to_avg = ["reward", "position_error"]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }

    # TODO: remove - temporary tracking cost accumulator
    _goal = np.array(config["cost_fn_params"]["goal_state"])
    _weights = np.array(config["cost_fn_params"]["state_weights"])
    _w_ctrl = config["cost_fn_params"]["weight_control"]
    _accum_cost = 0.0

    # Main training loop
    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)
    current_obs = get_obs_fn(state)

    for step in range(1, config["total_steps"] + 1):

        # Compute actions
        cost_params = {
            "dyn_params": train_state.params,
            "params_cov_model": train_state.covariance,
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
        _accum_cost += float(np.sum(_weights * (np.array(state) - _goal) ** 2) + _w_ctrl * np.sum(np.array(action) ** 2))

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
                cov_trace = (
                    jnp.trace(train_state.covariance)
                    if train_state.covariance is not None
                    else 0.0
                )
                # divide by number of model params (not normalizer params)
                cov_trace_per_param = cov_trace / train_state.covariance.shape[0]

                wandb.log(
                    {
                        "eval/multi_step_loss": multi_step_loss,
                        "eval/one_step_loss": one_step_loss,
                        "eval/cov_trace": cov_trace_per_param,
                        "eval/accumulated_cost": _accum_cost,
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
        help="Random seed.",
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

    for lam in lambdas:
        print(f"--- Starting run for lambda={lam} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = args.seed
        run_config["wandb_group"] = "planar_drone_wind"
        run_config["cost_fn_params"]["weight_info"] = lam

        if len(lambdas) > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_lam_{lam}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
