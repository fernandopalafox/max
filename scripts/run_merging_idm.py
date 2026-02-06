# run_merging_idm.py

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
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
import tempfile
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle


def plot_merging_trajectory(buffers, buffer_idx, config):
    """Plot bird's-eye view and speed profiles for the merging scenario."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    p_y_target = config["cost_fn_params"]["p_y_target"]
    time = np.arange(len(states)) * dt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Bird's-eye view (longitudinal x vs lateral y)
    axes[0].axhline(
        y=p_y_target, color="gray", linestyle="--",
        alpha=0.5, label="Target lane",
    )
    axes[0].axhline(
        y=p_y_target - 3.5, color="gray", linestyle=":",
        alpha=0.3, label="Merge lane",
    )

    # Ego trajectory
    axes[0].plot(
        states[:, 0], states[:, 1],
        "b-", linewidth=2, label="Ego",
    )
    axes[0].scatter(
        states[0, 0], states[0, 1],
        marker="o", s=100, color="green", zorder=5,
    )
    axes[0].scatter(
        states[-1, 0], states[-1, 1],
        marker="x", s=100, color="red", zorder=5,
    )

    # IDM vehicles (move only longitudinally at p_y_target)
    colors = ["orange", "purple", "brown"]
    labels = ["V2", "V3", "V4"]
    for j, (col, lbl) in enumerate(zip(colors, labels)):
        px_idx = 4 + 2 * j
        axes[0].plot(
            states[:, px_idx],
            [p_y_target] * len(states),
            color=col, linewidth=2, alpha=0.7, label=lbl,
        )

    axes[0].set_xlabel("Longitudinal position (m)")
    axes[0].set_ylabel("Lateral position (m)")
    axes[0].set_title("Bird's-eye View")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)

    # Right: Speeds over time
    axes[1].plot(time, states[:, 2], "b-", label="Ego vx")
    for j, (col, lbl) in enumerate(zip(colors, labels)):
        vx_idx = 5 + 2 * j
        axes[1].plot(
            time, states[:, vx_idx],
            color=col, label=f"{lbl} vx",
        )
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Speed (m/s)")
    axes[1].set_title("Vehicle Speeds")
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def make_merging_animation(buffers, buffer_idx, config, fps=30):
    """Create a GIF animation showing car blocks moving over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = config["env_params"]["dt"]
    p_y_target = config["cost_fn_params"]["p_y_target"]
    car_l, car_w = 4.0, 1.8  # car length and width for drawing

    # Subsample frames to keep gif manageable
    skip = max(1, len(states) // 200)
    frames = states[::skip]

    # Fixed y limits, x will track cars each frame
    y_lo, y_hi = p_y_target - 7, p_y_target + 5
    x_window = 60.0  # visible window width in meters

    fig, ax = plt.subplots(figsize=(12, 4))

    # Lane markings (drawn long enough to always be visible)
    all_px = np.concatenate([
        frames[:, 0], frames[:, 4], frames[:, 6], frames[:, 8],
    ])
    lane_x_lo, lane_x_hi = all_px.min() - 20, all_px.max() + 20
    for y_lane, ls in [
        (p_y_target + 1.75, "-"), (p_y_target - 1.75, "-"),
        (p_y_target - 3.5 + 1.75, ":"),
        (p_y_target - 3.5 - 1.75, ":"),
    ]:
        ax.plot([lane_x_lo, lane_x_hi], [y_lane, y_lane],
                color="gray", lw=1, ls=ls)
    ax.set_ylim(y_lo, y_hi)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    colors = {"ego": "dodgerblue", "V2": "orange",
              "V3": "purple", "V4": "brown"}
    rects = {}
    labels = {}
    for name in colors:
        r = Rectangle((0, 0), car_l, car_w, fc=colors[name],
                       ec="black", lw=0.8, alpha=0.85)
        ax.add_patch(r)
        rects[name] = r
        t = ax.text(0, 0, name, ha="center", va="center",
                    fontsize=7, fontweight="bold")
        labels[name] = t

    time_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                        fontsize=9)

    def update(i):
        s = frames[i]
        # Ego
        rects["ego"].set_xy(
            (s[0] - car_l / 2, s[1] - car_w / 2)
        )
        labels["ego"].set_position((s[0], s[1]))
        # IDM vehicles on target lane
        for j, name in enumerate(["V2", "V3", "V4"]):
            px = s[4 + 2 * j]
            py = p_y_target
            rects[name].set_xy(
                (px - car_l / 2, py - car_w / 2)
            )
            labels[name].set_position((px, py))
        # Pan x-axis to follow the leftmost car
        x_min = min(s[0], s[4], s[6], s[8])
        ax.set_xlim(x_min - 10, x_min - 10 + x_window)
        time_text.set_text(f"t = {i * skip * dt:.1f}s")
        return (
            list(rects.values())
            + list(labels.values())
            + [time_text]
        )

    # blit=False needed since we update axis limits each frame
    anim = FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 // fps, blit=False,
    )

    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    anim.save(tmp.name, writer="pillow", fps=fps)
    plt.close(fig)
    return tmp.name


def main(config, save_dir):
    wandb.init(
        project="merging_idm",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    # Ground truth parameters for evaluation (V3 & V4 only — V2 is fixed/known)
    true_T_vec = jnp.array(config["env_params"]["true_T_vec"][1:])
    true_b_vec = jnp.array(config["env_params"]["true_b_vec"][1:])
    true_k_lat = jnp.array(config["env_params"]["true_k_lat"])
    true_d0 = jnp.array(config["env_params"]["true_d0"])
    true_params = {
        "model": {
            "T": true_T_vec,
            "b": true_b_vec,
            "k_lat": true_k_lat,
            "d0": true_d0,
        },
        "normalizer": None,
    }

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
        shape=(
            config["eval_traj_horizon"],
            config["dim_action"],
        ),
        minval=jnp.array(
            config["normalization_params"]["action"]["min"]
        ),
        maxval=jnp.array(
            config["normalization_params"]["action"]["max"]
        ),
    )

    key, reset_key = jax.random.split(key)
    eval_trajectory = [reset_fn(reset_key)]
    current_state = eval_trajectory[0]
    for action in eval_actions:
        next_state, _, _, _, _, _ = step_fn(
            current_state, 0, action
        )
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
    planner, planner_state = init_planner(
        config, cost_fn, planner_key
    )

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
    reward_component_keys_to_avg = [
        "reward", "ego_py", "ego_vx",
    ]
    episode_reward_components = {
        info_key: 0.0
        for info_key in reward_component_keys_to_avg
    }

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
        planner_state = planner_state.replace(
            key=planner_key
        )

        actions, planner_state = planner.solve(
            planner_state, state, cost_params
        )
        action = actions[0][None, :]  # add agent dim

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
                    avg_val = (
                        episode_reward_components[info_key]
                        / episode_length
                    )
                    episode_log[
                        f"rewards/{info_key}"
                    ] = float(avg_val)
            wandb.log(episode_log, step=step)
            episode_length = 0
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0

        # Train policy
        if step % config["train_policy_freq"] == 0:
            pass

        # Train model
        if (
            step % config["train_model_freq"] == 0
            and buffer_idx >= 2
        ):
            train_data = {
                "states": buffers["states"][
                    0, buffer_idx - 2 : buffer_idx - 1, :
                ],
                "actions": buffers["actions"][
                    0, buffer_idx - 2 : buffer_idx - 1, :
                ],
                "next_states": buffers["states"][
                    0, buffer_idx - 1 : buffer_idx, :
                ],
            }
            train_state, loss = trainer.train(
                train_state, train_data, step=step
            )
            wandb.log(
                {"train/model_loss": float(loss)}, step=step
            )

            if step % config["eval_freq"] == 0:
                multi_step_loss = (
                    evaluator.compute_multi_step_loss(
                        train_state.params,
                        eval_trajectory_data,
                    )
                )
                one_step_loss = (
                    evaluator.compute_one_step_loss(
                        train_state.params,
                        eval_trajectory_data,
                    )
                )

                # Parameter difference from true dynamics
                diff_tree = jax.tree.map(
                    lambda x, y: x - y,
                    train_state.params,
                    true_params,
                )
                param_diff = sum(
                    jnp.linalg.norm(leaf)
                    for leaf in jax.tree_util.tree_leaves(
                        diff_tree
                    )
                )
                cov_trace = (
                    jnp.trace(train_state.covariance)
                    if train_state.covariance is not None
                    else 0.0
                )

                # Lane tracking error (mean |ego_py - p_y_target|)
                p_y_target = config["cost_fn_params"]["p_y_target"]
                buf_states = buffers["states"][0, :buffer_idx]
                ego_py_buf = buf_states[:, 1]
                lane_error = float(
                    jnp.mean(jnp.abs(ego_py_buf - p_y_target))
                )

                # Safety: mean sum of distances to V3 and V4
                ego_px = buf_states[:, 0]
                ego_py = buf_states[:, 1]
                d_v3 = jnp.sqrt(
                    (ego_px - buf_states[:, 6]) ** 2
                    + (ego_py - p_y_target) ** 2
                )
                d_v4 = jnp.sqrt(
                    (ego_px - buf_states[:, 8]) ** 2
                    + (ego_py - p_y_target) ** 2
                )
                safety_dist = float(jnp.mean(d_v3 + d_v4))

                # Individual parameter errors
                learned_T = train_state.params["model"]["T"]
                learned_b = train_state.params["model"]["b"]
                param_log = {
                    "eval/multi_step_loss": multi_step_loss,
                    "eval/one_step_loss": one_step_loss,
                    "eval/param_diff": param_diff,
                    "eval/cov_trace": cov_trace,
                    "eval/lane_error": lane_error,
                    "eval/safety_dist": safety_dist,
                }
                for j, name in enumerate(["v3", "v4"]):
                    param_log[f"params/T_{name}"] = float(
                        learned_T[j]
                    )
                    param_log[f"params/b_{name}"] = float(
                        learned_b[j]
                    )
                    param_log[f"params/T_{name}_err"] = float(
                        jnp.abs(learned_T[j] - true_T_vec[j])
                    )
                    param_log[f"params/b_{name}_err"] = float(
                        jnp.abs(learned_b[j] - true_b_vec[j])
                    )
                learned_k_lat = train_state.params["model"]["k_lat"]
                learned_d0 = train_state.params["model"]["d0"]
                param_log["params/k_lat"] = float(learned_k_lat)
                param_log["params/d0"] = float(learned_d0)
                param_log["params/k_lat_err"] = float(
                    jnp.abs(learned_k_lat - true_k_lat)
                )
                param_log["params/d0_err"] = float(
                    jnp.abs(learned_d0 - true_d0)
                )
                wandb.log(param_log, step=step)

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
        run_name = config.get(
            "wandb_run_name",
            f"merging_idm_model_{config['seed']}",
        )
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        print(
            f"\nSaving final model parameters to {save_path}..."
        )
        dynamics_params_np = jax.device_get(
            train_state.params
        )
        file_path = os.path.join(
            save_path, "dynamics_params.pkl"
        )
        with open(file_path, "wb") as f:
            pickle.dump(dynamics_params_np, f)
        print(f"Dynamics parameters saved to {file_path}")
        if train_state.covariance is not None:
            cov_path = os.path.join(
                save_path, "param_covariance.pkl"
            )
            cov_np = jax.device_get(train_state.covariance)
            with open(cov_path, "wb") as f:
                pickle.dump(cov_np, f)
            print(
                f"Parameter covariance saved to {cov_path}"
            )

    # Plot and log trajectory
    if buffer_idx > 0:
        print("\nGenerating trajectory plot...")
        fig = plot_merging_trajectory(
            buffers, buffer_idx, config
        )
        wandb.log(
            {
                "trajectory/merging_plot": wandb.Image(fig),
            },
            step=config["total_steps"],
        )
        plt.close(fig)
        print("Trajectory plot logged to wandb.")

        # Animation
        print("Generating merging animation...")
        gif_path = make_merging_animation(
            buffers, buffer_idx, config
        )
        wandb.log(
            {
                "trajectory/animation": wandb.Video(
                    gif_path, format="gif"
                ),
            },
            step=config["total_steps"],
        )
        os.remove(gif_path)
        print("Animation logged to wandb.")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run merging IDM experiments."
    )
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
        help="List of weight_info (lambda) values to sweep over.",
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
        help="Directory to save learned dynamics parameters.",
    )
    args = parser.parse_args()

    # Load config from JSON file
    config_path = os.path.join(
        os.path.dirname(__file__),
        "..", "configs", "merging_idm.json",
    )
    with open(config_path, "r") as f:
        CONFIG = json.load(f)

    run_name_base = args.run_name or "merging_idm"

    if args.lambdas is None:
        lambdas = [CONFIG["cost_fn_params"]["weight_info"]]
    else:
        lambdas = args.lambdas

    for lam in lambdas:
        print(f"--- Starting run for lambda={lam} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = args.seed
        run_config["wandb_group"] = "merging_idm"
        run_config["cost_fn_params"]["weight_info"] = lam

        if len(lambdas) > 1:
            run_config["wandb_run_name"] = (
                f"{run_name_base}_lam_{lam}"
            )
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")
