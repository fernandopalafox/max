# finetune_cheetah.py

# Enable deterministic GPU operations for debugging (set before importing JAX)
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

import time

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.environments import make_cheetah_env, EnvParams
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.critics import init_critic
from max.policies import init_policy
from max.rewards import init_reward_model
from max.trainers import init_trainer
from max.samplers import init_sampler
from max.dynamics_evaluators import init_evaluator
from max.planners import init_planner
import argparse
import copy
import os
import pickle
import json

from max.visualizers import create_cheetah_xy_animation


def plot_cheetah_velocity(buffers, buffer_idx, config):
    """Plot forward velocity over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = 0.01
    time_axis = np.arange(buffer_idx) * dt
    forward_vel = states[:, 8]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time_axis, forward_vel, label="Forward Velocity", color="blue", linewidth=2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Velocity (m/s)")
    ax.set_title("Cheetah Forward Velocity")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig


def plot_action_trajectory(buffers, buffer_idx, config):
    """Plot applied actions over time."""
    actions = np.array(buffers["actions"][0, :buffer_idx, :])
    dt = 0.01
    time_axis = np.arange(buffer_idx) * dt
    dim_action = actions.shape[1]

    fig, axes = plt.subplots(dim_action, 1, figsize=(12, 2 * dim_action), sharex=True)
    if dim_action == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(time_axis, actions[:, i], lw=1.5)
        ax.set_ylabel(f"a{i}")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Applied Actions", fontsize=10)
    plt.tight_layout()
    return fig


def plot_state_components(buffers, buffer_idx, config):
    """Plot joint angles and velocities over time."""
    states = np.array(buffers["states"][0, :buffer_idx, :])
    dt = 0.01
    time_axis = np.arange(buffer_idx) * dt
    state_labels = config.get("state_labels", [f"s{i}" for i in range(17)])

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    ax = axes[0]
    for i in range(8):
        ax.plot(time_axis, states[:, i], label=state_labels[i], alpha=0.8)
    ax.set_ylabel("Joint Angle (rad)")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Joint Positions")

    ax = axes[1]
    for i in range(8, 17):
        ax.plot(time_axis, states[:, i], label=state_labels[i], alpha=0.8)
    ax.set_ylabel("Velocity")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="upper right", ncol=3, fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_title("Velocities")

    plt.tight_layout()
    return fig


def main(config):
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    save_dir = config.get("save_dir", None)
    plot_run = config.get("plot_run", True)
    plot_eval = config.get("plot_eval", False)

    # ---- Initialize environment ----
    print(f"[{time.time()-t0:.2f}s] Initializing environment...")
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)
    print(f"[{time.time()-t0:.2f}s] Environment initialized")

    # ---- TDMPC2 component initialization ----
    print(f"[{time.time()-t0:.2f}s] Initializing normalizer...")
    normalizer, norm_params = init_normalizer(config)

    print(f"[{time.time()-t0:.2f}s] Initializing encoder...")
    key, enc_key = jax.random.split(key)
    encoder, enc_params = init_encoder(enc_key, config, normalizer, norm_params)

    print(f"[{time.time()-t0:.2f}s] Initializing dynamics...")
    key, dyn_key = jax.random.split(key)
    dynamics, dyn_params = init_dynamics(dyn_key, config, normalizer, norm_params, encoder=encoder)

    print(f"[{time.time()-t0:.2f}s] Initializing critic...")
    key, critic_key = jax.random.split(key)
    critic, critic_params = init_critic(critic_key, config)

    print(f"[{time.time()-t0:.2f}s] Initializing policy...")
    key, policy_key = jax.random.split(key)
    policy, policy_params = init_policy(policy_key, config)

    print(f"[{time.time()-t0:.2f}s] Initializing reward model...")
    reward_model, reward_params = init_reward_model(config, encoder=encoder)

    # ---- Build unified parameters dict ----
    parameters = {
        "encoder":    enc_params,
        "dynamics":   dyn_params,
        "reward":     reward_params,
        "critic":     critic_params,
        "ema_critic": copy.deepcopy(critic_params),
        "policy":     policy_params,
        "normalizer": {**norm_params, "q_scale": jnp.array(1.0)},
    }

    # ---- Initialize TDMPC2 trainer ----
    print(f"[{time.time()-t0:.2f}s] Initializing trainer...")
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        trainer_key, config, encoder, dynamics, critic, policy, reward_model, parameters
    )
    print(f"[{time.time()-t0:.2f}s] Trainer initialized")

    # ---- Initialize sampler ----
    sampler = init_sampler(config["sampler"])

    # Log trainable parameter counts
    enc_n = sum(x.size for x in jax.tree_util.tree_leaves(enc_params["encoder"]))
    dyn_n = sum(x.size for x in jax.tree_util.tree_leaves(dyn_params["mean"]))
    cri_n = sum(x.size for x in jax.tree_util.tree_leaves(critic_params))
    pol_n = sum(x.size for x in jax.tree_util.tree_leaves(policy_params))
    rew_n = sum(x.size for x in jax.tree_util.tree_leaves(reward_params))
    total_n = enc_n + dyn_n + cri_n + pol_n + rew_n
    wandb.config.update({
        "num_params_encoder": enc_n,
        "num_params_dynamics": dyn_n,
        "num_params_critic": cri_n,
        "num_params_policy": pol_n,
        "num_params_reward": rew_n,
        "num_params_total": total_n,
    })
    print(f"Trainable params: encoder={enc_n}, dynamics={dyn_n}, critic={cri_n}, policy={pol_n}, reward={rew_n}, total={total_n}")

    # ---- Initialize evaluator (TDMPC2 MPPI evaluator) ----
    print(f"[{time.time()-t0:.2f}s] Initializing evaluator...")
    evaluator = init_evaluator(
        config,
        encoder=encoder,
        dynamics=dynamics,
        reward=reward_model,
        critic=critic,
        policy=policy,
    )
    print(f"[{time.time()-t0:.2f}s] Evaluator initialized")

    # ---- Initialize MPPI planner ----
    print(f"[{time.time()-t0:.2f}s] Initializing MPPI planner...")
    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(
        config,
        key=planner_key,
        encoder=encoder,
        dynamics=dynamics,
        reward=reward_model,
        critic=critic,
        policy=policy,
    )
    print(f"[{time.time()-t0:.2f}s] Planner initialized")

    # ---- Initialize buffer ----
    print(f"[{time.time()-t0:.2f}s] Initializing buffers...")
    buffers = init_jax_buffers(
        config["num_agents"],
        config["buffer_size"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0
    print(f"[{time.time()-t0:.2f}s] Buffers initialized")

    print(f"Starting TDMPC2 cheetah finetuning for {config['total_steps']} steps")

    episode_length = 0
    episode_total_reward = 0.0

    # # Initial evaluation
    # print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    # eval_results = evaluator.evaluate(parameters)
    # initial_metrics = {
    #     k: v for k, v in eval_results.items() if isinstance(v, (int, float))
    # }
    # wandb.log(initial_metrics, step=0)
    # print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")

    # if plot_eval and "trajectory" in eval_results:
    #     traj = eval_results["trajectory"]
    #     full_states = np.concatenate([traj.qpos, traj.qvel], axis=-1)
    #     gif_path = create_cheetah_xy_animation(full_states)
    #     wandb.log({"eval/animation": wandb.Video(gif_path, fps=20, format="gif")}, step=0)

    # full_states_for_animation = []

    # ---- Main training loop ----
    print(f"[{time.time()-t0:.2f}s] Starting main loop...")
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    current_obs = get_obs_fn(mjx_data).squeeze()

    t_planner = 0.0
    t_step = 0.0
    t_train = 0.0
    t_eval = 0.0

    train_freq = config.get("train_freq", 1)

    for step in range(1, config["total_steps"] + 1):
        step_start = time.time()

        full_state = jnp.concatenate([mjx_data.qpos, mjx_data.qvel])
        full_states_for_animation.append(np.array(full_state))

        # ---- Planning step (MPPI in latent space) ----
        _t0 = time.time()
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)
        # MPPI solve: cost_params = full parameters dict
        actions, planner_state = planner.solve(planner_state, current_obs, parameters)
        action = actions[0][None, :]  # (1, dim_a) with agent dim
        t_planner += time.time() - _t0

        # ---- Environment step ----
        _t0 = time.time()
        mjx_data, next_obs, rewards, terminated, truncated, _ = step_fn(
            mjx_data, episode_length, action
        )
        t_step += time.time() - _t0
        next_obs = next_obs.squeeze()
        done = terminated or truncated
        episode_length += 1
        episode_total_reward += float(rewards[0])

        # ---- Buffer update ----
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs[None, :],
            action,
            rewards,
            float(done),
        )
        buffer_idx += 1
        current_obs = next_obs

        # ---- Episode reset ----
        if done:
            key, reset_key = jax.random.split(key)
            mjx_data = reset_fn(reset_key)
            current_obs = get_obs_fn(mjx_data).squeeze()
            wandb.log({
                "episode/length": episode_length,
                "rewards/episode_reward": episode_total_reward,
            }, step=step)
            episode_length = 0
            episode_total_reward = 0.0

        # ---- Training step ----
        dt_train = 0.0
        if step % train_freq == 0:
            _t0 = time.time()
            key, sample_key = jax.random.split(key)
            train_data = sampler.sample(sample_key, buffers, buffer_idx)
            if train_data is not None:
                key, train_key = jax.random.split(key)
                train_state, parameters, metrics = trainer.train(
                    train_state, train_data, parameters, train_key
                )
                wandb.log(
                    {f"train/{k}": float(v) for k, v in metrics.items()},
                    step=step
                )
            dt_train = time.time() - _t0
            t_train += dt_train

        # ---- Evaluation ----
        dt_eval = 0.0
        if step % config["eval_freq"] == 0:
            _t0 = time.time()
            eval_results = evaluator.evaluate(parameters)
            dt_eval = time.time() - _t0
            t_eval += dt_eval

            metrics_to_log = {
                k: v for k, v in eval_results.items() if isinstance(v, (int, float))
            }
            wandb.log(metrics_to_log, step=step)

            if plot_eval and "trajectory" in eval_results:
                traj = eval_results["trajectory"]
                full_states = np.concatenate([traj.qpos, traj.qvel], axis=-1)
                gif_path = create_cheetah_xy_animation(full_states)
                wandb.log(
                    {"eval/animation": wandb.Video(gif_path, fps=20, format="gif")},
                    step=step
                )

        step_total = time.time() - step_start
        print(
            f"[Step {step}] total={step_total:.3f}s | "
            f"planner={t_planner/(step):.3f}s avg, "
            f"train={dt_train:.3f}s, eval={dt_eval:.3f}s"
        )

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_jax_buffers(
                config["num_agents"],
                config["buffer_size"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    # ---- Final timing summary ----
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total planner time: {t_planner:.2f}s")
    print(f"Total step time:    {t_step:.2f}s")
    print(f"Total train time:   {t_train:.2f}s")
    print(f"Total eval time:    {t_eval:.2f}s")
    print(f"======================\n")

    # ---- Save parameters ----
    if save_dir:
        run_name = config.get("wandb_run_name", f"cheetah_tdmpc2_{config['seed']}")
        save_path = os.path.join(save_dir, run_name)
        os.makedirs(save_path, exist_ok=True)
        print(f"\nSaving parameters to {save_path}...")
        params_np = jax.device_get(parameters)
        file_path = os.path.join(save_path, "parameters.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(params_np, f)
        print(f"Parameters saved to {file_path}")

    # ---- Plots ----
    if plot_run and buffer_idx > 0:
        fig = plot_cheetah_velocity(buffers, buffer_idx, config)
        wandb.log({"trajectory/velocity_plot": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)

        fig = plot_state_components(buffers, buffer_idx, config)
        wandb.log({"trajectory/state_components": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)

        fig = plot_action_trajectory(buffers, buffer_idx, config)
        wandb.log({"trajectory/actions": wandb.Image(fig)}, step=config["total_steps"])
        plt.close(fig)

        # if len(full_states_for_animation) > 0:
        #     full_states_array = np.array(full_states_for_animation)
        #     gif_path = create_cheetah_xy_animation(full_states_array)
        #     wandb.log(
        #         {"trajectory/animation": wandb.Video(gif_path, fps=20, format="gif")},
        #         step=config["total_steps"]
        #     )

    print("Run complete.")


def run_sweep():
    """Entry point for wandb sweep agents."""
    wandb.init()

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "cheetah.json"
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)

    run_config = copy.deepcopy(full_config["finetuning"])

    for key, value in wandb.config.items():
        keys = key.split(".")
        target = run_config
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value

    main(run_config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TDMPC2 cheetah finetuning.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument(
        "--config",
        type=str,
        default="cheetah.json",
        help="Config filename in configs folder.",
    )
    args = parser.parse_args()

    if os.environ.get("WANDB_SWEEP_ID"):
        run_sweep()
    else:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", args.config
        )
        with open(config_path, "r") as f:
            full_config = json.load(f)
        CONFIG = full_config["finetuning"]

        run_name_base = args.run_name or "cheetah_tdmpc2"

        base_key = jax.random.key(CONFIG["seed"])
        seed_keys = jax.random.split(base_key, args.num_seeds)
        seeds = [int(jax.random.bits(k)) for k in seed_keys]

        for seed_idx, seed in enumerate(seeds, start=1):
            print(f"--- Starting run seed {seed_idx}/{args.num_seeds} ---")
            run_config = copy.deepcopy(CONFIG)
            run_config["seed"] = seed
            run_name = run_name_base
            if args.num_seeds > 1:
                run_name = f"{run_name}_{seed_idx}"
            run_config["wandb_run_name"] = run_name

            wandb.init(
                project=run_config.get("wandb_project", "cheetah_tdmpc2"),
                config=run_config,
                name=run_config.get("wandb_run_name"),
                reinit=True,
            )
            main(run_config)
            wandb.finish()

        print("All experiments complete.")
