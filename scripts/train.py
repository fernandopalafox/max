# train.py

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

import time

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import wandb
from max.buffers import init_buffer, update_buffer
from max.utilities import count_parameters
from max.environments import init_env
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.critics import init_critic
from max.policies import init_policy
from max.rewards import init_reward_model
from max.trainers import init_trainer
from max.samplers import init_sampler
from max.evaluators import init_evaluator
from max.planners import init_planner
import argparse
import copy
import os
import pickle
import json
from datetime import datetime

from max.visualizers import create_cheetah_xy_animation



def main(config):
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    save_dir = config.get("save_dir", None)
    plot_eval = config.get("plot_eval", False)
    checkpoint_enabled = config.get("checkpoint_enabled", False)
    checkpoint_freq = config.get("checkpoint_freq", 100000)

    # Create timestamped run directory
    run_dir = None
    if save_dir:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        if checkpoint_enabled:
            print(f"Checkpointing every {checkpoint_freq} steps to {run_dir}/")

    # ---- Environment ----
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # ---- Model components ----
    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_parameters    = init_encoder(enc_key, config)
    dynamics,     dyn_parameters    = init_dynamics(dyn_key, config)
    critic,       critic_parameters = init_critic(critic_key, config)
    policy,       policy_parameters = init_policy(policy_key, config)
    reward_model, reward_parameters = init_reward_model(config)

    # ---- Parameters dict ----
    parameters = {
        "mean": {
            "encoder":    enc_parameters,
            "dynamics":   dyn_parameters,
            "reward":     reward_parameters,
            "critic":     critic_parameters,
            "ema_critic": copy.deepcopy(critic_parameters),
            "policy":     policy_parameters,
        },
        "normalizer": {"q_scale": jnp.array(config["normalizer"]["critic"]["q_scale_init"])},
    }

    # ---- Trainer ----
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        trainer_key, config, encoder, dynamics, critic, policy, reward_model, parameters
    )

    # ---- Sampler, evaluator, planner, buffer ----
    sampler = init_sampler(config["sampler"])

    evaluator = init_evaluator(
        config,
        encoder=encoder,
        dynamics=dynamics,
        reward=reward_model,
        critic=critic,
        policy=policy,
    )

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

    buffers = init_buffer(config)
    buffer_idx = 0

    # ---- Parameter count ----
    total_n = count_parameters(parameters["mean"])
    wandb.config.update({"num_params_total": total_n})
    print(f"[{time.time()-t0:.2f}s] Components ready  (total={total_n:,})")

    print(f"Starting TDMPC2 cheetah finetuning for {config['max_steps']} steps")

    episode_length = 0
    episode_total_reward = 0.0

    # Initial evaluation
    print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    eval_results = evaluator.evaluate(parameters)
    initial_metrics = {
        k: v for k, v in eval_results.items() if isinstance(v, (int, float))
    }
    wandb.log(initial_metrics, step=0)
    print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")

    if plot_eval and "trajectory" in eval_results:
        traj = eval_results["trajectory"]
        full_states = np.concatenate([traj.qpos, traj.qvel], axis=-1)
        gif_path = create_cheetah_xy_animation(full_states)
        wandb.log({"eval/animation": wandb.Video(gif_path, format="gif")}, step=0)

    # ---- Main training loop ----
    print(f"[{time.time()-t0:.2f}s] Starting main loop...")
    key, reset_key = jax.random.split(key)
    mjx_data = reset_fn(reset_key)
    current_obs = get_obs_fn(mjx_data).squeeze()

    t_planner = 0.0
    t_step = 0.0
    t_buffer = 0.0
    t_train = 0.0
    t_eval = 0.0

    train_freq = config.get("train_freq", 1)

    for step in range(1, config["max_steps"] + 1):
        step_start = time.time()

        # ---- Planning step (MPPI in latent space) ----
        _t0 = time.time()
        key, planner_key = jax.random.split(key)
        planner_state = planner_state.replace(key=planner_key)
        actions, planner_state = planner.solve(planner_state, current_obs, parameters)
        action = actions[0][None, :]  # (1, dim_a) with agent dim
        dt_planner = time.time() - _t0
        t_planner += dt_planner

        # ---- Environment step ----
        _t0 = time.time()
        mjx_data, next_obs, rewards, terminated, truncated, _ = step_fn(
            mjx_data, episode_length, action
        )
        dt_step = time.time() - _t0
        t_step += dt_step
        next_obs = next_obs.squeeze()
        done = terminated or truncated
        episode_length += 1
        episode_total_reward += float(rewards[0])

        # ---- Buffer update ----
        _t0 = time.time()
        buffers = update_buffer(
            buffers,
            buffer_idx,
            current_obs[None, :],
            action,
            rewards,
            float(done),
        )
        buffer_idx += 1
        dt_buffer = time.time() - _t0
        t_buffer += dt_buffer
        current_obs = next_obs

        # ---- Episode reset ----
        if done:
            key, reset_key = jax.random.split(key)
            mjx_data = reset_fn(reset_key)
            current_obs = get_obs_fn(mjx_data).squeeze()
            wandb.log({
                "episodes/length": episode_length,
                "episodes/reward": episode_total_reward,
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
                    {k: float(v) for k, v in metrics.items()},
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
                    {"eval/animation": wandb.Video(gif_path, format="gif")},
                    step=step
                )

        step_total = time.time() - step_start
        print(
            f"[Step {step}] total={step_total:.3f}s | "
            f"planner={dt_planner:.3f}s, step={dt_step:.3f}s, "
            f"buffer={dt_buffer:.3f}s, train={dt_train:.3f}s, eval={dt_eval:.3f}s"
        )

        # ---- Checkpoint ----
        if checkpoint_enabled and run_dir and step % checkpoint_freq == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(jax.device_get(parameters), f)
            print(f"Checkpoint saved: {ckpt_path}")

        # Handle buffer overflow
        if buffer_idx >= config["buffer_size"]:
            buffers = init_buffer(config)
            buffer_idx = 0

    # ---- Final timing summary ----
    print(f"\n=== TIMING SUMMARY ===")
    print(f"Total planner time: {t_planner:.2f}s")
    print(f"Total step time:    {t_step:.2f}s")
    print(f"Total buffer time:  {t_buffer:.2f}s")
    print(f"Total train time:   {t_train:.2f}s")
    print(f"Total eval time:    {t_eval:.2f}s")
    print(f"======================\n")

    # ---- Save final parameters ----
    if run_dir:
        file_path = os.path.join(run_dir, "final.pkl")
        print(f"\nSaving final parameters to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(jax.device_get(parameters), f)
        print(f"Parameters saved to {file_path}")

    print("Run complete.")


def run_sweep():
    """Entry point for wandb sweep agents."""
    wandb.init()

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", "cheetah.json"
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)

    run_config = copy.deepcopy(full_config["training"])

    for key, value in wandb.config.items():
        keys = key.split(".")
        target = run_config
        for k in keys[:-1]:
            target = target[k]
        target[keys[-1]] = value

    main(run_config)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TDMPC2 training.")
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
        CONFIG = full_config["training"]

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
