# train.py

import os
# os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"
os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import jax

import jax.numpy as jnp
import numpy as np
import wandb
from max.buffers import init_buffer, episodes_from_buffer
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
from max.rollouts import init_rollout, prefill_buffer
import argparse
import copy
import os
import pickle
import json
from datetime import datetime

from max.visualizers import init_visualizer


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
    pretrained_params = {}
    if path := config.get("pretrained_path"):
        with open(path, "rb") as f:
            pretrained_params = pickle.load(f)["mean"]
        print(f"Loaded pretrained parameters from {path}")

    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_parameters    = init_encoder(enc_key, config,    pretrained=pretrained_params.get("encoder"))
    dynamics,     dyn_parameters    = init_dynamics(dyn_key, config,   pretrained=pretrained_params.get("dynamics"))
    critic,       critic_parameters = init_critic(critic_key, config,  pretrained=pretrained_params.get("critic"))
    policy,       policy_parameters = init_policy(policy_key, config,  pretrained=pretrained_params.get("policy"))
    reward_model, reward_parameters = init_reward_model(config,        pretrained=pretrained_params.get("reward"))

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
        "normalizer": {"q_scale": jnp.array(config["normalizer"]["critic"]["q_scale_init"], dtype=jnp.float32)},
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
    visualizer = init_visualizer(config) if plot_eval else None

    # ---- Parameter count ----
    total_n = count_parameters(parameters["mean"])
    wandb.config.update({"num_params_total": total_n})
    print(f"[{time.time()-t0:.2f}s] Components ready  (total={total_n:,})")

    print(f"Starting TDMPC2 cheetah training for {config['max_steps']} steps")

    # ---- Initial evaluation ----
    print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    eval_results = evaluator.evaluate(parameters)
    initial_metrics = {
        k: v for k, v in eval_results.items() if isinstance(v, (int, float))
    }
    wandb.log(initial_metrics, step=0)
    print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")

    if visualizer is not None and "trajectory" in eval_results:
        video_path = visualizer.visualize(eval_results["trajectory"])
        wandb.log({"eval/animation": wandb.Video(video_path, format="mp4")}, step=0)
        print(f"[{time.time()-t0:.2f}s] Animation logged")

    # ---- Build rollout ----
    key, rollout_key = jax.random.split(key)
    rollout, rollout_state = init_rollout(
        rollout_key, config,
        reset_fn, step_fn, get_obs_fn,
        planner, planner_state,
        trainer, train_state,
        sampler,
        parameters, buffers,
    )
    print(f"[{time.time()-t0:.2f}s] Rollout initialized")

    # ---- Pre-fill buffer with random actions ----
    sampler_cfg = config["sampler"]
    min_buffer_size = sampler_cfg.get(
        "min_buffer_size",
        sampler_cfg["batch_size"] + sampler_cfg["horizon"],
    )
    print(f"[{time.time()-t0:.2f}s] Pre-filling buffer ({min_buffer_size} steps)...")
    rollout_state = prefill_buffer(
        rollout_state,
        reset_fn, step_fn, get_obs_fn,
        dim_a=config["dim_action"],
        min_buffer_size=min_buffer_size,
        buffer_size=config["buffer_size"],
    )
    print(f"[{time.time()-t0:.2f}s] Buffer pre-filled (buffer_idx={int(rollout_state.buffer_idx)})")

    # ---- Chunk loop (jax.lax.scan per chunk) ----
    chunk_size = config["chunk_size"]
    num_chunks = config["max_steps"] // chunk_size
    eval_every = config["eval_freq"] // chunk_size
    checkpoint_chunk_freq = max(1, checkpoint_freq // chunk_size)

    print(f"[{time.time()-t0:.2f}s] Starting scan loop ({num_chunks} chunks of {chunk_size} steps)...")
    print(f"  First chunk triggers JIT compilation — expect a delay.")

    buffer_size = config["buffer_size"]
    ep_partial_reward = 0.0
    ep_partial_len = 0

    for chunk_idx in range(1, num_chunks + 1):
        prev_idx = int(rollout_state.buffer_idx)
        t_chunk = time.time()
        rollout_state, chunk_out = rollout.scan_fn(rollout_state)
        jax.block_until_ready(chunk_out)
        dt = time.time() - t_chunk
        step = chunk_idx * chunk_size

        # ---- Log mean train metrics for this chunk ----
        mean_metrics = {k: float(jnp.mean(v)) for k, v in chunk_out.train_metrics.items()}
        wandb.log(mean_metrics, step=step)

        # ---- Log episode stats from buffer ----
        curr_idx = int(rollout_state.buffer_idx)
        episodes, ep_partial_reward, ep_partial_len = episodes_from_buffer(
            rollout_state.buffers, prev_idx, curr_idx, buffer_size,
            ep_partial_reward, ep_partial_len,
        )
        if episodes:
            wandb.log(
                {
                    "episodes/reward": float(np.mean([ep["episodes/reward"] for ep in episodes])),
                    "episodes/length": float(np.mean([ep["episodes/length"] for ep in episodes])),
                },
                step=step,
            )

        # ---- Evaluation ----
        sps = chunk_size / dt
        if chunk_idx % eval_every == 0:
            t_eval = time.time()
            eval_results = evaluator.evaluate(rollout_state.parameters)
            dt_eval = time.time() - t_eval

            metrics_to_log = {
                k: v for k, v in eval_results.items() if isinstance(v, (int, float))
            }
            wandb.log(metrics_to_log, step=step)

            if visualizer is not None and "trajectory" in eval_results:
                video_path = visualizer.visualize(eval_results["trajectory"])
                wandb.log(
                    {"eval/animation": wandb.Video(video_path, format="mp4")},
                    step=step,
                )

            print(
                f"[Step {step}] chunk={dt:.2f}s ({sps:.0f} steps/s) | eval={dt_eval:.2f}s"
            )
        else:
            print(f"[Step {step}] chunk={dt:.2f}s ({sps:.0f} steps/s)")

        # ---- Checkpoint ----
        if checkpoint_enabled and run_dir and chunk_idx % checkpoint_chunk_freq == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(jax.device_get(rollout_state.parameters), f)
            print(f"Checkpoint saved: {ckpt_path}")

    # ---- Save final parameters ----
    if run_dir:
        file_path = os.path.join(run_dir, "final.pkl")
        print(f"\nSaving final parameters to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(jax.device_get(rollout_state.parameters), f)
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
