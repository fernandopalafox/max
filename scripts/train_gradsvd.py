# train_gradsvd.py
# Adaptation training script using LoRA-XS initialized from scaled gradient SVD.
# Three changes from train.py:
#   1. Imports init_gradsvd_lora_xs_dynamics instead of init_dynamics
#   2. Loads grad_capture.pkl (g_sum_scaled) from --grad-capture-path
#   3. Passes grad_avg to init_gradsvd_lora_xs_dynamics when initializing dynamics

import os
import sys
import argparse as _ap

_pp = _ap.ArgumentParser(add_help=False)
_pp.add_argument("--gpu", type=str, default=None)
_pre, _ = _pp.parse_known_args()
if _pre.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pre.gpu

os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.cache/jax_cache")
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
from max.dynamics_gradsvd import init_gradsvd_lora_xs_dynamics
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

    save_dir = config["save_dir"]
    plot_eval = config["plot_eval"]
    save_checkpoints = config["save_checkpoints"]
    save_final = config["save_final"]
    checkpoint_freq = config["checkpoint_freq"]

    # Create timestamped run directory
    run_dir = None
    if save_dir:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        if save_checkpoints:
            print(f"Checkpointing every {checkpoint_freq} steps to {run_dir}/")

    # ---- Environment ----
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # ---- Model components ----
    pretrained_params = {}
    if path := config["pretrained_path"]:
        with open(path, "rb") as f:
            pretrained_params = pickle.load(f)["mean"]
        print(f"Loaded pretrained parameters from {path}")

    # ---- Load gradient capture data (g_sum_scaled = Adam updates) ----
    with open(config["grad_capture_path"], "rb") as f:
        gc = pickle.load(f)
    grad_avg = gc["g_sum_scaled"]
    print(f"Loaded gradient data from {config['grad_capture_path']}")

    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_parameters    = init_encoder(enc_key, config,    pretrained=pretrained_params.get("encoder"))
    dynamics,     dyn_parameters    = init_gradsvd_lora_xs_dynamics(
        dyn_key, config, pretrained=pretrained_params.get("dynamics"), grad_avg=grad_avg
    )
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

    print(f"Starting loraxs_gradsvd adaptation for {config['max_steps']} steps")

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

    # ---- Pre-fill buffer ----
    prefill_buffer_size = config["prefill_buffer_size"]
    prefill_planner = planner if config["prefill_with_planner"] else None
    print(f"[{time.time()-t0:.2f}s] Pre-filling buffer ({prefill_buffer_size} steps)...")
    rollout_state = prefill_buffer(
        rollout_state,
        reset_fn, step_fn, get_obs_fn,
        dim_a=config["dim_action"],
        prefill_buffer_size=prefill_buffer_size,
        buffer_size=config["buffer_size"],
        action_min=config["action_min"],
        action_max=config["action_max"],
        planner=prefill_planner,
    )
    print(f"[{time.time()-t0:.2f}s] Buffer pre-filled (buffer_idx={int(rollout_state.buffer_idx)})")

    # ---- Initial SGD burst ----
    sgd_burst_steps = config["sgd_burst_steps"]
    if sgd_burst_steps > 0:
        print(f"[{time.time()-t0:.2f}s] Initial SGD burst ({sgd_burst_steps} steps)...")
        pretrain_fn = jax.jit(trainer.train)
        for _ in range(sgd_burst_steps):
            key, sample_key, train_key = jax.random.split(key, 3)
            train_data = sampler.sample_jit(sample_key, rollout_state.buffers, rollout_state.buffer_idx)
            new_train_state, new_parameters, _ = pretrain_fn(
                rollout_state.train_state, train_data, rollout_state.parameters, train_key
            )
            rollout_state = rollout_state._replace(
                train_state=new_train_state,
                parameters=new_parameters,
            )
        print(f"[{time.time()-t0:.2f}s] Initial SGD burst complete")

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
        sps = chunk_size / dt
        wandb.log({**mean_metrics, "train/steps_per_second": sps}, step=step)

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
        if save_checkpoints and run_dir and chunk_idx % checkpoint_chunk_freq == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(jax.device_get(rollout_state.parameters), f)
            print(f"Checkpoint saved: {ckpt_path}")

    # ---- Save final parameters ----
    if save_final:
        assert run_dir, "save_final requires save_dir to be set"
        file_path = os.path.join(run_dir, "final.pkl")
        print(f"\nSaving final parameters to {file_path}...")
        with open(file_path, "wb") as f:
            pickle.dump(jax.device_get(rollout_state.parameters), f)
        print(f"Parameters saved to {file_path}")

    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LoRA-XS adaptation with gradient SVD initialization.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default="adapt_full_with_loraxs_gradsvd_pt.json",
        help="Config filename or absolute path.",
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--save-dir", type=str, default=None, help="Override training.save_dir in config.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed in config.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Override training.pretrained_path in config.")
    parser.add_argument("--grad-capture-path", type=str, default=None, help="Override training.grad_capture_path in config.")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["training"]

    if args.save_dir is not None:
        CONFIG["save_dir"] = args.save_dir
    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.pretrained_path is not None:
        CONFIG["pretrained_path"] = args.pretrained_path
    if args.grad_capture_path is not None:
        CONFIG["grad_capture_path"] = args.grad_capture_path

    run_name_base = args.run_name or "loraxs_gradsvd"
    num_seeds = CONFIG["num_seeds"]

    base_key = jax.random.key(CONFIG["seed"])
    seed_keys = jax.random.split(base_key, num_seeds)
    seeds = [int(jax.random.bits(k)) for k in seed_keys]

    for seed_idx, seed in enumerate(seeds, start=1):
        print(f"--- Starting run {seed_idx}/{num_seeds} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = seed
        run_name = run_name_base
        if num_seeds > 1:
            run_name = f"{run_name}_{seed_idx}"
        run_config["wandb_run_name"] = run_name

        wandb.init(
            project=run_config["wandb_project"],
            config=run_config,
            name=run_config["wandb_run_name"],
            group=run_config.get("wandb_group"),
            reinit=True,
        )
        main(run_config)
        wandb.finish()

    print("All experiments complete.")
