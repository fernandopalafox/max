# train_grad_capture.py
# Modified training script that captures gradient sums for subspace alignment analysis.
# Three changes from train.py:
#   1. Uses init_grad_capture_trainer instead of init_trainer
#   2. Captures w_initial (initial Dense kernel weights) before training
#   3. Saves grad_capture.pkl after training completes

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
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.critics import init_critic
from max.policies import init_policy
from max.rewards import init_reward_model
from max.trainers_grad_capture import init_grad_capture_trainer
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

    # ---- Trainer (grad capture variant) ----
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_grad_capture_trainer(
        trainer_key, config, encoder, dynamics, critic, policy, reward_model, parameters
    )

    # ---- Capture initial Dense kernel weights before any training ----
    dense_names = sorted([
        k for k in parameters["mean"]["dynamics"]["params"]
        if k.startswith("Dense_")
    ])
    w_initial = {
        name: np.array(parameters["mean"]["dynamics"]["params"][name]["kernel"])
        for name in dense_names
    }
    print(f"Captured w_initial for layers: {dense_names}")

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

    # If capture_last_steps > 0, reset gradient accumulators at the right chunk boundary
    # so only the final capture_last_steps steps are accumulated.
    capture_last_steps = config.get("capture_last_steps", 0)
    capture_last_chunks = capture_last_steps // chunk_size if capture_last_steps > 0 else 0
    reset_at_chunk = num_chunks - capture_last_chunks if capture_last_chunks > 0 else None
    if reset_at_chunk is not None:
        print(f"[{time.time()-t0:.2f}s] Will reset gradient accumulators at step {reset_at_chunk * chunk_size} "
              f"(capturing last {capture_last_steps} steps only)")

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

        # Reset gradient accumulators after completing the chunk that ends at reset_at_chunk * chunk_size
        if reset_at_chunk is not None and chunk_idx == reset_at_chunk:
            zero_sums = jax.tree_util.tree_map(jnp.zeros_like, rollout_state.train_state.g_sum_unscaled)
            rollout_state = rollout_state._replace(
                train_state=rollout_state.train_state.replace(
                    g_sum_unscaled=zero_sums,
                    g_sum_scaled=jax.tree_util.tree_map(jnp.zeros_like, rollout_state.train_state.g_sum_scaled),
                )
            )
            print(f"[Step {step}] Gradient accumulators reset — capturing last {capture_last_steps} steps.")

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

    # ---- Save gradient capture data ----
    if run_dir:
        grad_path = os.path.join(run_dir, "grad_capture.pkl")
        print(f"Saving gradient capture data to {grad_path}...")
        w_final = {
            name: np.array(rollout_state.parameters["mean"]["dynamics"]["params"][name]["kernel"])
            for name in dense_names
        }
        grad_data = {
            "w_initial": w_initial,
            "w_final": w_final,
            "g_sum_unscaled": jax.device_get(rollout_state.train_state.g_sum_unscaled),
            "g_sum_scaled": jax.device_get(rollout_state.train_state.g_sum_scaled),
        }
        with open(grad_path, "wb") as f:
            pickle.dump(grad_data, f)
        print(f"Gradient data saved to {grad_path}")

    print("Run complete.")


if __name__ == "__main__":
    import sys
    import shutil
    import subprocess
    import tempfile

    parser = argparse.ArgumentParser(description="Run TDMPC2 training with gradient capture.")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--config",
        type=str,
        default="pretrain_grad_capture_tdmpc2.json",
        help="Config filename or absolute path.",
    )
    parser.add_argument("--gpu", type=str, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--save-dir", type=str, default=None, help="Override training.save_dir in config.")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed in config.")
    parser.add_argument("--pretrained-path", type=str, default=None, help="Override training.pretrained_path in config.")
    parser.add_argument("--capture-last-steps", type=int, default=None,
                        help="Only accumulate gradients in the final N steps (resets accumulators at step max_steps-N).")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max_steps in config.")
    args = parser.parse_args()

    config_path = os.path.join(
        os.path.dirname(__file__), "..", "configs", args.config
    )
    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["training"]

    if args.capture_last_steps is not None:
        CONFIG["capture_last_steps"] = args.capture_last_steps
    if args.max_steps is not None:
        CONFIG["max_steps"] = args.max_steps

    if args.save_dir is not None:
        CONFIG["save_dir"] = args.save_dir
    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.pretrained_path is not None:
        CONFIG["pretrained_path"] = args.pretrained_path

    run_name_base = args.run_name or "grad_capture"
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
