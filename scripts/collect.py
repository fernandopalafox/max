# collect.py

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
import copy
import json
import pickle
import argparse
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np
import wandb

from max.buffers import init_buffer, episodes_from_buffer
from max.environments import init_env
from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.critics import init_critic
from max.policies import init_policy
from max.rewards import init_reward_model
from max.evaluators import init_evaluator
from max.planners import init_planner
from max.rollouts import init_collect_rollout
from max.visualizers import init_visualizer


def _save_buffer(buffers, buffer_idx, buffer_size, run_dir, config):
    n_written = int(buffer_idx)
    n_valid = min(n_written, buffer_size)
    wrapped = n_written > buffer_size
    write_ptr = n_written % buffer_size

    buf_np = jax.device_get(buffers)
    buf_path = os.path.join(run_dir, "buffer.npz")
    np.savez_compressed(
        buf_path,
        states=buf_np["states"][:, :n_valid, :],
        actions=buf_np["actions"][:, :n_valid, :],
        rewards=buf_np["rewards"][:, :n_valid],
        dones=buf_np["dones"][:n_valid],
        n_steps=np.array(n_valid),
        wrapped=np.array(wrapped),
        write_ptr=np.array(write_ptr),
    )

    meta = {
        "n_steps": n_valid,
        "wrapped": wrapped,
        "write_ptr": write_ptr,
        "dim_state": config["dim_state"],
        "dim_action": config["dim_action"],
        "num_agents": config["num_agents"],
        "explore": config["collect"]["explore"],
        "pretrained_path": config["pretrained_path"],
        "seed": config["seed"],
        "environment": config["environment"],
    }
    meta_path = os.path.join(run_dir, "metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Buffer saved: {n_valid} steps ({'wrapped' if wrapped else 'sequential'}) -> {buf_path}")


def main(config):
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    save_dir = config["save_dir"]
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, run_timestamp)
    os.makedirs(run_dir, exist_ok=True)

    # ---- Environment ----
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # ---- Model components ----
    with open(config["pretrained_path"], "rb") as f:
        pretrained_params = pickle.load(f)["mean"]
    print(f"Loaded pretrained parameters from {config['pretrained_path']}")

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

    print(f"[{time.time()-t0:.2f}s] Components ready")

    # ---- Evaluator, planner, buffer ----
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
        config, key=planner_key,
        encoder=encoder, dynamics=dynamics, reward=reward_model, critic=critic, policy=policy,
    )

    buffers = init_buffer(config)
    visualizer = init_visualizer(config) if config["plot_eval"] else None

    # ---- Initial evaluation ----
    print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    eval_results = evaluator.evaluate(parameters)
    initial_metrics = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
    wandb.log(initial_metrics, step=0)
    if visualizer is not None and "trajectory" in eval_results:
        video_path = visualizer.visualize(eval_results["trajectory"])
        wandb.log({"eval/animation": wandb.Video(video_path, format="mp4")}, step=0)
    print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")

    # ---- Build collect rollout ----
    key, rollout_key = jax.random.split(key)
    rollout, rollout_state = init_collect_rollout(
        rollout_key, config,
        reset_fn, step_fn, get_obs_fn,
        planner, planner_state,
        parameters, buffers,
    )
    print(f"[{time.time()-t0:.2f}s] Rollout initialized (explore={config['collect']['explore']})")

    # ---- Chunk loop ----
    collect_steps = config["collect"]["max_steps"]
    chunk_size = config["chunk_size"]
    num_chunks = collect_steps // chunk_size
    eval_every = config["eval_freq"] // chunk_size
    buffer_size = config["buffer_size"]

    print(f"[{time.time()-t0:.2f}s] Starting collection ({num_chunks} chunks of {chunk_size} steps)...")
    print(f"  First chunk triggers JIT compilation — expect a delay.")

    ep_partial_reward = 0.0
    ep_partial_len = 0

    for chunk_idx in range(1, num_chunks + 1):
        prev_idx = int(rollout_state.buffer_idx)
        t_chunk = time.time()
        rollout_state, chunk_out = rollout.scan_fn(rollout_state)
        jax.block_until_ready(chunk_out)
        dt = time.time() - t_chunk
        step = chunk_idx * chunk_size
        sps = chunk_size / dt

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

        if chunk_idx % eval_every == 0:
            t_eval = time.time()
            eval_results = evaluator.evaluate(rollout_state.parameters)
            dt_eval = time.time() - t_eval
            metrics_to_log = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
            wandb.log(metrics_to_log, step=step)
            if visualizer is not None and "trajectory" in eval_results:
                video_path = visualizer.visualize(eval_results["trajectory"])
                wandb.log({"eval/animation": wandb.Video(video_path, format="mp4")}, step=step)
            print(f"[Step {step}] chunk={dt:.2f}s ({sps:.0f} steps/s) | eval={dt_eval:.2f}s")
        else:
            print(f"[Step {step}] chunk={dt:.2f}s ({sps:.0f} steps/s)")

    # ---- Save buffer ----
    _save_buffer(rollout_state.buffers, rollout_state.buffer_idx, buffer_size, run_dir, config)
    print("Collection complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run data collection with a pretrained model.")
    parser.add_argument("--config", type=str, default="pretrain_dense_tdmpc2.json",
                        help="Config filename (relative to configs/) or absolute path.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed in config.")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Override training.pretrained_path in config.")
    parser.add_argument("--save-dir", type=str, default=None, help="Override training.save_dir in config.")
    parser.add_argument("--collect-steps", type=int, default=None,
                        help="Override training.collect.max_steps in config.")
    args = parser.parse_args()

    if os.path.isabs(args.config):
        config_path = args.config
    else:
        config_path = os.path.join(os.path.dirname(__file__), "..", "configs", args.config)

    with open(config_path, "r") as f:
        full_config = json.load(f)
    CONFIG = full_config["training"]

    if args.seed is not None:
        CONFIG["seed"] = args.seed
    if args.pretrained_path is not None:
        CONFIG["pretrained_path"] = args.pretrained_path
    if args.save_dir is not None:
        CONFIG["save_dir"] = args.save_dir
    if args.collect_steps is not None:
        CONFIG["collect"]["max_steps"] = args.collect_steps

    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=CONFIG.get("wandb_run_name"),
        group=CONFIG.get("wandb_group"),
    )
    main(CONFIG)
    wandb.finish()
