# finetune.py

import os
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
from max.visualizers import init_visualizer


def main(config):
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    save_dir = config["save_dir"]
    save_checkpoints = config["save_checkpoints"]
    save_final = config["save_final"]
    checkpoint_freq = config["checkpoint_freq"]

    run_dir = None
    if save_dir:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(save_dir, run_timestamp)
        os.makedirs(run_dir, exist_ok=True)
        if save_checkpoints:
            print(f"Checkpointing every {checkpoint_freq} steps to {run_dir}/")

    # ---- Load offline buffer ----
    buffer_path = config["finetune"]["buffer_path"]
    data = np.load(buffer_path)
    buffers = {
        "states":  jnp.array(data["states"]),
        "actions": jnp.array(data["actions"]),
        "rewards": jnp.array(data["rewards"]),
        "dones":   jnp.array(data["dones"]),
    }
    buffer_idx = int(data["n_steps"])
    print(f"[{time.time()-t0:.2f}s] Loaded buffer: {buffer_idx} steps from {buffer_path}")

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

    # ---- Trainer and sampler ----
    key, trainer_key = jax.random.split(key)
    trainer, train_state = init_trainer(
        trainer_key, config, encoder, dynamics, critic, policy, reward_model, parameters
    )
    sampler = init_sampler(config["sampler"])

    # ---- Evaluator and planner ----
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
    visualizer = init_visualizer(config) if config["plot_eval"] else None

    print(f"[{time.time()-t0:.2f}s] Components ready")

    # ---- Initial evaluation ----
    print(f"[{time.time()-t0:.2f}s] Running initial evaluation...")
    eval_results = evaluator.evaluate(parameters)
    initial_metrics = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
    wandb.log(initial_metrics, step=0)
    if visualizer is not None and "trajectory" in eval_results:
        video_path = visualizer.visualize(eval_results["trajectory"])
        wandb.log({"eval/animation": wandb.Video(video_path, format="mp4")}, step=0)
    print(f"[{time.time()-t0:.2f}s] Initial evaluation complete")

    # ---- Finetuning loop ----
    max_steps = config["finetune"]["max_steps"]
    eval_freq = config["eval_freq"]
    train_fn = jax.jit(trainer.train)

    print(f"[{time.time()-t0:.2f}s] Starting finetuning ({max_steps} steps)...")
    print(f"  First step triggers JIT compilation — expect a delay.")

    accumulated_metrics = {}

    for step in range(1, max_steps + 1):
        key, sample_key, train_key = jax.random.split(key, 3)
        train_data = sampler.sample_jit(sample_key, buffers, buffer_idx)
        train_state, parameters, train_metrics = train_fn(
            train_state, train_data, parameters, train_key
        )

        for k, v in train_metrics.items():
            accumulated_metrics.setdefault(k, []).append(v)

        if step % eval_freq == 0:
            jax.block_until_ready(parameters)
            mean_metrics = {k: float(jnp.mean(jnp.stack(vs))) for k, vs in accumulated_metrics.items()}
            accumulated_metrics = {}

            t_eval = time.time()
            eval_results = evaluator.evaluate(parameters)
            dt_eval = time.time() - t_eval

            eval_scalar = {k: v for k, v in eval_results.items() if isinstance(v, (int, float))}
            wandb.log({**mean_metrics, **eval_scalar}, step=step)

            if visualizer is not None and "trajectory" in eval_results:
                video_path = visualizer.visualize(eval_results["trajectory"])
                wandb.log({"eval/animation": wandb.Video(video_path, format="mp4")}, step=step)

            print(f"[Step {step}] eval={dt_eval:.2f}s | reward={eval_scalar.get('eval/episode_reward', float('nan')):.3f}")

        if save_checkpoints and run_dir and step % checkpoint_freq == 0:
            ckpt_path = os.path.join(run_dir, f"step_{step}.pkl")
            with open(ckpt_path, "wb") as f:
                pickle.dump(jax.device_get(parameters), f)
            print(f"Checkpoint saved: {ckpt_path}")

    # ---- Save final parameters ----
    if save_final:
        assert run_dir, "save_final requires save_dir to be set"
        file_path = os.path.join(run_dir, "final.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(jax.device_get(parameters), f)
        print(f"Parameters saved to {file_path}")

    print("Finetuning complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune a model on an offline dataset.")
    parser.add_argument("--config", type=str, default="finetune_cartpole.json",
                        help="Config filename (relative to configs/) or absolute path.")
    parser.add_argument("--gpu", type=str, default=None, help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--seed", type=int, default=None, help="Override training.seed in config.")
    parser.add_argument("--pretrained-path", type=str, default=None,
                        help="Override training.pretrained_path in config.")
    parser.add_argument("--save-dir", type=str, default=None, help="Override training.save_dir in config.")
    parser.add_argument("--buffer-path", type=str, default=None,
                        help="Override training.finetune.buffer_path in config.")
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
    if args.buffer_path is not None:
        CONFIG["finetune"]["buffer_path"] = args.buffer_path

    wandb.init(
        project=CONFIG["wandb_project"],
        config=CONFIG,
        name=CONFIG.get("wandb_run_name"),
        group=CONFIG.get("wandb_group"),
    )
    main(CONFIG)
    wandb.finish()
