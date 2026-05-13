#!/usr/bin/env python3
"""
Dominance contest active info-gathering example.

Ego agent estimates opponent quality θ = [q, K] online via EKF while
planning actions that trade off task cost (c·x1) against info gain.

Usage:
    conda run -n max python scripts/run_animals.py configs/animals.json
"""

import os
import sys
import json
import argparse

os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser("~/.cache/jax_cache")
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import jax
import jax.numpy as jnp
import wandb

from max.environments import init_env
from max.dynamics import create_animals_dynamics
from max.rewards import init_animals_reward
from max.trainers import init_trainer
from max.planners import init_planner


def main(config):
    key = jax.random.key(config["seed"])

    reset_fn, step_fn, get_obs_fn = init_env(config)

    dynamics, dyn_params = create_animals_dynamics(config)
    reward_model, reward_params = init_animals_reward(config)

    parameters = {
        "mean": {"dynamics": dyn_params, "reward": reward_params},
        "normalizer": {},
    }

    # init_trainer mutates parameters to add ["covariance"]
    trainer, train_state = init_trainer(key, config, dynamics=dynamics, init_params=parameters)

    key, planner_key = jax.random.split(key)
    planner, planner_state = init_planner(
        config, key=planner_key, dynamics=dynamics, reward=reward_model
    )

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)

    for step in range(1, config["total_steps"] + 1):
        action_seqs, _, planner_state = planner.solve(planner_state, state, parameters)
        action = action_seqs[0:1]  # (1, dim_a)

        state, obs, reward, done, truncated, info = step_fn(state, step, action)

        # Generate noisy observation outside JIT boundary
        key, noise_key = jax.random.split(key)
        noise = float(config["environment"]["sigma_v"]) * jax.random.normal(noise_key)
        true_K = float(info["true_K"])
        true_q = float(info["true_q"])
        x1_pre = float(info["x1_pre"])
        y_obs = true_K * (true_q - x1_pre) + float(noise)

        train_data = {"x1": info["x1_pre"], "y_obs": jnp.array(y_obs)}
        train_state, parameters, metrics = trainer.train(train_state, train_data, parameters, key)

        wandb.log(
            {
                "belief/q_hat": float(parameters["mean"]["dynamics"]["q"]),
                "belief/K_hat": float(parameters["mean"]["dynamics"]["K"]),
                "belief/cov_trace": float(jnp.trace(parameters["covariance"])),
                "env/x1": float(state[0]),
                "env/x2": float(state[1]),
                **{k: float(v) for k, v in metrics.items()},
            },
            step=step,
        )

        if step % 20 == 0:
            q_hat = float(parameters["mean"]["dynamics"]["q"])
            K_hat = float(parameters["mean"]["dynamics"]["K"])
            cov_trace = float(jnp.trace(parameters["covariance"]))
            print(f"step {step:3d}  x1={float(state[0]):.3f}  q̂={q_hat:.3f}  K̂={K_hat:.3f}  cov={cov_trace:.4f}")


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to JSON config file")
    parser.add_argument("--name", type=str, default=None, help="wandb run name (overrides config)")
    parser.add_argument("--info_weight", type=float, default=None, help="override planner info_weight")
    parser.add_argument("--total_steps", type=int, default=None, help="override total_steps")
    parser.add_argument("--horizon", type=int, default=None, help="override planner horizon")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if args.info_weight is not None:
        config["planner"]["info_weight"] = args.info_weight
    if args.total_steps is not None:
        config["total_steps"] = args.total_steps
    if args.horizon is not None:
        config["planner"]["horizon"] = args.horizon

    ts = datetime.now().strftime("%m%d_%H%M%S")
    iw = config["planner"]["info_weight"]
    q = config["environment"]["true_q"]
    K = config["environment"]["true_K"]
    auto_name = f"iw{iw}_q{q}_K{K}_{ts}"
    run_name = args.name if args.name else auto_name

    wandb.init(
        project=config["wandb_project"],
        name=run_name,
        config=config,
    )
    main(config)
    wandb.finish()
