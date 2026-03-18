# finetune_cheetah.py

import os
os.environ['XLA_FLAGS'] = '--xla_gpu_deterministic_ops=true'
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.45"

import time

import jax
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

import jax.numpy as jnp
import numpy as np
import wandb
from max.normalizers import init_normalizer
from max.buffers import init_buffer, update_buffer
from max.utilities import count_parameters
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



def main(config):
    t0 = time.time()
    wandb.config.update(config, allow_val_change=True)
    key = jax.random.key(config["seed"])

    save_dir = config.get("save_dir", None)
    plot_eval = config.get("plot_eval", False)

    # ---- Environment ----
    env_params = EnvParams(**config["env_params"])
    reset_fn, step_fn, get_obs_fn = make_cheetah_env(env_params)

    # ---- Normalizer ----
    normalizer, norm_params = init_normalizer(config)

    # ---- Model components ----
    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)
    encoder,      enc_params    = init_encoder(enc_key, config, normalizer)
    dynamics,     dyn_params    = init_dynamics(dyn_key, config, normalizer, norm_params)
    critic,       critic_params = init_critic(critic_key, config)
    policy,       policy_params = init_policy(policy_key, config)
    reward_model, reward_params = init_reward_model(config, encoder=encoder)

    # ---- Parameters dict ----
    parameters = {
        "mean": {
            "encoder":    enc_params,
            "dynamics":   dyn_params,
            "reward":     reward_params,
            "critic":     critic_params,
            "ema_critic": copy.deepcopy(critic_params),
            "policy":     policy_params,
        },
        "normalizer": {**norm_params, "q_scale": jnp.array(1.0)},
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
        wandb.log({"eval/animation": wandb.Video(gif_path, fps=20, format="gif")}, step=0)

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
        t_buffer += time.time() - _t0
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
