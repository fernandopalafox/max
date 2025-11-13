# ippo_tracking.py

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from max.normalizers import (
    init_normalizer,
    init_rolling_return_normalizer,
    update_rolling_return_stats,
)
from max.buffers import init_jax_buffers, update_buffer_dynamic
from max.policies import init_policy
from max.policy_trainers import init_policy_trainer
from max.policy_evaluators import evaluate_policy
from max.environments import init_env
import argparse
import copy
import time
import functools
import os
import pickle

CONFIG = {
    "env_name": "pursuit_evasion",
    "env_params": {
       "num_agents": 3,
       "box_half_width": 1.0,
       "max_episode_steps": 25,
       "dt": 0.1,
       "max_accel": 2.0,
       "pursuer_max_accel": 3.0,
       "evader_max_accel": 4.0,
       "pursuer_max_speed": 1.0,
       "evader_max_speed": 1.3,
       "pursuer_size": 0.075,
       "evader_size": 0.05,
       "reward_shaping_k1": 1.0,
       "reward_shaping_k2": 1.0,
       "reward_collision_penalty": 1.0,
    },
    "total_steps": 200_000,
    "num_agents": 3,
    "dim_state": 14,
    "dim_action": 2,
    "train_freq": 1,
    "train_policy_freq": 2048,
    "normalize_freq": 1000000,
    "eval_freq": 100,
    "eval_traj_horizon": 100,
    "normalization": {"method": "static"},
    "normalization_params": {
        "state": {
            "min": [
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
                -1.5,
                -1.5,
                -1.0,
                -1.0,
            ],
            "max": [
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
                1.5,
                1.5,
                1.0,
                1.0,
            ],
        },
        "action": {
            "min": [-2.0, -2.0],
            "max": [2.0, 2.0],
        },
    },
    "policy": "actor-critic",
    "policy_params": {
        "hidden_layers": [64, 64],
    },
    "policy_trainer": "ippo",
    "policy_trainer_params": {
        "actor_lr": 3e-4,
        "critic_lr": 1e-3,
        "ppo_lambda": 0.95,
        "ppo_gamma": 0.99,
        "clip_epsilon": 0.2,
        "n_epochs": 4,
        "mini_batch_size": 64,
        "entropy_coef": 0.01,
        "value_coef": 0.5,
        "max_grad_norm": 0.5,
    },
    "policy_evaluator_params": {
        "n_episodes": 10,
    },
    "reward_scaling_discount_factor": 0.99,
    "reward_clip": 100.0,
}


def main(config, save_dir):
    wandb.init(
        project="pe_again",
        config=config,
        group=config.get("wandb_group"),
        name=config.get("wandb_run_name"),
        reinit=True,
    )
    key = jax.random.key(config["seed"])

    print("Setting up Multi-Agent Tracking environment...")
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Initialize policy, trainer, normalizer
    normalizer, norm_params = init_normalizer(config)

    cost_fn = None
    dynamics_model = None
    key, policy_key = jax.random.split(key)
    policy, policy_state = init_policy(
        policy_key, config, dynamics_model, cost_fn, normalizer, norm_params
    )

    key, policy_trainer_key = jax.random.split(key)
    policy_trainer, policy_train_state = init_policy_trainer(
        config, policy, policy_state.params, policy_trainer_key
    )

    # Initialize reward normalizer
    reward_normalizer, reset_rolling_return, reward_norm_params = (
        init_rolling_return_normalizer(
            num_agents=config["num_agents"],
            gamma=config["reward_scaling_discount_factor"],
            clip_range=config["reward_clip"],
        )
    )

    def _policy_step(params, state, dyn_params, key):
        action, _ = policy.select_action(params, state, dyn_params, key)
        value = policy.evaluate_value(params, state, dyn_params)
        log_pi = policy.compute_log_prob(params, state, action, dyn_params)
        return action, value, log_pi

    vmapped_step = jax.vmap(_policy_step, in_axes=(0, 0, None, 0))

    @functools.partial(jax.jit, static_argnums=None)
    def jitted_policy_step(params, states, dyn_params, keys):
        return vmapped_step(params, states, dyn_params, keys)

    key, eval_key = jax.random.split(key)
    n_eval_episodes = config.get("policy_evaluator_params", {}).get(
        "n_episodes", 10
    )

    # Init evaluation
    print("Running initial policy evaluation...")
    eval_results = evaluate_policy(
        (reset_fn, step_fn, get_obs_fn),
        policy,
        policy_train_state.params,
        eval_key,
        config["policy_evaluator_params"]["n_episodes"],
        dyn_params=None,
        config=config,
    )

    eval_log = {}
    for agent_idx in range(config["num_agents"]):
        agent_key = f"agent_{agent_idx}"
        if f"{agent_key}/mean_return" in eval_results:
            eval_log[f"eval/{agent_key}/mean_return"] = eval_results[
                f"{agent_key}/mean_return"
            ]
            eval_log[f"eval/{agent_key}/std_return"] = eval_results[
                f"{agent_key}/std_return"
            ]

    wandb.log(eval_log, step=0)
    print(
        f"Initial Evaluation | Mean Return: {eval_results['mean_return']:.2f} +/- {eval_results['std_return']:.2f}"
    )

    key, reset_key = jax.random.split(key)
    state = reset_fn(reset_key)

    current_obs = get_obs_fn(state)  # Already replicated for all agents

    # Initialize IPPO buffer with JAX arrays
    num_agents = config["num_agents"]
    buffers = init_jax_buffers(
        config["num_agents"],
        config["train_policy_freq"],
        config["dim_state"],
        config["dim_action"],
    )
    buffer_idx = 0

    print(
        f"Starting online training for {config['total_steps']} steps "
        f"with train_policy_freq = {config['train_policy_freq']}..."
    )

    episode_length = 0

    # === 1. MODIFIED: Initialize reward component accumulators ===
    # These keys must match the `info` dict from the environment
    # reward_component_keys_to_avg = [
    #     "reward_blocker_avg",
    #     "reward_seeker",
    #     "shaping_reward_blocker",
    #     "shaping_reward_seeker",
    #     "terminal_reward_blocker",
    #     "terminal_reward_seeker",
    #     "oob_penalty_blocker_avg",
    #     "oob_penalty_seeker",
    # ]
    reward_component_keys_to_avg = [
        "reward_pursuer_avg",
        "reward_evader",
    ]
    episode_reward_components = {
        info_key: 0.0 for info_key in reward_component_keys_to_avg
    }
    # Add collision separately to sum it, not average it
    episode_reward_components["collision_sum"] = 0.0
    # =======================================================

    rollout_start_time = time.time()
    time_policy_inference = 0.0
    time_env_step = 0.0
    time_buffering = 0.0
    time_reward_norm = 0.0
    time_env_reset = 0.0
    time_wandb_log = 0.0
    time_train_prep = 0.0

    for step in range(1, config["total_steps"] + 1):

        key, action_key, reset_key = jax.random.split(key, 3)

        # Select action
        t_start_policy = time.time()
        dyn_params = None

        agent_action_keys = jax.random.split(action_key, num_agents)

        # Use current_obs which is already replicated for all agents
        actions, values, log_pis = jitted_policy_step(
            policy_train_state.params,
            current_obs,
            dyn_params,
            agent_action_keys,
        )
        time_policy_inference += time.time() - t_start_policy

        # Execute action in environment
        t_start_env = time.time()
        state, next_obs, rewards, terminated, truncated, info = step_fn(
            state, episode_length, actions
        )
        time_env_step += time.time() - t_start_env

        # === 2. Accumulate reward components (No change needed) ===
        for info_key in reward_component_keys_to_avg:
            episode_reward_components[info_key] += info[info_key]
        episode_reward_components["collision_sum"] += jnp.where(
            info["collision"], 1.0, 0.0
        )
        # ==========================================================

        done = terminated or truncated

        episode_length += 1

        t_start_reward_norm = time.time()

        # Update normalizer parameters with new rewards
        reward_norm_params = update_rolling_return_stats(
            reward_norm_params, rewards
        )

        # Normalize rewards
        scaled_rewards = reward_normalizer.normalize(
            reward_norm_params, rewards
        )
        time_reward_norm += time.time() - t_start_reward_norm

        # Add data to the IPPO buffer
        t_start_buffer = time.time()
        buffers = update_buffer_dynamic(
            buffers,
            buffer_idx,
            current_obs,
            actions,
            scaled_rewards,
            log_pis,
            values,
            float(done),
        )
        buffer_idx += 1
        time_buffering += time.time() - t_start_buffer

        current_obs = next_obs

        if done:
            # --- START ENV RESET TIMING ---
            t_start_env_reset = time.time()

            state = reset_fn(reset_key)  # Use the key from the top of the loop

            current_obs = get_obs_fn(state)
            time_env_reset += time.time() - t_start_env_reset

            # --- START WANDB LOG TIMING ---
            t_start_wandb_log = time.time()
            # Log per-agent episode metrics
            episode_log = {"episode/length": episode_length}
            for agent_idx in range(config["num_agents"]):
                episode_log[f"episode/agent_{agent_idx}_return"] = float(
                    reward_norm_params["rolling_return"][agent_idx]
                )
                episode_log[f"episode/agent_{agent_idx}_rolling_mean"] = float(
                    reward_norm_params["mean"][agent_idx]
                )
                episode_log[f"episode/agent_{agent_idx}_rolling_var"] = float(
                    reward_norm_params["var"][agent_idx]
                )

            # === 3. Log average reward components and reset (No change needed) ===
            if episode_length > 0:  # Avoid division by zero
                for info_key in reward_component_keys_to_avg:
                    avg_val = (
                        episode_reward_components[info_key] / episode_length
                    )
                    episode_log[f"rewards/{info_key}"] = float(avg_val)

            # Log total collisions (sum, not average)
            episode_log["rewards/total_collisions"] = float(
                episode_reward_components["collision_sum"]
            )

            wandb.log(episode_log, step=step)
            time_wandb_log += time.time() - t_start_wandb_log

            # Reset rolling return for new episode
            reward_norm_params = reset_rolling_return(reward_norm_params)
            episode_length = 0

            # Reset accumulators
            for info_key in episode_reward_components:
                episode_reward_components[info_key] = 0.0
            # =======================================================

        # Train policy
        if step % config["train_policy_freq"] == 0:
            rollout_end_time = time.time()

            t_start_train_prep = time.time()
            # Use current_obs which is already replicated for all agents
            last_values = jax.vmap(
                policy.evaluate_value, in_axes=(0, 0, None)
            )(policy_train_state.params, current_obs, dyn_params)

            # Slice buffers to get valid data (up to buffer_idx)
            policy_train_data = {
                "states": buffers["states"][:, :buffer_idx, :],
                "actions": buffers["actions"][:, :buffer_idx, :],
                "rewards": buffers["rewards"][:, :buffer_idx],
                "dones": buffers["dones"][:buffer_idx],
                "log_pis_old": buffers["log_pis_old"][:, :buffer_idx],
                "values_old": buffers["values_old"][:, :buffer_idx],
                "last_value": last_values,
            }
            time_train_prep += time.time() - t_start_train_prep

            total_rollout_time = rollout_end_time - rollout_start_time

            time_other = (
                total_rollout_time
                - time_policy_inference
                - time_env_step
                - time_buffering
                - time_reward_norm
                - time_env_reset
                - time_wandb_log
                - time_train_prep
            )

            print(
                f"\n[DEBUG] Rollout ({config['train_policy_freq']} steps) took: {total_rollout_time:.4f}s"
            )
            print(f"    -> Policy Inference: {time_policy_inference:.4f}s")
            print(f"    -> Env Step:         {time_env_step:.4f}s")
            print(f"    -> Buffering:        {time_buffering:.4f}s")
            # --- ADDED PRINTS ---
            print(f"    -> Reward Norm:      {time_reward_norm:.4f}s")
            print(f"    -> Env Resets:       {time_env_reset:.4f}s")
            print(f"    -> Wandb Logging:    {time_wandb_log:.4f}s")
            print(f"    -> Train Prep:       {time_train_prep:.4f}s")
            print(f"    -> Other (JAX keys, etc): {time_other:.4f}s")
            # --------------------

            train_start_time = time.time()

            policy_train_state, policy_metrics = policy_trainer.train(
                policy_train_state, policy_train_data
            )

            train_end_time = time.time()
            print(
                f"[DEBUG] Training ({config['policy_trainer_params']['n_epochs']} epochs) took: {train_end_time - train_start_time:.4f}s"
            )

            rollout_start_time = time.time()
            time_policy_inference = 0.0
            time_env_step = 0.0
            time_buffering = 0.0
            time_reward_norm = 0.0
            time_env_reset = 0.0
            time_wandb_log = 0.0
            time_train_prep = 0.0

            train_log = {
                "policy/policy_loss": policy_metrics.get("policy_loss", 0.0),
                "policy/value_loss": policy_metrics.get("value_loss", 0.0),
                "policy/entropy": policy_metrics.get("entropy", 0.0),
                "policy/mean_value_target": policy_metrics.get(
                    "mean_value_target", 0.0
                ),
            }
            for agent_idx in range(config["num_agents"]):
                agent_key = f"agent_{agent_idx}"
                if f"{agent_key}/policy_loss" in policy_metrics:
                    train_log[f"policy/{agent_key}/policy_loss"] = (
                        policy_metrics[f"{agent_key}/policy_loss"]
                    )
                    train_log[f"policy/{agent_key}/value_loss"] = (
                        policy_metrics[f"{agent_key}/value_loss"]
                    )
                    train_log[f"policy/{agent_key}/entropy"] = policy_metrics[
                        f"{agent_key}/entropy"
                    ]

            wandb.log(train_log, step=step)

            print(
                f"Step {step} | Policy Loss: {policy_metrics.get('policy_loss', 0.0):.4f} | "
                f"Value Loss: {policy_metrics.get('value_loss', 0.0):.4f} | "
                f"Entropy: {policy_metrics.get('entropy', 0.0):.4f}"
            )

            # Policy Evaluation
            eval_results = evaluate_policy(
                (reset_fn, step_fn, get_obs_fn),
                policy,
                policy_train_state.params,
                eval_key,
                n_eval_episodes,
                dyn_params=None,
                config=config,
            )

            eval_log = {}
            for agent_idx in range(config["num_agents"]):
                agent_key = f"agent_{agent_idx}"
                if f"{agent_key}/mean_return" in eval_results:
                    eval_log[f"eval/{agent_key}/mean_return"] = eval_results[
                        f"{agent_key}/mean_return"
                    ]
                    eval_log[f"eval/{agent_key}/std_return"] = eval_results[
                        f"{agent_key}/std_return"
                    ]

            wandb.log(eval_log, step=step)
            print(
                f"Evaluation | Mean Return: {eval_results['mean_return']:.2f} +/- {eval_results['std_return']:.2f}"
            )

            # Reset buffers
            buffers = init_jax_buffers(
                config["num_agents"],
                config["train_policy_freq"],
                config["dim_state"],
                config["dim_action"],
            )
            buffer_idx = 0

    if save_dir:
        run_name = config.get(
            "wandb_run_name", f"ippo_tracking_seed_{config['seed']}"
        )
        save_path = os.path.join(save_dir, run_name)

        os.makedirs(save_path, exist_ok=True)

        print(f"\nSaving final policy parameters to {save_path}...")

        target_params_np = jax.device_get(policy_train_state.params)

        file_path = os.path.join(save_path, "policy_params.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(target_params_np, f)

        print(f"Policy parameters saved to {file_path}")

    wandb.finish()
    print("Run complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run IPPO"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Custom name for the W&B run.",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=1,
        help="Number of random seeds to run.",
    )
    parser.add_argument(
        "--meta-seed",
        type=int,
        default=42,
        help="A seed to generate the run seeds.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./trained_policies",
        help="Directory to save the trained policy parameters.",
    )
    args = parser.parse_args()

    if args.meta_seed is not None:
        rng = np.random.default_rng(args.meta_seed)
        seeds = rng.integers(low=0, high=2**32 - 1, size=args.num_seeds)
    else:
        seeds = range(args.num_seeds)

    for seed_idx, seed in enumerate(seeds):
        print(f"--- Starting run for seed #{seed} ---")
        run_config = copy.deepcopy(CONFIG)
        run_config["seed"] = int(seed)
        run_config["wandb_group"] = "ippo"

        if args.run_name:
            run_name_base = args.run_name
        else:
            run_name_base = "ippo"

        if args.num_seeds > 1:
            run_config["wandb_run_name"] = f"{run_name_base}_seed_{seed_idx}"
        else:
            run_config["wandb_run_name"] = run_name_base

        main(run_config, save_dir=args.save_dir)

    print("All experiments complete.")