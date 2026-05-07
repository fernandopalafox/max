# eval_geometry.py
"""
Evaluate LoRA-XS adapter geometry from a pretrained checkpoint.

Computes two metrics per checkpoint (averaged across adapted layers):
  diag_mean_AtA   : mean diagonal of A^T A  (target: 1.0)
  offdiag_rms_AtA : RMS off-diagonal of A^T A  (target: 0.0)
  diag_mean_Chat  : mean diagonal of C_hat = Bh^T Bh / N  (target: 1.0)
  offdiag_rms_Chat: RMS off-diagonal of C_hat  (target: 0.0)

C_hat is estimated using latent states from MPPI-planned rollouts with the
checkpoint's own policy/critic/encoder/reward — same setup as training eval —
so the observation distribution matches what the model actually sees.

Appends one row to a CSV results table.

Usage:
  python scripts/eval_geometry.py \\
      --trainer_type loraxs_regularized \\
      --run_name run_1 \\
      --checkpoint_path data/models/cheetah/baseline_loraxs_regularized/run_1/TIMESTAMP/final.pkl \\
      --csv results/geometry.csv
"""

import os
import sys
import csv
import argparse
import json
import pickle
import copy

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.critics import init_critic
from max.policies import init_policy
from max.rewards import init_reward_model
from max.environments import init_env
from max.planners import init_planner

# Map trainer_type to its pretraining config file
CONFIGS = {
    "loraxs":              "configs/pretrain_loraxs_tdmpc2.json",
    "loraxs_regularized":  "configs/pretrain_loraxs_regularized_tdmpc2.json",
    "loraxs_reg_Cfix":     "configs/pretrain_loraxs_reg_Cfix_tdmpc2.json",
    "loraxs_init":         "configs/pretrain_loraxs_tdmpc2.json",  # SVD init from dense checkpoint
}


def _matrix_metrics(M, rank):
    """Return (diag_mean, offdiag_rms) for a (rank, rank) matrix."""
    diag = jnp.diag(M)
    offdiag = M * (1.0 - jnp.eye(rank))
    return float(jnp.mean(diag)), float(jnp.sqrt(jnp.mean(offdiag ** 2)))


def _normalize_adapter_keys(adapter):
    """Rename old P_i/Q_i keys to new A_i/B_i convention if needed."""
    if any(k.startswith("P_") for k in adapter):
        renamed = {}
        for k, v in adapter.items():
            if k.startswith("P_"):
                renamed["B_" + k[2:]] = v
            elif k.startswith("Q_"):
                renamed["A_" + k[2:]] = v
            else:
                renamed[k] = v
        return renamed
    return adapter


def _collect_planner_data(config, full_params, encoder, dynamics, reward, critic, policy, n_samples, key):
    """
    Run MPPI-guided rollouts and return (latents, actions) pairs.
    Mirrors the training evaluator exactly: uses evaluator config for env/planner setup.
    """
    evaluator_cfg = config.get("evaluator", {})
    max_steps = evaluator_cfg.get("max_steps", config["environment"]["max_episode_steps"])

    # Build env config with evaluator overrides (but we always use training env here)
    reset_fn, step_fn, get_obs_fn = init_env(config)

    # Build planner config with evaluator overrides
    planner_config = {**config}
    planner_overrides = evaluator_cfg.get("planner", {})
    if planner_overrides:
        planner_config["planner"] = {**config["planner"], **planner_overrides}

    key, planner_key = jax.random.split(key)
    planner, init_planner_state = init_planner(
        planner_config, key=planner_key,
        encoder=encoder, dynamics=dynamics, reward=reward, critic=critic, policy=policy,
    )

    encode_single = jax.jit(
        lambda obs: encoder.encode(full_params["mean"]["encoder"], obs)
    )

    latents = []
    actions_list = []

    while len(latents) < n_samples:
        key, reset_key, pk = jax.random.split(key, 3)
        env_state = reset_fn(reset_key)
        planner_state = init_planner_state.replace(key=pk)

        for _ in range(max_steps):
            if len(latents) >= n_samples:
                break
            obs = get_obs_fn(env_state).squeeze(0)
            z = encode_single(obs)
            planned_actions, planner_state = planner.solve(planner_state, obs, full_params)
            a = planned_actions[0]
            latents.append(z)
            actions_list.append(a)
            env_state, _, _, terminated, truncated, _ = step_fn(env_state, len(latents), a[None, :])
            if terminated or truncated:
                break

    return (
        jnp.stack(latents[:n_samples]),   # (n_samples, latent_dim)
        jnp.stack(actions_list[:n_samples]),  # (n_samples, dim_action)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_type", required=True,
                        choices=["loraxs", "loraxs_regularized", "loraxs_reg_Cfix", "loraxs_init"])
    parser.add_argument("--run_name", required=True, help="Label for this run, e.g. run_1")
    parser.add_argument("--checkpoint_path", required=True, help="Path to final.pkl")
    parser.add_argument("--csv", default="results/geometry.csv",
                        help="CSV file to append results to")
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="Number of (z, a) pairs for C_hat estimation")
    args = parser.parse_args()

    is_init = args.trainer_type == "loraxs_init"

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(repo_root, CONFIGS[args.trainer_type])
    with open(config_path) as f:
        config = json.load(f)["training"]

    adapt_layers = config["dynamics"]["adapt_layers"]
    rank         = config["dynamics"]["rank"]

    with open(args.checkpoint_path, "rb") as f:
        ckpt = pickle.load(f)

    key = jax.random.key(42)
    key, enc_key, dyn_key, critic_key, policy_key = jax.random.split(key, 5)

    if is_init:
        # Dense checkpoint: SVD-initialize adapter, keep dense policy/encoder/critic/reward
        dense_mean = ckpt["mean"]
        encoder,      enc_params    = init_encoder(enc_key, config,    pretrained=dense_mean.get("encoder"))
        dynamics,     dyn_params    = init_dynamics(dyn_key, config,   pretrained=dense_mean.get("dynamics"))
        critic,       critic_params = init_critic(critic_key, config,  pretrained=dense_mean.get("critic"))
        policy,       policy_params = init_policy(policy_key, config,  pretrained=dense_mean.get("policy"))
        reward_model, reward_params = init_reward_model(config,        pretrained=dense_mean.get("reward"))

        # A^T A metrics from SVD-initialized adapter
        diag_AtA_per_layer, offdiag_AtA_per_layer = [], []
        for i in adapt_layers:
            W = jnp.array(dense_mean["dynamics"]["params"][f"Dense_{i}"]["kernel"])
            U, S, Vh = jnp.linalg.svd(W, full_matrices=False)
            A_i = Vh[:rank, :].T  # (d_out, rank)
            d, o = _matrix_metrics(A_i.T @ A_i, rank)
            diag_AtA_per_layer.append(d)
            offdiag_AtA_per_layer.append(o)
    else:
        mean = ckpt["mean"]
        mean["dynamics"]["adapter"] = _normalize_adapter_keys(mean["dynamics"]["adapter"])

        encoder,      enc_params    = init_encoder(enc_key, config,    pretrained=mean.get("encoder"))
        dynamics,     dyn_params    = init_dynamics(dyn_key, config,   pretrained=mean.get("dynamics"))
        critic,       critic_params = init_critic(critic_key, config,  pretrained=mean.get("critic"))
        policy,       policy_params = init_policy(policy_key, config,  pretrained=mean.get("policy"))
        reward_model, reward_params = init_reward_model(config,        pretrained=mean.get("reward"))

        diag_AtA_per_layer, offdiag_AtA_per_layer = [], []
        for i in adapt_layers:
            A_i = jnp.array(mean["dynamics"]["adapter"][f"A_{i}"])  # (d_out, rank)
            d, o = _matrix_metrics(A_i.T @ A_i, rank)
            diag_AtA_per_layer.append(d)
            offdiag_AtA_per_layer.append(o)

    diag_mean_AtA   = float(jnp.mean(jnp.array(diag_AtA_per_layer)))
    offdiag_rms_AtA = float(jnp.mean(jnp.array(offdiag_AtA_per_layer)))

    # Build full params dict for the planner (same structure as train.py)
    full_params = {
        "mean": {
            "encoder":    enc_params,
            "dynamics":   dyn_params,
            "reward":     reward_params,
            "critic":     critic_params,
            "ema_critic": copy.deepcopy(critic_params),
            "policy":     policy_params,
        },
        "normalizer": ckpt.get("normalizer", {
            "q_scale": jnp.array(config["normalizer"]["critic"]["q_scale_init"], dtype=jnp.float32)
        }),
    }

    key, rollout_key = jax.random.split(key)
    z_data, a_data = _collect_planner_data(
        config, full_params, encoder, dynamics, reward_model, critic, policy,
        args.n_samples, rollout_key,
    )

    # C_hat from bottleneck activations on the collected (z, a) pairs
    infer_batch = jax.jit(jax.vmap(dynamics.predict_with_activations, in_axes=(None, 0, 0)))
    _, bh_dict = infer_batch(dyn_params, z_data, a_data)

    diag_Chat_per_layer, offdiag_Chat_per_layer = [], []
    for i in adapt_layers:
        Bh = bh_dict[i]                         # (n_samples, rank)
        C_hat = Bh.T @ Bh / args.n_samples
        d, o = _matrix_metrics(C_hat, rank)
        diag_Chat_per_layer.append(d)
        offdiag_Chat_per_layer.append(o)

    diag_mean_Chat   = float(jnp.mean(jnp.array(diag_Chat_per_layer)))
    offdiag_rms_Chat = float(jnp.mean(jnp.array(offdiag_Chat_per_layer)))

    # Append row to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
    write_header = not os.path.exists(args.csv)
    with open(args.csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "trainer_type", "run_name",
                "diag_mean_AtA", "offdiag_rms_AtA",
                "diag_mean_Chat", "offdiag_rms_Chat",
            ])
        writer.writerow([
            args.trainer_type, args.run_name,
            f"{diag_mean_AtA:.4f}",  f"{offdiag_rms_AtA:.4f}",
            f"{diag_mean_Chat:.4f}", f"{offdiag_rms_Chat:.4f}",
        ])

    print(f"trainer_type={args.trainer_type}  run_name={args.run_name}")
    print(f"  A^T A  : diag_mean={diag_mean_AtA:.4f}  offdiag_rms={offdiag_rms_AtA:.4f}")
    print(f"  C_hat  : diag_mean={diag_mean_Chat:.4f}  offdiag_rms={offdiag_rms_Chat:.4f}")
    print(f"Row appended to {args.csv}")


if __name__ == "__main__":
    main()
