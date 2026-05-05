# eval_geometry.py
"""
Evaluate LoRA-XS adapter geometry from a pretrained checkpoint.

Computes two metrics per checkpoint (averaged across adapted layers):
  diag_mean_AtA   : mean diagonal of A^T A  (target: 1.0)
  offdiag_rms_AtA : RMS off-diagonal of A^T A  (target: 0.0)
  diag_mean_Chat  : mean diagonal of C_hat = Z^T Z / N  (target: 1.0)
  offdiag_rms_Chat: RMS off-diagonal of C_hat  (target: 0.0)

C_hat is estimated using latent states from real environment rollouts with the
saved encoder and policy, so the observation distribution matches training.

Appends one row to a CSV results table.

Usage:
  python scripts/eval_geometry.py \\
      --trainer_type loraxs_regularized \\
      --run_name run_1 \\
      --checkpoint_path data/models/cheetah/baseline_loraxs_regularized/run_1 \\
      --csv results/geometry.csv
"""

import os
import sys
import csv
import argparse
import json
import pickle

import jax
import jax.numpy as jnp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from max.dynamics import init_dynamics
from max.encoders import init_encoder
from max.environments import init_env

# Map trainer_type to its config file
CONFIGS = {
    "dense":               "configs/pretrain_dense_tdmpc2.json",
    "loraxs":              "configs/pretrain_loraxs_tdmpc2.json",
    "loraxs_regularized":  "configs/pretrain_loraxs_regularized_tdmpc2.json",
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
                renamed["B_" + k[2:]] = v   # P_i (d_in, rank)  -> B_i
            elif k.startswith("Q_"):
                renamed["A_" + k[2:]] = v   # Q_i (d_out, rank) -> A_i
            else:
                renamed[k] = v
        return renamed
    return adapter


def _collect_latents(config, enc_params, n_samples, key):
    """Run random-action rollouts and return encoded latent states."""
    reset_fn, step_fn, get_obs_fn = init_env(config)
    encoder, _ = init_encoder(key, config)
    encode_fn = jax.jit(lambda obs: encoder.encode(enc_params, obs.squeeze(0)))

    dim_action = config["dim_action"]
    latents = []

    key, rk = jax.random.split(key)
    env_state = reset_fn(rk)
    obs = get_obs_fn(env_state)

    while len(latents) < n_samples:
        z = encode_fn(obs)
        latents.append(z)

        key, ak, sk = jax.random.split(key, 3)
        action = jax.random.uniform(ak, (dim_action,), minval=-1.0, maxval=1.0)
        env_state, obs, _, terminated, truncated, _ = step_fn(env_state, len(latents), action)

        if terminated or truncated:
            key, rk = jax.random.split(key)
            env_state = reset_fn(rk)
            obs = get_obs_fn(env_state)

    return jnp.stack(latents[:n_samples])  # (n_samples, latent_dim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_type", required=True,
                        choices=["dense", "loraxs", "loraxs_regularized"])
    parser.add_argument("--run_name", required=True,
                        help="Label for this run, e.g. run_1")
    parser.add_argument("--checkpoint_path", required=True,
                        help="Directory containing final.pkl")
    parser.add_argument("--csv", default="results/geometry.csv",
                        help="CSV file to append results to")
    parser.add_argument("--n_samples", type=int, default=1024,
                        help="Environment samples for C_hat estimation")
    args = parser.parse_args()

    if args.trainer_type == "dense":
        print("dense trainer has no adapter matrices — nothing to evaluate.")
        return

    # Load config
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    config_path = os.path.join(repo_root, CONFIGS[args.trainer_type])
    with open(config_path) as f:
        config = json.load(f)["training"]

    adapt_layers = config["dynamics"]["adapt_layers"]
    rank         = config["dynamics"]["rank"]
    dim_action   = config["dim_action"]

    # Load checkpoint
    ckpt_file = os.path.join(args.checkpoint_path, "final.pkl")
    with open(ckpt_file, "rb") as f:
        params = pickle.load(f)

    dyn_params = params["mean"]["dynamics"]
    dyn_params["adapter"] = _normalize_adapter_keys(dyn_params["adapter"])

    # ---- A^T A metrics (direct from params, no data needed) ----
    diag_AtA_per_layer, offdiag_AtA_per_layer = [], []
    for i in adapt_layers:
        A_i = jnp.array(dyn_params["adapter"][f"A_{i}"])  # (d_out, rank)
        AtA = A_i.T @ A_i                                  # (rank, rank)
        d, o = _matrix_metrics(AtA, rank)
        diag_AtA_per_layer.append(d)
        offdiag_AtA_per_layer.append(o)

    diag_mean_AtA   = float(jnp.mean(jnp.array(diag_AtA_per_layer)))
    offdiag_rms_AtA = float(jnp.mean(jnp.array(offdiag_AtA_per_layer)))

    # ---- C_hat metrics (real env latents through predict_with_activations) ----
    key = jax.random.key(42)
    z_real = _collect_latents(config, params["mean"]["encoder"], args.n_samples, key)

    key, ka = jax.random.split(key)
    a_rand = jax.random.uniform(ka, (args.n_samples, dim_action), minval=-1.0, maxval=1.0)

    dynamics, _ = init_dynamics(key, config, pretrained=None)
    infer_batch = jax.jit(jax.vmap(dynamics.predict_with_activations, in_axes=(None, 0, 0)))
    _, bh_dict = infer_batch(dyn_params, z_real, a_rand)

    diag_Chat_per_layer, offdiag_Chat_per_layer = [], []
    for i in adapt_layers:
        Bh = bh_dict[i]                         # (n_samples, rank)
        C_hat = Bh.T @ Bh / args.n_samples      # (rank, rank)
        d, o = _matrix_metrics(C_hat, rank)
        diag_Chat_per_layer.append(d)
        offdiag_Chat_per_layer.append(o)

    diag_mean_Chat   = float(jnp.mean(jnp.array(diag_Chat_per_layer)))
    offdiag_rms_Chat = float(jnp.mean(jnp.array(offdiag_Chat_per_layer)))

    # ---- Append row to CSV ----
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
