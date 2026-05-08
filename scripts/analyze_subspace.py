# analyze_subspace.py
# Compute subspace alignment between W_final, delta_W, G_sum_unscaled, G_sum_scaled
# for each dynamics Dense layer, aggregated across seeds.
#
# Usage:
#   python scripts/analyze_subspace.py \
#       --dirs data/models/cheetah/grad_capture/run_1/YYYYMMDD_HHMMSS \
#              data/models/cheetah/grad_capture/run_2/YYYYMMDD_HHMMSS \
#              ... \
#       --rank 16

import argparse
import os
import pickle
import numpy as np
from glob import glob


def subspace_alignment(U1, U2):
    """Mean cosine of principal angles between column spaces of U1 and U2.

    Both U1, U2 must have orthonormal columns (output of SVD).
    Returns a scalar in [0, 1]: 1.0 = perfect alignment.
    """
    M = U1.T @ U2
    sigma = np.linalg.svd(M, compute_uv=False)
    return float(np.mean(np.clip(sigma, 0.0, 1.0)))


def top_r_left_svecs(M, r):
    """Top-r left singular vectors of matrix M (m x n)."""
    U, _, _ = np.linalg.svd(M, full_matrices=False)
    return U[:, :r]


def analyze_seed(pkl_path, r, wfinal_pkl_path=None):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if wfinal_pkl_path is not None:
        with open(wfinal_pkl_path, "rb") as f:
            wfinal_data = pickle.load(f)
        data["w_final"] = wfinal_data["w_final"]

    results = {}
    layer_names = sorted(data["w_initial"].keys())

    for name in layer_names:
        W_init = data["w_initial"][name].astype(np.float64)
        W_final = data["w_final"][name].astype(np.float64)
        G_unscaled = data["g_sum_unscaled"][name].astype(np.float64)
        G_scaled = data["g_sum_scaled"][name].astype(np.float64)
        delta_W = W_final - W_init

        r_eff = min(r, W_final.shape[0], W_final.shape[1])

        U_Wf = top_r_left_svecs(W_final, r_eff)
        U_dW = top_r_left_svecs(delta_W, r_eff)
        U_Gu = top_r_left_svecs(G_unscaled, r_eff)
        U_Gs = top_r_left_svecs(G_scaled, r_eff)

        results[name] = {
            "Wf_vs_Gu": subspace_alignment(U_Wf, U_Gu),   # tests Observation 3.1
            "dW_vs_Gu": subspace_alignment(U_dW, U_Gu),   # raw gradient predicts trajectory
            "dW_vs_Gs": subspace_alignment(U_dW, U_Gs),   # optimizer scaling impact
            "Wf_vs_dW": subspace_alignment(U_Wf, U_dW),   # initial weight noise
        }
    return results, layer_names


def find_grad_capture_pkl(run_dir):
    """Find grad_capture.pkl under run_dir (looks one level deep for timestamped subdirs)."""
    direct = os.path.join(run_dir, "grad_capture.pkl")
    if os.path.exists(direct):
        return direct
    # Look in timestamped subdirectories
    candidates = sorted(glob(os.path.join(run_dir, "*", "grad_capture.pkl")))
    if candidates:
        return candidates[-1]  # take most recent
    raise FileNotFoundError(f"No grad_capture.pkl found under {run_dir}")


def main():
    parser = argparse.ArgumentParser(description="Subspace alignment analysis for dynamics MLP.")
    parser.add_argument(
        "--dirs", nargs="+", required=True,
        help="Run directories (each containing grad_capture.pkl or a timestamped subdir)."
    )
    parser.add_argument("--rank", type=int, default=16, help="Number of singular vectors (r).")
    parser.add_argument(
        "--wfinal-dirs", nargs="+", default=None,
        help="Dirs with grad_capture.pkl to source w_final from (matched by index to --dirs). "
             "Use when gradient data and final weights come from different runs.",
    )
    args = parser.parse_args()

    if args.wfinal_dirs is not None and len(args.wfinal_dirs) != len(args.dirs):
        raise ValueError("--wfinal-dirs must have the same number of entries as --dirs")

    all_results = []
    layer_names = None

    for i, run_dir in enumerate(args.dirs):
        try:
            pkl_path = find_grad_capture_pkl(run_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

        wfinal_pkl_path = None
        if args.wfinal_dirs is not None:
            wfinal_pkl_path = find_grad_capture_pkl(args.wfinal_dirs[i])

        results, names = analyze_seed(pkl_path, args.rank, wfinal_pkl_path=wfinal_pkl_path)
        all_results.append(results)
        if layer_names is None:
            layer_names = names
        print(f"  Analyzed: {pkl_path}")

    if not all_results:
        print("No data found.")
        return

    pairs = ["Wf_vs_Gu", "dW_vs_Gu", "dW_vs_Gs", "Wf_vs_dW"]
    pair_labels = {
        "Wf_vs_Gu": "U_Wfinal vs U_Gunscaled  (Obs 3.1)",
        "dW_vs_Gu": "U_deltaW vs U_Gunscaled  (raw grad predicts path)",
        "dW_vs_Gs": "U_deltaW vs U_Gscaled    (optimizer impact)",
        "Wf_vs_dW": "U_Wfinal vs U_deltaW     (init noise)",
    }

    n_seeds = len(all_results)
    print(f"\nSubspace Alignment Results (r={args.rank}, n_seeds={n_seeds})\n")
    print(f"Metric: mean cosine of principal angles in [0, 1] (1.0 = perfect alignment)\n")

    for pair in pairs:
        print(f"### {pair_labels[pair]}\n")
        header = "| Layer    | " + " | ".join(f"seed {i+1}" for i in range(n_seeds)) + " | mean ± std |"
        sep = "|" + "|".join(["-" * (len(h) + 2) for h in ["Layer    "] + [f"seed {i+1}" for i in range(n_seeds)] + ["mean ± std"]]) + "|"
        print(header)
        print(sep)

        for layer in layer_names:
            vals = [r[layer][pair] for r in all_results]
            mean_val = np.mean(vals)
            std_val = np.std(vals)
            seed_cols = " | ".join(f"{v:.4f}" for v in vals)
            print(f"| {layer:8s} | {seed_cols} | {mean_val:.4f} ± {std_val:.4f} |")
        print()

    # Also print a compact summary table
    print("### Summary (mean ± std across seeds)\n")
    col_w = max(len(p) for p in pair_labels.values()) + 2
    header_parts = ["Layer    "] + list(pair_labels.values())
    print("| Layer    | " + " | ".join(pair_labels.values()) + " |")
    print("|" + "|".join(["-" * 10] + ["-" * (len(v) + 2) for v in pair_labels.values()]) + "|")

    for layer in layer_names:
        row_parts = [f"| {layer:8s}"]
        for pair in pairs:
            vals = [r[layer][pair] for r in all_results]
            row_parts.append(f" {np.mean(vals):.4f} ± {np.std(vals):.4f} ")
        print("|".join(row_parts) + "|")

    print()


if __name__ == "__main__":
    main()
