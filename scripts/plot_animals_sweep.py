#!/usr/bin/env python3
"""
Plot final belief error and covariance trace vs info_weight from run_animals.py log files.

Supports overlaying multiple sweeps with different labels:
    python scripts/plot_animals_sweep.py --sweep "label:/tmp/animals_iw*.log" --sweep "label2:/tmp/animals2_iw*.log"

Or single sweep (original usage):
    python scripts/plot_animals_sweep.py /tmp/animals_iw*.log
"""

import re
import sys
import glob
import argparse
import matplotlib.pyplot as plt


TRUE_Q = 5.0
TRUE_K = 1.0
KEEP = {0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50}
COLORS = ["steelblue", "tomato", "seagreen", "darkorange"]


def parse_final_step(path):
    q_hat = K_hat = cov_trace = None
    with open(path) as f:
        for line in f:
            m = re.search(r"q̂=(-?[\d.]+)\s+K̂=(-?[\d.]+)\s+cov=([\d.]+)", line)
            if m:
                q_hat = float(m.group(1))
                K_hat = float(m.group(2))
                cov_trace = float(m.group(3))
    return q_hat, K_hat, cov_trace


def parse_iw_from_path(path):
    m = re.search(r"iw([\d.]+?)(?:\.log|$)", path)
    return float(m.group(1)) if m else None


def load_sweep(paths):
    results = []
    for path in sorted(paths, key=parse_iw_from_path):
        iw = parse_iw_from_path(path)
        q_hat, K_hat, cov_trace = parse_final_step(path)
        if iw is not None and q_hat is not None and iw in KEEP:
            results.append((iw, q_hat, K_hat, cov_trace))
    return results


def plot_sweep(axes, results, label, color):
    ax1, ax2, ax3 = axes
    iws = [r[0] for r in results]
    q_errs = [abs(r[1] - TRUE_Q) for r in results]
    K_errs = [abs(r[2] - TRUE_K) for r in results]
    cov_traces = [r[3] for r in results]
    ax1.plot(iws, q_errs, "o-", color=color, label=label)
    ax2.plot(iws, K_errs, "o-", color=color, label=label)
    ax3.plot(iws, cov_traces, "o-", color=color, label=label)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="append", metavar="LABEL:GLOB",
                        help="label:glob pair, can be repeated for multiple sweeps")
    parser.add_argument("logs", nargs="*", help="log files for single unlabelled sweep")
    args = parser.parse_args()

    sweeps = []
    if args.sweep:
        for s in args.sweep:
            label, pattern = s.split(":", 1)
            paths = glob.glob(pattern)
            sweeps.append((label, paths))
    elif args.logs:
        sweeps.append(("", args.logs))
    else:
        print("No input files provided.")
        sys.exit(1)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    all_iws = []
    for i, (label, paths) in enumerate(sweeps):
        results = load_sweep(paths)
        if not results:
            print(f"No valid results for sweep '{label}'")
            continue
        print(f"\n--- {label or 'sweep'} ---")
        for r in results:
            print(f"  iw={r[0]:5.1f}  q̂={r[1]:.3f}  K̂={r[2]:.3f}  cov={r[3]:.4f}")
        all_iws.extend(r[0] for r in results)
        plot_sweep(axes, results, label, COLORS[i % len(COLORS)])

    max_iw = max(all_iws)
    xticks = list(range(0, int(max_iw) + 1, 10))
    show_legend = len(sweeps) > 1 or sweeps[0][0]

    for ax, ylabel, title in [
        (ax1, "|q̂ − q|", "Belief error: opponent quality q"),
        (ax2, "|K̂ − K|", "Belief error: opponent reactivity K"),
        (ax3, "tr(P)", "Final covariance trace"),
    ]:
        ax.set_xlabel("Info-gathering weight λ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(left=0, right=max_iw)
        ax.set_xticks(xticks)
        ax.set_ylim(bottom=0)
        if show_legend:
            ax.legend()

    fig.tight_layout()
    out = "logs/animals/belief_sweep_h_comparison.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
