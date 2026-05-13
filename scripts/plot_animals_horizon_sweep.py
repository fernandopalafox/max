#!/usr/bin/env python3
"""
Plot final belief error and covariance trace vs planning horizon.

Usage:
    python scripts/plot_animals_horizon_sweep.py /tmp/animals_h*.log
"""

import re
import sys
import matplotlib.pyplot as plt


TRUE_Q = 5.0
TRUE_K = 1.0


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


def parse_horizon_from_path(path):
    m = re.search(r"h([\d]+?)(?:\.log|$)", path)
    return int(m.group(1)) if m else None


def main(log_paths):
    results = []
    for path in sorted(log_paths, key=parse_horizon_from_path):
        h = parse_horizon_from_path(path)
        q_hat, K_hat, cov_trace = parse_final_step(path)
        if h is not None and q_hat is not None:
            results.append((h, q_hat, K_hat, cov_trace))
            print(f"h={h:3d}  q̂={q_hat:.3f}  K̂={K_hat:.3f}  cov={cov_trace:.4f}")

    if not results:
        print("No valid log files found.")
        return

    hs = [r[0] for r in results]
    q_errs = [abs(r[1] - TRUE_Q) for r in results]
    K_errs = [abs(r[2] - TRUE_K) for r in results]
    cov_traces = [r[3] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    ax1, ax2, ax3 = axes

    xticks = list(range(0, max(hs) + 1, 5))

    for ax, vals, ylabel, title in [
        (ax1, q_errs, "|q̂ − q|", "Belief error: opponent quality q"),
        (ax2, K_errs, "|K̂ − K|", "Belief error: opponent reactivity K"),
        (ax3, cov_traces, "tr(P)", "Final covariance trace"),
    ]:
        ax.plot(hs, vals, "o-", color="steelblue")
        ax.set_xlabel("Planning horizon")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xlim(left=0, right=max(hs) + 1)
        ax.set_xticks(xticks)
        ax.set_ylim(bottom=0)

    fig.tight_layout()
    out = "logs/animals/horizon_sweep.png"
    fig.savefig(out, dpi=150)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
