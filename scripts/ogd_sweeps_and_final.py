#!/usr/bin/env python3
"""
OGD LR sweeps + final 5-seed comparison.

Phase 1: Sweep LR for all 4 OGD methods (each in own wandb project).
Phase 2: Update experiment_log.md Key Results table with best OGD LRs.
Phase 3: Run all 7 methods with 5 seeds in streaming-adaptation-final.
"""

import json
import re
import subprocess
import os
import numpy as np
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
LOG_PATH = REPO / "experiment_log.md"
SUMMARY_PATH = REPO / "ogd_runs_progress.md"  # live progress log; final results go to experiment_log.md

BASE_CONFIGS = {
    "ogd_lastlayer": REPO / "configs" / "stream_ogd.json",
    "ogd_dense":     REPO / "configs" / "stream_ogd_dense.json",
    "ogd_loraxs":    REPO / "configs" / "stream_ogd_lora.json",
    "ogd_tinylora":  REPO / "configs" / "stream_ogd_tiny_lora.json",
    "ekf_loraxs":    REPO / "configs" / "stream_ekf_lora_best_full.json",
    "ekf_tinylora":  REPO / "configs" / "stream_ekf_tiny_lora_best.json",
    "ekf_lastlayer": REPO / "configs" / "stream_ekf_lastlayer_best.json",
}

# Initial LR sweep range. Adam for dynamics adaptation: start low, up to 1e-2.
# If best is at the top of the range we extend upward by a decade until we find a peak.
OGD_INITIAL_LRS = [1e-5, 1e-4, 1e-3, 1e-2]
MAX_ADAM_LR = 1.0  # Adam rarely stable above this for neural net weights

summary = []


def log(msg):
    print(msg, flush=True)
    summary.append(msg)


def write_summary():
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")


def lr_run_name(lr):
    """Human-readable run name for a learning rate."""
    s = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")
    return f"lr{s}"


def run_experiment(config_dict, run_name, num_seeds):
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_dict, f, indent=2)
        tmp = f.name
    try:
        result = subprocess.run(
            ["conda", "run", "-n", "max", "python", str(REPO / "scripts" / "train.py"),
             "--config", tmp, "--run-name", run_name, "--num-seeds", str(num_seeds)],
            capture_output=True, text=True, cwd=REPO
        )
        out = result.stdout + result.stderr
        if result.returncode != 0:
            log(f"  ERROR {run_name}: exit {result.returncode}")
            log("\n".join(out.splitlines()[-20:]))
        return out
    finally:
        os.unlink(tmp)


def parse_rewards(output):
    rewards = []
    for line in output.splitlines():
        if "eval/episode_reward" in line:
            m = re.search(r"eval/episode_reward\s+([\d.]+)", line)
            if m:
                rewards.append(float(m.group(1)))
    return rewards


def fmt(rewards):
    if not rewards:
        return "NO RESULTS"
    a = np.array(rewards)
    return f"mean={a.mean():.1f} ±{a.std():.1f} (n={len(a)})"


def mean_or_neg(rewards):
    return np.mean(rewards) if rewards else -1.0


def sweep_lr(base, project, num_seeds=3):
    """
    Logarithmic LR sweep with upward extension.
    Starts with OGD_INITIAL_LRS. If the best LR is at the top of the tested
    range, extends upward by a decade until the peak is interior or we hit
    MAX_ADAM_LR. Returns (best_lr, sweep_results_dict).
    """
    sweep_results = {}

    for lr in OGD_INITIAL_LRS:
        cfg = json.loads(json.dumps(base))
        cfg["training"]["wandb_project"] = project
        cfg["training"]["trainer"]["lr"] = lr
        out = run_experiment(cfg, lr_run_name(lr), num_seeds=num_seeds)
        rewards = parse_rewards(out)
        sweep_results[lr] = rewards
        log(f"  lr={lr:.1e}: {fmt(rewards)}")

    # Extend upward if best is at the top boundary
    while True:
        best_lr = max(sweep_results, key=lambda k: mean_or_neg(sweep_results[k]))
        top_lr  = max(sweep_results.keys())
        if best_lr < top_lr or top_lr >= MAX_ADAM_LR:
            break  # peak is interior, or we've hit the Adam stability ceiling
        next_lr = top_lr * 10
        if next_lr > MAX_ADAM_LR:
            break
        log(f"  Best is at top (lr={best_lr:.1e}) — extending to lr={next_lr:.1e}")
        cfg = json.loads(json.dumps(base))
        cfg["training"]["wandb_project"] = project
        cfg["training"]["trainer"]["lr"] = next_lr
        out = run_experiment(cfg, lr_run_name(next_lr), num_seeds=num_seeds)
        rewards = parse_rewards(out)
        sweep_results[next_lr] = rewards
        log(f"  lr={next_lr:.1e}: {fmt(rewards)}")

    best_lr = max(sweep_results, key=lambda k: mean_or_neg(sweep_results[k]))
    return best_lr, sweep_results


# ─── Phase 1: OGD LR Sweeps ──────────────────────────────────────────────────

ogd_sweep_configs = {
    "ogd_lastlayer": "ogd-lastlayer-lr",
    "ogd_dense":     "ogd-dense-lr",
    "ogd_loraxs":    "ogd-loraxs-lr",
    "ogd_tinylora":  "ogd-tinylora-lr",
}

best_ogd_lrs = {}

for method_key, project in ogd_sweep_configs.items():
    log(f"\n## OGD LR Sweep: {method_key} → project={project}")
    with open(BASE_CONFIGS[method_key]) as f:
        base = json.load(f)

    best_lr, sweep_results = sweep_lr(base, project, num_seeds=3)
    best_ogd_lrs[method_key] = best_lr
    log(f"  → Best lr: {best_lr:.1e}  ({fmt(sweep_results[best_lr])})")
    write_summary()

log(f"\n## OGD Best LRs Summary")
for k, v in best_ogd_lrs.items():
    log(f"  {k}: {v:.0e}")
write_summary()


# ─── Phase 2: Update experiment_log.md ───────────────────────────────────────

log("\n## Updating experiment_log.md Key Results table...")

method_labels = {
    "ogd_lastlayer": "OGD last-layer",
    "ogd_dense":     "OGD full network",
    "ogd_loraxs":    "OGD LoRA-XS",
    "ogd_tinylora":  "OGD TinyLoRA",
}

updates = []
for key, label in method_labels.items():
    lr = best_ogd_lrs[key]
    updates.append(f"  {label}: best lr = {lr:.0e}")

log("\n".join(updates))

# Append update note to log
note = f"\n\n### OGD LR Sweep Results — {datetime.now().strftime('%Y-%m-%d')}\n"
note += "Best LRs found:\n"
for key, label in method_labels.items():
    note += f"- {label}: lr={best_ogd_lrs[key]:.0e}\n"
note += "\n*(Key Results table should be manually updated with final 5-seed results from streaming-adaptation-final)*\n"

with open(LOG_PATH, "a") as f:
    f.write(note)

write_summary()


# ─── Phase 3: Final 5-Seed Comparison ────────────────────────────────────────

FINAL_PROJECT = "streaming-adaptation-final"
log(f"\n## Phase 3: Final comparison → {FINAL_PROJECT}")

final_runs = [
    ("ekf-loraxs",    "ekf_loraxs",    {}),
    ("ekf-lastlayer", "ekf_lastlayer", {}),
    ("ekf-tinylora",  "ekf_tinylora",  {}),
    ("ogd-loraxs",    "ogd_loraxs",    {"lr": best_ogd_lrs["ogd_loraxs"]}),
    ("ogd-tinylora",  "ogd_tinylora",  {"lr": best_ogd_lrs["ogd_tinylora"]}),
    ("ogd-lastlayer", "ogd_lastlayer", {"lr": best_ogd_lrs["ogd_lastlayer"]}),
    ("ogd-dense",     "ogd_dense",     {"lr": best_ogd_lrs["ogd_dense"]}),
]

final_results = {}

for run_name, config_key, lr_override in final_runs:
    with open(BASE_CONFIGS[config_key]) as f:
        cfg = json.load(f)
    cfg["training"]["wandb_project"] = FINAL_PROJECT
    if lr_override:
        cfg["training"]["trainer"]["lr"] = lr_override["lr"]

    log(f"  Running {run_name} (5 seeds)...")
    out = run_experiment(cfg, run_name, num_seeds=5)
    rewards = parse_rewards(out)
    final_results[run_name] = rewards
    log(f"  {run_name}: {fmt(rewards)}")
    write_summary()


# ─── Final Summary ────────────────────────────────────────────────────────────

log(f"\n---\n## Final Results — {FINAL_PROJECT} — {datetime.now().strftime('%Y-%m-%d')}")
log("\n| Method | Trainer | Mean | Std | Seeds |")
log("|---|---|---|---|---|")

rows = [
    ("LoRA-XS",      "EKF", "ekf-loraxs"),
    ("last-layer",   "EKF", "ekf-lastlayer"),
    ("TinyLoRA",     "EKF", "ekf-tinylora"),
    ("LoRA-XS",      "OGD", "ogd-loraxs"),
    ("TinyLoRA",     "OGD", "ogd-tinylora"),
    ("last-layer",   "OGD", "ogd-lastlayer"),
    ("full network", "OGD", "ogd-dense"),
]

for method, trainer, key in rows:
    rewards = final_results.get(key, [])
    if rewards:
        a = np.array(rewards)
        log(f"| {method} | {trainer} | {a.mean():.1f} | ±{a.std():.1f} | {len(a)} |")
    else:
        log(f"| {method} | {trainer} | — | — | — |")

log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
write_summary()

# ─── Append concise results to experiment_log.md ─────────────────────────────

entry = f"\n\n---\n\n## Final Comparison — {FINAL_PROJECT} — {datetime.now().strftime('%Y-%m-%d')}\n\n"
entry += "All methods, 5 seeds each. OGD LRs tuned via sweep (see `ogd-*-lr` wandb projects).\n\n"
entry += "| Method | Trainer | Mean | Std | Seeds |\n"
entry += "|---|---|---|---|---|\n"

for method, trainer, key in rows:
    rewards = final_results.get(key, [])
    if rewards:
        a = np.array(rewards)
        entry += f"| {method} | {trainer} | {a.mean():.1f} | ±{a.std():.1f} | {len(a)} |\n"
    else:
        entry += f"| {method} | {trainer} | — | — | — |\n"

entry += "\n**OGD best LRs from sweep:**\n"
for key, label in method_labels.items():
    entry += f"- {label}: lr={best_ogd_lrs[key]:.0e}\n"

with open(LOG_PATH, "a") as f:
    f.write(entry)

print(f"\nResults appended to {LOG_PATH}")
print(f"Progress log at {SUMMARY_PATH}")
