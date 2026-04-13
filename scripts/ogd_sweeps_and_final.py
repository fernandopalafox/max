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
SUMMARY_PATH = REPO / "final_comparison_summary.md"

BASE_CONFIGS = {
    "ogd_lastlayer": REPO / "configs" / "stream_ogd.json",
    "ogd_dense":     REPO / "configs" / "stream_ogd_dense.json",
    "ogd_loraxs":    REPO / "configs" / "stream_ogd_lora.json",
    "ogd_tinylora":  REPO / "configs" / "stream_ogd_tiny_lora.json",
    "ekf_loraxs":    REPO / "configs" / "stream_ekf_lora_best_full.json",
    "ekf_tinylora":  REPO / "configs" / "stream_ekf_tiny_lora_best.json",
    "ekf_lastlayer": REPO / "configs" / "stream_ekf_lastlayer_best.json",
}

OGD_LRS = [1e-5, 1e-4, 3e-4, 1e-3, 1e-2]

summary = []


def log(msg):
    print(msg, flush=True)
    summary.append(msg)


def write_summary():
    SUMMARY_PATH.write_text("\n".join(summary) + "\n")


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


# ─── Phase 1: OGD LR Sweeps ──────────────────────────────────────────────────

ogd_sweep_configs = {
    "ogd_lastlayer": ("ogd-lastlayer-lr", {}),
    "ogd_dense":     ("ogd-dense-lr",     {}),
    "ogd_loraxs":    ("ogd-loraxs-lr",    {}),
    "ogd_tinylora":  ("ogd-tinylora-lr",  {}),
}

best_ogd_lrs = {}

for method_key, (project, overrides) in ogd_sweep_configs.items():
    log(f"\n## OGD LR Sweep: {method_key} → project={project}")
    with open(BASE_CONFIGS[method_key]) as f:
        base = json.load(f)

    sweep_results = {}
    for lr in OGD_LRS:
        run_name = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
        cfg = json.loads(json.dumps(base))
        cfg["training"]["wandb_project"] = project
        cfg["training"]["trainer"]["lr"] = lr
        for k, v in overrides.items():
            cfg["training"][k] = v

        out = run_experiment(cfg, run_name, num_seeds=3)
        rewards = parse_rewards(out)
        sweep_results[lr] = rewards
        log(f"  lr={lr:.0e}: {fmt(rewards)}")

    best_lr = max(sweep_results, key=lambda k: np.mean(sweep_results[k]) if sweep_results[k] else -1)
    best_ogd_lrs[method_key] = best_lr
    log(f"  → Best lr: {best_lr:.0e}  ({fmt(sweep_results[best_lr])})")
    write_summary()

log(f"\n## OGD Best LRs Summary")
for k, v in best_ogd_lrs.items():
    log(f"  {k}: {v:.0e}")
write_summary()


# ─── Phase 2: Update experiment_log.md ───────────────────────────────────────

log("\n## Updating experiment_log.md Key Results table...")

log_text = LOG_PATH.read_text()

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
print(f"\nSummary written to {SUMMARY_PATH}")
