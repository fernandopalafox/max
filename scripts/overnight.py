#!/usr/bin/env python3
"""
Master overnight script.
Runs all experiments sequentially, makes decisions after each phase,
and writes a final summary to overnight_summary.md.
"""

import json
import re
import subprocess
import sys
import tempfile
import os
import numpy as np
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).parent.parent
SUMMARY_PATH = REPO / "overnight_summary.md"
BASE_EKF_LORA = REPO / "configs" / "stream_ekf_lora.json"
BASE_EKF_TINY = REPO / "configs" / "stream_ekf_tiny_lora_best.json"
BASE_OGD_TINY = REPO / "configs" / "stream_ogd_tiny_lora.json"

summary_lines = [f"# Overnight Results — {datetime.now().strftime('%Y-%m-%d')}\n"]


def log(msg):
    print(msg, flush=True)
    summary_lines.append(msg)


def write_summary():
    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n")


def run_experiment(config_dict, run_name, num_seeds, extra_seed=False):
    """Write temp config, run experiment, return stdout lines."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_dict, f, indent=2)
        tmp_path = f.name
    try:
        cmd = [
            "conda", "run", "-n", "max",
            "python", str(REPO / "scripts" / "train.py"),
            "--config", tmp_path,
            "--run-name", run_name,
            "--num-seeds", str(num_seeds),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
        output = result.stdout + result.stderr
        if result.returncode != 0:
            log(f"  ERROR running {run_name}: exit code {result.returncode}")
            log(f"  Last 20 lines:\n" + "\n".join(output.splitlines()[-20:]))
        return output
    finally:
        os.unlink(tmp_path)


def parse_rewards(output):
    """Extract final eval/episode_reward float values (not sparklines)."""
    rewards = []
    for line in output.splitlines():
        if "eval/episode_reward" in line:
            m = re.search(r"eval/episode_reward\s+([\d.]+)", line)
            if m:
                rewards.append(float(m.group(1)))
    return rewards


def summarize(rewards):
    if not rewards:
        return "NO RESULTS"
    arr = np.array(rewards)
    return f"mean={arr.mean():.1f} ±{arr.std():.1f} (n={len(arr)})"


def pick_best(results: dict):
    """Return key with highest mean reward."""
    return max(results, key=lambda k: np.mean(results[k]) if results[k] else -1)


# ─── Load base configs ────────────────────────────────────────────────────────

with open(BASE_EKF_LORA) as f:
    ekf_lora_cfg = json.load(f)

with open(BASE_EKF_TINY) as f:
    ekf_tiny_cfg = json.load(f)

with open(BASE_OGD_TINY) as f:
    ogd_tiny_cfg = json.load(f)


# ─── Step 1A: Extra seeds for stream-ekf-lora ────────────────────────────────

log("\n## Step 1A: Extra seeds — stream-ekf-lora")
cfg = json.loads(json.dumps(ekf_lora_cfg))
cfg["training"]["seed"] = 100
out = run_experiment(cfg, "stream-ekf-lora", num_seeds=2)
rewards_ekf_lora_extra = parse_rewards(out)
log(f"  Extra seeds: {summarize(rewards_ekf_lora_extra)}")
log(f"  Raw: {rewards_ekf_lora_extra}")
write_summary()


# ─── Step 1B: Extra seeds for stream-ekf-tiny-lora ───────────────────────────

log("\n## Step 1B: Extra seeds — stream-ekf-tiny-lora")
cfg = json.loads(json.dumps(ekf_tiny_cfg))
cfg["training"]["seed"] = 100
out = run_experiment(cfg, "stream-ekf-tiny-lora", num_seeds=2)
rewards_ekf_tiny_extra = parse_rewards(out)
log(f"  Extra seeds: {summarize(rewards_ekf_tiny_extra)}")
log(f"  Raw: {rewards_ekf_tiny_extra}")
write_summary()


# ─── Step 1C: OGD + TinyLoRA tuned ──────────────────────────────────────────

log("\n## Step 1C: OGD + TinyLoRA tuned")
cfg = json.loads(json.dumps(ogd_tiny_cfg))
cfg["training"]["dynamics"]["svd_rank"] = 64
cfg["training"]["dynamics"]["steering_dim"] = 16
cfg["training"]["dynamics"]["adapt_layers"] = [0, 1, 2, 3]
out = run_experiment(cfg, "stream-ogd-tiny-lora", num_seeds=3)
rewards_ogd_tiny = parse_rewards(out)
log(f"  Results: {summarize(rewards_ogd_tiny)}")
log(f"  Raw: {rewards_ogd_tiny}")
write_summary()


# ─── LoRA-XS Phase 1: Spatial Placement ──────────────────────────────────────

log("\n## LoRA-XS Phase 1: Spatial Placement (r=16, u fixed, lr=1000)")

phase1_configs = {
    "p1-input":      [0],
    "p1-transition": [1, 2],
    "p1-output":     [3],
    "p1-full":       [0, 1, 2, 3],
}
phase1_results = {}

for run_name, adapt_layers in phase1_configs.items():
    cfg = json.loads(json.dumps(ekf_lora_cfg))
    cfg["training"]["wandb_project"] = "loraxs-placement"
    cfg["training"]["dynamics"]["svd_rank"] = 16
    cfg["training"]["dynamics"]["adapt_layers"] = adapt_layers
    out = run_experiment(cfg, run_name, num_seeds=3)
    rewards = parse_rewards(out)
    phase1_results[run_name] = rewards
    log(f"  {run_name} adapt={adapt_layers}: {summarize(rewards)}")

# Decision: pick best; if within 20 pts of best, use most expressive (full)
best_p1 = pick_best(phase1_results)
best_p1_mean = np.mean(phase1_results[best_p1]) if phase1_results[best_p1] else 0
full_mean = np.mean(phase1_results["p1-full"]) if phase1_results["p1-full"] else 0
if best_p1 != "p1-full" and (best_p1_mean - full_mean) > 20:
    chosen_adapt = phase1_configs[best_p1]
    log(f"  → Winner: {best_p1} adapt={chosen_adapt}")
else:
    chosen_adapt = [0, 1, 2, 3]
    log(f"  → Results close or full wins → using adapt=[0,1,2,3] (most expressive)")

write_summary()


# ─── LoRA-XS Phase 2: Rank Sweep ─────────────────────────────────────────────

log(f"\n## LoRA-XS Phase 2: Rank Sweep (adapt={chosen_adapt}, lr=1000)")

ranks = [4, 8, 16, 32, 48]
phase2_results = {}

for r in ranks:
    run_name = f"p2-r{r}"
    cfg = json.loads(json.dumps(ekf_lora_cfg))
    cfg["training"]["wandb_project"] = "loraxs-placement"
    cfg["training"]["dynamics"]["svd_rank"] = r
    cfg["training"]["dynamics"]["adapt_layers"] = chosen_adapt
    out = run_experiment(cfg, run_name, num_seeds=3)
    rewards = parse_rewards(out)
    phase2_results[run_name] = rewards
    log(f"  {run_name}: {summarize(rewards)}")

best_p2 = pick_best(phase2_results)
# If within 20 pts of best, prefer highest rank (most expressive)
best_p2_mean = np.mean(phase2_results[best_p2]) if phase2_results[best_p2] else 0
chosen_rank = None
for r in reversed(ranks):  # highest first
    run_name = f"p2-r{r}"
    mean = np.mean(phase2_results[run_name]) if phase2_results[run_name] else 0
    if (best_p2_mean - mean) <= 20:
        chosen_rank = r
        break
if chosen_rank is None:
    chosen_rank = int(best_p2.split("-r")[1])

log(f"  → Winner: r={chosen_rank}")
write_summary()


# ─── LoRA-XS Phase 3: LR Sweep ───────────────────────────────────────────────

log(f"\n## LoRA-XS Phase 3: LR Sweep (adapt={chosen_adapt}, r={chosen_rank})")

lrs = [1, 10, 100, 500, 1000, 5000]
phase3_results = {}

for lr in lrs:
    run_name = f"p3-lr{lr}"
    cfg = json.loads(json.dumps(ekf_lora_cfg))
    cfg["training"]["wandb_project"] = "loraxs-placement"
    cfg["training"]["dynamics"]["svd_rank"] = chosen_rank
    cfg["training"]["dynamics"]["adapt_layers"] = chosen_adapt
    cfg["training"]["trainer"]["lr"] = lr
    out = run_experiment(cfg, run_name, num_seeds=3)
    rewards = parse_rewards(out)
    phase3_results[run_name] = rewards
    log(f"  {run_name}: {summarize(rewards)}")

best_p3 = pick_best(phase3_results)
best_lr = int(best_p3.split("-lr")[1])
log(f"  → Winner: lr={best_lr}")
write_summary()


# ─── Final Summary ────────────────────────────────────────────────────────────

log("\n---\n## Final Summary")

log("\n### fewshot-cheetah — Updated Results (5 seeds each for top configs)")
log(f"stream-ekf-lora extra seeds (seeds 4-5): {summarize(rewards_ekf_lora_extra)}")
log(f"stream-ekf-tiny-lora extra seeds (seeds 4-5): {summarize(rewards_ekf_tiny_extra)}")
log(f"stream-ogd-tiny-lora (tuned, 3 seeds): {summarize(rewards_ogd_tiny)}")

log("\n### LoRA-XS Best Config (loraxs-placement)")
log(f"adapt_layers: {chosen_adapt}")
log(f"svd_rank (r): {chosen_rank}")
log(f"EKF lr: {best_lr}")
best_loraxs_rewards = phase3_results[best_p3]
log(f"eval/episode_reward: {summarize(best_loraxs_rewards)}")

log(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
write_summary()
print(f"\nSummary written to {SUMMARY_PATH}")
