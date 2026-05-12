import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
project = "clearoboticslab/finetune_cartpole"

run_names = {
    "loraxs_ogd": {
        1e-6: "loraxs_ogd_lr0p003x", 3e-6: "loraxs_ogd_lr0p01x",
        1e-5: "loraxs_ogd_lr0p033x", 3e-5: "loraxs_ogd_lr0p1x",
        1e-4: "loraxs_ogd_lr0p33x",  3e-4: "loraxs_ogd",
        1e-3: "loraxs_ogd_lr3p3x",   3e-3: "loraxs_ogd_lr10x",
        1e-2: "loraxs_ogd_lr33x",    3e-2: "loraxs_ogd_lr100x",
        1e-1: "loraxs_ogd_lr333x",
    },
    "dense_ogd": {
        1e-6: "dense_ogd_lr0p003x", 3e-6: "dense_ogd_lr0p01x",
        1e-5: "dense_ogd_lr0p033x", 3e-5: "dense_ogd_lr0p1x",
        1e-4: "dense_ogd_lr0p33x",  3e-4: "dense_ogd",
        1e-3: "dense_ogd_lr3p3x",   3e-3: "dense_ogd_lr10x",
        1e-2: "dense_ogd_lr33x",    3e-2: "dense_ogd_lr100x",
        1e-1: "dense_ogd_lr333x",
    },
    "lora_ogd": {
        1e-6: "lora_ogd_lr0p003x", 3e-6: "lora_ogd_lr0p01x",
        1e-5: "lora_ogd_lr0p033x", 3e-5: "lora_ogd_lr0p1x",
        1e-4: "lora_ogd_lr0p33x",  3e-4: "lora_ogd",
        1e-3: "lora_ogd_lr3p3x",   3e-3: "lora_ogd_lr10x",
    },
}

# Fetch summary for every run name we need
name_to_summary = {}
runs = api.runs(project)
for run in runs:
    name_to_summary[run.name] = run.summary

colors = {"loraxs_ogd": "#e15759", "dense_ogd": "#4e79a7", "lora_ogd": "#59a14f"}
labels = {"loraxs_ogd": "LoRA-XS OGD", "dense_ogd": "Dense OGD", "lora_ogd": "LoRA OGD (rank 64)"}

fig, ax = plt.subplots(figsize=(7, 4))

for method, lr_map in run_names.items():
    lrs, losses = [], []
    for lr, name in sorted(lr_map.items()):
        summary = name_to_summary.get(name, {})
        val = summary.get("losses/consistency")
        if val is not None:
            lrs.append(lr)
            losses.append(float(val))
        else:
            print(f"missing: {name}")
    ax.plot(lrs, losses, marker="o", color=colors[method], label=labels[method], linewidth=2, markersize=5)

ax.set_xscale("log")
ax.set_xlabel("Learning rate", fontsize=12)
ax.set_ylabel("Final consistency loss", fontsize=12)
ax.set_title("CartPole 5× pole — consistency loss vs LR", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)

plt.tight_layout()
out = "logs/lr_sweep_consistency.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")
