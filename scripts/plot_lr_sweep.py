import matplotlib.pyplot as plt
import numpy as np

data = {
    "loraxs_ogd": {
        1e-6: 550, 3e-6: 513, 1e-5: 503, 3e-5: 510,
        1e-4: 496, 3e-4: 485, 1e-3: 468, 3e-3: 379,
        1e-2: 528, 3e-2: 441, 1e-1: 459,
    },
    "dense_ogd": {
        1e-6: 523, 3e-6: 512, 1e-5: 520, 3e-5: 522,
        1e-4: 525, 3e-4: 524, 1e-3: 438, 3e-3: 325,
        1e-2: 436, 3e-2: 349, 1e-1: 396,
    },
    "lora_ogd": {
        1e-6: 506, 3e-6: 500, 1e-5: 476, 3e-5: 427,
        1e-4: 478, 3e-4: 525, 1e-3: 476, 3e-3: 495,
    },
}

colors = {"loraxs_ogd": "#e15759", "dense_ogd": "#4e79a7", "lora_ogd": "#59a14f"}
labels = {"loraxs_ogd": "LoRA-XS OGD", "dense_ogd": "Dense OGD", "lora_ogd": "LoRA OGD (rank 64)"}

fig, ax = plt.subplots(figsize=(7, 4))

for name, points in data.items():
    lrs = sorted(points)
    rewards = [points[lr] for lr in lrs]
    ax.plot(lrs, rewards, marker="o", color=colors[name], label=labels[name], linewidth=2, markersize=5)

ORACLE_REWARD = 501.5   # mean of 4 seeds, 500k-transition buffer, frozen E_A, OGD lr=3e-4
SKYLINE_REWARD = 974.0  # mean of 4 seeds, trained from scratch on Task B

ax.set_xscale("log")
ax.set_xlabel("Learning rate", fontsize=12)
ax.set_ylabel("Final eval reward", fontsize=12)
ax.set_title("CartPole 5× pole — OGD LR sweep", fontsize=13)

xlim = (5e-7, 2e-1)
ax.axhline(ORACLE_REWARD, color="gray", linestyle="--", linewidth=1.5,
           label=f"Oracle (500k transitions, frozen $E_A$): {ORACLE_REWARD:.0f}")
ax.axhline(SKYLINE_REWARD, color="black", linestyle="--", linewidth=1.5,
           label=f"Skyline (scratch on Task B): {SKYLINE_REWARD:.0f}")

ax.legend(fontsize=10)
ax.grid(True, which="both", alpha=0.3)
ax.set_ylim(300, 1050)

plt.tight_layout()
out = "logs/lr_sweep.png"
plt.savefig(out, dpi=150)
print(f"saved {out}")
