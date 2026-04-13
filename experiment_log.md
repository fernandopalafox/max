# Experiment Log

All experiments use: cheetah env, mass_scale=0.1, EKF=`ekf_efficient`, streaming setup (latest sampler, batch_size=1, horizon=1), 10k steps, 3 seeds unless noted.
Metric: `eval/episode_reward` (mean ± std).

---

## Project: `fewshot-cheetah`

### Streaming baseline comparisons

| Run | Dynamics | Trainer | lr | Mean | Std | Params | Notes |
|---|---|---|---|---|---|---|---|
| `stream-ekf` | last_layer | EKF | 1000 | — | — | ~8384 | original baseline |
| `stream-ogd` | last_layer | OGD | 4e-5 | — | — | ~8384 | |
| `stream-ogd-dense` | dense (full) | OGD | 3e-4 | — | — | ~400k | |
| `stream-ekf-lora` | LoRA-XS | EKF | 1000 | ~598 | — | 9216 | all layers, r=48, 3+2 seeds |
| `stream-ogd-lora` | LoRA-XS | OGD | 4e-5 | ~505 | — | 9216 | all layers, r=48 |
| `stream-ekf-tiny-lora` | TinyLoRA | EKF | 5000 | 496 | ±31 | 64 | best config: r=64,u=16,all layers, 3+2 seeds |
| `stream-ogd-tiny-lora` | TinyLoRA | OGD | 4e-5 | 496 | — | 64 | r=64, u=8, all layers (pre-tuning) |
| `stream-ogd-tiny-lora` (tuned) | TinyLoRA | OGD | 4e-5 | 469 | ±7 | 64 | r=64, u=16, all layers |

### Tuned LoRA-XS runs

| Run | adapt_layers | r | lr | Mean | Std | Params |
|---|---|---|---|---|---|---|
| `stream-ekf-lora-best` | [1,2] | 48 | 10 | 580 | ±18 | 4608 |
| `stream-ekf-lora-best-full` | [0,1,2,3] | 48 | 10 | **616** | ±48 | 9216 |
| `stream-ekf-lastlayer` | — | — | 7000 | 481 | ±77 | ~8384 |

---

## Project: `tinylora-placement`

Goal: Optimal spatial placement, rank r, steering dim u, and EKF lr for TinyLoRA.

### Phase 1 — Spatial Placement (u=64, r=2, lr=1000)

| adapt_layers | Mean | Std |
|---|---|---|
| [0] | 431 | ±63 |
| [1,2] | 424 | ±25 |
| [3] | 407 | ±29 |
| [0,1,2,3] | 415 | ±63 |

→ **Full [0,1,2,3]** (most expressive, results within noise)

### Phase 2A — Rank Sweep (u=16, adapt=[0,1,2,3], lr=1000)

| r | Mean | Std |
|---|---|---|
| 1 | 428 | ±96 |
| 2 | 444 | ±47 |
| 4 | 388 | ±25 |
| 8 | 444 | ±51 |
| 16 | 391 | ±81 |
| 32 | 398 | ±32 |
| 48 | 405 | ±83 |
| **64** | **482** | **±33** |

→ **r=64** (max possible, clear winner)

### Phase 2B — Steering Dim Sweep (r=64, adapt=[0,1,2,3], lr=1000)

| u | Mean | Std |
|---|---|---|
| **16** | **482** | **±33** |
| 64 | 449 | ±25 |
| 128 | 430 | ±13 |
| 256 | 422 | ±72 |

→ **u=16** (smaller u better — EKF with sparse data)

### Phase 3 — LR Sweep (r=64, u=16, adapt=[0,1,2,3])

| lr | Mean | Std |
|---|---|---|
| 1 | 489 | ±43 |
| 10 | 453 | ±55 |
| 100 | 433 | ±26 |
| 500 | 492 | ±27 |
| 1000 | 482 | ±33 |
| **5000** | **496** | **±31** |

→ **lr=5000**

**Best TinyLoRA config:** r=64, u=16, adapt=[0,1,2,3], lr=5000 → **496 ±31** (64 params)

---

## Project: `loraxs-placement`

Goal: Optimal spatial placement, rank r, and EKF lr for LoRA-XS (same sweep structure as TinyLoRA).

### Phase 1 — Spatial Placement (r=16, r_init_std=1e-5, lr=1000)

| adapt_layers | Mean | Std |
|---|---|---|
| [0] | 459 | ±36 |
| **[1,2]** | **472** | **±15** |
| [3] | 431 | ±14 |
| [0,1,2,3] | 418 | ±104 |

→ **[1,2]** (transition layers win, unlike TinyLoRA)

### Phase 2 — Rank Sweep (adapt=[1,2], lr=1000)

| r | Mean | Std | Params |
|---|---|---|---|
| 4 | 444 | ±47 | 32 |
| 8 | 425 | ±46 | 128 |
| 16 | 472 | ±15 | 512 |
| 32 | 543 | ±54 | 2048 |
| **48** | **559** | **±14** | 4608 |

→ **r=48** (clear trend: higher rank = better)

### Phase 3 — LR Sweep (adapt=[1,2], r=48)

| lr | Mean | Std |
|---|---|---|
| 1 | 520 | ±27 |
| **10** | **581** | **±15** |
| 100 | 528 | ±70 |
| 500 | 507 | ±74 |
| 1000 | 559 | ±14 |
| 5000 | 427 | ±21 |

→ **lr=10**

**Best LoRA-XS config:** adapt=[1,2], r=48, lr=10 → **581 ±15** (4608 params)

---

## Project: `lastlayer-sweep` — IN PROGRESS

Goal: Find best EKF lr for dense_last_layer dynamics.
Fixed: EKF trainer, last_layer dynamics, same streaming setup.

| lr | Mean | Std | Status |
|---|---|---|---|
| 1 | 446 | ±58 | done |
| 10 | 439 | ±65 | done |
| 100 | 461 | ±57 | done |
| 500 | 486 | ±28 | done |
| 1000 | 483 | ±53 | done |
| 5000 | 513 | ±86 | done |
| **7000** | **537** | **±17** | done ← best (3 seeds; 5-seed recheck gave 481 ±77) |
| 10000 | 472 | ±106 | done |
| 15000 | 474 | ±38 | done |
| 20000 | 223 | — | done (mostly collapsing) |
| 30000+ | — | — | abandoned (unstable) |

---

## Key Results (Best Configs)

Methods we care most about, with best hyperparameters from sweeps. All in `fewshot-cheetah`. LR for OGD methods was not swept — set by param-count scaling from dense lr=3e-4.

| Method | Trainer | Mean | Std | Seeds | Params | Best Hyperparams | wandb run |
|---|---|---|---|---|---|---|---|
| LoRA-XS | EKF | **616** | ±48 | 3 | 9,216 | adapt=[0,1,2,3], r=48, lr=10 | `stream-ekf-lora-best-full` |
| LoRA-XS | EKF | 580 | ±18 | 3 | 4,608 | adapt=[1,2], r=48, lr=10 | `stream-ekf-lora-best` |
| EKF last-layer | EKF | 481 | ±77 | 5 | ~8,384 | lr=7000 | `stream-ekf-lastlayer` |
| TinyLoRA | EKF | 496 | ±31 | 3 | **64** | adapt=[0,1,2,3], r=64, u=16, lr=5000 | `stream-ekf-tiny-lora` |
| OGD LoRA-XS | OGD | ~505 | — | 3 | 9,216 | adapt=[0,1,2,3], lr=4e-5 *(placement not swept for OGD)* | `stream-ogd-lora` |
| OGD TinyLoRA | OGD | 469 | ±7 | 3 | **64** | adapt=[0,1,2,3], r=64, u=16, lr=4e-5 | `stream-ogd-tiny-lora` |
| OGD last-layer | OGD | — | — | 3 | ~8,384 | lr=4e-5 *(not swept)* | `stream-ogd` |
| OGD full network | OGD | — | — | 3 | ~400k | lr=3e-4 *(not swept)* | `stream-ogd-dense` |

**Notes:**
- OGD last-layer and OGD dense results not precisely logged — check wandb `fewshot-cheetah`
- OGD LoRA-XS placement was never swept (only EKF was); [1,2] may improve OGD performance too
- EKF last-layer at 5 seeds shows high variance (±77), suggesting lr=7000 is near stability limit

---

## Sweep Details
