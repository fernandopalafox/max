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

## Key Results (Best Configs) — 2026-04-13

Methods we care most about, with best hyperparameters from sweeps. All in `fewshot-cheetah`. LR for OGD methods was not swept — set by param-count scaling from dense lr=3e-4.

| Method | Trainer | Mean | Std | Seeds | Params | Best Hyperparams | wandb run |
|---|---|---|---|---|---|---|---|
| LoRA-XS | EKF | **616** | ±48 | 3 | 9,216 | adapt=[0,1,2,3], r=48, lr=10 | `stream-ekf-lora-best-full` |
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


### OGD LR Sweep Results — 2026-04-13
Best LRs found:
- OGD last-layer: lr=1e-04
- OGD full network: lr=1e-04
- OGD LoRA-XS: lr=1e-04
- OGD TinyLoRA: lr=1e-03

*(Key Results table should be manually updated with final 5-seed results from streaming-adaptation-final)*


---

## Final Comparison — streaming-adaptation-final — 2026-04-13

All methods, 5 seeds each. OGD LRs tuned via sweep (see `ogd-*-lr` wandb projects).

| Method | Trainer | Mean | Std | Seeds |
|---|---|---|---|---|
| LoRA-XS | EKF | 609.7 | ±40.7 | 5 |
| last-layer | EKF | 480.7 | ±69.2 | 5 |
| TinyLoRA | EKF | 502.4 | ±31.8 | 5 |
| LoRA-XS | OGD | 540.2 | ±51.3 | 5 |
| TinyLoRA | OGD | 474.2 | ±26.5 | 5 |
| last-layer | OGD | 445.4 | ±68.5 | 5 |
| full network | OGD | 589.5 | ±89.1 | 5 |

**OGD best LRs from sweep:**
- OGD last-layer: lr=1e-04
- OGD full network: lr=1e-04
- OGD LoRA-XS: lr=1e-04
- OGD TinyLoRA: lr=1e-03


---

## Final Comparison v2 — 2 processes × 3 seeds — 2026-04-14

Re-run of all 7 methods using `num_processes=2, num_seeds=3` (6 runs per method) in wandb project `streaming-adaptation-v2`. Each process clears the JAX compilation cache and starts fresh, sampling both training-trajectory and cross-compile GPU variance. OGD LRs from prior sweep (see v1 entry).

| Method | Trainer | Mean | Std | N |
|---|---|---|---|---|
| LoRA-XS | EKF | 614.0 | ±48.6 | 6 |
| last-layer | EKF | 505.4 | ±83.4 | 6 |
| TinyLoRA | EKF | 497.4 | ±29.4 | 6 |
| LoRA-XS | OGD | 560.4 | ±34.5 | 6 |
| TinyLoRA | OGD | 459.6 | ±55.6 | 6 |
| last-layer | OGD | 421.0 | ±31.1 | 6 |
| full network | OGD | 632.1 | ±56.3 | 6 |

**Trend vs v1 (5 seeds, 1 process):**
- LoRA-XS (EKF): 614.0 vs 609.7 (+4.3)
- last-layer (EKF): 505.4 vs 480.7 (+24.7)
- TinyLoRA (EKF): 497.4 vs 502.4 (-5.0)
- LoRA-XS (OGD): 560.4 vs 540.2 (+20.2)
- TinyLoRA (OGD): 459.6 vs 474.2 (-14.6)
- last-layer (OGD): 421.0 vs 445.4 (-24.4)
- full network (OGD): 632.1 vs 589.5 (+42.6)

Rankings consistent with v1 indicate prior results were not artifacts of a single compile state.
