# Overnight Results — 2026-04-13


## Step 1A: Extra seeds — stream-ekf-lora
  Extra seeds: mean=597.1 ±11.0 (n=2)
  Raw: [608.08173, 586.13098]

## Step 1B: Extra seeds — stream-ekf-tiny-lora
  Extra seeds: mean=456.5 ±36.7 (n=2)
  Raw: [419.81812, 493.13239]

## Step 1C: OGD + TinyLoRA tuned
  Results: mean=468.5 ±6.6 (n=3)
  Raw: [477.50879, 462.09262, 465.75259]

## LoRA-XS Phase 1: Spatial Placement (r=16, u fixed, lr=1000)
  p1-input adapt=[0]: mean=458.8 ±36.2 (n=3)
  p1-transition adapt=[1, 2]: mean=472.3 ±14.5 (n=3)
  p1-output adapt=[3]: mean=430.6 ±13.5 (n=3)
  p1-full adapt=[0, 1, 2, 3]: mean=417.7 ±103.7 (n=3)
  → Winner: p1-transition adapt=[1, 2]

## LoRA-XS Phase 2: Rank Sweep (adapt=[1, 2], lr=1000)
  p2-r4: mean=443.6 ±47.0 (n=3)
  p2-r8: mean=424.7 ±45.7 (n=3)
  p2-r16: mean=472.3 ±14.5 (n=3)
  p2-r32: mean=543.1 ±54.3 (n=3)
  p2-r48: mean=558.9 ±14.4 (n=3)
  → Winner: r=48

## LoRA-XS Phase 3: LR Sweep (adapt=[1, 2], r=48)
  p3-lr1: mean=520.3 ±26.5 (n=3)
  p3-lr10: mean=580.5 ±15.1 (n=3)
  p3-lr100: mean=527.7 ±70.0 (n=3)
  p3-lr500: mean=506.5 ±74.1 (n=3)
  p3-lr1000: mean=558.9 ±14.4 (n=3)
  p3-lr5000: mean=427.0 ±21.3 (n=3)
  → Winner: lr=10

---
## Final Summary

### fewshot-cheetah — Updated Results (5 seeds each for top configs)
stream-ekf-lora extra seeds (seeds 4-5): mean=597.1 ±11.0 (n=2)
stream-ekf-tiny-lora extra seeds (seeds 4-5): mean=456.5 ±36.7 (n=2)
stream-ogd-tiny-lora (tuned, 3 seeds): mean=468.5 ±6.6 (n=3)

### LoRA-XS Best Config (loraxs-placement)
adapt_layers: [1, 2]
svd_rank (r): 48
EKF lr: 10
eval/episode_reward: mean=580.5 ±15.1 (n=3)

Completed: 2026-04-13 02:55
