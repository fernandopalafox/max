# Active vs Passive Learning in Pursuit-Evasion

**Date**: January 7, 2026

## Goal

Demonstrate that **active information-gathering** (learning opponent parameters while evading) leads to better exploitation than **passive learning**. Specifically, we want the evader to discover and exploit a unicycle pursuer's turning weakness (staying behind it where it can't reach you).

---

## What We Tried

### 1. Baseline Investigation

- **Finding**: Perfect info achieves ~57-114 avg distance by staying behind the unicycle
- **Problem**: Passive and active learners only achieve ~3-4 distance despite eventually learning correct parameters
- **Root cause**: Path dependence — by the time parameters converge (~step 95-200), the geometric advantage is already lost. Also a catch-22: to learn parameters, unicycle needs to turn, but if you're behind it, it doesn't turn.

### 2. Warmstart Hypothesis (❌ Didn't help)

- Tested whether iCEM warmstart was trapping planners in local minima
- Added warmstart toggle to `planners.py`
- **Result**: Warmstart OFF was actually worse
- **Reverted** all warmstart changes

### 3. Aggressive Pursuer (✅ Success!)

- Created `configs/unicycle_aggressive.json` with `true_theta1 = 15.0` (vs original 5.0)
- **Results**:
  - Active learning: **16.43** mean distance (exploits successfully!)
  - Perfect info: **15.15** mean distance
  - Passive learning: **2.69** mean distance
- **Takeaway**: With a more aggressive pursuer, active learning matches perfect info performance while passive learning fails.

### 4. Slow Turner Config (❌ Not exploitable as hoped)

- Created `configs/unicycle_slow_turn.json` with `weight_w = 2.0` (high turn penalty)
- **Result**: All cases got similar ~5 distance. The slow turning wasn't creating an exploitable asymmetry in the same way.

### 5. Learnable Turn Penalty (🔄 Work in progress)

- Created `configs/unicycle_learn_turn.json`
- Made `theta2` control the turn penalty instead of acceleration scaling
- Added `theta2_role` config option in `dynamics.py`:
  - `"accel_scaling"` (default): theta2 scales acceleration
  - `"turn_penalty"`: theta2 is the angular velocity penalty weight

- **Problem**: EKF learned negative theta2 values (-0.78), which is nonsensical for a penalty weight

- **Fix**: Added exponential parameterization `jnp.exp(theta2) * w**2` to ensure positivity
  - `init_theta2 = -2.3` → exp(-2.3) ≈ 0.1 (thinks unicycle turns easily)
  - `true_theta2 = 0.693` → exp(0.693) ≈ 2.0 (actually slow to turn)

- **Current status**: Still not learning theta2 correctly. Needs more debugging.

---

## Key Files Modified

| File | Changes |
|------|---------|
| `max/dynamics.py` | Added `theta2_role` config, exp() parameterization for turn penalty |
| `scripts/run_unicycle_comparison.py` | Added `--config` flag |
| `configs/unicycle_aggressive.json` | Created (theta1=15) |
| `configs/unicycle_slow_turn.json` | Created (weight_w=2.0) |
| `configs/unicycle_learn_turn.json` | Created (learnable turn penalty) |

---

## Key Commands

```bash
# Run aggressive config
python scripts/run_unicycle_comparison.py --config unicycle_aggressive --seeds 42

# Run learn_turn config
python scripts/run_unicycle_comparison.py --config unicycle_learn_turn --seeds 42

# Generate GIFs
python scripts/animate_trajectory.py comparison_results_aggressive/unicycle_trajectories_seed_42.npz --output-dir comparison_results_aggressive
```

---

## Bottom Line

**Aggressive pursuer config works!** Active learning achieves near-perfect-info performance (16.43 vs 15.15) while passive learning fails (2.69). The learnable turn penalty approach needs more work — EKF gradient flow for theta2 needs debugging.

---

## Next Steps

1. Debug why EKF isn't learning theta2 correctly with exp() parameterization
2. Check gradient magnitudes for theta2 vs theta1
3. Consider whether the turn penalty has enough observability (does changing it affect observed trajectories enough?)

---

## Session Update (Jan 7, 2026 - later)

### 6. Relaxed MPC Clamping Experiment

Investigated whether the MPC output clamping was affecting results. Changed clamp bounds from:
- `max_angular_vel`: 2.0 → 10.0 rad/s
- `max_accel`: 5.0 → 20.0 m/s²

**Results with relaxed clamps:**

| Config | Case | Mean Dist | Control Cost | Total Cost |
|--------|------|-----------|--------------|------------|
| Aggressive (θ₁=15) | Perfect Info | 1.06 | 284 | 20 |
| | Passive | 1.06 | 292 | 27 |
| | Active | 1.42 | **2237** | **1882** |
| Default (θ₁=5) | Perfect Info | 3.24 | 620 | -189 |
| | Passive | 1.86 | 385 | -80 |
| | Active | 2.57 | **1710** | **1069** |

**Problem**: With looser clamps, active learning "thrashes" — uses ~8x more control effort than perfect info to gather information, destroying actual evasion performance. Also theta2 diverged to -0.10 (should be 1.0).

**Diagnosis**: The `weight_info` parameter (currently 10.0) is still too high relative to control penalty when the system has more freedom to move. The old tighter clamps were inadvertently regularizing the behavior.

### TODO for next session

1. **Reduce `weight_info`** significantly (try 1.0 or 0.1) to prevent info-gathering from dominating
2. Alternatively, keep tighter clamps as implicit regularization
3. The fundamental issue: exploration bonus needs to be balanced so it helps learning without sacrificing primary objective
