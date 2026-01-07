# Pursuit-Evasion Comparison Script Notes

## Overview

We created `scripts/run_lqr_comparison.py` to compare four different evader strategies against an LQR-controlled pursuer:

1. **Perfect Info**: Evader knows the true Q/R matrices of the pursuer's LQR controller
2. **No Learning**: Evader has wrong Q/R estimates and never updates them
3. **Passive Learning**: Evader has wrong Q/R but learns via EKF (no exploration incentive)
4. **Active Learning**: Evader has wrong Q/R, learns via EKF, and uses info-gathering cost term

## Key Files Modified

### `max/environments.py`
- **Bug fix in `reset_fn`**: The original code created state as `[positions, velocities]` but the dynamics expected `[evader_pos, evader_vel, pursuer_pos, pursuer_vel]`. Fixed by interleaving correctly:
```python
evader_state = jnp.concatenate([agent_positions[0, 0:2], jnp.zeros(2)])
pursuer_state = jnp.concatenate([agent_positions[0, 2:4], jnp.zeros(2)])
return jnp.concatenate([evader_state, pursuer_state])
```

### `max/costs.py`
- Added `_stage_cost_evasion_only()` helper function
- Added `evasion_only` cost type in `init_cost()` for cases without info-gathering term

### `scripts/run_lqr_comparison.py`
- New comparison script with four run functions
- All run functions return `(states, actions)` tuples for full cost analysis
- `plot_comparison()`: Trajectory plots with velocity arrows
- `plot_metrics()`: Three subplots showing distance, cumulative distance, and cumulative control cost
- Summary statistics include control cost and total cost

### `scripts/analyze_trajectory.py`
- Updated to handle new npz format with `_states` and `_actions` suffixes
- Now computes and displays control cost and total cost

### `configs/lqr.json`
- **Changed `weight_info` from 1000.0 to 10.0** (see experimental findings below)

## Cost Structure

The evader's cost function is:
```
cost = -dist_sq + weight_control * ||u||^2 + exploration_term (for active learning)
```

Where:
- `-dist_sq`: Negative squared distance (evader wants to maximize distance)
- `weight_control * ||u||^2`: Penalizes control effort
- `exploration_term`: `-weight_info * info_gain` (only in active learning)

From `configs/lqr.json`:
- `weight_control = 10.0`
- `weight_info = 10.0` (reduced from 1000.0)
- True pursuer params: `q_diag = [10, 10, 1, 1]`, `r_diag = [0.1, 0.1]`
- Initial (wrong) params: `q_diag = [1, 1, 1, 1]`, `r_diag = [1, 1]`

## Experimental Findings

### The Control Cost Paradox

Initially, we observed that Active Learning appeared to outperform Perfect Info on distance metrics, which seemed paradoxical. Investigation revealed that **we weren't tracking control costs**.

When we added control cost tracking, we discovered:

#### With `weight_info = 1000.0` (original):

| Case | Cumulative Dist | Control Cost | Total Cost |
|------|-----------------|--------------|------------|
| Perfect Info | 449.72 | 171.48 | **-278.24** |
| No Learning | 719.75 | 1155.90 | 436.16 |
| Passive Learning | 478.58 | 324.73 | -153.85 |
| Active Learning | 910.20 | **8080.67** | **7170.47** |

**Problem**: Active Learning was spending ~47x more control effort than Perfect Info! The info-gathering term dominated the cost function, causing the evader to "thrash around" to generate informative data at the expense of its actual evasion objective.

#### With `weight_info = 10.0` (fixed):

| Case | Cumulative Dist | Control Cost | Total Cost |
|------|-----------------|--------------|------------|
| Perfect Info | 449.72 | 171.48 | **-278.24** |
| No Learning | 719.75 | 1155.90 | 436.16 |
| Passive Learning | 478.58 | 324.73 | -153.85 |
| Active Learning | 486.21 | 338.69 | -147.51 |

**Result**: With balanced weights, Active Learning now has similar control cost to Passive Learning (~339 vs ~325) and achieves reasonable evasion performance.

### Key Insights

1. **Perfect Info is still the best** (total cost -278) as expected - knowing the true model allows optimal evasion with minimal control effort.

2. **No Learning is catastrophic** - wrong model estimates lead to both poor evasion AND high control costs (total cost +436).

3. **Passive Learning helps** - EKF updates improve the model passively, reducing total cost to -154.

4. **Active Learning with proper tuning** - slightly *worse* than Passive (-148 vs -154), though the difference is minor (~4%). The exploration bonus provides no benefit in this scenario, possibly because:
   - The pursuit-evasion dynamics naturally generate informative trajectories
   - The EKF converges quickly enough without explicit exploration
   - The horizon is short enough that learning benefits don't compound

5. **Hyperparameter sensitivity**: The `weight_info` parameter is critical. Too high and exploration dominates; too low and there's no exploration benefit. The ratio `weight_info / weight_control` determines the exploration-exploitation trade-off.

## Usage

```bash
python scripts/run_lqr_comparison.py --seed 42
python scripts/analyze_trajectory.py  # Analyze saved results
```

Output:
- `comparison_results/trajectories_seed_42.npz` (states and actions for all cases)
- `comparison_results/comparison_seed_42.png` (trajectory plots)
- `comparison_results/comparison_seed_42_metrics.png` (metrics over time)

## Notes on Implementation

- `init_env()` mutates config by popping `true_q_diag`/`true_r_diag`, so we use `copy.deepcopy(config)` before each call
- All four cases use the same random key and initial state for fair comparison
- The pursuer always uses its true LQR controller; only the evader's model of the pursuer varies

## Future Work

1. **Test multiple seeds** - We only ran seed 42. Different initial conditions (starting positions, relative distances) might change whether Active Learning helps.

2. **Sweep `weight_info`** - Try values between 0 and 100 to find if there's a sweet spot where Active Learning outperforms Passive. Current result (10.0) shows no benefit.

3. **Longer horizons** - 250 steps may be too short for learning benefits to compound. Try 500-1000 steps to see if Active Learning pulls ahead once the model is better learned.

4. **Investigate No Learning control cost** - Why does No Learning use ~7x more control (1156 vs 171) than Perfect Info? The wrong model seems to make the evader "work harder" in unproductive ways. Understanding this could inform when learning matters most.

5. **Track parameter convergence** - Log the EKF's estimate of Q/R over time to see how quickly Passive vs Active Learning converge to the true values.

6. **Different initial parameter errors** - Current setup has 10x error in Q and R. Try smaller/larger errors to see when active exploration becomes valuable.
