# Wandb Visualizations for Lambda Sweep Experiments

## Overview

The script `scripts/run_unicycle_wandb.py` runs experiments comparing different values of λ (the information-gathering weight) and logs results to Weights & Biases for visualization.

## What it Compares

- **λ=0**: Evasion only (no information-gathering term)
- **λ=10, 100, ...**: Different weights on the directed information cost term
- **random** (optional): Random actions baseline (no planning)

## Metrics Logged

Per step:
- `eval/cov_trace` - Trace of the parameter covariance matrix (total uncertainty)
- `eval/param_diff` - L2 norm of parameter estimation error (distance from true θ)

At end:
- `trajectory/xy_plot` - X-Y trajectory plot as an image

## Run Names

- `random` - Random actions baseline
- `λ=0` - No information gathering
- `λ=10` - Moderate exploration
- `λ=100` - Aggressive exploration

## Usage

```bash
# Default: λ=0, 10, 100
python scripts/run_unicycle_wandb.py

# Include random baseline
python scripts/run_unicycle_wandb.py --include-random

# Custom lambda values
python scripts/run_unicycle_wandb.py --lambdas "0,5,10,50"

# Use different config
python scripts/run_unicycle_wandb.py --config unicycle_aggressive

# Multiple seeds for error bars
python scripts/run_unicycle_wandb.py --num-seeds 5 --include-random

# Custom project/group
python scripts/run_unicycle_wandb.py --project my-project --group exp1
```

## Expected Plots in Wandb

1. **Covariance Trace vs Step** (`eval/cov_trace`)
   - Shows how uncertainty decreases over time
   - Higher λ should reduce uncertainty faster (more exploration)

2. **Parameter Error vs Step** (`eval/param_diff`)
   - Shows convergence to true parameters
   - Higher λ should converge faster (if not too high)

## Notes

- All runs in a sweep share the same initial state for fair comparison
- The config key is stored as `λ` (unicode) in wandb config
- Group name defaults to `λ_sweep`
