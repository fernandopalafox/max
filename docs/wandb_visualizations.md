# Wandb Visualizations for Lambda Sweep Experiments

## Overview

The script `scripts/run_unicycle_wandb.py` runs experiments comparing different values of λ (the information-gathering weight) and logs results to Weights & Biases for visualization.

## What it Compares

- **λ=0**: Evasion only (no information-gathering term)
- **λ=10, 100, ...**: Different weights on the directed information cost term
- **random** (optional): Random actions baseline (no planning)

## Metrics Logged

### EKF Metrics (per step)
- `eval/cov_trace` - Trace of the parameter covariance matrix (total uncertainty)
- `eval/param_diff` - L2 norm of parameter estimation error (distance from true θ)

### Dynamics Model Evaluation (periodic)
- `eval/one_step_loss` - MSE for single-step predictions (local model accuracy)
- `eval/multi_step_loss` - MSE over full trajectory rollout (tests compounding error)

These evaluate the learned dynamics model against a fixed evaluation trajectory rolled out from the initial state using random actions.

### At end
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

# Extended lambda sweep
python scripts/run_unicycle_wandb.py --lambdas "0,10,100,1000"

# Quick test run (50 steps)
python scripts/run_unicycle_wandb.py --steps 50 --lambdas "0,10"

# Use different config
python scripts/run_unicycle_wandb.py --config unicycle_aggressive

# Multiple seeds for error bars
python scripts/run_unicycle_wandb.py --num-seeds 5 --include-random

# Custom project/group
python scripts/run_unicycle_wandb.py --project my-project --group exp1

# Override total steps
python scripts/run_unicycle_wandb.py --steps 200 --lambdas "0,10,100"
```

### Manual Initial Conditions

Override starting positions and heading for controlled experiments:

```bash
# Set evader at origin, pursuer at (5,0) facing evader (head-on)
python scripts/run_unicycle_wandb.py --lambdas "0,10,100" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 3.14

# Pursuer facing away (must turn around)
python scripts/run_unicycle_wandb.py --lambdas "0,10,100" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 0

# Perpendicular approach
python scripts/run_unicycle_wandb.py --lambdas "0,10,100" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 1.57

# Close quarters
python scripts/run_unicycle_wandb.py --lambdas "0,10,100" \
  --evader-pos "0,0" --pursuer-pos "2,0" --pursuer-heading 3.14

# Just override heading (keep random positions)
python scripts/run_unicycle_wandb.py --lambdas "0,10,100" \
  --pursuer-heading 1.57
```

### Regime-Oriented Experiments

Compare normal vs aggressive pursuer across different starting configurations:

```bash
# Head-on collision course
python scripts/run_unicycle_wandb.py --config unicycle --lambdas "0,10,100,1000" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 3.14 \
  --group "head_on_normal"

python scripts/run_unicycle_wandb.py --config unicycle_aggressive --lambdas "0,10,100,1000" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 3.14 \
  --group "head_on_aggressive"

# Pursuer facing away (tests turn rate learning)
python scripts/run_unicycle_wandb.py --config unicycle --lambdas "0,10,100,1000" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 0 \
  --group "facing_away_normal"

python scripts/run_unicycle_wandb.py --config unicycle_aggressive --lambdas "0,10,100,1000" \
  --evader-pos "0,0" --pursuer-pos "5,0" --pursuer-heading 0 \
  --group "facing_away_aggressive"
```

### Testing Seeds

Preview initial conditions for different seeds:

```bash
# Check seeds 0-10
python scripts/test_seeds.py --seeds "0,1,2,3,4,5,6,7,8,9,10"

# Use a specific seed
python scripts/run_unicycle_wandb.py --seed 42 --lambdas "0,10,100"
```

## Expected Plots in Wandb

1. **Covariance Trace vs Step** (`eval/cov_trace`)
   - Shows how uncertainty decreases over time
   - Higher λ should reduce uncertainty faster (more exploration)

2. **Parameter Error vs Step** (`eval/param_diff`)
   - Shows convergence to true parameters
   - Higher λ should converge faster (if not too high)

3. **One-Step Loss vs Step** (`eval/one_step_loss`)
   - Shows local dynamics model accuracy
   - Lower is better - model predicts next state well given current state

4. **Multi-Step Loss vs Step** (`eval/multi_step_loss`)
   - Shows trajectory prediction accuracy (errors compound)
   - More sensitive metric - tests if model is good enough for planning

## Config Options

The evaluation behavior can be controlled via config:
- `eval_traj_horizon` - Length of evaluation trajectory (default: 50)
- `eval_freq` - How often to log eval metrics (default: `train_model_freq`)

## CLI Arguments Reference

| Argument | Description | Example |
|----------|-------------|---------|
| `--config` | Config file name (without .json) | `unicycle_aggressive` |
| `--lambdas` | Comma-separated λ values | `"0,10,100,1000"` |
| `--steps` | Override total_steps | `200` |
| `--seed` | Random seed | `42` |
| `--num-seeds` | Run multiple seeds | `5` |
| `--meta-seed` | Seed for generating run seeds | `42` |
| `--project` | Wandb project name | `my-project` |
| `--group` | Wandb group name | `head_on_normal` |
| `--include-random` | Add random baseline | (flag) |
| `--evader-pos` | Evader start position | `"0,0"` |
| `--pursuer-pos` | Pursuer start position | `"5,0"` |
| `--pursuer-heading` | Pursuer start heading (radians) | `3.14` |

## Logged Initial State Config

When runs are logged to wandb, the following initial state info is included:
- `initial/evader_x`, `initial/evader_y`
- `initial/pursuer_x`, `initial/pursuer_y`
- `initial/pursuer_heading`, `initial/pursuer_speed`
- `initial/distance`

These can be used for filtering and grouping runs in wandb.

## Notes

- All runs in a sweep share the same initial state for fair comparison
- The config key is stored as `λ` (unicode) in wandb config
- Group name defaults to `λ_sweep`
- Uses non-interactive matplotlib backend (`Agg`) to avoid threading issues with wandb
- Available configs: `unicycle`, `unicycle_aggressive`, `unicycle_slow_turn`, `unicycle_learn_turn`
