#!/bin/bash
set -e

BASE_CONFIG=/home/fp5275/repos/max/configs/stream_gd_loraxs_maml.json

for LR in 1e-5 1e-4 1e-3 1e-2 0.1 1.0; do
    TMP=$(mktemp /tmp/sweep_lr_XXXXXX.json)
    python3 -c "
import json
cfg = json.load(open('$BASE_CONFIG'))
t = cfg['training']
t['wandb_project'] = 'lr_sweep'
t['num_seeds'] = 3
t['num_processes'] = 1
t['max_steps'] = 10000
t['eval_freq'] = 1000
t['plot_eval'] = False
t['environment']['cheetah_mass_scale'] = 0.1
t['evaluator']['environment']['cheetah_mass_scale'] = 0.1
t['trainer']['meta_lr_inner'] = $LR
t['save_dir'] = '/tmp/sweep_lr_$LR'
json.dump(cfg, open('$TMP', 'w'), indent=2)
"
    echo "=== lr=$LR ==="
    conda run -n max --no-capture-output python /home/fp5275/repos/max/scripts/train.py \
        --config "$TMP" --run-name "sweep-lr-$LR"
    rm -f "$TMP"
done
