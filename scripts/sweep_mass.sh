#!/bin/bash
set -e

BASE_CONFIG=/home/fp5275/repos/max/configs/stream_gd_loraxs_maml.json

for MASS in 0.1; do
    TMP=$(mktemp /tmp/sweep_mass_XXXXXX.json)
    python3 -c "
import json, sys
cfg = json.load(open('$BASE_CONFIG'))
t = cfg['training']
t['wandb_project'] = 'mass_scale_sweep'
t['num_seeds'] = 1
t['num_processes'] = 1
t['max_steps'] = 10000
t['eval_freq'] = 1000
t['plot_eval'] = False
t['environment']['cheetah_mass_scale'] = $MASS
t['evaluator']['environment']['cheetah_mass_scale'] = $MASS
t['save_dir'] = '/tmp/sweep_mass_$MASS'
json.dump(cfg, open('$TMP', 'w'), indent=2)
"
    echo "=== mass_scale=$MASS ==="
    conda run -n max --no-capture-output python /home/fp5275/repos/max/scripts/train.py \
        --config "$TMP" --run-name "sweep-mass-$MASS"
    rm -f "$TMP"
done
