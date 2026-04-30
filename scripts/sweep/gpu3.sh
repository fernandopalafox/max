#!/bin/bash
set -e
cd /home/fp5275/repos/max

# === Phase 1: dense pretrain seed=49 run_6, then finetune dynamics r4 ===

python3 -c "
import json
with open('configs/pretrain_dense_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 49
c['training']['save_dir'] = './data/models/cheetah/baseline_new/run_6'
with open('/tmp/gpu3_dense_r6.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu3_dense_r6.json --run-name tdmpc2_r6 --gpu 3

DENSE_CKPT=$(find data/models/cheetah/baseline_new/run_6 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_full_dynamics_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu3_adapt_dynamics_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$DENSE_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu3_adapt_dynamics_r4.json --run-name dynamics_pt_r4 --gpu 3

# === Phase 2, seed=52, run 9 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 52
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_9'
with open('/tmp/gpu3_loraxs_r9.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu3_loraxs_r9.json --run-name loraxs_tdmpc2_r9 --gpu 3

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_9 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu3_adapt_loraxs_r9.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu3_adapt_loraxs_r9.json --run-name loraxs_pt_r9 --gpu 3

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 52
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_9'
with open('/tmp/gpu3_fomaml_r9.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu3_fomaml_r9.json --run-name loraxs_fomaml_r9 --gpu 3

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_9 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu3_adapt_fomaml_r9.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu3_adapt_fomaml_r9.json --run-name loraxs_pt_fomaml_r9 --gpu 3

# === Phase 2, seed=53, run 10 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 53
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_10'
with open('/tmp/gpu3_loraxs_r10.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu3_loraxs_r10.json --run-name loraxs_tdmpc2_r10 --gpu 3

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_10 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu3_adapt_loraxs_r10.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu3_adapt_loraxs_r10.json --run-name loraxs_pt_r10 --gpu 3

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 53
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_10'
with open('/tmp/gpu3_fomaml_r10.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu3_fomaml_r10.json --run-name loraxs_fomaml_r10 --gpu 3

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_10 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu3_adapt_fomaml_r10.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu3_adapt_fomaml_r10.json --run-name loraxs_pt_fomaml_r10 --gpu 3

echo "GPU 3 complete."
