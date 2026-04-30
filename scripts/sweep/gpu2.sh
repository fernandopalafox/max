#!/bin/bash
set -e
cd /home/fp5275/repos/max

# === Phase 1: dense pretrain seed=48 run_5, then finetune dynamics r3 ===

python3 -c "
import json
with open('configs/pretrain_dense_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 48
c['training']['save_dir'] = './data/models/cheetah/baseline_new/run_5'
with open('/tmp/gpu2_dense_r5.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu2_dense_r5.json --run-name tdmpc2_r5 --gpu 2

DENSE_CKPT=$(find data/models/cheetah/baseline_new/run_5 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_full_dynamics_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu2_adapt_dynamics_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$DENSE_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu2_adapt_dynamics_r3.json --run-name dynamics_pt_r3 --gpu 2

# === Phase 2, seed=50, run 7 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 50
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_7'
with open('/tmp/gpu2_loraxs_r7.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu2_loraxs_r7.json --run-name loraxs_tdmpc2_r7 --gpu 2

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_7 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu2_adapt_loraxs_r7.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu2_adapt_loraxs_r7.json --run-name loraxs_pt_r7 --gpu 2

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 50
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_7'
with open('/tmp/gpu2_fomaml_r7.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu2_fomaml_r7.json --run-name loraxs_fomaml_r7 --gpu 2

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_7 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu2_adapt_fomaml_r7.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu2_adapt_fomaml_r7.json --run-name loraxs_pt_fomaml_r7 --gpu 2

# === Phase 2, seed=51, run 8 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 51
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_8'
with open('/tmp/gpu2_loraxs_r8.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu2_loraxs_r8.json --run-name loraxs_tdmpc2_r8 --gpu 2

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_8 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu2_adapt_loraxs_r8.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu2_adapt_loraxs_r8.json --run-name loraxs_pt_r8 --gpu 2

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 51
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_8'
with open('/tmp/gpu2_fomaml_r8.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu2_fomaml_r8.json --run-name loraxs_fomaml_r8 --gpu 2

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_8 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu2_adapt_fomaml_r8.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu2_adapt_fomaml_r8.json --run-name loraxs_pt_fomaml_r8 --gpu 2

echo "GPU 2 complete."
