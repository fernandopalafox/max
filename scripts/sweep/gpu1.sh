#!/bin/bash
set -e
cd /home/fp5275/repos/max

# === Phase 1: dense pretrain seed=47 run_4, then finetune dynamics r2 ===

python3 -c "
import json
with open('configs/pretrain_dense_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 47
c['training']['save_dir'] = './data/models/cheetah/baseline_new/run_4'
with open('/tmp/gpu1_dense_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu1_dense_r4.json --run-name tdmpc2_r4 --gpu 1

DENSE_CKPT=$(find data/models/cheetah/baseline_new/run_4 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_full_dynamics_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu1_adapt_dynamics_r2.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$DENSE_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu1_adapt_dynamics_r2.json --run-name dynamics_pt_r2 --gpu 1

# === Phase 2, seed=48, run 5 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 48
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_5'
with open('/tmp/gpu1_loraxs_r5.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu1_loraxs_r5.json --run-name loraxs_tdmpc2_r5 --gpu 1

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_5 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu1_adapt_loraxs_r5.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu1_adapt_loraxs_r5.json --run-name loraxs_pt_r5 --gpu 1

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 48
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_5'
with open('/tmp/gpu1_fomaml_r5.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu1_fomaml_r5.json --run-name loraxs_fomaml_r5 --gpu 1

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_5 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu1_adapt_fomaml_r5.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu1_adapt_fomaml_r5.json --run-name loraxs_pt_fomaml_r5 --gpu 1

# === Phase 2, seed=49, run 6 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 49
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_6'
with open('/tmp/gpu1_loraxs_r6.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu1_loraxs_r6.json --run-name loraxs_tdmpc2_r6 --gpu 1

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_6 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu1_adapt_loraxs_r6.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu1_adapt_loraxs_r6.json --run-name loraxs_pt_r6 --gpu 1

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 49
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_6'
with open('/tmp/gpu1_fomaml_r6.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu1_fomaml_r6.json --run-name loraxs_fomaml_r6 --gpu 1

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_6 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu1_adapt_fomaml_r6.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu1_adapt_fomaml_r6.json --run-name loraxs_pt_fomaml_r6 --gpu 1

echo "GPU 1 complete."
