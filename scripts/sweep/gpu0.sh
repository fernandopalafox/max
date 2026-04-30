#!/bin/bash
set -e
cd /home/fp5275/repos/max

# === Phase 1: dense pretrain seed=46 run_3, then finetune dynamics r1 ===

python3 -c "
import json
with open('configs/pretrain_dense_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 46
c['training']['save_dir'] = './data/models/cheetah/baseline_new/run_3'
with open('/tmp/gpu0_dense_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu0_dense_r3.json --run-name tdmpc2_r3 --gpu 0

DENSE_CKPT=$(find data/models/cheetah/baseline_new/run_3 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_full_dynamics_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu0_adapt_dynamics_r1.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$DENSE_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu0_adapt_dynamics_r1.json --run-name dynamics_pt_r1 --gpu 0

# === Phase 2, seed=46, run 3 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 46
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_3'
with open('/tmp/gpu0_loraxs_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu0_loraxs_r3.json --run-name loraxs_tdmpc2_r3 --gpu 0

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_3 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu0_adapt_loraxs_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu0_adapt_loraxs_r3.json --run-name loraxs_pt_r3 --gpu 0

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 46
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_3'
with open('/tmp/gpu0_fomaml_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu0_fomaml_r3.json --run-name loraxs_fomaml_r3 --gpu 0

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_3 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu0_adapt_fomaml_r3.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu0_adapt_fomaml_r3.json --run-name loraxs_pt_fomaml_r3 --gpu 0

# === Phase 2, seed=47, run 4 ===

python3 -c "
import json
with open('configs/pretrain_loraxs_tdmpc2.json') as f:
    c = json.load(f)
c['training']['seed'] = 47
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs/run_4'
with open('/tmp/gpu0_loraxs_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu0_loraxs_r4.json --run-name loraxs_tdmpc2_r4 --gpu 0

LORAXS_CKPT=$(find data/models/cheetah/baseline_loraxs/run_4 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu0_adapt_loraxs_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$LORAXS_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu0_adapt_loraxs_r4.json --run-name loraxs_pt_r4 --gpu 0

python3 -c "
import json
with open('configs/pretrain_loraxs_bgd_fomaml.json') as f:
    c = json.load(f)
c['training']['seed'] = 47
c['training']['save_dir'] = './data/models/cheetah/baseline_loraxs_fomaml_bgd/run_4'
with open('/tmp/gpu0_fomaml_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
"
conda run -n max python scripts/train.py --config /tmp/gpu0_fomaml_r4.json --run-name loraxs_fomaml_r4 --gpu 0

FOMAML_CKPT=$(find data/models/cheetah/baseline_loraxs_fomaml_bgd/run_4 -name "final.pkl" | sort | tail -1)
python3 -c "
import json, sys
with open('configs/adapt_loraxs_pt_fomaml.json') as f:
    c = json.load(f)
c['training']['pretrained_path'] = sys.argv[1]
with open('/tmp/gpu0_adapt_fomaml_r4.json', 'w') as f:
    json.dump(c, f, indent=2)
" "$FOMAML_CKPT"
conda run -n max python scripts/train.py --config /tmp/gpu0_adapt_fomaml_r4.json --run-name loraxs_pt_fomaml_r4 --gpu 0

echo "GPU 0 complete."
