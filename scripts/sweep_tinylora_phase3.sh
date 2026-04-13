#!/bin/bash
# Phase 3: EKF LR sweep with best config from Phases 1+2.
# Fixed: r=64, u=16, adapt_layers=[0,1,2,3]. 3 seeds each.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/stream_ekf_tiny_lora.json"
LRS=(1 10 100 500 1000 5000)

for lr in "${LRS[@]}"; do
    run_name="p3-lr${lr}"
    tmp_config="/tmp/tinylora_${run_name}.json"

    echo "=========================================="
    echo "Starting ${run_name}  lr=${lr}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['wandb_project'] = 'tinylora-placement'
c['training']['dynamics']['svd_rank'] = 64
c['training']['dynamics']['steering_dim'] = 16
c['training']['dynamics']['adapt_layers'] = [0, 1, 2, 3]
c['training']['trainer']['lr'] = ${lr}
with open('${tmp_config}', 'w') as f:
    json.dump(c, f, indent=2)
"

    conda run -n max python scripts/train.py \
        --config "${tmp_config}" \
        --run-name "${run_name}" \
        --num-seeds 3

    rm -f "${tmp_config}"
    echo "Done: ${run_name}"
done

echo "Phase 3 complete."
