#!/bin/bash
# High-LR sweep for ekf_batch, batch_size=1. Runs sequentially.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/cheetah_ekf_batch.json"
LRS=(3000 10000 30000 100000)

for lr in "${LRS[@]}"; do
    run_name="batch-ekf-lr${lr}"
    tmp_config="/tmp/ekf_lr_high_${lr}.json"

    echo "=========================================="
    echo "Starting LR=${lr}  run=${run_name}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['trainer']['lr'] = ${lr}
c['training']['sampler']['batch_size'] = 1
with open('${tmp_config}', 'w') as f:
    json.dump(c, f, indent=2)
"

    conda run -n max python scripts/train.py \
        --config "${tmp_config}" \
        --run-name "${run_name}" \
        --num-seeds 3

    rm -f "${tmp_config}"
    echo "Done: LR=${lr}"
done

echo "Sweep complete."
