#!/bin/bash
# LR sweep for ekf_batch trainer, batch_size=8. Runs sequentially, one job at a time.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/cheetah_ekf_batch.json"
LRS=(0.001 0.01 0.1 1.0 10.0 100.0 1000.0)

for lr in "${LRS[@]}"; do
    run_name="batch-ekf-b8-lr${lr}"
    tmp_config="/tmp/ekf_lr_sweep_b8_${lr}.json"

    echo "=========================================="
    echo "Starting LR=${lr}  run=${run_name}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['trainer']['lr'] = ${lr}
c['training']['sampler']['batch_size'] = 8
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
