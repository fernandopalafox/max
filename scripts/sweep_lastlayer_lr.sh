#!/bin/bash
# LR sweep for EKF + dense_last_layer dynamics.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/stream_ekf.json"
LRS=(1 10 100 500 1000 5000)

for lr in "${LRS[@]}"; do
    run_name="lastlayer-lr${lr}"
    tmp_config="/tmp/lastlayer_lr${lr}.json"

    echo "=========================================="
    echo "Starting ${run_name}  lr=${lr}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['wandb_project'] = 'lastlayer-sweep'
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

echo "Last-layer LR sweep complete."
