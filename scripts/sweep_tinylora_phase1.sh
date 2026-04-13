#!/bin/bash
# Phase 1: Spatial placement — which layers to adapt with TinyLoRA.
# Fixed: u=64, r=2, EKF lr=1000. 3 seeds each.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/stream_ekf_tiny_lora.json"

declare -A RUNS
RUNS[p1-input]='[0]'
RUNS[p1-transition]='[1, 2]'
RUNS[p1-output]='[3]'
RUNS[p1-full]='[0, 1, 2, 3]'

for run_name in p1-input p1-transition p1-output p1-full; do
    adapt_layers="${RUNS[$run_name]}"
    tmp_config="/tmp/tinylora_${run_name}.json"

    echo "=========================================="
    echo "Starting ${run_name}  adapt_layers=${adapt_layers}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['wandb_project'] = 'tinylora-placement'
c['training']['dynamics']['svd_rank'] = 2
c['training']['dynamics']['steering_dim'] = 64
c['training']['dynamics']['adapt_layers'] = ${adapt_layers}
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

echo "Phase 1 complete."
