#!/bin/bash
# Phase 2A: Sweep frozen rank r with best layer config from Phase 1.
# Fixed: u=16, adapt_layers=[0], EKF lr=1000. 3 seeds each.

set -e
cd "$(dirname "$0")/.."

BASE_CONFIG="configs/stream_ekf_tiny_lora.json"
RANKS=(1 2 4 8)

for r in "${RANKS[@]}"; do
    run_name="p2a-r${r}"
    tmp_config="/tmp/tinylora_${run_name}.json"

    echo "=========================================="
    echo "Starting ${run_name}  r=${r}"
    echo "=========================================="

    python3 -c "
import json
with open('${BASE_CONFIG}') as f:
    c = json.load(f)
c['training']['wandb_project'] = 'tinylora-placement'
c['training']['dynamics']['svd_rank'] = ${r}
c['training']['dynamics']['steering_dim'] = 16
c['training']['dynamics']['adapt_layers'] = [0, 1, 2, 3]
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

echo "Phase 2A complete."
