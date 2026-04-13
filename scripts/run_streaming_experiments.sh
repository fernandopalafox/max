#!/bin/bash
# Streaming data comparison: EKF vs OGD (last-layer) vs OGD (full network).
# All use latest sampler (single-point updates). Sequential, one job at a time.

set -e
cd "$(dirname "$0")/.."

conda run -n max python scripts/train.py --config stream_ekf.json       --run-name stream-ekf       --num-seeds 3
conda run -n max python scripts/train.py --config stream_ogd.json       --run-name stream-ogd       --num-seeds 3
conda run -n max python scripts/train.py --config stream_ogd_dense.json --run-name stream-ogd-dense --num-seeds 3

echo "All streaming experiments complete."
