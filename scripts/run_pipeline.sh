#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON:-python}"
device="${DEVICE:-auto}"
output_root="${repository}/artifacts/multimodal/training_runs"
seeds=(11 23 37 51 73)

cd "${repository}"
"${python_bin}" -m src.multimodal.train \
  --config configs/multimodal/deployment.json \
  --seeds "${seeds[@]}" \
  --epochs 200 \
  --patience 30 \
  --batch-size 512 \
  --objective nnpu \
  --transformer-layers 2 \
  --modality-dropout 0 \
  --device "${device}" \
  --network-dir data/processed \
  --network-variant complete \
  --output-root "${output_root}"

for seed in "${seeds[@]}"; do
  "${python_bin}" -m src.multimodal.export_candidates \
    --run-dir "${output_root}/nnpu_seed${seed}" \
    --top-k 10
done
