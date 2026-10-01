#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {s|b|l|g} DATA_DIR OUTPUT_DIR" >&2
  exit 2
fi

variant="$1"
data_dir="$2"
output_dir="$3"

case "$variant" in
  s) backbone=dinov2_vits14; layer=11 ;;
  b) backbone=dinov2_vitb14; layer=11 ;;
  l) backbone=dinov2_vitl14; layer=23 ;;
  g) backbone=dinov2_vitg14; layer=39 ;;
  *) echo "unknown variant: $variant" >&2; exit 2 ;;
esac

python_bin="${PYTHON_BIN:-python}"
gpu_ids="${GPU_IDS:-0,1,2,3}"
nproc="${NPROC:-4}"
batch_size="${BATCH_SIZE:-256}"
steps="${STEPS:-200000}"

export CUDA_VISIBLE_DEVICES="$gpu_ids"
exec "$python_bin" -m torch.distributed.run --standalone --nproc_per_node="$nproc" \
  -m coverlock.train \
  --variant "$variant" --data "$data_dir" --output "$output_dir" \
  --backbone "$backbone" --layer "$layer" --batch-size "$batch_size" \
  --steps "$steps" --precision bf16

