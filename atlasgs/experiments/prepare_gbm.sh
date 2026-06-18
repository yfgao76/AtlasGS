#!/usr/bin/env bash

set -eo pipefail

SRC_ROOT=${SRC_ROOT:?Set SRC_ROOT to the GBM source directory}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/data/gbm_medgs}
PROJECT_ROOT=${PROJECT_ROOT:-${ROOT}}
WORKERS=${WORKERS:-12}

mkdir -p "${ROOT}/logs"
mkdir -p "${OUT_ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
conda activate medgs
set -u

echo "Host: $(hostname)"
echo "GPUs: ${GPU_COUNT:-auto}"
echo "Workers: ${WORKERS}"

cd "${PROJECT_ROOT}"
python -m atlasgs.ops.prepare_gbm_medgs \
  --src "${SRC_ROOT}" \
  --out "${OUT_ROOT}" \
  --seed 0 \
  --train-ratio 0.8 \
  --workers "${WORKERS}"

python -m atlasgs.ops.batch_degrade_modalities \
  --data-root "${OUT_ROOT}" \
  --modalities t2,flair \
  --factors 7 \
  --sigma-z 1.0 \
  --mode avgpool

echo "Done. Output: ${OUT_ROOT}"
