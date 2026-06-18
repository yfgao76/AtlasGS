#!/usr/bin/env bash

set -eo pipefail

SRC_ROOT=${SRC_ROOT:?Set SRC_ROOT to the HCP/FOMO source directory}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/data/hcp_medgs}
PROJECT_ROOT=${PROJECT_ROOT:-${ROOT}}
SYNTHSTRIP_SIF=${SYNTHSTRIP_SIF:-}
WORKERS=${WORKERS:-16}

ALIGN_METHOD=${ALIGN_METHOD:-rigid}
TRAIN_RATIO=${TRAIN_RATIO:-0.5}
SEED=${SEED:-0}
DWI_PREF=${DWI_PREF:-900,2000,0}
DEGRADE_FACTORS=${DEGRADE_FACTORS:-3}
DEGRADE_MODALITIES=${DEGRADE_MODALITIES:-dwi,asl}
SIGMA_Z=${SIGMA_Z:-1.0}
DEGRADE_MODE=${DEGRADE_MODE:-avgpool}
ROTATE_CCW_90=${ROTATE_CCW_90:-1}
SYNTHSTRIP_GPU=${SYNTHSTRIP_GPU:-0}
NO_SKULLSTRIP=${NO_SKULLSTRIP:-0}
MAX_CASES=${MAX_CASES:-0}
ALLOW_MISSING_ASL=${ALLOW_MISSING_ASL:-0}
ALLOW_MISSING_DWI=${ALLOW_MISSING_DWI:-0}

mkdir -p "${ROOT}/logs"
mkdir -p "${OUT_ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
if [[ -n "${SYNTHSTRIP_SIF}" ]] && command -v module >/dev/null 2>&1; then
  module load "${APPTAINER_MODULE:-apptainer/1.0.1}" >/dev/null 2>&1 || true
fi
conda activate medgs
set -u

echo "Host: $(hostname)"
echo "Workers: ${WORKERS}"
echo "ALIGN_METHOD=${ALIGN_METHOD}"
echo "DWI_PREF=${DWI_PREF}"
echo "OUT_ROOT=${OUT_ROOT}"

cd "${PROJECT_ROOT}"

CMD=(
  python -m atlasgs.ops.prepare_hcp_medgs
  --src "${SRC_ROOT}"
  --out "${OUT_ROOT}"
  --seed "${SEED}"
  --train-ratio "${TRAIN_RATIO}"
  --align-method "${ALIGN_METHOD}"
  --dwi-pref "${DWI_PREF}"
  --degrade-factors "${DEGRADE_FACTORS}"
  --degrade-modalities "${DEGRADE_MODALITIES}"
  --sigma-z "${SIGMA_Z}"
  --degrade-mode "${DEGRADE_MODE}"
  --max-cases "${MAX_CASES}"
)

if [[ -n "${SYNTHSTRIP_SIF}" ]]; then
  CMD+=(--apptainer-sif "${SYNTHSTRIP_SIF}")
fi

if [[ "${ROTATE_CCW_90}" == "1" ]]; then
  CMD+=(--rotate-ccw-90)
else
  CMD+=(--no-rotate-ccw-90)
fi

if [[ "${SYNTHSTRIP_GPU}" == "1" ]]; then
  CMD+=(--synthstrip-gpu)
fi

if [[ "${NO_SKULLSTRIP}" == "1" ]]; then
  CMD+=(--no-skullstrip)
fi

if [[ "${ALLOW_MISSING_ASL}" == "1" ]]; then
  CMD+=(--allow-missing-asl)
fi

if [[ "${ALLOW_MISSING_DWI}" == "1" ]]; then
  CMD+=(--allow-missing-dwi)
fi

"${CMD[@]}"

echo "Done. Output: ${OUT_ROOT}"
