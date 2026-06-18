#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/gbm_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs/gbm_medgs_t2_flair_z7}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}

SUBJECT_ID=${SUBJECT_ID:-}
FACTOR_Z=${FACTOR_Z:-7}
INTERP_FACTOR=${INTERP_FACTOR:-7}
TRAIN_ITER=${TRAIN_ITER:-30000}
T1GUIDED_ITER=${T1GUIDED_ITER:-10000}
T1G_FEATURE_L2=${T1G_FEATURE_L2:-1e-5}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ -z "${SUBJECT_ID}" ]]; then
  echo "[ERROR] Set SUBJECT_ID, e.g. SUBJECT_ID=UPENN-GBM-00022_11"
  exit 1
fi

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

echo "Host: $(hostname)"
echo "Run mode: bash"
nvidia-smi || true

cmd=(
  python -m atlasgs.experiments.run_subject
  --subject-id "${SUBJECT_ID}"
  --data-root "${DATA_ROOT}"
  --out-root "${OUT_ROOT}"
  --medgs-root "${MEDGS_ROOT}"
  --target-modalities "flair,t2"
  --factor-z "${FACTOR_Z}"
  --interp-factor "${INTERP_FACTOR}"
  --train-iterations "${TRAIN_ITER}"
  --t1guided-iterations "${T1GUIDED_ITER}"
  --t1guided-feature-l2 "${T1G_FEATURE_L2}"
  --seed "${SEED}"
  --run-lr-fusion
  --no-run-alpine-a2
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"

echo "[DONE] GBM one-subject run completed."
