#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/hcp_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs/hcp_medgs_asl_dwi_z3}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}

SUBJECT_ID=${SUBJECT_ID:-sub-04_ses-01}
FACTOR_Z=${FACTOR_Z:-3}
INTERP_FACTOR=${INTERP_FACTOR:-3}
TRAIN_ITER=${TRAIN_ITER:-30000}
T1GUIDED_ITER=${T1GUIDED_ITER:-10000}
T1G_FEATURE_L2=${T1G_FEATURE_L2:-1e-5}
T1G_TIME_LR=${T1G_TIME_LR:-5e-4}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-0}

if [[ -z "${SUBJECT_ID}" ]]; then
  echo "[ERROR] Set SUBJECT_ID, e.g. SUBJECT_ID=sub-XXXX_ses-YY"
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
  --target-modalities "asl,dwi"
  --factor-z "${FACTOR_Z}"
  --interp-factor "${INTERP_FACTOR}"
  --train-iterations "${TRAIN_ITER}"
  --t1guided-iterations "${T1GUIDED_ITER}"
  --t1guided-feature-l2 "${T1G_FEATURE_L2}"
  --t1guided-time-lr "${T1G_TIME_LR}"
  --t1guided-time-init-auto
  --seed "${SEED}"
  --run-lr-fusion
  --no-run-alpine-a2
  --apply-target-brain-mask
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"

echo "[DONE] HCP one-subject run completed."
