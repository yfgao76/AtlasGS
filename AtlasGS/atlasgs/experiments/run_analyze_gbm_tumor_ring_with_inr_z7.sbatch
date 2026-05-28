#!/bin/bash
#SBATCH --job-name=gbm_roi_z7
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/gbm_roi_z7_%j.out
#SBATCH --error=logs/gbm_roi_z7_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/gbm_medgs_all_methods_z7}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/gbm_medgs}
INR_FLAIR_ROOT=${INR_FLAIR_ROOT:-${ROOT}/external_metrics/brain-gs/inr_gbm_flair_ds7}
INR_T2_ROOT=${INR_T2_ROOT:-${ROOT}/external_metrics/brain-gs/inr_gbm_t2_ds7}
FACTOR_Z=${FACTOR_Z:-7}
RING_INNER_MM=${RING_INNER_MM:-5}
RING_OUTER_MM=${RING_OUTER_MM:-10}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

python -m atlasgs.eval.analyze_gbm_tumor_roi_with_inr \
  --out-root "${OUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --inr-flair-root "${INR_FLAIR_ROOT}" \
  --inr-t2-root "${INR_T2_ROOT}" \
  --factor-z "${FACTOR_Z}" \
  --ring-inner-mm "${RING_INNER_MM}" \
  --ring-outer-mm "${RING_OUTER_MM}"
