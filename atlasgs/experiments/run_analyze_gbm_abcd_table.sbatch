#!/bin/bash
#SBATCH --job-name=tbl_gbm_abcd
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tbl_gbm_abcd_%j.out
#SBATCH --error=logs/tbl_gbm_abcd_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
GBM_DATA_ROOT=${GBM_DATA_ROOT:-${ROOT}/data/gbm_medgs}
GBM_OUT_ROOT=${GBM_OUT_ROOT:-${ROOT}/outputs_dataset/gbm_medgs_all_methods_z7}
GBM_TEST_CSV=${GBM_TEST_CSV:-${GBM_DATA_ROOT}/test.csv}
GBM_SA_FLAIR_ROOT=${GBM_SA_FLAIR_ROOT:-${ROOT}/external_metrics/brain-gs/sa_inr_gbm_flair_ds7/predictions}
GBM_SA_T2_ROOT=${GBM_SA_T2_ROOT:-${ROOT}/external_metrics/brain-gs/sa_inr_gbm_t2_ds7/predictions}
GBM_FACTOR_Z=${GBM_FACTOR_Z:-7}

HCP_DATA_ROOT=${HCP_DATA_ROOT:-${ROOT}/data/hcp_medgs}
HCP_OUT_ROOT=${HCP_OUT_ROOT:-${ROOT}/outputs_dataset/hcp_medgs_all_methods_z3}
HCP_TEST_CSV=${HCP_TEST_CSV:-${HCP_DATA_ROOT}/test.csv}
HCP_SA_DWI_ROOT=${HCP_SA_DWI_ROOT:-${ROOT}/external_metrics/brain-gs/sa_inr_hcp_dwi_ds3/predictions}
HCP_SA_ASL_ROOT=${HCP_SA_ASL_ROOT:-${ROOT}/external_metrics/brain-gs/sa_inr_hcp_asl_ds3/predictions}
HCP_FACTOR_Z=${HCP_FACTOR_Z:-3}

OUT_JSON=${OUT_JSON:-${ROOT}/outputs_dataset/gbm_abcd_table_metrics.json}
OUT_CSV=${OUT_CSV:-${ROOT}/outputs_dataset/gbm_abcd_table_metrics.csv}
NUM_WORKERS=${NUM_WORKERS:-8}
EXCLUDE_SUBJECT_SUBSTRINGS=${EXCLUDE_SUBJECT_SUBSTRINGS:-}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

CMD=(
  python -m atlasgs.eval.analyze_gbm_abcd_table
  --gbm-data-root "${GBM_DATA_ROOT}"
  --gbm-out-root "${GBM_OUT_ROOT}"
  --gbm-test-csv "${GBM_TEST_CSV}"
  --gbm-sa-flair-root "${GBM_SA_FLAIR_ROOT}"
  --gbm-sa-t2-root "${GBM_SA_T2_ROOT}"
  --gbm-factor-z "${GBM_FACTOR_Z}"
  --hcp-data-root "${HCP_DATA_ROOT}"
  --hcp-out-root "${HCP_OUT_ROOT}"
  --hcp-test-csv "${HCP_TEST_CSV}"
  --hcp-sa-dwi-root "${HCP_SA_DWI_ROOT}"
  --hcp-sa-asl-root "${HCP_SA_ASL_ROOT}"
  --hcp-factor-z "${HCP_FACTOR_Z}"
  --num-workers "${NUM_WORKERS}"
  --exclude-subject-substrings "${EXCLUDE_SUBJECT_SUBSTRINGS}"
  --out-json "${OUT_JSON}"
  --out-csv "${OUT_CSV}"
)

echo "[CMD] ${CMD[*]}"
"${CMD[@]}"
