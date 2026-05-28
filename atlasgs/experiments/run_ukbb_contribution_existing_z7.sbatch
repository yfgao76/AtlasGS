#!/bin/bash
#SBATCH --job-name=ukbb_contrib_existing
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --mem=24G
#SBATCH --output=logs/ukbb_contrib_existing_%j.out
#SBATCH --error=logs/ukbb_contrib_existing_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/ukbb_medgs_all_methods/z7}
ANALYSIS_ROOT=${ANALYSIS_ROOT:-${ROOT}/outputs_contribution/ukbb_z7_topology_latent/existing_full}
FACTOR_Z=${FACTOR_Z:-7}
mkdir -p "${ROOT}/logs" "${ANALYSIS_ROOT}"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

python -m atlasgs.eval.analyze_contribution_ablation \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --factor-z "${FACTOR_Z}" \
  --out-json "${ANALYSIS_ROOT}/contribution_raw_split_z${FACTOR_Z}.json" \
  --out-csv "${ANALYSIS_ROOT}/contribution_raw_split_z${FACTOR_Z}.csv"

echo "[DONE] contribution analysis: ${ANALYSIS_ROOT}"
