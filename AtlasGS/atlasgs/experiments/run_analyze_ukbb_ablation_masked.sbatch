#!/bin/bash
#SBATCH --job-name=ukbb_abla_mask
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ukbb_abla_mask_%j.out
#SBATCH --error=logs/ukbb_abla_mask_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/ukbb_medgs_all_methods}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}

OUT_JSON=${OUT_JSON:-${OUT_ROOT}/ukbb_ablation_masked_summary.json}
OUT_CSV=${OUT_CSV:-${OUT_ROOT}/ukbb_ablation_masked_summary.csv}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

python -m atlasgs.eval.analyze_ukbb_ablation_masked \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --factors "3,5,7" \
  --out-json "${OUT_JSON}" \
  --out-csv "${OUT_CSV}"

