#!/bin/bash
#SBATCH --job-name=ukbb_anlz
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/ukbb_analyze_extinr_%j.out
#SBATCH --error=logs/ukbb_analyze_extinr_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/ukbb_medgs_all_methods}
INR_BASE=${INR_BASE:-${ROOT}/external_metrics/brain-gs}
FACTORS=${FACTORS:-3,5,7}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

cmd=(
  python -m atlasgs.eval.analyze_ukbb_with_external_inr
  --out-root "${OUT_ROOT}"
  --inr-base "${INR_BASE}"
  --factors "${FACTORS}"
)
echo "[CMD] ${cmd[*]}"
"${cmd[@]}"
