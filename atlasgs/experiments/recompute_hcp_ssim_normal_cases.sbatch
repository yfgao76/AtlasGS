#!/bin/bash
#SBATCH --job-name=hcp_ssim_fix
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/hcp_ssim_fix_%j.out
#SBATCH --error=logs/hcp_ssim_fix_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/hcp_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/hcp_medgs_all_methods_z3}
TEST_CSV=${TEST_CSV:-$DATA_ROOT/test.csv}
FACTOR_Z=${FACTOR_Z:-3}
NORMAL_THRESHOLD=${NORMAL_THRESHOLD:-0.95}
EXCLUDE_SUBSTR=${EXCLUDE_SUBSTR:-sub-06}

OUT_JSON=${OUT_JSON:-$OUT_ROOT/ssim_recomputed_normal_cases_medgs_ours.json}
OUT_CSV=${OUT_CSV:-$OUT_ROOT/ssim_recomputed_normal_cases_medgs_ours.csv}

mkdir -p "$ROOT/logs"
cd "$ROOT"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

cmd=(
  python -m atlasgs.eval.recompute_hcp_ssim_medgs_ours
  --data-root "$DATA_ROOT"
  --out-root "$OUT_ROOT"
  --test-csv "$TEST_CSV"
  --factor-z "$FACTOR_Z"
  --normal-threshold "$NORMAL_THRESHOLD"
  --exclude-subject-substrings "$EXCLUDE_SUBSTR"
  --modalities "dwi,asl"
  --methods "medgs_single,ours"
  --out-json "$OUT_JSON"
  --out-csv "$OUT_CSV"
)

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"

echo "[DONE] Wrote: $OUT_JSON"
echo "[DONE] Wrote: $OUT_CSV"
