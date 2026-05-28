#!/bin/bash
#SBATCH --job-name=ukbb_overlay_z7
#SBATCH --output=logs/ukbb_overlay_z7_%j.out
#SBATCH --error=logs/ukbb_overlay_z7_%j.err
#SBATCH --partition=defq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

set -euo pipefail

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
cd "${ROOT}"

OUT_OVERLAY_ROOT=${OUT_OVERLAY_ROOT:-${ROOT}/outputs_dataset/ukbb_medgs_all_methods_overlays_z7_orth}
INR_ROOT=${INR_ROOT:-${ROOT}/external_metrics/brain-gs/inr_ukbb_ds7}
SA_ROOT=${SA_ROOT:-${ROOT}/external_metrics/brain-gs/sa_inr_ukbb_ds7}
mkdir -p "${OUT_OVERLAY_ROOT}"

python -m atlasgs.eval.make_ukbb_z7_orth_overlays \
  --data-root "${ROOT}/data/ukbb_medgs" \
  --out-root "${ROOT}/outputs_dataset/ukbb_medgs_all_methods" \
  --factor-z 7 \
  --overlay-root "${OUT_OVERLAY_ROOT}" \
  --inr-root "${INR_ROOT}" \
  --sa-root "${SA_ROOT}" \
  --overwrite \
  --dpi 260 \
  --panel-width 4.4 \
  --panel-height 4.4 \
  --summary-json "${OUT_OVERLAY_ROOT}/overlay_orth_all_summary.json"
