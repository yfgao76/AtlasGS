#!/usr/bin/env bash

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
