#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/hcp_medgs_all_methods_z3}
NORMAL_THR=${NORMAL_THR:-0.95}
SA_ASL_JSON=${SA_ASL_JSON:-${ROOT}/external_metrics/brain-gs/sa_inr_hcp_asl_ds3/eval_SA_INR_asl_ds3.json}
SA_DWI_JSON=${SA_DWI_JSON:-${ROOT}/external_metrics/brain-gs/sa_inr_hcp_dwi_ds3/eval_SA_INR_dwi_ds3.json}
SKIP_BSPLINE=${SKIP_BSPLINE:-0}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

cmd=(
  python -m atlasgs.eval.analyze_hcp_normal_ssim_with_external_inr
  --out-root "${OUT_ROOT}"
  --modalities "asl,dwi"
  --normal-threshold "${NORMAL_THR}"
  --sa-asl-json "${SA_ASL_JSON}"
  --sa-dwi-json "${SA_DWI_JSON}"
)
if [[ "${SKIP_BSPLINE}" == "1" ]]; then
  cmd+=(--skip-bspline)
fi

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"
