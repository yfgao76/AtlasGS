#!/usr/bin/env bash

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_contribution/ukbb_z7_topology_latent/seed_cost_pilot_2500}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}
CSV=${CSV:?Set CSV to a subject list with a subject_id column}
MAX_SUBJECTS=${MAX_SUBJECTS:-2}
SEEDS=${SEEDS:-0,1,2}
GPU_IDS=${GPU_IDS:-0,1}
ITERATIONS=${ITERATIONS:-2500}
FACTOR_Z=${FACTOR_Z:-7}
PERSIST_LAMBDA=${PERSIST_LAMBDA:-10}

mkdir -p "${ROOT}/logs" "${OUT_ROOT}"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

echo "Host: $(hostname)"
echo "Run mode: bash"
nvidia-smi || true

python -m atlasgs.experiments.run_ukbb_contribution_pilot \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_ROOT}" \
  --medgs-root "${MEDGS_ROOT}" \
  --csv "${CSV}" \
  --max-subjects "${MAX_SUBJECTS}" \
  --seeds "${SEEDS}" \
  --gpu-ids "${GPU_IDS}" \
  --factor-z "${FACTOR_Z}" \
  --iterations "${ITERATIONS}" \
  --persist-lambda "${PERSIST_LAMBDA}"

IFS=',' read -r -a seed_arr <<< "${SEEDS}"
for seed in "${seed_arr[@]}"; do
  seed="$(echo "${seed}" | xargs)"
  [[ -z "${seed}" ]] && continue
  seed_root="${OUT_ROOT}/seed${seed}"
  if [[ -d "${seed_root}" ]]; then
    python -m atlasgs.eval.analyze_contribution_ablation \
      --data-root "${DATA_ROOT}" \
      --out-root "${seed_root}" \
      --factor-z "${FACTOR_Z}" \
      --out-json "${OUT_ROOT}/contribution_seed${seed}_z${FACTOR_Z}.json" \
      --out-csv "${OUT_ROOT}/contribution_seed${seed}_z${FACTOR_Z}.csv"
  fi
done

echo "[DONE] contribution pilot: ${OUT_ROOT}"
