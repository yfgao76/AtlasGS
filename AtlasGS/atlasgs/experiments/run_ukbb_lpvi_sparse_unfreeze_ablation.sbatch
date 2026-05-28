#!/bin/bash
#SBATCH --job-name=ukbb_lpvi_abla
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/ukbb_lpvi_abla_%j.out
#SBATCH --error=logs/ukbb_lpvi_abla_%j.err

set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}
OUT_BASE=${OUT_BASE:-${ROOT}/outputs_contribution/ukbb_z7_lpvi_sparse_unfreeze}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}
SUBJECT_ID=${SUBJECT_ID:-1936135}
FACTOR_Z=${FACTOR_Z:-7}
TRAIN_ITER=${TRAIN_ITER:-4000}
T1GUIDED_ITER=${T1GUIDED_ITER:-2500}
TOPO_ITER=${TOPO_ITER:-2500}
PERSIST_LAMBDA=${PERSIST_LAMBDA:-10}
KEEP_RATIO=${KEEP_RATIO:-0.30}
MIN_KEEP=${MIN_KEEP:-120000}
LPVI_ANCHORS=${LPVI_ANCHORS:-768}
LPVI_MAX_NEW=${LPVI_MAX_NEW:-15000}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-1}

mkdir -p "${ROOT}/logs" "${OUT_BASE}"
cd "${ROOT}"
export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

echo "Host: $(hostname)"
echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-none}"
nvidia-smi || true

run_variant() {
  local name="$1"; shift
  local out_root="${OUT_BASE}/${name}"
  mkdir -p "${out_root}"
  local cmd=(
    python -m atlasgs.experiments.run_subject
    --subject-id "${SUBJECT_ID}"
    --data-root "${DATA_ROOT}"
    --out-root "${out_root}"
    --medgs-root "${MEDGS_ROOT}"
    --target-modalities flair
    --factor-z "${FACTOR_Z}"
    --interp-factor "${FACTOR_Z}"
    --train-iterations "${TRAIN_ITER}"
    --t1guided-iterations "${T1GUIDED_ITER}"
    --run-t1guided-topology
    --t1guided-topology-iterations "${TOPO_ITER}"
    --t1guided-topology-lambda-dssim 0.2
    --t1guided-topology-interp-lambda-dssim -1.0
    --t1guided-topology-persist-lambda "${PERSIST_LAMBDA}"
    --t1guided-topology-preprune-keep-ratio "${KEEP_RATIO}"
    --t1guided-topology-preprune-min-keep "${MIN_KEEP}"
    --t1guided-topology-lpvi-anchor-samples "${LPVI_ANCHORS}"
    --t1guided-topology-lpvi-max-new-points "${LPVI_MAX_NEW}"
    --t1guided-time-init-auto
    --seed "${SEED}"
    --no-run-alpine-a2
    "$@"
  )
  if [[ "${OVERWRITE}" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  echo "[VARIANT ${name}] ${cmd[*]}"
  "${cmd[@]}"
}

# Sparse LPVI while keeping the scaffold geometry frozen.
run_variant sparse_frozen

# Sparse LPVI with a small, anchored xyz update only in the topology branch.
run_variant sparse_unfreeze_xyz \
  --t1guided-topology-no-freeze-xyz \
  --t1guided-topology-xyz-lr 1e-5 \
  --t1guided-topology-lambda-xyz-anchor 0.05 \
  --t1guided-topology-lambda-xyz-zero-mean 0.02 \
  --t1guided-topology-lambda-xyz-max 0.05 \
  --t1guided-topology-xyz-max-mm 0.015 \
  --t1guided-topology-xyz-delta-clamp-mm 0.02 \
  --t1guided-topology-grad-clip 1.0

# Same xyz relaxation plus very small covariance LR to test whether shape updates help.
run_variant sparse_unfreeze_xyzcov \
  --t1guided-topology-no-freeze-xyz \
  --t1guided-topology-no-freeze-cov \
  --t1guided-topology-xyz-lr 1e-5 \
  --t1guided-topology-cov-lr 1e-5 \
  --t1guided-topology-lambda-xyz-anchor 0.05 \
  --t1guided-topology-lambda-xyz-zero-mean 0.02 \
  --t1guided-topology-lambda-xyz-max 0.05 \
  --t1guided-topology-xyz-max-mm 0.015 \
  --t1guided-topology-xyz-delta-clamp-mm 0.02 \
  --t1guided-topology-grad-clip 1.0

python -m atlasgs.eval.analyze_contribution_ablation \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_BASE}/sparse_frozen" \
  --factor-z "${FACTOR_Z}" \
  --out-json "${OUT_BASE}/summary_sparse_frozen_z${FACTOR_Z}.json" \
  --out-csv "${OUT_BASE}/summary_sparse_frozen_z${FACTOR_Z}.csv"
python -m atlasgs.eval.analyze_contribution_ablation \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_BASE}/sparse_unfreeze_xyz" \
  --factor-z "${FACTOR_Z}" \
  --out-json "${OUT_BASE}/summary_sparse_unfreeze_xyz_z${FACTOR_Z}.json" \
  --out-csv "${OUT_BASE}/summary_sparse_unfreeze_xyz_z${FACTOR_Z}.csv"
python -m atlasgs.eval.analyze_contribution_ablation \
  --data-root "${DATA_ROOT}" \
  --out-root "${OUT_BASE}/sparse_unfreeze_xyzcov" \
  --factor-z "${FACTOR_Z}" \
  --out-json "${OUT_BASE}/summary_sparse_unfreeze_xyzcov_z${FACTOR_Z}.json" \
  --out-csv "${OUT_BASE}/summary_sparse_unfreeze_xyzcov_z${FACTOR_Z}.csv"

echo "[DONE] sparse LPVI unfreeze ablation: ${OUT_BASE}"
