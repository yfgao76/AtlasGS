#!/bin/bash
#SBATCH --job-name=medgs_gbm_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:2
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/medgs_gbm_all_%j.out
#SBATCH --error=logs/medgs_gbm_all_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/gbm_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs_dataset/gbm_medgs_all_methods_z7}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}
CSV=${CSV:-${ROOT}/data/gbm_medgs/test.csv}

FACTOR_Z=${FACTOR_Z:-7}
INTERP_FACTOR=${INTERP_FACTOR:-7}
TRAIN_ITER=${TRAIN_ITER:-10000}
T1GUIDED_ITER=${T1GUIDED_ITER:-10000}
LATENT_ITER=${LATENT_ITER:-10000}
TOPO_ITER=${TOPO_ITER:-10000}
OUR_ITER=${OUR_ITER:-10000}
T1_REFINE_ITER=${T1_REFINE_ITER:-10000}
TOPO_PERSIST_LAMBDA=${TOPO_PERSIST_LAMBDA:-10}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-0}
MAX_SUBJECTS=${MAX_SUBJECTS:-0}
PARALLEL_SUBJECTS=${PARALLEL_SUBJECTS:-2}
GPU_IDS=${GPU_IDS:-}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

cmd=(
  python -m atlasgs.experiments.run_all_methods_dataset
  --data-root "${DATA_ROOT}"
  --out-root "${OUT_ROOT}"
  --medgs-root "${MEDGS_ROOT}"
  --csv "${CSV}"
  --target-modalities "flair,t2"
  --factor-z "${FACTOR_Z}"
  --interp-factor "${INTERP_FACTOR}"
  --train-iterations "${TRAIN_ITER}"
  --t1guided-iterations "${T1GUIDED_ITER}"
  --t1guided-latent-iterations "${LATENT_ITER}"
  --t1guided-topology-iterations "${TOPO_ITER}"
  --t1guided-our-iterations "${OUR_ITER}"
  --t1guided-refine-t1
  --t1guided-refine-t1-iterations "${T1_REFINE_ITER}"
  --t1guided-topology-persist-lambda "${TOPO_PERSIST_LAMBDA}"
  --no-run-t1guided-latent
  --no-run-t1guided-topology
  --run-t1guided-our
  --no-run-lr-fusion
  --no-run-alpine-a2
  --parallel-subjects "${PARALLEL_SUBJECTS}"
  --seed "${SEED}"
)

if [[ "${MAX_SUBJECTS}" != "0" ]]; then
  cmd+=(--max-subjects "${MAX_SUBJECTS}")
fi
if [[ -n "${GPU_IDS}" ]]; then
  cmd+=(--gpu-ids "${GPU_IDS}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"
