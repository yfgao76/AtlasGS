#!/bin/bash
#SBATCH --job-name=medgs_ukbb_all
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:l40s:2
#SBATCH --time=168:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/medgs_ukbb_all_%j.out
#SBATCH --error=logs/medgs_ukbb_all_%j.err

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}
OUT_ROOT_BASE=${OUT_ROOT_BASE:-${OUT_ROOT:-${ROOT}/outputs_dataset/ukbb_medgs_all_methods}}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}
CSV=${CSV:-${ROOT}/data/ukbb_medgs/test.csv}

FACTOR_LIST=${FACTOR_LIST:-3,5,7}
TRAIN_ITER=${TRAIN_ITER:-10000}
T1GUIDED_ITER=${T1GUIDED_ITER:-10000}
LATENT_ITER=${LATENT_ITER:-10000}
TOPO_ITER=${TOPO_ITER:-10000}
OUR_ITER=${OUR_ITER:-10000}
T1_REFINE_ITER=${T1_REFINE_ITER:-10000}
TOPO_PERSIST_LAMBDA=${TOPO_PERSIST_LAMBDA:-10}
ALPINE_A2_PRETRAIN=${ALPINE_A2_PRETRAIN:-1000}
ALPINE_A2_FIT=${ALPINE_A2_FIT:-2000}
ALPINE_A2_BATCH=${ALPINE_A2_BATCH:-32768}
ALPINE_A2_CHUNK=${ALPINE_A2_CHUNK:-262144}
SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-1}
MAX_SUBJECTS=${MAX_SUBJECTS:-0}
PARALLEL_SUBJECTS=${PARALLEL_SUBJECTS:-2}
GPU_IDS=${GPU_IDS:-0,1}

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
  --medgs-root "${MEDGS_ROOT}"
  --csv "${CSV}"
  --target-modalities "flair"
  --train-iterations "${TRAIN_ITER}"
  --t1guided-iterations "${T1GUIDED_ITER}"
  --t1guided-latent-iterations "${LATENT_ITER}"
  --t1guided-topology-iterations "${TOPO_ITER}"
  --t1guided-our-iterations "${OUR_ITER}"
  --t1guided-refine-t1
  --t1guided-refine-t1-iterations "${T1_REFINE_ITER}"
  --t1guided-topology-persist-lambda "${TOPO_PERSIST_LAMBDA}"
  --alpine-a2-pretrain-iters "${ALPINE_A2_PRETRAIN}"
  --alpine-a2-fit-iters "${ALPINE_A2_FIT}"
  --alpine-a2-batch-size "${ALPINE_A2_BATCH}"
  --alpine-a2-chunk-size "${ALPINE_A2_CHUNK}"
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

IFS=',' read -r -a factor_arr <<< "${FACTOR_LIST}"
for factor in "${factor_arr[@]}"; do
  factor="$(echo "${factor}" | xargs)"
  [[ -z "${factor}" ]] && continue
  out_root="${OUT_ROOT_BASE}/z${factor}"
  mkdir -p "${out_root}"
  cmd_z=("${cmd[@]}" --out-root "${out_root}" --factor-z "${factor}" --interp-factor "${factor}")
  echo "[CMD z${factor}] ${cmd_z[*]}"
  "${cmd_z[@]}"
done
