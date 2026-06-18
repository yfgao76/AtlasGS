#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ATLASGS_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-${ROOT}/data/ukbb_medgs}
OUT_ROOT=${OUT_ROOT:-${ROOT}/outputs/ukbb_medgs_all_methods_toy_z7}
MEDGS_ROOT=${MEDGS_ROOT:-${ROOT}/external/MedGS}

SUBJECT_ID=${SUBJECT_ID:-1936135}
TARGET_MODALITIES=${TARGET_MODALITIES:-flair}
FACTOR_Z=${FACTOR_Z:-7}
INTERP_FACTOR=${INTERP_FACTOR:-7}

TRAIN_ITER=${TRAIN_ITER:-5000}
T1GUIDED_ITER=${T1GUIDED_ITER:-1500}
LATENT_ITER=${LATENT_ITER:-1500}
TOPO_ITER=${TOPO_ITER:-1500}
OUR_ITER=${OUR_ITER:-1500}

# Keep topology persist scale aligned with prior run medgs_ukbb_topo_3681591.out.
TOPO_PERSIST_LAMBDA=${TOPO_PERSIST_LAMBDA:-10}
TOPO_DSSIM=${TOPO_DSSIM:-0.2}
TOPO_INTERP_DSSIM=${TOPO_INTERP_DSSIM:--1.0}

ALPINE_PRETRAIN=${ALPINE_PRETRAIN:-300}
ALPINE_FIT=${ALPINE_FIT:-600}
ALPINE_BATCH=${ALPINE_BATCH:-16384}
ALPINE_CHUNK=${ALPINE_CHUNK:-131072}

SEED=${SEED:-0}
OVERWRITE=${OVERWRITE:-1}

mkdir -p "${ROOT}/logs"
cd "${ROOT}"

export BASHRCSOURCED=1
set +u
source ~/.bashrc >/dev/null 2>&1 || true
set -u
conda activate medgs

echo "Host: $(hostname)"
echo "Run mode: bash"
nvidia-smi || true

cmd=(
  python -m atlasgs.experiments.run_subject
  --subject-id "${SUBJECT_ID}"
  --data-root "${DATA_ROOT}"
  --out-root "${OUT_ROOT}"
  --medgs-root "${MEDGS_ROOT}"
  --target-modalities "${TARGET_MODALITIES}"
  --factor-z "${FACTOR_Z}"
  --interp-factor "${INTERP_FACTOR}"
  --train-iterations "${TRAIN_ITER}"
  --t1guided-iterations "${T1GUIDED_ITER}"
  --run-t1guided-latent
  --t1guided-latent-iterations "${LATENT_ITER}"
  --run-t1guided-topology
  --t1guided-topology-iterations "${TOPO_ITER}"
  --t1guided-topology-lambda-dssim "${TOPO_DSSIM}"
  --t1guided-topology-interp-lambda-dssim "${TOPO_INTERP_DSSIM}"
  --t1guided-topology-persist-lambda "${TOPO_PERSIST_LAMBDA}"
  --run-t1guided-our
  --t1guided-our-iterations "${OUR_ITER}"
  --run-lr-fusion
  --alpine-a2-pretrain-iters "${ALPINE_PRETRAIN}"
  --alpine-a2-fit-iters "${ALPINE_FIT}"
  --alpine-a2-batch-size "${ALPINE_BATCH}"
  --alpine-a2-chunk-size "${ALPINE_CHUNK}"
  --seed "${SEED}"
)

if [[ "${OVERWRITE}" == "1" ]]; then
  cmd+=(--overwrite)
fi

echo "[CMD] ${cmd[*]}"
"${cmd[@]}"

echo "[DONE] UKBB toy-all-methods run finished."
