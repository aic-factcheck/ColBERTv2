# THIS is sourced by all ctk scripts for PAR6
# initializes environment and sets all parameters

# if PROJECT_DIR is not defined, then expect we are in ${PROJECT_DIR}/slurm
if [[ -z "${PROJECT_DIR}" ]]; then
    export PROJECT_DIR="$(dirname "$(pwd)")"
fi

export PROJECT_DIR="/home/drchajan/devel/python/FC/experimental-martin/ColBERT"

if [ -f "${PROJECT_DIR}/init_environment_amd.sh" ]; then
   source "${PROJECT_DIR}/init_environment_amd.sh"
fi

cd ${PROJECT_DIR}

# problems with BLAS, see: https://github.com/awslabs/autogluon/issues/1020#issuecomment-926089808
export OPENBLAS_NUM_THREADS=1
export GOTO_NUM_THREADS=1
export OMP_NUM_THREADS=1

# ============== Martin's recommendation
export MAXLEN=180
export BS=32
export DEVICES="0"
export NODES=1
# export DIM=128
export DIM=32
export MAXSTEPS=200000
# =======================================

# Default Parameters
# lr 3*10e−6, bs 32, embedd dim 128, 200k iterations/training steps
# new: dim32

export LANG=cs
export DATE=20230220
export ROOT="/mnt/data/factcheck/wiki/$LANG/$DATE/colbert"
export DATA_ROOT="/mnt/data/factcheck/wiki/$LANG/$DATE/paragraphs"

export EXPERIMENT="wiki_${LANG}_${DATE}-${DIM}"

export CHECKPOINT="${ROOT}/${EXPERIMENT}/train.py/${EXPERIMENT}.l2/checkpoints/colbert.dnn"
export TRIPLES="${ROOT}/train-triples_merged_and_shuffled.jsonl"

# ----- CREATE INDEX
export COLLECTION="${ROOT}/collection_filtered.jsonl"
export INDEX_ROOT="${ROOT}/indexes/"
export INDEX_NAME="${EXPERIMENT}.L2.${DIM}x200k"

# ----- CREATE FAISS INDEX
