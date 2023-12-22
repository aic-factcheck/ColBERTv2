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

export ROOT="/mnt/data/cro/factcheck/v1/colbertv2"
export DATA_ROOT="/mnt/data/cro/factcheck/v1"

export EXPERIMENT="cro_v1-${DIM}"

export CHECKPOINT="${ROOT}${EXPERIMENT}/train.py/${EXPERIMENT}.l2/checkpoints/colbert.dnn"
export TRIPLES="${DATA_ROOT}/colbert/train-triples_merged_and_shuffled.tsv"

# ----- CREATE INDEX
export COLLECTION="${DATA_ROOT}/colbert/collection_filtered.tsv"
export INDEX_ROOT="${ROOT}indexes/"
export INDEX_NAME="${EXPERIMENT}.L2.${DIM}x200k"

# ----- CREATE FAISS INDEX

# ----- RETRIEVE
export SPLIT="test"
export QUERIES="${DATA_ROOT}/ctk-data/${SPLIT}_queries.tsv"

# ----- RERANK
export TOPK="${ROOT}${EXPERIMENT}/retrieve.py/test/unordered.tsv"

# ----- CONVERT PREDICTIONS
export CLAIMS="${DATA_ROOT}/ctk-data/${SPLIT}.jsonl"
export OLD2NEWID="${DATA_ROOT}/interim/old-id2new-id.tsv"
export RANKING="${ROOT}${EXPERIMENT}/retrieve.py/2023-01-10_19.42.03/ranking.tsv"
export OUTPUT="${ROOT}/predictions/${SPLIT}_colbert_${EXPERIMENT}_k500.jsonl"