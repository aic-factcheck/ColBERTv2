#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amd
#SBATCH --mem=196G
#SBATCH --out=../logs/rest_colbert.%j.out

# if PROJECT_DIR is not defined, then expect we are in ${PROJECT_DIR}/slurm
if [[ -z "${PROJECT_DIR}" ]]; then
    export PROJECT_DIR="$(dirname "$(pwd)")"
fi

cd ${PROJECT_DIR}

ml Anaconda3
eval "$(conda shell.bash hook)"
conda activate colbert_cpu

export PYTHONPATH=.:$PYTHONPATH
python scripts/rest_api.py cfg/rest_api/wiki_en_20230220.json



