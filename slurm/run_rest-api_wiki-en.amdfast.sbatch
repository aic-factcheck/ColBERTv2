#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=4
#SBATCH --partition=amdfast
#SBATCH --mem=160G
#SBATCH --out=../logs/rest_colbert_en.%j.out

# if PROJECT_DIR is not defined, then expect we are in ${PROJECT_DIR}/slurm
if [[ -z "${PROJECT_DIR}" ]]; then
    export PROJECT_DIR="$(dirname "$(pwd)")"
fi

if [ -f "${PROJECT_DIR}/init_environment_hflarge_amd.sh" ]; then
    source "${PROJECT_DIR}/init_environment_hflarge_amd.sh"
fi

cd ${PROJECT_DIR}
pwd

ml GCC/11.2.0

export PYTHONPATH=.:$PYTHONPATH
# python scripts/rest_api.py cfg/rest_api/wiki_en_20230220.json
# python scripts/rest_api.py cfg/rest_api/wiki_en_20240201.json
python scripts/rest_api.py cfg/rest_api/wiki_en_20240201_en_500tokens.json



