#!/usr/bin/env bash

## Logging
#SBATCH --job-name=typewriter-fetching
#SBATCH --output=slurm-runs/fetching.txt

## Email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=benjamin.sparks@stud.uni-heidelberg.de

## Resources - Serial execution
#SBATCH --ntasks=1
#SBATCH --time=1:00:00

(cd data/paper-dataset; sh cloner.sh)

source .venv/bin/activate
pip install -r requirements.txt



python TW_extractor.py \
    --o dataset \
    --d data/paper-dataset/Repos \
    --w 10 \
    --c 0