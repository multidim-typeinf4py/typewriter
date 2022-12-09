#!/usr/bin/env bash

## Logging
#SBATCH --job-name=typewriter-training
#SBATCH --output=slurm-runs/training.txt

## Email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=benjamin.sparks@stud.uni-heidelberg.de

## Execution Resources - Serial execution
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1


## Time & Space Resources
#SBATCH --mem=128000
#SBATCH --time=4:00:00

source .venv/bin/activate
pip install -r requirements.txt

python TW_model.py --o dataset