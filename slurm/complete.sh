#!/usr/bin/env bash

## Logging
#SBATCH --job-name=typewriter-training
#SBATCH --output=slurm-runs/training.txt

## Email
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=benjamin.sparks@stud.uni-heidelberg.de

## Resources - Serial execution, with max 8 cores per task
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00




source .venv/bin/activate
pip install -r requirements.txt

python main_TW.py