#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-a5000-q
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=train.out

#sbatch train_slurm.sh

source .venv/bin/activate
python3 train.py