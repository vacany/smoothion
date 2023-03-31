#!/bin/bash
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8

#SBATCH --mem=60G
#SBATCH --partition=amdgpulong
#SBATCH --gres=gpu:1

#SBATCH --error=logs/Flow_gen_%j.out
#SBATCH --output=logs/Flow_gen_%j.out


ml torchsparse

cd $HOME

python -u motion_supervision/generate_priors.py

