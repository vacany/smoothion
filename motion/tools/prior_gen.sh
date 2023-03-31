#!/bin/bash
#SBATCH --time=72:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6

#SBATCH --mem=40G
#SBATCH --partition=amdlong

#SBATCH --error=logs/data_preprocess_%a.out
#SBATCH --output=logs/data_preprocess_%a.out


ml torchsparse

cd $HOME

python -u motion_supervision/preprocess_data.py $SLURM_ARRAY_TASK_ID
