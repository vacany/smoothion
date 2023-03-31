#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

#SBATCH --mem=200G
#SBATCH --partition=amd
#SBATCH --output=logs/ego_spread.out
#SBATCH --error=logs/ego_spread.out

ml PyTorch
ml matplotlib

cd $HOME/projects

python -u motion_segmentation/run_dynamic_detection.py
