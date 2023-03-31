#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

#SBATCH --mem=100G
#SBATCH --partition=amd
#SBATCH --output=logs/ego_%j.out
#SBATCH --error=logs/ego_%j.out

ml SciPy-bundle
ml matplotlib
ml PyTorch

cd $HOME/projects

python -u motion_segmentation/run_ego_supervision.py 4 #$erasor_arg
