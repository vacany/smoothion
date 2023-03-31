#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=10

#SBATCH --mem=200G
#SBATCH --partition=amd
#SBATCH --output=logs/ego_%j.out
#SBATCH --error=logs/ego_%j.out

ml SciPy-bundle
ml matplotlib
ml PyTorch

cd $HOME/projects

python -u motion_segmentation/ego_supervision/ego_supervision.py 4
