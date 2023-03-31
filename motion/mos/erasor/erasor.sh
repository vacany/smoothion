#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

#SBATCH --mem=100G
#SBATCH --partition=amdlong
#SBATCH --output=logs/erasor_%j.out
#SBATCH --error=logs/erasor_%j.out

ml SciPy-bundle
ml matplotlib

cd $HOME/projects
pwd

python -u motion_segmentation/erasor.py 4 #$erasor_arg
