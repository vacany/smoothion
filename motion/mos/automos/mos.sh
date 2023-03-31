#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=5

#SBATCH --mem=60G
#SBATCH --partition=amdfast
#SBATCH --output=logs/mos_%j.out
#SBATCH --error=logs/mos_%j.out

ml SciPy-bundle
ml matplotlib
ml tqdm

cd $HOME/projects
pwd

python -u motion_segmentation/mos.py 4 #$erasor_arg
