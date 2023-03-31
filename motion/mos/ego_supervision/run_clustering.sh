#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8

#SBATCH --mem=100G
#SBATCH --partition=amd
#SBATCH --output=logs/clustering_%j.out
#SBATCH --error=logs/clustering_%j.out

ml SciPy-bundle
ml matplotlib
ml PyTorch

#ml OpenCV/4.5.1-foss-2020b-contrib


cd $HOME/projects

python -u motion_segmentation/ego_supervision/common.py 0
