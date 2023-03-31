#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4

#SBATCH --mem=60G
#SBATCH --partition=amd
#SBATCH --output=logs/removert_post_%j.out
#SBATCH --error=logs/removert_post_%j.out

#ml SciPy-bundle
#ml matplotlib
#ml PyTorch

ml OpenCV/4.5.1-foss-2020b-contrib


cd $HOME/projects

python -u motion_segmentation/removert/transform_data.py 12
