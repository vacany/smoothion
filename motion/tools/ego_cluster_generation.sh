#!/bin/bash
#SBATCH --time=24:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8

#SBATCH --mem=60G
#SBATCH --partition=amd

#SBATCH --error=logs/instance_%j.out
#SBATCH --output=logs/instance_%j.out


ml PyTorch3D/0.7.1-foss-2021b-CUDA-11.4.1
ml OpenCV/4.5.5-foss-2021b-CUDA-11.4.1-contrib

cd $HOME

python -u motion_supervision/ego_loss.py

