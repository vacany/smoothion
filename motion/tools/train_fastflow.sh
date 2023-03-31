#!/bin/bash
#SBATCH --time=4:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

#SBATCH --mem=60G
#SBATCH --partition=amdgpufast
#SBATCH --gres=gpu:1

#SBATCH --error=logs/fastflow_waymo_toy_%a_%j.out
#SBATCH --output=logs/fastflow_waymo_toy_%a_%j.out


ml torchsparse

cd $HOME/motion_supervision

python -u train.py $SLURM_ARRAY_TASK_ID

# this argument will then specify the config file or exp
