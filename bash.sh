#!/bin/bash

#SBATCH -A deep_learning
#SBATCH -n 2
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=2048 #SBATCH --tmp=2048
#SBATCH --job-name=dl
#SBATCH --output=dl-approachX.out #SBATCH --error=dl-approachX.err #SBATCH --open-mode=truncate