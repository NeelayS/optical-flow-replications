#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=dicl-run1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=outs/dicl/run1.out

module load cuda/10.2
cd ..
python dicl.py