#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=dicl-run1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=../outs/dicl/run1.out

module load cuda/10.2
cd ../
python train.py --train_cfg "configs/dicl_trainer.yaml" \
                --model "DICL" \
                --log_dir "./logs/dicl/run1" \
                --ckpt_dir "./ckpts/dicl/run1"