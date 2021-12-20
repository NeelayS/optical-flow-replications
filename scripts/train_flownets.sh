#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=flownets-run1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=outs/flownets/run1.out

module load cuda/10.2
cd ../
python train.py --train_cfg "configs/trainers/flownets_trainer.yaml" \
                --model "FlowNetS" \
                --log_dir "./logs/flownets/run1" \
                --ckpt_dir "./ckpts/flownets/run1"