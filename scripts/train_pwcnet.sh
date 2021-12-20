#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=pwcnet-run1
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=outs/pwcnet/run1.out

module load cuda/10.2
cd ../
python train.py --train_cfg "configs/trainers/pwcnet_trainer.yaml" \
                --model "PWCNet" \
                --log_dir "./logs/pwcnet/run1" \
                --ckpt_dir "./ckpts/pwcnet/run1"