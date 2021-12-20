#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=raft-run2
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=outs/raft/run2.out

module load cuda/10.2
cd ../
python train.py --train_cfg "configs/trainers/raft_trainer.yaml" \
                --model "RAFT" \
                --log_dir "./logs/raft/run2" \
                --ckpt_dir "./ckpts/raft/run2"