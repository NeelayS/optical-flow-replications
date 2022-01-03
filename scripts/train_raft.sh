#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=raft-run4
#SBATCH --partition=gpu
#SBATCH --mem=24G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=../outs/raft/run4.out

module load cuda/10.2
cd ../
python train.py --train_cfg "./configs/raft_trainer.yaml" \
                --model "RAFT" \
                --log_dir "./logs/raft/run4" \
                --ckpt_dir "./ckpts/raft/run4" \
                # --model_cfg "./configs/models/raft_128.yaml" \
