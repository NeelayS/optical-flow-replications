#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --job-name=resume-raft-run3
#SBATCH --partition=gpu
#SBATCH --mem=32G
#SBATCH --gres=gpu:v100-sxm2:1
#SBATCH --output=../outs/raft/resume_run3.out

module load cuda/10.2
cd ../
python train.py --train_cfg "./configs/raft_trainer.yaml" \
                --model "RAFT" \
                --log_dir "./logs/raft/run3" \
                --ckpt_dir "./ckpts/raft/run3" \
                --resume True \
                --resume_ckpt "./ckpts/raft/run3/raft_epochs93.pth" \
                --dataset "FlyingChairs" \
