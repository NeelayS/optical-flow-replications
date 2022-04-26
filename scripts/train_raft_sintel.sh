#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft-default-sintel
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/sintel-run1.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_sintel_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/run1/sintel" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/run1/sintel" \
                --epochs 50 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/raft/best_ckpts/run1/raft_best2.pth" \
                --device "0" 