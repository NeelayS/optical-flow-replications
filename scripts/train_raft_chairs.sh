#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/chairs-run2.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_chairs_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/run2/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/run2/chairs" \
                --epochs 100 \
                --batch_size 16 \
                --device "0" 