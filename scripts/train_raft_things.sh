#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=raft-default-things
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/things-run2.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_things_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/run2/things" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/run2/things" \
                --epochs 50 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/raft/run1/chairs/raft_best_final.pth" \
                --batch_size 16 \
                --device "0" 