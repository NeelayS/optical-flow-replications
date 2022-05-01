#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft-default-chairs-ddp
#SBATCH --partition=jiang
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=4
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/chairs-ddp.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_chairs_ddp_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --device "all" \
                --world_size 4 \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/ddp/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/ddp/chairs" \
                --epochs 100 