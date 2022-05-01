#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft-default-chairs-ddp-mp
#SBATCH --partition=jiang
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:2
#SBATCH --cpus-per-task=2
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/chairs-ddp-mp.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_chairs_ddp_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --device "all" \
                --world_size 2 \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/ddp-mp/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/ddp-mp/chairs" \
                --mixed_precision True \
                --epochs 100 