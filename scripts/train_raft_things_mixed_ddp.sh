#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=raft-default-things-ddp-mp
#SBATCH --partition=jiang
#SBATCH --mem=64G
#SBATCH --gres=gpu:a6000:4
#SBATCH --cpus-per-task=4
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/things-ddp-mp.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_things_ddp_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/ddp-mp/things" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/ddp-mp/things" \
                --epochs 100 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/raft/ddp-mp/chairs/raft_best.pth" \
                --world_size 4 \
                --mixed_precision True \
                --device "all" 