#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=resume-raft-default-things
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/resume_things.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_things_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft" \
                --resume True \
                --resume_epochs 50 \
                --resume_ckpt "../../../Share/optical_flow/replications/ckpts/raft/things/raft_epochs5.pth" \
                --device "0" 