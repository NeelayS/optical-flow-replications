#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=resume-raft-default-sintel
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/raft/resume-sintel-run1.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/raft_sintel_trainer.yaml" \
                --model "RAFT" \
                --model_cfg "configs/models/raft_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/raft/run1/sintel" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/raft/run1/sintel" \
                --resume True \
                --resume_epochs 150 \
                --resume_ckpt "../../../Share/optical_flow/replications/ckpts/raft/run1/sintel/raft_epoch99.pth" \
                --device "0" 