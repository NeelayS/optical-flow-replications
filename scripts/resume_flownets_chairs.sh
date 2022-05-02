#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=resume-flownets-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownets/resume_chairs.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_chairs_trainer.yaml" \
                --model "FlowNetS" \
                --model_cfg "configs/models/flownets_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownets/run1/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownets/run1/chairs" \
                --resume True \
                --resume_epochs 429 \
                --resume_ckpt "../../../Share/optical_flow/replications/ckpts/flownets/run1/flownets_epoch71.pth" \
                --device "0" 