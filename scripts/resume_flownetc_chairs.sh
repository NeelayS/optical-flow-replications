#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=resume-flownetc-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownetc/resume_chairs.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_chairs_trainer.yaml" \
                --model "FlowNetC" \
                --model_cfg "configs/models/flownetc_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownetc/run1/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownetc/run1/chairs" \
                --resume True \
                --resume_epochs 436 \
                --resume_ckpt "../../../Share/optical_flow/replications/ckpts/flownetc/run1/flownetc_epoch64.pth" \
                --device "0" 