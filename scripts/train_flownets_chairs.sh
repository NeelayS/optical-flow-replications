#!/bin/bash

#SBATCH --time=96:00:00
#SBATCH --job-name=flownets-default-chairs-run1
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownets/chairs-run1.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_chairs_trainer.yaml" \
                --model "FlowNetS" \
                --model_cfg "configs/models/flownets_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownets/run1" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownets/run1" \
                --epochs 500 \
                --batch_size 16 \
                --device "0" 