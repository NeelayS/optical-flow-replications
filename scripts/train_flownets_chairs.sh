#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=flownets-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownets/chairs.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_trainer.yaml" \
                --model "FlowNetS" \
                --model_cfg "configs/models/flownets_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownets" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownets" \
                --epochs 100 \
                --batch_size 10 \
                --device "0" 