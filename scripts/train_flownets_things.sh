#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=flownets-default-things
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownets/things.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_trainer.yaml" \
                --model "FlowNetS" \
                --model_cfg "configs/models/flownets_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownets/things" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownets/things" \
                --epochs 50 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/flownets/chairs/flownets_best.pth" \
                --device "0" 