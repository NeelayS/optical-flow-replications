#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=flownetc-default-things
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/flownetc/things.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/flownet_trainer.yaml" \
                --model "FlowNetC" \
                --model_cfg "configs/models/flownetc_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/flownetc/things" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/flownetc/things" \
                --epochs 50 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/flownetc/chairs/flownetc_best.pth" \
                --device "0" 