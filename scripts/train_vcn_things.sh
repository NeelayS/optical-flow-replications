#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=vcn-default-things
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/vcn/things.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/vcn_trainer.yaml" \
                --model "VCN" \
                --model_cfg "configs/models/vcn_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/vcn/things" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/vcn/things" \
                --epochs 50 \
                --model_ckpt "../../../Share/optical_flow/replications/ckpts/vcn/chairs/vcn_best.pth" \
                --batch_size 16 \
                --device "0" 