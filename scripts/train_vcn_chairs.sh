#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=vcn-default-chairs-run1
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/vcn/chairs-run1.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/vcn_chairs_trainer.yaml" \
                --model "VCN" \
                --model_cfg "configs/models/vcn_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/vcn/chairs/run1" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/vcn/chairs/run1" \
                --epochs 100 \
                --batch_size 16 \
                --target_scale_factor 20 \
                --device "0" 