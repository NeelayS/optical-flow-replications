#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=vcn-default-chairs-run4
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/vcn/chairs-run4.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/vcn_chairs_trainer_default.yaml" \
                --model "VCN" \
                --model_cfg "configs/models/vcn_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/vcn/chairs/run4" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/vcn/chairs/run4" \
                --epochs 200 \
                --batch_size 16 \
                --lr 0.0001 \
                --device "0" 