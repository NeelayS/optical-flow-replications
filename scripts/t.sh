#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --job-name=vcn-default-chairs-run2
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/vcn/chairs-run2.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/vcn_chairs_trainer.yaml" \
                --model "VCN" \
                --model_cfg "configs/models/vcn_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/vcn/chairs/run2" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/vcn/chairs/run2" \
                --epochs 100 \
                --batch_size 16 \
                --device "0" 