#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=dicl-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/dicl/chairs.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/dicl_trainer.yaml" \
                --model "DICL" \
                --model_cfg "configs/models/dicl_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/dicl" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/dicl" \
                --epochs 100 \
                --batch_size 10 \
                --device "0" 