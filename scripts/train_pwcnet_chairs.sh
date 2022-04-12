#!/bin/bash

#SBATCH --time=72:00:00
#SBATCH --job-name=pwcnet-default-chairs
#SBATCH --partition=jiang
#SBATCH --mem=32G
#SBATCH --gres=gpu:a6000:1
#SBATCH --output=../../../../Share/optical_flow/replications/outs/pwcnet/chairs.out


module load cuda/11.3
cd ..
python train.py --train_cfg "configs/pwcnet_trainer.yaml" \
                --model "PWCNet" \
                --model_cfg "configs/models/pwcnet_default.yaml" \
                --log_dir "../../../Share/optical_flow/replications/logs/pwcnet/chairs" \
                --ckpt_dir "../../../Share/optical_flow/replications/ckpts/pwcnet/chairs" \
                --epochs 100 \
                --batch_size 10 \
                --device "0" 