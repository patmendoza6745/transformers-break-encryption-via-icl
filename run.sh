#!/bin/bash

# lr-1e-3
CUDA_VISIBLE_DEVICES=0
nohup python train.py  \
    config/train_gpt2.py \
    --wandb_run_name='lr-1e-3' \
    --learning_rate=1e-3 \
    --batch_size=64 \
    --gradient_accumulation_steps=1 \
    --max_iters=20000 \
    --lr_decay_iters=20000 \
    --eval_interval=5000 \
    --eval_iters=50 \
    --log_interval=1 \
    --weight_decay=1e-1 > logs/lr-1e-3.log 2>&1 &
