#! /bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py --wandb_log=False