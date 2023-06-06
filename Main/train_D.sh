#!/bin/bash

python train_D.py \
    --input_dir "/home/vlsilab/Dataset/Img2Pos_train/" \
    --bchm_dir "/home/vlsilab/Dataset/Img2Pos_test/AFLW2000_all-crop/" \
    \
    --ckpt_dir "../Log/D_training_log/img_npy_3/ckpt/" \
    --img_dir "../Log/D_training_log/img_npy_3/img/" \
    --tblog_dir "../Log/D_training_log/img_npy_3/tblog/" \
    \
    --ndf 64 \
    --epochs 300 \
    --batch_size 16 \
    --beta1 0.9 \
    --beta2 0.99 \
    --lr 0.0001 \
    --decay_step 30 \
    \
    --device_id 0 1 2 \
    --mode 'train' \
    --print_freq 100. \
    --ckpt_freq 10 \
    --validate_freq 1
