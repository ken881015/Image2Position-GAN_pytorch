#!/bin/bash

python train_GAN.py \
        --input_dir "/home/vlsilab/Dataset/Img2Pos_train/" \
        --bchm_dir "/home/vlsilab/Dataset/Img2Pos_test/AFLW2000_all-crop/" \
        \
        --ckpt_dir "../Log/GAN_training_log/img_npy_4/ckpt/" \
        --img_dir "../Log/GAN_training_log/img_npy_4/img/" \
        --tblog_dir "../Log/GAN_training_log/img_npy_4/tblog/" \
        \
        --ngf 64 \
        --ndf 64 \
        \
        --epochs 300 \
        --lr 0.0001 \
        --decay_step 30 \
        --beta1 0.9 \
        --beta2 0.99 \
        --l1_weight 1 \
        --gan_weight 1 \
        --batch_size 16 \
        \
        --gamma 1.5 \
        --lambda_k 0.001 \
        \
        --device_id 0 1 2 \
        --mode 'export' \
        --print_freq 100 \
        --ckpt_freq 5 \
        --validate_freq 1 \
        --bm_version "ori" \
        --gan_pre_trained