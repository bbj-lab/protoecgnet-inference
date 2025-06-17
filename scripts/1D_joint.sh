#!/bin/bash

python ../src/main.py \
    --job_name "1D_rhythm_train_01" \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.0001 \
    --checkpoint_dir "/gpfs/data/bbj-lab/users/chend5/experiments/checkpoints" \
    --log_dir "/gpfs/data/bbj-lab/users/chend5/experiments/logs" \
    --save_top_k 3 \
    --patience 10 \
    --resume_checkpoint False \
    --training_stage "joint" \
    --dimension "1D" \
    --backbone "resnet1d18" \
    --single_class_prototype_per_class 5 \
    --joint_prototypes_per_border 0 \
    --sampling_rate 100 \
    --label_set "1" \
    --save_weights True \
    --seed 42 \
    --num_workers 4 \
    --dropout 0.35 \
    --l2 0.00017 \
    --scheduler_type "CosineAnnealingLR" \
    --custom_groups True \
    --proto_time_len 32 \
    --proto_dim 512 

