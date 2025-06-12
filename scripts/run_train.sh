#!/bin/bash

sbatch train_david.sh \
    --job_name "2D_morphology_train1" \
    --epochs 200 \
    --batch_size 32 \
    --lr 0.0001 \
    --checkpoint_dir "/path/to/experiments/checkpoints" \
    --log_dir "/path/to/experiments/logs" \
    --save_top_k 3 \
    --patience 10 \
    --resume_checkpoint False \
    --training_stage "joint" \
    --dimension "2D" \
    --backbone "resnet18" \
    --single_class_prototype_per_class 18 \
    --joint_prototypes_per_border 0 \
    --sampling_rate 100 \
    --label_set "3" \
    --save_weights True \
    --seed 42 \
    --num_workers 4 \
    --dropout 0.35 \
    --l2 0.00017 \
    --scheduler_type "CosineAnnealingLR" \
    --custom_groups True \
    --proto_time_len 3 \
    --proto_dim 512 \
    --pretrained_weights "/path/to/pretrained_weights.ckpt"

