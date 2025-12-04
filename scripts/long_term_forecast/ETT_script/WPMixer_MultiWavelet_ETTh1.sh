#!/bin/bash

# WPMixer with Multi-Wavelet Decomposition
# This script uses multiple wavelets (db2, db3, db4) with weighted combination

export CUDA_VISIBLE_DEVICES=0

model_name=WPMixer_MultiWavelet

# Dataset and prediction lengths
dataset=ETTh1
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.000242438 0.000201437 0.000132929 0.000239762)
batches=(256 256 256 256)
epochs=(30 30 30 30)
dropouts=(0.4 0.05 0.0 0.2)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 128)
patiences=(12 12 12 12)

# Multi-Wavelet parameters are set in WPMixer_MultiWavelet.py
# Default: wavelet_names=['db2', 'db3', 'db4'], equal weights
# You can customize by modifying the Model class initialization

# Loop over datasets and prediction lengths
for i in "${!pred_lens[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --model_id wpmixer_multiwavelet \
        --model $model_name \
        --task_name long_term_forecast \
        --data $dataset \
        --seq_len ${seq_lens[$i]} \
        --pred_len ${pred_lens[$i]} \
        --label_len 0 \
        --d_model ${d_models[$i]} \
        --patch_len ${patch_lens[$i]} \
        --batch_size ${batches[$i]} \
        --learning_rate ${learning_rates[$i]} \
        --lradj ${lradjs[$i]} \
        --dropout ${dropouts[$i]} \
        --patience ${patiences[$i]} \
        --train_epochs ${epochs[$i]} \
        --use_amp
done

