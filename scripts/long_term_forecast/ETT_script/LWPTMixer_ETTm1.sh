#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=LWPTMixer
dataset=ETTm1

seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.001277976 0.002415901 0.001594735 0.002011441)
batches=(256 256 256 256)
epochs=(80 80 80 80)
dropouts=(0.4 0.4 0.4 0.4)
patch_lens=(48 48 48 48)
lradjs=(type3 type3 type3 type3)
d_models=(256 128 256 128)
patiences=(12 12 12 12)

for i in "${!pred_lens[@]}"; do
    python -u run.py \
        --is_training 1 \
        --root_path ./data/ETT/ \
        --data_path ETTm1.csv \
        --model_id LWPTMixer_${dataset}_${seq_lens[$i]}_${pred_lens[$i]} \
        --model $model_name \
        --task_name long_term_forecast \
        --data $dataset \
        --seq_len ${seq_lens[$i]} \
        --pred_len ${pred_lens[$i]} \
        --label_len 0 \
        --features M \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model ${d_models[$i]} \
        --d_ff 2048 \
        --e_layers 2 \
        --d_layers 1 \
        --patch_len ${patch_lens[$i]} \
        --batch_size ${batches[$i]} \
        --learning_rate ${learning_rates[$i]} \
        --lradj ${lradjs[$i]} \
        --dropout ${dropouts[$i]} \
        --patience ${patiences[$i]} \
        --train_epochs ${epochs[$i]} \
        --use_amp \
        --des 'NeuralDWAV' \
        --itr 1
done


