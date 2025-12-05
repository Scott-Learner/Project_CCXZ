#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=LWPTMixer

# ETTh1 - Prediction lengths: 96, 192, 336, 720
for pred_len in 96
do
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path ./data/ETT/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_512_${pred_len} \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len 512 \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers 2 \
      --d_layers 1 \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --d_model 256 \
      --d_ff 2048 \
      --des 'NeuralDWAV' \
      --itr 1 \
      --batch_size 16 \
      --learning_rate 0.001 \
      --train_epochs 10 \
      --patience 3 \
      --patch_len 16
done


