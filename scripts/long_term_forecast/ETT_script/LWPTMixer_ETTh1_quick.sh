#!/bin/bash

# LWPTMixer quick test with optimized hyperparameters (based on WPMixer)
export CUDA_VISIBLE_DEVICES=0

model_name=LWPTMixer

# Quick test: only pred_len=96
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --model_id LWPTMixer_ETTh1_512_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 512 \
  --label_len 0 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 2048 \
  --patch_len 16 \
  --batch_size 256 \
  --learning_rate 0.000242438 \
  --dropout 0.4 \
  --patience 12 \
  --train_epochs 30 \
  --lradj type3 \
  --use_amp \
  --des 'NeuralDWAV' \
  --itr 1


