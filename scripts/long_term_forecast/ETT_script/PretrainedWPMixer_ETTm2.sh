#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=PretrainedWPMixer

# Dataset
dataset=ETTm2

# Parameters aligned with WPMixer_ETTm2.sh
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.00076587 0.000275775 0.000234608 0.001039536)
batches=(256 256 256 256)
epochs=(80 80 80 80)
dropouts=(0.4 0.2 0.4 0.4)
patch_lens=(48 48 48 48)
lradjs=(type3 type3 type3 type3)
d_models=(256 256 256 256)
patiences=(12 12 12 12)
strides=(24 24 24 24)

# PretrainedWPMixer specific: pretrained checkpoint path
pretrained_ckpt=./checkpoints/autoencoding/ETTm2

# Wavelet parameters (must match autoencoding checkpoint)
wavelet_level=1  # ETTm2 autoencoding was trained with level=1
wavelet_name=db2

# Loop over prediction lengths
for i in "${!pred_lens[@]}"; do
	python -u run.py \
		--is_training 1 \
		--root_path ./data/ETT/ \
		--data_path ETTm2.csv \
		--model_id PreWPMixer_${dataset}_${pred_lens[$i]} \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--label_len 0 \
		--d_model ${d_models[$i]} \
		--patch_len ${patch_lens[$i]} \
		--patch_stride ${strides[$i]} \
		--batch_size ${batches[$i]} \
		--learning_rate ${learning_rates[$i]} \
		--lradj ${lradjs[$i]} \
		--dropout ${dropouts[$i]} \
		--patience ${patiences[$i]} \
		--train_epochs ${epochs[$i]} \
		--pretrained_ckpt $pretrained_ckpt \
		--wavelet_level $wavelet_level \
		--wavelet_name $wavelet_name \
		--checkpoints ./checkpoints/ \
		--use_amp
done

