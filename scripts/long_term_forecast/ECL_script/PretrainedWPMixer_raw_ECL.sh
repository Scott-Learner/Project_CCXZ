#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=PretrainedWPMixer_raw

# Dataset
dataset=electricity

# Parameters aligned with WPMixer ECL configuration
seq_lens=(512 512 512 512)
pred_lens=(96 192 336 720)
learning_rates=(0.00328086 0.000493286 0.002505375 0.001977516)
batches=(32 32 32 32)
epochs=(100 100 100 100)
dropouts=(0.1 0.1 0.2 0.1)
patch_lens=(16 16 16 16)
lradjs=(type3 type3 type3 type3)
d_models=(32 32 32 32)
patiences=(12 12 12 12)
strides=(8 8 8 8)

# PretrainedWPMixer_raw specific: pretrained checkpoint path
pretrained_ckpt=./checkpoints/autoencoding/ECL

# Wavelet parameters (must match autoencoding checkpoint)
wavelet_level=2  # ECL autoencoding was trained with level=2
wavelet_name=db2

echo "============================================================"
echo "PretrainedWPMixer_raw for ECL Dataset"
echo "Using serial checkpoints from: $pretrained_ckpt"
echo "Expected format: channel_{0..320}_checkpoint.pth"
echo "============================================================"

# Loop over prediction lengths
for i in "${!pred_lens[@]}"; do
	echo ""
	echo "Running experiment ${i}/${#pred_lens[@]}: pred_len=${pred_lens[$i]}"
	echo "------------------------------------------------------------"
	
	python -u run.py \
		--is_training 1 \
		--root_path ./data/electricity/ \
		--data_path electricity.csv \
		--model_id PreWPMixer_raw_${dataset}_${pred_lens[$i]} \
		--model $model_name \
		--task_name long_term_forecast \
		--data $dataset \
		--seq_len ${seq_lens[$i]} \
		--pred_len ${pred_lens[$i]} \
		--label_len 0 \
		--enc_in 321 \
		--dec_in 321 \
		--c_out 321 \
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

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"

