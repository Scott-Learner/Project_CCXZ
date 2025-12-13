#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# Model name
model_name=PretrainedWPMixer

# Dataset
dataset=ETTh1

# Parameters aligned with WPMixer_ETTh1.sh
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
strides=(8 8 8 8)

# PretrainedWPMixer specific: using PARALLEL checkpoint format
pretrained_ckpt=./checkpoints/autoencoding/ETTh1

# Wavelet parameters (must match autoencoding checkpoint)
wavelet_level=2  # ETTh1 autoencoding was trained with level=2
wavelet_name=db2

echo "============================================================"
echo "PretrainedWPMixer for ETTh1 Dataset (Parallel Checkpoint)"
echo "Using parallel checkpoint from: $pretrained_ckpt/parallel_checkpoint.pth"
echo "NO AMP mode (full precision)"
echo "============================================================"

# Loop over prediction lengths
for i in "${!pred_lens[@]}"; do
	echo ""
	echo "Running experiment $((i+1))/${#pred_lens[@]}: pred_len=${pred_lens[$i]}"
	echo "------------------------------------------------------------"
	
	python -u run.py \
		--is_training 1 \
		--root_path ./data/ETT/ \
		--data_path ETTh1.csv \
		--model_id PreWPMixer_parallel_${dataset}_${pred_lens[$i]} \
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
		--checkpoints ./checkpoints/
done

echo ""
echo "============================================================"
echo "All experiments completed!"
echo "============================================================"

