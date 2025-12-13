#!/bin/bash

# NeuralDWAV Autoencoding Script - Using Real Dataset
# Per-channel independent training with wavelet comparison

export CUDA_VISIBLE_DEVICES=0

# Basic config
task_name=autoencoding
model_name=NeuralDWAV
data=ETTh1

# Data paths
root_path='./data/ETT/'
data_path='ETTh1.csv'
features='M'
target='OT'
freq='h'

# Autoencoding parameters
seq_len=512             # Use shorter sequence for real data
level=3                # Wavelet decomposition level
num_channels=7         # Number of channels (ETTh1 has 7 features)
archi=DWT              # Architecture: DWT or WPT
wavelet=db2            # Wavelet type: db2, db3, db4, etc.

# Training parameters
learning_rate=0.0005    # Lower learning rate for real data
lambda_l1=1.0          # L1 sparsity regularization weight
batch_size=128       # Batch size
train_epochs=10     # Fewer epochs for real data (each epoch goes through full dataset)
patience=3            # Early stopping patience

# Experiment settings
itr=1
des='autoencoding_realdata'

echo "========================================"
echo "NeuralDWAV Autoencoding Training"
echo "Using Real Dataset: $data"
echo "========================================"
echo "Data Path: $root_path$data_path"
echo "Signal Length: $seq_len"
echo "Level: $level"
echo "Wavelet: $wavelet"
echo "Channels: $num_channels"
echo "Batch Size: $batch_size"
echo "Learning Rate: $learning_rate"
echo "Lambda L1: $lambda_l1"
echo "Epochs per channel: $train_epochs"
echo "========================================"

python -u run_autoencoding.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id autoencoding_${data}_real_L${level}_${wavelet} \
  --model $model_name \
  --data $data \
  --root_path $root_path \
  --data_path $data_path \
  --features $features \
  --target $target \
  --freq $freq \
  --seq_len $seq_len \
  --level $level \
  --archi $archi \
  --wavelet $wavelet \
  --num_channels $num_channels \
  --learning_rate $learning_rate \
  --lambda_l1 $lambda_l1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --patience $patience \
  --gpu 0 \
  --checkpoints './checkpoints/' \
  --des $des \
  --itr $itr \
  --use_real_data
