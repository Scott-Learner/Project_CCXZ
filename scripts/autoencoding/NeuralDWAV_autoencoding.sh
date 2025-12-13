#!/bin/bash

# NeuralDWAV Autoencoding Script
# Per-channel independent training with wavelet comparison
# Following LDWT_main.py DESPAWN approach

export CUDA_VISIBLE_DEVICES=0

# Basic config
task_name=autoencoding
model_name=NeuralDWAV
data=ETTh1

# Data paths
root_path='./data/ETT/'
data_path='ETTh1.csv'
features='M'

# Autoencoding parameters
seq_len=8192           # Signal length (2^13)
level=3                # Wavelet decomposition level
num_channels=7         # Number of channels
archi=DWT              # Architecture: DWT or WPT
wavelet=db2            # Wavelet type: db2, db3, db4, etc.

# Training parameters
learning_rate=0.01     # Optimizer learning rate
lambda_l1=1.0          # L1 sparsity regularization weight
batch_size=32          # Batch size
train_epochs=1000      # Train epochs per channel
patience=20            # Early stopping patience

# Experiment settings
itr=1
des='autoencoding_db4'

# Use real data or dummy generator (remove --use_real_data to use dummy generator)
use_real_data_flag=""   # Set to "--use_real_data" to use real dataset

echo "========================================"
echo "NeuralDWAV Autoencoding Training"
echo "========================================"
echo "Data: $data"
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
  --model_id autoencoding_${data}_${seq_len}_L${level} \
  --model $model_name \
  --data $data \
  --root_path $root_path \
  --data_path $data_path \
  --features $features \
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
  $use_real_data_flag


