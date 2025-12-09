#!/bin/bash

# NeuralDWAV Autoencoding Script
# Per-channel independent training with wavelet comparison
# Following LDWT_main.py DESPAWN approach

export CUDA_VISIBLE_DEVICES=0

# Task name
task_name=autoencoding
model_name=NeuralDWAV
data=ETTh1

# Signal parameters
seq_len=8192           # 2^13 - signal length
level=3                # Wavelet decomposition level (changed to 3 for faster testing)
num_channels=7         # Number of channels (like ETT data)
archi=DWT              # Architecture: DWT or WPT

# Training parameters (following LDWT_main DESPAWN)
learning_rate=0.01     # Same as DESPAWN
lambda_l1=1.0          # L1 sparsity regularization weight
batch_size=32           # Small batch size works better
train_epochs=1000      # 1000 epochs per channel

# Other settings
patience=20
itr=1
des='autoencoding_db4'

echo "========================================"
echo "NeuralDWAV Autoencoding Training"
echo "========================================"
echo "Signal Length: $seq_len"
echo "Level: $level"
echo "Channels: $num_channels"
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
  --seq_len $seq_len \
  --level $level \
  --archi $archi \
  --num_channels $num_channels \
  --learning_rate $learning_rate \
  --lambda_l1 $lambda_l1 \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --patience $patience \
  --gpu 0 \
  --checkpoints './checkpoints/' \
  --des $des \
  --itr $itr


