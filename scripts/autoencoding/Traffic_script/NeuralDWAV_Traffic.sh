#!/bin/bash

# NeuralDWAV Autoencoding - Traffic Dataset
# Per-channel independent training with validation-based model selection

export CUDA_VISIBLE_DEVICES=0

# Basic config
task_name=autoencoding
model_name=NeuralDWAV
data=Traffic

# Data paths
root_path='./data/traffic/'
data_path='traffic.csv'
features='M'
target='OT'
freq='h'

# Autoencoding parameters
seq_len=1200            # Sequence length for real data (same as WPMixer)
level=1                 # Wavelet decomposition level (same as WPMixer)
num_channels=862        # Traffic has 862 features
archi=DWT               # Architecture: DWT or WPT
wavelet=db3             # Wavelet type (same as WPMixer: db3)

# Training parameters
learning_rate=0.0002    # Learning rate
lambda_l1=1.0           # L1 sparsity regularization weight
batch_size=32           # Smaller batch size due to many channels
train_epochs=10         # Training epochs
patience=10             # Early stopping patience

# Experiment settings
itr=1
des='Traffic_autoencoding'

echo "========================================"
echo "NeuralDWAV Autoencoding - Traffic"
echo "========================================"
echo "Data Path: $root_path$data_path"
echo "Signal Length: $seq_len"
echo "Level: $level"
echo "Wavelet: $wavelet"
echo "Channels: $num_channels"
echo "Batch Size: $batch_size"
echo "Learning Rate: $learning_rate"
echo "Lambda L1: $lambda_l1"
echo "Epochs: $train_epochs"
echo "========================================"

python -u run_autoencoding.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id ${data}_L${level}_${wavelet} \
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
  --save_model \
  --use_real_data
