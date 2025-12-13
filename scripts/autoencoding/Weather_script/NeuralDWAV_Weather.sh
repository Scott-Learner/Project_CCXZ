#!/bin/bash

# NeuralDWAV Autoencoding - Weather Dataset
# Per-channel independent training with validation-based model selection

export CUDA_VISIBLE_DEVICES=0

# Basic config
task_name=autoencoding
model_name=NeuralDWAV
data=Weather

# Data paths
root_path='./data/weather/'
data_path='weather.csv'
features='M'
target='OT'
freq='h'

# Autoencoding parameters
seq_len=512             # Sequence length for real data (same as WPMixer)
level=2                 # Wavelet decomposition level (same as WPMixer)
num_channels=21         # Weather has 21 features
archi=DWT               # Architecture: DWT or WPT
wavelet=db3             # Wavelet type (same as WPMixer: db3)

# Training parameters
learning_rate=0.0002    # Learning rate
lambda_l1=1.0           # L1 sparsity regularization weight
batch_size=128          # Batch size
train_epochs=10         # Training epochs
patience=10             # Early stopping patience

# Experiment settings
itr=1
des='Weather_autoencoding'

echo "========================================"
echo "NeuralDWAV Autoencoding - Weather"
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

