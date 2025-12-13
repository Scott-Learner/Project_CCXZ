#!/bin/bash

# NeuralDWAV Autoencoding - ETTm2 Dataset
# Per-channel independent training with validation-based model selection

export CUDA_VISIBLE_DEVICES=0

# Basic config
task_name=autoencoding
model_name=NeuralDWAV
data=ETTm2

# Data paths
root_path='./data/ETT/'
data_path='ETTm2.csv'
features='M'
target='OT'
freq='t'  # 15min frequency (same as WPMixer)

# Autoencoding parameters
seq_len=512             # Sequence length for real data (same as WPMixer)
level=1                 # Wavelet decomposition level (same as WPMixer)
num_channels=7          # ETTm2 has 7 features
archi=DWT               # Architecture: DWT or WPT
wavelet=db2             # Wavelet type (same as WPMixer: db2)

# Training parameters
learning_rate=0.0002    # Learning rate
lambda_l1=1.0           # L1 sparsity regularization weight
batch_size=128          # Batch size
train_epochs=10         # Training epochs
patience=10             # Early stopping patience

# Experiment settings
itr=1
des='ETTm2_autoencoding'

echo "========================================"
echo "NeuralDWAV Autoencoding - ETTm2"
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

