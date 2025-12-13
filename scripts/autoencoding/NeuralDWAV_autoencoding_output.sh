#!/bin/bash

# NeuralDWAV Autoencoding with Output Script

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
seq_len=512
level=3
num_channels=1
archi=DWT
wavelet=db2

# Training parameters
learning_rate=0.005
lambda_l1=1.0
batch_size=8
train_epochs=1000
patience=20

# Output settings
save_wavelet_sweeps=true
sweeps_outdir='./wavelet_sweeps/'

# Experiment settings
itr=1
des='autoencoding_output'

echo "=========================================="
echo "NeuralDWAV Autoencoding with Output"
echo "=========================================="
echo "Data: $data"
echo "Wavelet: $wavelet"
echo "Level: $level"
echo "Epochs: $train_epochs"
echo "Output Dir: $sweeps_outdir"
echo "=========================================="

python -u run_autoencoding_output.py \
  --task_name $task_name \
  --is_training 1 \
  --model_id autoencoding_${data}_${wavelet}_output \
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
  --use_real_data \
  --save_wavelet_sweeps \
  --sweeps_outdir $sweeps_outdir
