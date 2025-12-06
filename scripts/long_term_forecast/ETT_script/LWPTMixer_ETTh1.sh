#!/bin/bash

# LWPTMixer training script for ETTh1
# Uses optimized hyperparameters based on WPMixer

export CUDA_VISIBLE_DEVICES=0

model_name=LWPTMixer
dataset=ETTh1

# Optimized hyperparameters for each prediction length
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

# Loop over all prediction lengths
for i in "${!pred_lens[@]}"; do
    echo "================================================"
    echo "Training LWPTMixer: pred_len=${pred_lens[$i]}"
    echo "================================================"
    
    python -u run.py \
        --is_training 1 \
        --root_path ./data/ETT/ \
        --data_path ETTh1.csv \
        --model_id LWPTMixer_${dataset}_${seq_lens[$i]}_${pred_lens[$i]} \
        --model $model_name \
        --task_name long_term_forecast \
        --data $dataset \
        --seq_len ${seq_lens[$i]} \
        --pred_len ${pred_lens[$i]} \
        --label_len 0 \
        --features M \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model ${d_models[$i]} \
        --d_ff 2048 \
        --e_layers 2 \
        --d_layers 1 \
        --patch_len ${patch_lens[$i]} \
        --batch_size ${batches[$i]} \
        --learning_rate ${learning_rates[$i]} \
        --lradj ${lradjs[$i]} \
        --dropout ${dropouts[$i]} \
        --patience ${patiences[$i]} \
        --train_epochs ${epochs[$i]} \
        --use_amp \
        --des 'NeuralDWAV' \
        --itr 1
    
    echo ""
done

echo "================================================"
echo "✅ All training completed!"
echo "================================================"

