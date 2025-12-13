#!/bin/bash

# Run NeuralDWAV Autoencoding on all ETT datasets
# This script will run autoencoding on ETTh1, ETTh2, ETTm1, ETTm2

echo "=========================================="
echo "Running NeuralDWAV Autoencoding on all ETT datasets"
echo "=========================================="

# ETTh1
echo ""
echo "Starting ETTh1..."
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTh1.sh

# ETTh2
echo ""
echo "Starting ETTh2..."
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTh2.sh

# ETTm1
echo ""
echo "Starting ETTm1..."
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTm1.sh

# ETTm2
echo ""
echo "Starting ETTm2..."
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTm2.sh

echo ""
echo "=========================================="
echo "All ETT datasets completed!"
echo "=========================================="
