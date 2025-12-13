#!/bin/bash

# Run NeuralDWAV Autoencoding on all datasets
# This script will run autoencoding on all available datasets

echo "=========================================="
echo "Running NeuralDWAV Autoencoding on all datasets"
echo "=========================================="

# ETT datasets
echo ""
echo "=== ETT Datasets ==="
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTh1.sh
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTh2.sh
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTm1.sh
bash ./scripts/autoencoding/ETT_script/NeuralDWAV_ETTm2.sh

# Weather
echo ""
echo "=== Weather Dataset ==="
bash ./scripts/autoencoding/Weather_script/NeuralDWAV_Weather.sh

# ECL (Electricity) - Large dataset, may take longer
echo ""
echo "=== ECL (Electricity) Dataset ==="
bash ./scripts/autoencoding/ECL_script/NeuralDWAV_ECL.sh

# Traffic - Large dataset, may take longer
echo ""
echo "=== Traffic Dataset ==="
bash ./scripts/autoencoding/Traffic_script/NeuralDWAV_Traffic.sh

echo ""
echo "=========================================="
echo "All datasets completed!"
echo "=========================================="
