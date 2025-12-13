#!/bin/bash

# Master script to run complete pipeline for all ETT datasets
# Pipeline: LWPTMixer → Autoencoding → PretrainedWPMixer (Parallel)

export CUDA_VISIBLE_DEVICES=0

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Log directory
LOG_DIR="./logs/full_pipeline_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "============================================================"
echo "          ETT Complete Pipeline Experiment                  "
echo "============================================================"
echo "Pipeline stages:"
echo "  1. LWPTMixer training (baseline)"
echo "  2. Autoencoding training (self-supervised)"
echo "  3. PretrainedWPMixer training (using parallel checkpoint)"
echo ""
echo "Datasets: ETTh1, ETTh2, ETTm1, ETTm2"
echo "Log directory: $LOG_DIR"
echo "============================================================"
echo ""

# Function to run a stage
run_stage() {
    local dataset=$1
    local stage_name=$2
    local script_path=$3
    local log_file=$4
    
    echo -e "${BLUE}[$dataset]${NC} Stage: ${YELLOW}$stage_name${NC}"
    echo "  Script: $script_path"
    echo "  Log: $log_file"
    echo "  Start time: $(date)"
    
    if [ ! -f "$script_path" ]; then
        echo -e "${RED}  ERROR: Script not found!${NC}"
        return 1
    fi
    
    bash "$script_path" > "$log_file" 2>&1
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}  ✓ Completed successfully${NC}"
    else
        echo -e "${RED}  ✗ Failed with exit code $exit_code${NC}"
        echo -e "${RED}  Check log: $log_file${NC}"
        return 1
    fi
    
    echo "  End time: $(date)"
    echo ""
    return 0
}

# Datasets to process
datasets=(ETTh1 ETTh2 ETTm1 ETTm2)

# Main pipeline
for dataset in "${datasets[@]}"; do
    echo "============================================================"
    echo -e "${GREEN}Processing dataset: $dataset${NC}"
    echo "============================================================"
    echo ""
    
    # Stage 1: LWPTMixer (baseline)
    run_stage "$dataset" \
              "LWPTMixer Training" \
              "scripts/long_term_forecast/ETT_script/LWPTMixer_${dataset}.sh" \
              "$LOG_DIR/${dataset}_1_lwptmixer.log"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Stopping pipeline for $dataset due to LWPTMixer failure${NC}"
        continue
    fi
    
    # Stage 2: Autoencoding (self-supervised learning)
    run_stage "$dataset" \
              "Autoencoding (Self-supervised)" \
              "scripts/autoencoding/ETT_script/NeuralDWAV_${dataset}.sh" \
              "$LOG_DIR/${dataset}_2_autoencoding.log"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}Stopping pipeline for $dataset due to Autoencoding failure${NC}"
        continue
    fi
    
    # Stage 3: PretrainedWPMixer (with parallel checkpoint)
    run_stage "$dataset" \
              "PretrainedWPMixer (Parallel Checkpoint)" \
              "scripts/long_term_forecast/ETT_script/PretrainedWPMixer_${dataset}_parallel.sh" \
              "$LOG_DIR/${dataset}_3_pretrained.log"
    
    if [ $? -ne 0 ]; then
        echo -e "${RED}PretrainedWPMixer failed for $dataset${NC}"
        continue
    fi
    
    echo -e "${GREEN}✓ All stages completed for $dataset${NC}"
    echo ""
done

# Summary
echo "============================================================"
echo "               Pipeline Execution Summary                   "
echo "============================================================"
echo ""

for dataset in "${datasets[@]}"; do
    echo -e "${BLUE}$dataset:${NC}"
    
    stages=("1_lwptmixer" "2_autoencoding" "3_pretrained")
    stage_names=("LWPTMixer" "Autoencoding" "PretrainedWPMixer")
    
    for i in "${!stages[@]}"; do
        log_file="$LOG_DIR/${dataset}_${stages[$i]}.log"
        if [ -f "$log_file" ]; then
            # Check if log contains errors
            if grep -q "Error\|ERROR\|Failed\|FAILED" "$log_file"; then
                echo -e "  ${stage_names[$i]}: ${RED}✗ Failed${NC}"
            else
                echo -e "  ${stage_names[$i]}: ${GREEN}✓ Success${NC}"
            fi
        else
            echo -e "  ${stage_names[$i]}: ${YELLOW}- Not run${NC}"
        fi
    done
    echo ""
done

echo "All logs saved to: $LOG_DIR"
echo "============================================================"
echo "Pipeline completed at: $(date)"
echo "============================================================"


