# Repository Simplification Summary

## Overview
This repository has been simplified to focus **only on forecasting tasks** (long-term and short-term forecasting).

## Changes Made

### 1. Removed Experiment Files
Deleted the following task-specific experiment files from `exp/`:
- ❌ `exp_anomaly_detection.py` (anomaly detection)
- ❌ `exp_classification.py` (classification)
- ❌ `exp_imputation.py` (imputation)

### 2. Kept Experiment Files
Retained the following forecasting experiment files:
- ✅ `exp_long_term_forecasting.py` (long-term forecasting)
- ✅ `exp_short_term_forecasting.py` (short-term forecasting)
- ✅ `exp_basic.py` (base class)

### 3. Removed Script Directories
Deleted the following script directories from `scripts/`:
- ❌ `anomaly_detection/` (all anomaly detection scripts)
- ❌ `classification/` (all classification scripts)
- ❌ `exogenous_forecast/` (exogenous forecasting scripts)
- ❌ `imputation/` (all imputation scripts)

### 4. Kept Script Directories
Retained the following forecasting script directories:
- ✅ `long_term_forecast/` (all long-term forecasting scripts)
- ✅ `short_term_forecast/` (all short-term forecasting scripts)

### 5. Simplified `run.py`
Updated the main entry point to:
- Removed imports for deleted experiment classes
- Updated help text to only mention forecasting tasks
- Removed task selection branches for non-forecasting tasks
- Removed unused argument parsers (mask_rate, anomaly_ratio)

## Verification

All forecasting functionality has been verified to work correctly:
- ✓ Long-term forecasting experiment initializes successfully
- ✓ Short-term forecasting experiment initializes successfully
- ✓ No breaking changes to existing forecasting workflows

## Usage

### Long-Term Forecasting
```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id your_model_id \
  --model WPMixer_MultiWavelet \
  --data ETTh1 \
  --root_path ./data/ETT/ \
  --data_path ETTh1.csv \
  --features M \
  --seq_len 512 \
  --pred_len 96 \
  --d_model 256 \
  ...
```

### Short-Term Forecasting
```bash
python run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --model_id your_model_id \
  --model TimesNet \
  --data M4 \
  --seasonal_patterns Monthly \
  ...
```

## Directory Structure After Simplification

```
Project_CCXZ/
├── exp/
│   ├── exp_basic.py                    ✓ Base class
│   ├── exp_long_term_forecasting.py    ✓ Long-term forecasting
│   └── exp_short_term_forecasting.py   ✓ Short-term forecasting
├── scripts/
│   ├── long_term_forecast/             ✓ Long-term scripts
│   └── short_term_forecast/            ✓ Short-term scripts
├── run.py                               ✓ Simplified entry point
└── ...
```

## Notes
- All existing checkpoints and results are preserved
- Model implementations remain unchanged
- Data loaders remain unchanged
- Only task-specific experiment classes and scripts were removed
