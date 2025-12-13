# Project_CCXZ: Learnable Wavelet Transformer for Time-Series Forecasting
We try our best to beat SOTA in this project for Penn course ESE_5380.
This repository is modified based on the Time-Series-Library by Tsinghua University.


## Usage

1. Install Python 3.8. For convenience, execute the following command.

```
pip install -r requirements.txt
```

2. Prepare Data. You can obtain the well-preprocessed datasets from [[Google Drive]](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing), [[Baidu Drive]](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy) or [[Hugging Face]](https://huggingface.co/datasets/thuml/Time-Series-Library). Then place the downloaded data in the folder`./dataset`. Here is a summary of supported datasets.

<p align="center">
<img src=".\pic\dataset.png" height = "200" alt="" align=center />
</p>

3. Train and evaluate the model. We provide the experiment scripts for all benchmarks under the folder `./scripts/`. You can reproduce the experiment results as the following examples:

```bash
# long-term forecast
bash ./scripts/long_term_forecast/ETT_script/TimesNet_ETTh1.sh
# short-term forecast
bash ./scripts/short_term_forecast/TimesNet_M4.sh
# imputation
bash ./scripts/imputation/ETT_script/TimesNet_ETTh1.sh
# anomaly detection
bash ./scripts/anomaly_detection/PSM/TimesNet.sh
# classification
bash ./scripts/classification/TimesNet.sh
```

### WPMixer with Pretrained Wavelet Decomposition

Our **PretrainedWPMixer** model integrates pretrained wavelet decomposition learned from self-supervised autoencoding. Here's the complete training pipeline:

#### Option A: Automated Pipeline (Recommended)

Run the complete pipeline for all ETT datasets:

```bash
bash scripts/run_all_ETT_experiments.sh
```

This script automatically performs for each dataset (ETTh1, ETTh2, ETTm1, ETTm2):
1. Train LWPTMixer baseline
2. Train self-supervised wavelet autoencoder (generates parallel checkpoints)
3. Train PretrainedWPMixer using the pretrained wavelet weights

#### Option B: Manual Step-by-Step Training

**Step 1: Train Baseline WPMixer**

```bash
# Example: ETTh1 dataset
bash scripts/long_term_forecast/ETT_script/WPMixer_ETTh1.sh
```

**Step 2: Train Self-Supervised Wavelet Autoencoder**

```bash
# This generates pretrained wavelet weights for decomposition
bash scripts/autoencoding/ETT_script/NeuralDWAV_ETTh1.sh

# Checkpoint will be saved to: ./checkpoints/autoencoding/ETTh1/parallel_checkpoint.pth
```

**Step 3: Train PretrainedWPMixer with Frozen Wavelet Weights**

```bash
# This model loads and freezes the pretrained wavelet weights
bash scripts/long_term_forecast/ETT_script/PretrainedWPMixer_ETTh1_parallel.sh

# The wavelet parameters are frozen by default (freeze_wavelet=True)
```

#### Key Features

- **Frozen Wavelet Layers**: The pretrained wavelet decomposition is frozen during forecasting training
- **Parallel Checkpoints**: Supports efficient parallel training across multiple channels
- **Consistent Normalization**: Both autoencoder and forecaster use the same normalization scheme
- **Verified Weight Freezing**: The model automatically verifies that wavelet parameters are correctly frozen

#### Performance

Our experiments show:
- **WPMixer**: Baseline performance (best)
- **PretrainedWPMixer**: Comparable to WPMixer (0.5-2% difference)
- **LWPTMixer**: Slightly lower than PretrainedWPMixer (1-3% difference)

See `result_long_term_forecast.txt` for detailed metrics on all ETT datasets.

4. Develop your own model.

- Add the model file to the folder `./models`. You can follow the `./models/Transformer.py`.
- Include the newly added model in the `Exp_Basic.model_dict` of  `./exp/exp_basic.py`.
- Create the corresponding scripts under the folder `./scripts`.

Note: 

(1) About classification: Since we include all five tasks in a unified code base, the accuracy of each subtask may fluctuate but the average performance can be reproduced (even a bit better). We have provided the reproduced checkpoints [here](https://github.com/thuml/Time-Series-Library/issues/494).

(2) About anomaly detection: Some discussion about the adjustment strategy in anomaly detection can be found [here](https://github.com/thuml/Anomaly-Transformer/issues/14). The key point is that the adjustment strategy corresponds to an event-level metric.


## Acknowledgement

This library is constructed based on the following repos:

- Forecasting: https://github.com/thuml/Autoformer.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://github.com/thuml/Flowformer.

All the experiment datasets are public, and we obtain them from the following links:

- Long-term Forecasting and Imputation: https://github.com/thuml/Autoformer.

- Short-term Forecasting: https://github.com/ServiceNow/N-BEATS.

- Anomaly Detection: https://github.com/thuml/Anomaly-Transformer.

- Classification: https://www.timeseriesclassification.com/.
