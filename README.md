# Wavelet-Based Multi-Scale Recurrence Analysis for Early Prediction of Atrial Fibrillation

## KEN3500 Bachelor's Thesis
### BSc Data Science and Artificial Intelligence
### Maastricht University
### Author: Filip Sulich
### Supervisors: Dr Joël Karel, Dr Pietro Bonizzi

---

## Project Overview

This repository contains the implementation of a machine learning-based approach for predicting atrial fibrilation (AF) using a late-fusion convolutional neural network (CNN)
and multi-scale unthresholded recurrence plots (URPs).

## Project Structure

### `src/CNN/`
Contains the CNN model architectures, cross-validation training setup, and evaluation functions.

**Key files:**
- `models/multi_scale_cnn.py` - Multi-scale CNN (7 branches for D2-D8 subbands with attention fusion)
- `models/full_band_cnn.py` - Full-band baseline CNN (the same architecture as the multi-scale branches but with single input)
- `training/unified_cross_validator.py` - Cross-validation setup for training every model configuration with patient-level fold grouping
- `dataset/dataset_generator.py` - Generates a dataset of URPs from ECG signals
- `utils/datasets.py` - PyTorch dataset classes for loading H5 data
- `utils/evaluation.py` - Model evaluation metrics and utilities

**Runnable scripts:**
- `tests/model_evaluator.py` - Evaluate trained models on the validation folds and save results for later analysis

### `src/wavelets/`
Wavelet transform implementations for signal decomposition.

**Key files:**
- `wavelet_transform.py` - Abstract base class for wavelet transforms
- `discrete_wavelet_transform.py` - DWT implementation (Discrete Wavelet Transform)
- `stationary_wavelet_transform.py` - SWT implementation (Stationary Wavelet Transform)

### `src/pipelines/`
Wavelet decomposition pipeline strategies.

**Key files:**
- `pipeline_wavelets.py` - Wrapper for different decomposition strategies (SWT-R, SWT-D, DWT-R)

### `src/recurrence_plot/`
Phase space reconstruction and recurrence plot generation.

**Key files:**
- `AMI.py` - Average Mutual Information for time delay estimation
- `FNN.py` - False Nearest Neighbors for embedding dimension estimation
- `recurrence_plot_generation.py` - Binary and unthresholded recurrence plot generation using the pyrqa package (https://pypi.org/project/PyRQA/)

### `src/wavelet_selection_tests/`
Experiments for selecting optimal wavelet function and decomposition strategy.

**Subdirectories:**
- `leakage_analysis/` - Tests for energy leakage across subbands
  - `leakage_tester.py` - Energy ratio computation using synthetic signals
  - `leakage_stats.py` - Statistical analysis (Kruskal-Wallis, Dunn post-hoc)
  - `leakage_plotter.py` - Visualization of leakage results
  - `main.py` - Run complete leakage analysis

- `urp_discrminability/` - Tests for URP discriminability using image texture and statistical features
  - `wavelet_evaluator.py` - Cross-validated AUC comparison across wavelets
  - `utrp_features.py` - Feature extraction from URPs
  - `main.py` - Run 5-fold CV training for the discriminability analysis (note it can take a while) 

**Key files:**
- `utils.py` - Subject grouping and effect size utilities

### `src/decomposition_strategy_validation/`
Validation of decomposition strategies using synthetic signals.

**Key files:**
- `experiment.py` - Main experiment runner for strategy validation
- `signals.py` - Synthetic signal generation with known frequency content
- `embedding.py` - Phase space embedding utilities
- `recurrence.py` - Recurrence plot computation
- `metrics.py` -  Structural similarity metrics for comparing per-scale RPs
- `ground_truth.py` - Ground truth data generation
- `statistical_tests.py` - Statistical comparison of strategies
- `plotting.py` - Visualization utilities

**Runnable scripts:**
- `run.py` - Run the experiment and save results with plots

## Dataset

**IMPORTANT:** The processed dataset is not included in this repository due to size constraints.

**Download the dataset:**
1. Download the URP data and model weights from Google Drive: **https://drive.google.com/drive/folders/1NplSsH6urf2LSVbM0s3QcqdQjDLAxCP7?usp=sharing**
2. Place the downloaded `data` folder into the `data_container/` directory
3. Run the scripts as needed to train models or evaluate results

**Note:** The raw ECG data is from the PAF Prediction Challenge Database (AFPDB) available at PhysioNet (https://physionet.org/content/afpdb/1.0.0/).

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate 

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies:**
- PyTorch >= 2.0
- numpy
- scipy
- scikit-learn
- h5py
- pywt (PyWavelets)
- pyrqa (for recurrence plot generation)
- matplotlib
- pandas

## Usage Examples
```bash
# Leakage analysis
cd src/wavelet_selection_tests/leakage_analysis
python main.py

# URP discriminability analysis
cd src/wavelet_selection_tests/urp_discrminability
python main.py

# Decomposition strategy validation
cd src/decomposition_strategy_validation
python run.py

# Model evaluation
cd src/CNN/tests
python model_evaluator.py

# Ablation study
cd src/CNN/ablation
python ablation_study.py
```

## Methodology

This implementation of the multi-scale recurrence analysis pipeline follows these steps :

1. **Wavelet Decomposition**: ECG signals are decomposed into frequency subbands (D2-D8) using Stationary Wavelet Transform (SWT) with db8 wavelet
2. **Phase Space Reconstruction**: Each subband is embedded using parameters (tau from AMI, m from FNN)
3. **Recurrence Plot Generation**: Unthresholded recurrence plots (URPs) are computed per wavelet scale using pyrqa
4. **Multi-Scale CNN**: URPs from all subbands are processed by separate CNN branches with late feature-fusion and attention mechanisms to modulate the contribution of each scale
5. **Classification**: Binary classification (NSR vs Pre-AF) using weighted cross-entropy loss

## Disclosure: Use of Generative AI

Generative AI (Claude model Opus 4.6) was used to assist only with the following aspects of this project:

**Code Documentation:**
- Generation of docstrings for Python functions and classes
- Formatting of documentation to maintain consistency across the codebase
- Clarity improvements in code comments

**Code Refactoring:**
- Help with removing code redundancy and optimization
- Suggestions for code organization and structure improvements

**README file**
- Structuring and formatting of this README.md file

## License
MIT License

## Contact
Author: f.sulich@student.maastrichtuniversity.nl
