# MIT-BIH Lead-Time-Based RQA Pipeline

Patient-wise analysis of RQA metrics at specific lead times before AF onset.

## Overview

This pipeline prepares MIT-BIH datasets for RQA analysis by:
1. Extracting pre-AF segments at specified lead times (5, 10, 15, 30, or 45 minutes before AF onset)
2. Extracting matched normal segments from the same patients (>60 min from any AF)
3. Applying advanced signal cleaning with SWT-based adaptive denoising
4. Exporting cleaned, normalized datasets ready for RQA computation

## Signal Processing Pipeline

The cleaning pipeline follows this exact order:

1. **Raw ECG signal**
2. **Basic filtering**:
   - Remove DC offset (mean subtraction)
   - Highpass filter (0.5 Hz) - removes baseline wander
   - Lowpass filter (40 Hz) - removes high-frequency noise
3. **SWT-based adaptive denoising** (optional):
   - Decompose signal using Stationary Wavelet Transform (level 4)
   - Estimate noise σ from detail coefficients (D1) using MAD
   - Compute universal threshold: λ = σ * sqrt(2 * log(N))
   - Apply soft thresholding to all detail subbands
   - Reconstruct denoised signal
4. **Z-score normalization**: (signal - mean) / std
5. **Ready for embedding & RP computation**

## Key Features

### SWT-Based Adaptive Denoising

The denoising method uses:

- **Noise estimation**: σ = MAD(D1) / 0.6745
  - MAD = Median Absolute Deviation
  - D1 = first detail level (highest frequency)

- **Universal threshold**: λ = σ * √(2 log N)
  - Balances noise reduction vs signal preservation
  - Adapts to signal length N

- **Soft thresholding** on detail coefficients:
  ```
  If |c| < λ:
      c_new = 0
  Else:
      c_new = sign(c) * (|c| - λ)
  ```
  - Smoothly shrinks coefficients toward zero
  - Preserves signal structure better than hard thresholding

### Patient-Wise Matching

For each patient at each lead time:
- Extracts N pre-AF segments at the specified lead time
- Extracts N matched normal segments from the same patient
  - Normal segments must be >60 minutes from any AF episode
  - Randomly sampled from safe normal periods
- Uses minimum count if insufficient segments available

## Configuration Parameters

All parameters are configurable in the main execution block:

```python
# Lead time selection
LEAD_TIME = 5  # 5, 10, 15, 30, or 45 minutes before AF

# Cleaning parameters
APPLY_CLEANING = True
HIGHPASS_CUTOFF = 0.5  # Hz
NOTCH_FREQ = 60.0  # Hz (powerline noise)
NOTCH_Q = 30.0
NORMALIZE_METHOD = 'zscore'  # Always use z-score
SWT_DENOISE = True  # Enable/disable SWT denoising
SWT_LEVEL = 4  # SWT decomposition level

# Matching parameters
NORMAL_BUFFER_MIN = 60  # Min distance (minutes) from AF for normal segments
```

## Usage

```python
# Initialize pipeline
pipeline = MITBIHPipeline(
    lead_time=5,
    apply_cleaning=True,
    swt_denoise=True,
    swt_level=4,
)

# Prepare dataset
preaf_segments, normal_segments = pipeline.prepare_dataset()

# Export to CSV
pipeline.export_to_csv()
```

## Output Format

### CSV Structure

- **Metadata columns**:
  - `segment_id`: Unique identifier
  - `patient_id`: Patient identifier (e.g., mitbih_04746)
  - `label`: 0 (normal) or 1 (pre-AF)
  - `lead_time`: Minutes before AF onset (0 for normal)
  - `episode_index`: Episode number (-1 for normal)
  - `lead_index`: ECG lead index (0)
  - `lead_name`: Lead name (ECG1)

- **Signal columns**: `s0` through `s{N-1}` where N ≈ 2500 samples (10 seconds at 250 Hz)

### Output Location

`/data/extracted/mitbih/results/dataset_lt{X}.csv`

Where X is the lead time (5, 10, 15, 30, or 45)

## Example Results (Lead Time 5 min)

```
Patients: 11
Pre-AF segments: 29
Normal segments: 29
Total: 58 segments

Signal properties:
- Duration: 10 seconds
- Sampling rate: 250 Hz
- Samples: ~2500 per segment
- Normalization: Z-score (mean=0, std=1)
```

## Implementation Details

### SWT Denoising in DataCleaner

Added three new methods to `DataCleaner` class:

1. `_estimate_noise_sigma(detail_coeffs)`: Estimates noise using MAD
2. `_soft_threshold(coeffs, threshold)`: Applies soft thresholding
3. `swt_adaptive_denoise(signal)`: Main denoising method

### Normal Segment Extraction

Method `_extract_normal_segments_for_patient()`:
- Loads raw MIT-BIH annotations
- Identifies Normal rhythm periods
- Calculates distance to nearest AF episode
- Randomly samples segments from safe periods
- Applies same cleaning pipeline as pre-AF segments

## Next Steps

After dataset preparation, the next phase is RQA computation:

1. **Single-scale RQA**: Direct embedding → RP → RQA metrics
2. **Multi-scale RQA**: SWT decomposition → RP per subband → RQA metrics
3. **Statistical comparison**: Patient-wise paired tests at each lead time
4. **Temporal analysis**: Track RQA metric changes across lead times

## Files

- `pipeline_mitbih.py`: Main pipeline implementation
- `../data_cleaning/data_cleaner.py`: Signal cleaning with SWT denoising
- `../wavelets/stationary_wavelet_transform.py`: SWT implementation
- `README.md`: This file
