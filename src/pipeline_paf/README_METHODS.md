# PAF Pipeline - 4 Methods for RQA Analysis

This pipeline computes RQA metrics using 4 different methods, comparing **Pre-AF patients vs Normal patients (n* only)**.

## Patient Filtering

- **Pre-AF segments:** All segments with label=1
- **Normal segments:** Only from patients with '_n' in their ID (e.g., paf_n01, paf_n20)
  - Excludes normal segments from pre-AF patients

## Methods

### Method 0: Single-scale RQA
- **File:** `single_scale_rqa.csv`
- **Approach:** Direct RQA on original signal
- **Parameters:** tau=12, m=7
- **Output columns:** `signal_epsilon`, `signal_recurrence_rate`, etc.

### Method 1: SWT + Reconstruction → RQA
- **File:** `multi_scale_swt_recon_rqa.csv`
- **Approach:**
  1. Decompose signal using Stationary Wavelet Transform (SWT)
  2. Reconstruct each subband by zeroing out all other coefficients
  3. Compute RQA from reconstructed subbands
- **Subbands:** D1, D2, D3, D4, A4
- **Output columns:** `D1_epsilon`, `D1_recurrence_rate`, ..., `A4_epsilon`, `A4_recurrence_rate`, etc.

### Method 2: SWT → Direct RQA from Coefficients
- **File:** `multi_scale_swt_direct_rqa.csv`
- **Approach:**
  1. Decompose signal using Stationary Wavelet Transform (SWT)
  2. Extract raw wavelet coefficients (cD for details, cA for approximation)
  3. Compute RQA directly from coefficients (no reconstruction)
- **Subbands:** D1, D2, D3, D4, A4
- **Output columns:** `D1_epsilon`, `D1_recurrence_rate`, ..., `A4_epsilon`, `A4_recurrence_rate`, etc.

### Method 3: DWT + Reconstruction → RQA
- **File:** `multi_scale_dwt_recon_rqa.csv`
- **Approach:**
  1. Decompose signal using Discrete Wavelet Transform (DWT)
  2. Reconstruct each subband by zeroing out all other coefficients
  3. Compute RQA from reconstructed subbands
- **Subbands:** D1, D2, D3, D4, A4
- **Output columns:** `D1_epsilon`, `D1_recurrence_rate`, ..., `A4_epsilon`, `A4_recurrence_rate`, etc.

## RQA Metrics Computed

For each method, the following metrics are computed:

1. **epsilon** - Threshold distance used for RP
2. **recurrence_rate** - Percentage of recurrent points
3. **determinism** - Percentage of recurrent points forming diagonal lines
4. **average_diagonal_line** - Average length of diagonal lines
5. **longest_diagonal_line** - Maximum length of diagonal lines
6. **divergence** - Inverse of longest diagonal line
7. **entropy_diagonal_lines** - Shannon entropy of diagonal line lengths
8. **laminarity** - Percentage of recurrent points forming vertical lines
9. **trapping_time** - Average length of vertical lines
10. **longest_vertical_line** - Maximum length of vertical lines
11. **entropy_vertical_lines** - Shannon entropy of vertical line lengths

## Subband Parameters

| Subband | Frequency Range | tau | m |
|---------|----------------|-----|---|
| D1      | 64-128 Hz      | 1   | 4 |
| D2      | 32-64 Hz       | 2   | 4 |
| D3      | 16-32 Hz       | 3   | 4 |
| D4      | 8-16 Hz        | 6   | 5 |
| A4      | 0-8 Hz         | 32  | 8 |

## CSV Output Format

Each CSV file contains:
- `segment_id`: Unique segment identifier
- `patient_id`: Patient identifier
- `label`: 0=Normal, 1=Pre-AF
- `single_scale`: True/False
- `method`: 'single_scale', 'swt_recon', 'swt_direct', or 'dwt_recon'
- Metric columns for each subband/signal

## Usage

```python
from pipeline_paf import PAFPipeline

pipeline = PAFPipeline()

# Run single method
results = pipeline.run_batch(method='swt_recon', filter_patients=True)
pipeline.results_to_csv(results, 'output/results.csv')

# Or run all methods (see __main__ section)
```

## Statistical Analysis

Use `rqa_stats_new.py` to analyze results:

```python
from utils.rqa_stats_new import RQAStatistics

# For multi-scale methods
stats = RQAStatistics('results_filtered/multi_scale_swt_recon_rqa.csv')
stats.print_multiscale_summary()

# For single-scale
stats = RQAStatistics('results_filtered/single_scale_rqa.csv')
stats.print_summary(prefix='signal')
```
