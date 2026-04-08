# Subband Distance Matrix Analysis for Wavelet Selection

## Overview

This module implements unthresholded recurrence plot (UTRP) geometry analysis to select the optimal wavelet for pre-AF detection. The approach analyzes distance matrix features from embedded wavelet subbands to distinguish pre-AF from NSR based on attractor dynamics.

## Theoretical Basis

### Pre-AF Flickering Hypothesis
- **Lan et al. (2020, Chaos 30, 053137)**: Pre-AF exhibits flickering between two attractor basins in atrial-frequency subbands
- This produces distinct geometric signatures in the phase space distance matrix

### Distance Matrix Metrics
1. **Mean pairwise distance** — Attractor size (larger in pre-AF due to bimodal basin structure)
2. **Distance variance** — Attractor dispersion (higher in pre-AF from flickering)
3. **Distance skewness** — Attractor bimodality (heavier tail from between-basin distances)
4. **Recurrence time entropy** — Temporal irregularity (pre-AF alternates between basins with different recurrence statistics)
   - Reference: Ngamga et al. (2012, PRE 85, 026217)
5. **Diagonal profile variance** — Multi-scale structure (complex RR(τ) profile from two attractors)

### Frequency Selectivity

**Signal Subbands** (expected large pre-AF vs NSR differences):
- D3: 8-16 Hz (atrial frequency band)
- D4: 4-8 Hz (lower atrial band)
- A4: 0-4 Hz (very low frequency, HRV band)

**Control Subbands** (expected small differences, negative controls):
- D1: 32-64 Hz (high frequency, QRS/noise)
- D2: 16-32 Hz (QRS harmonics)

**Wavelet Selection Criterion**:
```
Selectivity Ratio = (D3 + D4 + A4 composite score) / (D1 + D2 composite score)
```

Where composite score = sum of absolute group differences across all 5 metrics.

The best wavelet maximizes this ratio, showing strong separation in clinically relevant subbands while minimizing spurious broadband effects.

## File Structure

```
src/experiment2/
├── subband_distance_analyzer.py    # Main analyzer class
├── run_distance_analysis.py        # Runner script
├── test.py                          # Data loading utilities
└── README_distance_analysis.md     # This file
```

## Usage

### Quick Start

```python
from experiment2.subband_distance_analyzer import SubbandDistanceAnalyzer
from experiment2.test import load_segments_from_hdf5

# Load data
segments_pre_af, segments_nsr, _ = load_segments_from_hdf5('data.h5')

# Initialize analyzer
analyzer = SubbandDistanceAnalyzer(
    wavelets=['db2', 'db4', 'db6', 'sym4', 'sym6', 'coif1', 'coif2'],
    level=4,
    fs=128,
    signal_subbands=['D3', 'D4', 'A4'],
    control_subbands=['D1', 'D2']
)

# Estimate embedding parameters (once per subband)
sample_pre_af = dict(list(segments_pre_af.items())[:5])
sample_nsr = dict(list(segments_nsr.items())[:5])
analyzer.estimate_embedding_params(sample_pre_af, sample_nsr)

# Run full analysis
results = analyzer.run_full_analysis(
    segments_pre_af=segments_pre_af,
    segments_nsr=segments_nsr,
    output_dir='results'
)
```

### Run Complete Pipeline

```bash
cd src/experiment2
python run_distance_analysis.py
```

## Analysis Pipeline

### Step 1: Embedding Parameter Estimation
- Use AMI (Average Mutual Information) to find optimal time delay τ
- Use FNN (False Nearest Neighbors) to find optimal embedding dimension m
- Estimate **once per subband** using sample segments
- Parameters remain constant across all wavelets

### Step 2: Feature Extraction
For each segment:
1. Decompose via SWT reconstruction
2. Extract subbands D1, D2, D3, D4, A4
3. Z-score normalize each subband
4. Embed using cached τ and m
5. Compute pairwise Euclidean distance matrix
6. Extract 5 geometric features

### Step 3: Aggregation
- **Segment → Patient**: Median across segments (robust to outliers)
- **Patient → Cohort**: Median, mean, std for summary statistics

### Step 4: Statistical Testing
For each (wavelet, subband, metric) combination:
- Mann-Whitney U test (non-parametric)
- Cohen's d effect size
- p-values (no multiple comparison correction across wavelets — each is a separate confirmatory test)

### Step 5: Wavelet Selection
For each wavelet:
1. Compute absolute difference in cohort medians per (subband, metric)
2. Sum differences for signal subbands (D3+D4+A4) → signal_score
3. Sum differences for control subbands (D1+D2) → control_score
4. Compute selectivity_ratio = signal_score / control_score
5. Select wavelet with **highest selectivity ratio**

## Output Files

### 1. `distance_features_per_patient.csv`
Patient-level features for all wavelets.

Columns:
- `patient_id`: Patient identifier
- `wavelet`: Wavelet name
- `cohort`: 'pre_af' or 'nsr'
- `subband`: D1, D2, D3, D4, or A4
- `feature`: Feature name (mean_dist, var_dist, skew_dist, recurrence_time_entropy, diagonal_profile_var)
- `value`: Feature value (median across segments)

### 2. `statistical_comparison.csv`
Statistical test results for all combinations.

Columns:
- `wavelet`, `subband`, `feature`
- `pre_af_median`, `pre_af_mean`, `pre_af_std`
- `nsr_median`, `nsr_mean`, `nsr_std`
- `u_statistic`: Mann-Whitney U statistic
- `p_value`: Two-sided p-value
- `cohen_d`: Effect size

### 3. `wavelet_selection_scores.csv`
Composite scores and selectivity ratios (sorted by ratio, descending).

Columns:
- `wavelet`: Wavelet name
- `signal_score`: Sum of absolute differences in D3+D4+A4
- `control_score`: Sum of absolute differences in D1+D2
- `selectivity_ratio`: signal_score / control_score

### 4. `best_wavelet_summary.txt`
Human-readable summary with selected wavelet and top 5 candidates.

## Expected Outcomes

### Good Wavelet Properties
1. **Large signal_score**: Strong separation in D3, D4, A4 (atrial bands)
2. **Small control_score**: Minimal separation in D1, D2 (broadband controls)
3. **High selectivity_ratio**: Frequency-specific rather than generic differences

### Metric Interpretation

| Metric | Pre-AF Expected | NSR Expected | Rationale |
|--------|----------------|--------------|-----------|
| Mean distance | Higher | Lower | Bimodal attractor basin structure increases average separation |
| Distance variance | Higher | Lower | Flickering between basins increases dispersion |
| Distance skewness | Positive | ~Zero | Between-basin distances create heavy tail |
| Recurrence time entropy | Higher | Lower | Irregular recurrence from basin alternation |
| Diagonal profile variance | Higher | Lower | Two attractors produce complex RR(τ) with multiple timescales |

### Frequency Specificity

A valid result should show:
- **D3, D4, A4**: p < 0.05, Cohen's d > 0.5 for multiple metrics
- **D1, D2**: p > 0.05, Cohen's d < 0.3 (no significant broadband effects)

If D1 and D2 show comparable effect sizes to D3/D4, the wavelet is not frequency-selective and the result may be confounded by QRS morphology or noise.

## Advantages Over Binary RQA

1. **No epsilon threshold**: Unthresholded distance matrix avoids arbitrary threshold selection
2. **Richer information**: Continuous distances preserve fine-scale geometry lost in binarization
3. **Adaptive thresholds**: Recurrence time entropy uses row-wise median (local adaptation)
4. **Multi-scale analysis**: Diagonal profile captures RR(τ) structure without fixing a single recurrence criterion

## Implementation Notes

### Computational Complexity
- Distance matrix: O(N²) where N = embedded trajectory length
- N ≈ (segment_length - (m-1)×τ) ≈ 1000 points for 10s segments
- Memory: ~8 MB per distance matrix (1000×1000 float64)

### Edge Cases Handled
- Signals too short for embedding: Returns NaN, excluded from aggregation
- Zero variance subbands: Z-score normalization adds epsilon (1e-9)
- Empty recurrence time distributions: Returns entropy = 0.0
- Missing data: NaN values filtered before statistical tests

### Performance Tips
- Use `n_segments_per_record=10` for faster testing
- Full analysis with 12 wavelets × 50 patients × 180 segments ≈ 30-60 minutes
- Distance matrix computation is the bottleneck (parallelization possible)

## References

1. Lan, Y., et al. (2020). "Flickering as an early warning signal." *Chaos* 30(5), 053137.
2. Ngamga, E.J., et al. (2012). "Recurrence analysis of strange nonchaotic dynamics." *Physical Review E* 85, 026217.
3. Marwan, N., et al. (2007). "Recurrence plots for the analysis of complex systems." *Physics Reports* 438(5-6), 237-329.

## Contact / Issues

For questions or issues with the analysis, check:
- Embedding parameters: Are τ and m reasonable for your signal length?
- Data quality: Are segments properly preprocessed and normalized?
- Memory: Large datasets may require downsampling or batch processing