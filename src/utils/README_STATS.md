# RQA Statistics - Mann-Whitney U Test

This module performs statistical analysis on RQA metrics using the **Mann-Whitney U test** for independent samples.

## Key Changes

### Previous Approach (Removed)
- ❌ Wilcoxon signed-rank test (paired samples)
- ❌ Paired analysis assuming same patients have both normal and pre-AF segments

### Current Approach (Active)
- ✅ Mann-Whitney U test (independent samples)
- ✅ Pre-AF patients vs Normal patients (n* only)
- ✅ No pairing assumption - groups are independent

## Patient Filtering

The analysis now uses **independent groups**:

**Pre-AF Group:**
- All segments with `label = 1`
- From any pre-AF patient (e.g., `paf_p01`, `paf_p22`)

**Normal Group:**
- Only segments with `label = 0` **AND** `'_n'` in patient ID
- Examples: `paf_n01`, `paf_n20`, `paf_n33`
- **Excludes** normal segments from pre-AF patients

This ensures groups are independent (non-paired) as required for Mann-Whitney U test.

## Output Format

### Console Output
```
Metric                          N_preAF  N_normal      Pre-AF (mean±std)      Normal (mean±std)  Cohen d   p-value
------------------------------------------------------------------------------------------------------------------------
epsilon                              45         3      0.9876±0.1234          1.0234±0.0987      -0.345    0.0234
recurrence_rate                      45         3      0.0512±0.0067          0.0489±0.0045       0.456    0.1234
...
```

### CSV Output
Columns:
- `prefix`: Subband name (signal, D1, D2, D3, D4, A4)
- `metric`: RQA metric name
- `n_preaf`: Number of pre-AF patients
- `n_normal`: Number of normal (n*) patients
- `preaf_mean`, `preaf_std`, `preaf_median`: Pre-AF statistics
- `normal_mean`, `normal_std`, `normal_median`: Normal statistics
- `cohens_d`: Effect size (independent samples)
- `p_value`: Mann-Whitney U test p-value

## Usage

### Command Line
```python
python utils/rqa_stats_new.py
```

Edit the `method` variable in `__main__` to choose which CSV to analyze:
- `"single_scale"` - Single-scale RQA
- `"swt_recon"` - SWT + Reconstruction
- `"swt_direct"` - SWT Direct from coefficients
- `"dwt_recon"` - DWT + Reconstruction

### Programmatic
```python
from utils.rqa_stats_new import RQAStatistics

# Load data
stats = RQAStatistics("path/to/rqa_results.csv")

# Print summary
stats.print_summary(prefix="signal")  # For single-scale
stats.print_multiscale_summary()      # For multi-scale

# Get significant results
sig = stats.get_significant_results(
    alpha=0.05,
    prefixes=["D1", "D2", "D3", "D4", "A4"]
)

# Save to CSV
stats.save_results_csv(
    "output/statistics.csv",
    prefixes=["signal"]
)

# Generate plots
stats.plot_metrics("output/plots", prefix="signal")
```

## Statistical Methods

### Mann-Whitney U Test
- **Type:** Non-parametric test for independent samples
- **Null Hypothesis:** The two groups come from the same distribution
- **Alternative:** Two-sided test
- **Requirements:** At least 2 samples in each group

### Effect Size (Cohen's d)
For independent samples:
```
pooled_std = sqrt((std_preaf² + std_normal²) / 2)
cohens_d = (mean_preaf - mean_normal) / pooled_std
```

**Interpretation:**
- |d| < 0.2: Small effect
- |d| ≈ 0.5: Medium effect
- |d| > 0.8: Large effect

## Significant Results

Results with `p < 0.05` are considered statistically significant.

Example output:
```
Statistically Significant Results (p < 0.05):
================================================================================
  D3_determinism: p=0.0234, d=0.456, n_preaf=45, n_normal=3
  A4_laminarity: p=0.0456, d=-0.678, n_preaf=45, n_normal=3
```

## Notes

- The analysis requires **at least 2 patients in each group**
- With very small sample sizes (e.g., n_normal=3), interpret results cautiously
- Mann-Whitney U is robust to non-normal distributions
- No assumptions about paired/matched samples

## Files Generated

When running the main script, the following files are created:

- `stats_single_scale.csv` - Statistics for single-scale RQA
- `stats_swt_recon.csv` - Statistics for SWT + Reconstruction
- `stats_swt_direct.csv` - Statistics for SWT Direct
- `stats_dwt_recon.csv` - Statistics for DWT + Reconstruction
