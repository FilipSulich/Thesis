# MIT-BIH Pipeline Usage Guide

Complete guide for the MIT-BIH lead-time-based RQA analysis pipeline.

## Overview

The MIT-BIH pipeline consists of three main components:

1. **Dataset Preparation** (`pipeline_mitbih.py`) - Extract and clean segments
2. **RQA Analysis** (`rqa_pipeline_mitbih.py`) - Compute single and multi-scale RQA metrics
3. **Visualization** (`../visualizers/mitbih_visualizer.py`) - Visualize signals, cleaning, and RPs

## Workflow

### Step 1: Dataset Preparation

Extracts matched pre-AF and normal segments for a specific lead time.

```bash
cd /Users/filipsulich/Developer/uni/Thesis/Thesis/src/pipeline_mitbih
python pipeline_mitbih.py
```

**Configuration** (in main block):
```python
LEAD_TIME = 5  # 5, 10, 15, 30, or 45 minutes

# Cleaning parameters
SWT_DENOISE = True  # SWT-based adaptive denoising
HIGHPASS_CUTOFF = 0.5  # Hz
NORMALIZE_METHOD = 'zscore'  # Always z-score
```

**Output**:
- `results/dataset_lt5.csv` - Cleaned, normalized segments ready for RQA

**Processing pipeline**:
1. Raw ECG
2. Highpass filter (0.5 Hz)
3. Lowpass filter (40 Hz)
4. SWT adaptive denoising (soft thresholding)
5. Z-score normalization

### Step 2: RQA Analysis

Computes RQA metrics for single-scale and multi-scale analysis.

```bash
python rqa_pipeline_mitbih.py
```

**Configuration** (in main block):
```python
LEAD_TIME = 5  # Must match dataset

# Single-scale parameters
TIME_DELAY = 12
EMBEDDING_DIM = 7
EPSILON_METHOD = 'std'  # or 'max_distance'
EPSILON_FACTOR = 0.2

# Multi-scale subband parameters
subband_params = {
    "D1": {"tau": 6, "m": 7},
    "D2": {"tau": 4, "m": 7},
    "D3": {"tau": 2, "m": 7},
    "D4": {"tau": 2, "m": 7},
    "A4": {"tau": 16, "m": 8},
}
```

**Outputs**:
- `results/rqa_single_scale_lt5.csv` - Single-scale RQA metrics
- `results/rqa_multi_scale_lt5.csv` - Multi-scale RQA metrics per subband

**RQA Metrics Computed**:
- `recurrence_rate` - Percentage of recurrent points
- `determinism` - Percentage of recurrent points forming diagonal lines
- `average_diagonal_line` - Average length of diagonal lines
- `longest_diagonal_line` - Length of longest diagonal line
- `divergence` - Inverse of longest diagonal line
- `entropy_diagonal_lines` - Shannon entropy of diagonal line lengths
- `laminarity` - Percentage of recurrent points forming vertical lines
- `trapping_time` - Average length of vertical lines
- `longest_vertical_line` - Length of longest vertical line
- `entropy_vertical_lines` - Shannon entropy of vertical line lengths

### Step 3: Visualization

Visualize signals, cleaning steps, decomposition, and recurrence plots.

```python
from visualizers.mitbih_visualizer import MITBIHVisualizer

# Initialize visualizer
viz = MITBIHVisualizer(subband_params=subband_params)

# Get random signal at 5-min lead time
indices = viz.get_random_indices(lead_time=5, n=1)
idx = indices[0]

# Complete investigation
viz.investigate(
    idx=idx,
    epsilon_method='std',
    epsilon_factor=0.3,
    show_unthresholded=True
)
```

**Visualization methods**:

1. **`plot_cleaning_comparison(idx)`**
   - Shows 4-step cleaning pipeline
   - Raw → Filtered → Denoised → Normalized

2. **`plot_subbands(idx)`**
   - Shows original signal + 5 subbands (D1-D4, A4)
   - With statistics (std)

3. **`plot_subband_rps(idx)`**
   - Shows signal/subband + corresponding RPs
   - Original + D1 + D2 + D3 + D4 + A4
   - Option to show unthresholded (distance matrix) RPs

4. **`investigate(idx)`**
   - Runs all three visualizations above

## File Structure

```
pipeline_mitbih/
├── pipeline_mitbih.py          # Dataset preparation
├── rqa_pipeline_mitbih.py      # RQA computation
├── README.md                   # Technical documentation
├── USAGE.md                    # This file
└── __init__.py

results/
├── dataset_lt5.csv             # Cleaned segments (lead time 5)
├── rqa_single_scale_lt5.csv    # Single-scale RQA results
└── rqa_multi_scale_lt5.csv     # Multi-scale RQA results

visualizers/
└── mitbih_visualizer.py        # Visualization tools
```

## Example: Complete Analysis for Lead Time 5

```bash
cd /Users/filipsulich/Developer/uni/Thesis/Thesis/src/pipeline_mitbih

# Step 1: Prepare dataset
# Edit pipeline_mitbih.py: set LEAD_TIME = 5
python pipeline_mitbih.py

# Step 2: Run RQA analysis
# Edit rqa_pipeline_mitbih.py: set LEAD_TIME = 5
python rqa_pipeline_mitbih.py

# Step 3: Visualize (optional)
python -c "
from visualizers.mitbih_visualizer import MITBIHVisualizer
viz = MITBIHVisualizer(subband_params={
    'D1': {'tau': 6, 'm': 7},
    'D2': {'tau': 4, 'm': 7},
    'D3': {'tau': 2, 'm': 7},
    'D4': {'tau': 2, 'm': 7},
    'A4': {'tau': 16, 'm': 8},
})
indices = viz.get_random_indices(lead_time=5, n=1)
viz.investigate(indices[0], epsilon_factor=0.3, show_unthresholded=True)
"
```

## Processing All Lead Times

To process all lead times sequentially:

```bash
for LT in 5 10 15 30 45; do
    echo "Processing lead time: $LT minutes"

    # Update LEAD_TIME in both files
    sed -i '' "s/LEAD_TIME = .*/LEAD_TIME = $LT/" pipeline_mitbih.py
    sed -i '' "s/LEAD_TIME = .*/LEAD_TIME = $LT/" rqa_pipeline_mitbih.py

    # Run pipeline
    python pipeline_mitbih.py
    python rqa_pipeline_mitbih.py
done
```

## Output Data Format

### Dataset CSV (dataset_lt5.csv)

Columns:
- `segment_id` - Unique identifier
- `patient_id` - Patient identifier
- `label` - 0 (normal) or 1 (pre-AF)
- `lead_time` - Minutes before AF (0 for normal)
- `episode_index`, `lead_index`, `lead_name` - Metadata
- `s0` through `s{N-1}` - Signal samples (~2500 for 10s at 250Hz)

### Single-Scale RQA CSV

Columns:
- Metadata: `segment_id`, `patient_id`, `label`, `lead_time`, `single_scale`
- RQA metrics: `signal_recurrence_rate`, `signal_determinism`, etc.

### Multi-Scale RQA CSV

Columns:
- Metadata: `segment_id`, `patient_id`, `label`, `lead_time`, `single_scale`
- Subband metrics: `D1_recurrence_rate`, `D1_determinism`, ..., `A4_entropy_vertical_lines`

## Results Summary (Lead Time 5 min)

Example output from the pipeline:

```
Dataset Preparation:
  Patients: 11
  Pre-AF segments: 29
  Normal segments: 29
  Total: 58 segments

RQA Analysis:
  Single-scale: 58 results
  Multi-scale: 58 results (6 subbands each)

Processing time:
  Single-scale: ~6 seconds
  Multi-scale: ~40 seconds
```

## Statistical Analysis (Next Steps)

After running the pipeline for all lead times, use statistical analysis tools:

```python
from utils.rqa_stats_new import RQAStatistics

# Load results
stats = RQAStatistics('results/rqa_single_scale_lt5.csv')

# Print summary
stats.print_summary(prefix='signal')

# For multi-scale, analyze each subband
for subband in ['D1', 'D2', 'D3', 'D4', 'A4']:
    print(f"\n{subband} Statistics:")
    stats.print_summary(prefix=subband)
```

## Tips and Best Practices

1. **Always use z-score normalization** - Ensures consistency across segments
2. **SWT denoising recommended** - Improves signal quality before RQA
3. **Match parameters** - Ensure LEAD_TIME matches between scripts
4. **Save visualizations** - Use `save_path` parameter in visualization methods
5. **Process lead times sequentially** - Avoid mixing lead times in single run
6. **Check patient counts** - Some lead times may have fewer patients due to data availability

## Troubleshooting

**"Skipped N patients with insufficient segments"**
- Normal - some patients don't have normal segments far enough from AF
- Solution: Lower `NORMAL_BUFFER_MIN` (default 60 min) if needed

**"Signal too short for embedding"**
- Embedding parameters (tau, m) create vectors too long for signal
- Solution: Reduce tau or m for that subband

**RQA computation slow**
- Multi-scale takes ~6-7x longer than single-scale (6 subbands)
- Solution: Use `max_samples` parameter to test on subset first

**Visualization not showing**
- Running in non-interactive environment
- Solution: Save to file using `save_path` parameter
