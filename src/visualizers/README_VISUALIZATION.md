# MIT-BIH Visualization Guide

Complete guide for visualizing cleaned MIT-BIH signals with normal and pre-AF segments.

## Key Changes

The visualizer now **loads from CSV files** (not HDF5) which contain **both pre-AF and normal signals** that were matched per-patient during dataset preparation.

---

## Quick Start

### Basic Usage

```python
from visualizers.mitbih_visualizer import MITBIHVisualizer

# Initialize (loads from dataset_lt5.csv by default)
viz = MITBIHVisualizer(lead_time=5)

# Plot random pre-AF signal with subbands and RPs
viz.plot_subband_rps(label_filter=1)  # 1 = pre-AF

# Plot random normal signal with subbands and RPs
viz.plot_subband_rps(label_filter=0)  # 0 = normal
```

### Run Examples

```bash
cd /Users/filipsulich/Developer/uni/Thesis/Thesis/src/visualizers
python example_visualization.py
```

---

## Data Source

### CSV File Structure

**Input:** `results/dataset_lt{lead_time}.csv`

This CSV contains:
- **Pre-AF segments** (label=1): Z segments at specified lead time before AF onset
- **Normal segments** (label=0): Z matched normal segments from same patient (>60 min from AF)

**Columns:**
- `segment_id`, `patient_id`, `label`, `lead_time`, `episode_index`, `lead_index`, `lead_name`
- `s0`, `s1`, `s2`, ..., `sN` - Signal samples (already cleaned and normalized)

**Key Point:** Signals in CSV are **already cleaned** (filtered, denoised, normalized) during dataset preparation.

---

## API Reference

### Initialize Visualizer

```python
viz = MITBIHVisualizer(
    subband_params=None,           # Optional RQA parameters per subband
    csv_path=None,                 # Path to CSV (default: results/dataset_lt{lead_time}.csv)
    lead_time=5                    # Lead time to load (5, 10, 15, 30, or 45)
)
```

### Plot Subbands with RPs

```python
viz.plot_subband_rps(
    idx=None,                      # Specific index (if None, random from label_filter)
    label_filter=None,             # 0=normal, 1=pre-AF, None=any
    save_path=None                 # Optional save path
)
```

**Output:**
- 6 rows × 2 columns grid
- **Row 0:** Original signal + Original RP
- **Rows 1-5:** D1, D2, D3, D4, A4 subbands + their RPs

### Get Random Signals

```python
# Get random pre-AF signal indices
indices = viz.get_random_indices(label=1, n=3)

# Get random normal signal indices
indices = viz.get_random_indices(label=0, n=3)

# Get any signal at specific lead time
indices = viz.get_random_indices(lead_time=5, n=3)
```

### Other Visualization Methods

```python
# Plot subband decomposition (no RPs)
viz.plot_subbands(
    idx=42,
    apply_cleaning=False,          # Signals already cleaned
    normalize_per_subband=True,
    save_path='subbands.png'
)

# Plot cleaning comparison (uses raw data from HDF5)
viz.plot_cleaning_comparison(
    idx=42,
    wavelet='db4',
    save_path='cleaning.png'
)
```

---

## Examples

### Example 1: Compare Pre-AF vs Normal

```python
from visualizers.mitbih_visualizer import MITBIHVisualizer, OUTPUT_DIR

viz = MITBIHVisualizer(lead_time=5)
output_dir = OUTPUT_DIR / 'visualizations'
output_dir.mkdir(parents=True, exist_ok=True)

# Pre-AF
viz.plot_subband_rps(
    label_filter=1,
    save_path=output_dir / 'preaf_example.png'
)

# Normal
viz.plot_subband_rps(
    label_filter=0,
    save_path=output_dir / 'normal_example.png'
)
```

### Example 2: Generate Multiple Comparisons

```python
# Generate 3 pairs of pre-AF vs normal
for i in range(3):
    viz.plot_subband_rps(
        label_filter=1,
        save_path=output_dir / f'preaf_{i+1}.png'
    )

    viz.plot_subband_rps(
        label_filter=0,
        save_path=output_dir / f'normal_{i+1}.png'
    )
```

### Example 3: Specific Signal by Index

```python
# Get specific pre-AF signals
preaf_indices = viz.get_random_indices(label=1, n=5)

for idx in preaf_indices:
    viz.plot_subband_rps(
        idx=idx,
        save_path=output_dir / f'preaf_idx{idx}.png'
    )
```

### Example 4: Different Lead Times

```python
# Compare across lead times
for lt in [5, 10, 15]:
    viz_lt = MITBIHVisualizer(lead_time=lt)

    # Pre-AF at this lead time
    viz_lt.plot_subband_rps(
        label_filter=1,
        save_path=output_dir / f'preaf_lt{lt}.png'
    )

    # Normal at this lead time
    viz_lt.plot_subband_rps(
        label_filter=0,
        save_path=output_dir / f'normal_lt{lt}.png'
    )
```

---

## Subband Parameters (Optional)

You can specify custom RQA parameters for each subband:

```python
subband_params = {
    "original": {"tau": 21, "m": 7},
    "D1": {"tau": 6, "m": 7},
    "D2": {"tau": 4, "m": 7},
    "D3": {"tau": 2, "m": 7},
    "D4": {"tau": 2, "m": 7},
    "A4": {"tau": 16, "m": 8},
}

viz = MITBIHVisualizer(subband_params=subband_params, lead_time=5)
```

**Parameters:**
- `tau` - Time delay for phase space reconstruction
- `m` - Embedding dimension

---

## Output Interpretation

### Plot Layout

```
┌─────────────────┬─────────────────┐
│ Original Signal │  Original RP    │  Row 0
├─────────────────┼─────────────────┤
│ D1 (64-128 Hz)  │  D1 RP          │  Row 1
├─────────────────┼─────────────────┤
│ D2 (32-64 Hz)   │  D2 RP          │  Row 2
├─────────────────┼─────────────────┤
│ D3 (16-32 Hz)   │  D3 RP          │  Row 3
├─────────────────┼─────────────────┤
│ D4 (4-8 Hz)     │  D4 RP          │  Row 4
├─────────────────┼─────────────────┤
│ A4 (<4 Hz)      │  A4 RP          │  Row 5
└─────────────────┴─────────────────┘
```

### Signal Colors

- **Red:** Pre-AF signals (label=1)
- **Blue:** Normal signals (label=0)

### RP Title Information

Each RP shows:
- `tau` - Time delay used
- `m` - Embedding dimension used
- `RR` - Recurrence rate (percentage of recurrent points)

---

## Complete Workflow

```python
from visualizers.mitbih_visualizer import MITBIHVisualizer, OUTPUT_DIR
from pathlib import Path

# Configuration
LEAD_TIME = 5
N_EXAMPLES = 3

# Initialize
viz = MITBIHVisualizer(lead_time=LEAD_TIME)
output_dir = OUTPUT_DIR / 'visualizations' / f'lt{LEAD_TIME}'
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Loaded {len(viz.signals)} signals")
print(f"  Pre-AF: {sum(viz.labels == 1)}")
print(f"  Normal: {sum(viz.labels == 0)}")

# Generate examples
for i in range(N_EXAMPLES):
    print(f"\nGenerating pair {i+1}/{N_EXAMPLES}...")

    # Pre-AF
    viz.plot_subband_rps(
        label_filter=1,
        save_path=output_dir / f'preaf_{i+1}.png'
    )

    # Normal
    viz.plot_subband_rps(
        label_filter=0,
        save_path=output_dir / f'normal_{i+1}.png'
    )

print(f"\nDone! Visualizations saved to: {output_dir}")

# List files
for f in sorted(output_dir.glob('*.png')):
    print(f"  - {f.name}")
```

---

## Troubleshooting

### Error: "No signals found with label=0"

**Cause:** CSV file doesn't have normal signals, only pre-AF.

**Solution:** Run the dataset preparation pipeline first:
```python
from pipeline_mitbih.pipeline_mitbih import MITBIHPipeline

pipeline = MITBIHPipeline(lead_time=5)
pipeline.prepare_dataset()
pipeline.export_to_csv()
```

This creates `results/dataset_lt5.csv` with both pre-AF and normal segments.

### Error: "File not found: dataset_lt5.csv"

**Cause:** Dataset CSV doesn't exist for that lead time.

**Solution:** Specify the correct lead time or generate the dataset first.

### Signals look identical

**Cause:** Might be loading from wrong CSV or signals not properly prepared.

**Solution:** Check that:
1. CSV has both label=0 and label=1 rows
2. Signals are from different patients
3. Dataset was prepared correctly with matched segments

---

## Technical Notes

### Signal Processing

Signals in the CSV are **already processed** with this pipeline:
1. Raw ECG
2. Basic filtering (highpass + lowpass)
3. SWT adaptive denoising
4. Z-score normalization

### Subband Decomposition

For visualization:
1. Load cleaned signal from CSV
2. SWT decomposition (db4, level 4)
3. Reconstruct each subband independently
4. Z-score normalize each subband
5. Compute RP for each subband

### Frequency Bands

- **D1:** 64-128 Hz (high frequency, noise)
- **D2:** 32-64 Hz (medium-high)
- **D3:** 16-32 Hz (medium)
- **D4:** 4-8 Hz (respiratory sinus arrhythmia)
- **A4:** <4 Hz (very low frequency, trends)

---

## Summary

**Key Points:**
1. Visualizer loads from CSV files with both pre-AF and normal signals
2. Use `label_filter=1` for pre-AF, `label_filter=0` for normal
3. Signals are already cleaned in the CSV
4. Each plot shows original signal + 5 subbands with their RPs
5. Can compare across different lead times by changing initialization

**Quick commands:**
```python
# Pre-AF
viz = MITBIHVisualizer(lead_time=5)
viz.plot_subband_rps(label_filter=1)

# Normal
viz.plot_subband_rps(label_filter=0)
```

That's it! You now have visualizations for both pre-AF and normal signals.
