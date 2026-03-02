# Subband Extractor Guide

Complete guide for extracting decomposed, cleaned ECG signals into separate CSV files per subband.

## Overview

The `SubbandExtractor` class performs the following pipeline:

1. **Load cleaned signal** (from HDF5 or CSV)
2. **SWT decomposition** (Stationary Wavelet Transform)
3. **Reconstruct each subband** independently
4. **Z-score normalize** each subband separately
5. **Save to CSV** (one file per subband)

---

## Quick Start

### PAF Dataset

```bash
cd /Users/filipsulich/Developer/uni/Thesis/Thesis/src/extractors
python subband_extractor.py --dataset paf
```

### MIT-BIH Dataset (Single Lead Time)

```bash
python subband_extractor.py --dataset mitbih --lead_time 5
```

### MIT-BIH Dataset (All Lead Times)

```bash
python subband_extractor.py --dataset mitbih_all
```

---

## Output Files

### PAF Dataset

**Input:** `paf_segments_all_clean_new_cleaning_min_max_with_upper.h5`

**Output directory:** `data/extracted/paf/subbands/`

**Files generated:**
- `paf_D1.csv` - Detail 1 (64-128 Hz)
- `paf_D2.csv` - Detail 2 (32-64 Hz)
- `paf_D3.csv` - Detail 3 (16-32 Hz)
- `paf_D4.csv` - Detail 4 (4-8 Hz)
- `paf_A4.csv` - Approximation (<4 Hz)

### MIT-BIH Dataset

**Input:** `mitbih_cleaned_lt{X}.csv`

**Output directory:** `data/extracted/mitbih/subbands/`

**Files generated (per lead time):**
- `mitbih_lt5_D1.csv`
- `mitbih_lt5_D2.csv`
- `mitbih_lt5_D3.csv`
- `mitbih_lt5_D4.csv`
- `mitbih_lt5_A4.csv`

---

## CSV File Format

Each subband CSV contains:

**Metadata columns:**
- `segment_id` - Unique segment identifier
- `patient_id` - Patient identifier
- `record_id` - Record identifier
- `label` - 0 = normal, 1 = pre-AF

**Signal columns:**
- `s0`, `s1`, `s2`, ..., `sN` - Normalized subband signal samples

**Example:**
```
segment_id,patient_id,record_id,label,s0,s1,s2,...
paf_n01_s0_l0,paf_n01,n01,0,0.123,-0.456,0.789,...
paf_p02_s0_l0,paf_p01,p02,1,-0.234,0.567,-0.123,...
```

---

## Python API Usage

### Basic Usage

```python
from extractors.subband_extractor import SubbandExtractor
from pathlib import Path

# Initialize extractor
extractor = SubbandExtractor(wavelet='rbio3.9', level=4)

# Extract from HDF5
extractor.extract_from_hdf5(
    hdf5_path=Path('data/extracted/paf/paf_segments_all.h5'),
    output_dir=Path('data/extracted/paf/subbands'),
    dataset_name='paf',
    split='all'
)

# Extract from CSV
extractor.extract_from_csv(
    csv_path=Path('data/extracted/mitbih/mitbih_cleaned_lt5.csv'),
    output_dir=Path('data/extracted/mitbih/subbands'),
    dataset_name='mitbih_lt5'
)
```

### Load Subband Data

```python
# Load a specific subband
signals, labels, segment_ids, patient_ids, record_ids = extractor.load_subband_csv(
    csv_path=Path('data/extracted/paf/subbands/paf_D4.csv')
)

print(f"Loaded {len(signals)} D4 subband signals")
print(f"Signal shape: {signals.shape}")
print(f"Labels: {len(labels[labels==1])} pre-AF, {len(labels[labels==0])} normal")
```

### Process Single Signal

```python
import numpy as np

# Example: decompose a single cleaned signal
signal = np.random.randn(2500)  # Your cleaned signal

extractor = SubbandExtractor(wavelet='rbio3.9', level=4)
subbands = extractor.decompose_and_normalize(signal)

# Access individual subbands
d1 = subbands['D1']  # 64-128 Hz
d2 = subbands['D2']  # 32-64 Hz
d3 = subbands['D3']  # 16-32 Hz
d4 = subbands['D4']  # 4-8 Hz
a4 = subbands['A4']  # <4 Hz

print(f"D4 shape: {d4.shape}, mean: {np.mean(d4):.3f}, std: {np.std(d4):.3f}")
```

---

## Integration with RQA Pipeline

### Option 1: Use Subband CSVs Directly

```python
from extractors.subband_extractor import SubbandExtractor
from pipeline_paf.pipeline_paf import PAFRQAPipeline

# Extract subbands
extractor = SubbandExtractor()
extractor.extract_from_hdf5(...)

# Load D4 subband
signals, labels, segment_ids, patient_ids, record_ids = extractor.load_subband_csv(
    csv_path=Path('data/extracted/paf/subbands/paf_D4.csv')
)

# Run RQA on D4 subband only
# (signals are already normalized, ready for embedding)
```

### Option 2: On-the-fly Decomposition

```python
# In your RQA pipeline, decompose each signal
extractor = SubbandExtractor()

for signal in signals:
    subbands = extractor.decompose_and_normalize(signal)

    # Compute RQA for each subband
    for sb_name, sb_signal in subbands.items():
        # Embedding -> RP -> RQA
        pass
```

---

## Technical Details

### SWT Decomposition

**Wavelet:** `rbio3.9` (Reverse Biorthogonal)
**Level:** 4

**Frequency bands (assuming 250 Hz sampling):**
- **D1:** 64-128 Hz (high frequency, noise)
- **D2:** 32-64 Hz (medium-high frequency)
- **D3:** 16-32 Hz (medium frequency)
- **D4:** 4-8 Hz (low frequency, respiratory sinus arrhythmia)
- **A4:** <4 Hz (very low frequency, trends)

### Normalization

Each subband is independently z-score normalized:

```
normalized = (subband - mean(subband)) / std(subband)
```

**Properties after normalization:**
- Mean ≈ 0
- Standard deviation ≈ 1
- Preserves relative dynamics within each frequency band

### Reconstruction

For each subband, we:
1. Zero out all other coefficients
2. Reconstruct from zeroed coefficients
3. This isolates that specific frequency band

**Example for D4:**
```python
# Zero all coefficients except D4 detail
zeroed = [(zeros, zeros), (zeros, zeros), (zeros, zeros), (zeros, D4_detail), ...]
d4_signal = reconstruct(zeroed)
```

---

## Comparison: Single-Scale vs Multi-Scale

### Single-Scale Pipeline
```
Raw → Filter → SWT → Threshold → Reconstruct → ONE signal → Normalize → RQA
```

**Output:** 10 RQA metrics for full signal

### Multi-Scale Pipeline (using SubbandExtractor)
```
Raw → Filter → SWT → Threshold → Reconstruct →
  ├─ D1 → Normalize → RQA → 10 metrics
  ├─ D2 → Normalize → RQA → 10 metrics
  ├─ D3 → Normalize → RQA → 10 metrics
  ├─ D4 → Normalize → RQA → 10 metrics
  └─ A4 → Normalize → RQA → 10 metrics
```

**Output:** 50 RQA metrics total (10 per subband)

---

## Use Cases

### 1. Multi-Scale RQA Analysis

Extract subbands, compute RQA metrics for each:

```python
from extractors.subband_extractor import SubbandExtractor
from pathlib import Path

extractor = SubbandExtractor()

# Extract all subbands
extractor.extract_from_csv(
    csv_path=Path('data/mitbih_cleaned_lt5.csv'),
    output_dir=Path('data/subbands'),
    dataset_name='mitbih_lt5'
)

# Run RQA on each subband CSV
# Compare which frequency bands show strongest pre-AF differences
```

### 2. Frequency-Specific Analysis

Focus on specific frequency bands:

```python
# Only analyze D4 (respiratory sinus arrhythmia range)
signals_d4, labels, _, _, _ = extractor.load_subband_csv(
    Path('data/subbands/paf_D4.csv')
)

# Compare pre-AF vs normal in this frequency band
preaf_d4 = signals_d4[labels == 1]
normal_d4 = signals_d4[labels == 0]
```

### 3. Feature Engineering

Use subbands as features for machine learning:

```python
import pandas as pd

# Load all subbands
d1 = pd.read_csv('data/subbands/paf_D1.csv')
d2 = pd.read_csv('data/subbands/paf_D2.csv')
d3 = pd.read_csv('data/subbands/paf_D3.csv')
d4 = pd.read_csv('data/subbands/paf_D4.csv')
a4 = pd.read_csv('data/subbands/paf_A4.csv')

# Compute statistics per subband
features = pd.DataFrame({
    'segment_id': d1['segment_id'],
    'label': d1['label'],
    'd1_std': d1.filter(like='s').std(axis=1),
    'd2_std': d2.filter(like='s').std(axis=1),
    'd3_std': d3.filter(like='s').std(axis=1),
    'd4_std': d4.filter(like='s').std(axis=1),
    'a4_std': a4.filter(like='s').std(axis=1),
})
```

---

## Advanced Configuration

### Custom Wavelet

```python
# Use different wavelet
extractor = SubbandExtractor(wavelet='db4', level=4)
```

**Available wavelets:** 'rbio3.9', 'db4', 'sym4', 'coif4', etc.

### Custom Decomposition Level

```python
# 3-level decomposition (fewer subbands)
extractor = SubbandExtractor(wavelet='rbio3.9', level=3)
# Outputs: D1, D2, D3, A3
```

---

## Batch Processing

### Process All MIT-BIH Lead Times

```python
from extractors.subband_extractor import extract_all_mitbih_subbands

# Extract subbands for all lead times (5, 10, 15, 30, 45 min)
extract_all_mitbih_subbands()
```

**Output:**
```
data/extracted/mitbih/subbands/
├── mitbih_lt5_D1.csv
├── mitbih_lt5_D2.csv
├── ...
├── mitbih_lt10_D1.csv
├── mitbih_lt10_D2.csv
├── ...
└── mitbih_lt45_A4.csv
```

---

## Verification

### Check Output Files

```python
import pandas as pd
from pathlib import Path

output_dir = Path('data/extracted/paf/subbands')

for subband in ['D1', 'D2', 'D3', 'D4', 'A4']:
    csv_path = output_dir / f'paf_{subband}.csv'
    df = pd.read_csv(csv_path)

    signal_cols = [c for c in df.columns if c.startswith('s')]
    signals = df[signal_cols].values

    print(f"{subband}:")
    print(f"  Segments: {len(df)}")
    print(f"  Signal shape: {signals.shape}")
    print(f"  Mean: {signals.mean():.6f}")
    print(f"  Std: {signals.std():.6f}")
    print(f"  Pre-AF: {len(df[df['label']==1])}, Normal: {len(df[df['label']==0])}")
    print()
```

**Expected output:**
```
D1:
  Segments: 1234
  Signal shape: (1234, 2500)
  Mean: -0.000001
  Std: 1.000023
  Pre-AF: 567, Normal: 667

D2:
  Segments: 1234
  Signal shape: (1234, 2500)
  Mean: 0.000002
  Std: 0.999987
  Pre-AF: 567, Normal: 667
...
```

Note: Mean ≈ 0 and Std ≈ 1 confirms proper normalization.

---

## Troubleshooting

### Error: "Signal length changed after decomposition"

**Cause:** SWT may change signal length slightly due to padding.

**Solution:** This is expected and handled automatically. All subbands will have the same length for a given input signal.

### Error: "Memory error when processing large datasets"

**Cause:** Loading entire dataset into memory.

**Solution:** Process in batches:

```python
# Process CSV in chunks
for chunk in pd.read_csv(csv_path, chunksize=1000):
    # Process chunk
    pass
```

### Unexpected NaN values

**Cause:** Division by zero in normalization (std = 0).

**Solution:** Already handled with `+ 1e-8` epsilon in code. If issue persists, check input signal quality.

---

## Next Steps

1. **Extract subbands** for your dataset
2. **Run RQA analysis** on each subband CSV
3. **Compare results** across frequency bands
4. **Identify critical frequencies** showing strongest pre-AF differences
5. **Use for machine learning** as frequency-domain features

---

## Complete Example Workflow

```python
from extractors.subband_extractor import SubbandExtractor
from pathlib import Path
import pandas as pd

# 1. Initialize extractor
extractor = SubbandExtractor(wavelet='rbio3.9', level=4)

# 2. Extract subbands from PAF dataset
extractor.extract_from_hdf5(
    hdf5_path=Path('data/extracted/paf/paf_segments_all.h5'),
    output_dir=Path('data/extracted/paf/subbands'),
    dataset_name='paf',
    split='all'
)

# 3. Load specific subband
signals_d4, labels, segment_ids, patient_ids, record_ids = extractor.load_subband_csv(
    csv_path=Path('data/extracted/paf/subbands/paf_D4.csv')
)

print(f"Loaded D4 subband: {signals_d4.shape}")
print(f"Pre-AF segments: {(labels==1).sum()}")
print(f"Normal segments: {(labels==0).sum()}")

# 4. Verify normalization
print(f"Mean: {signals_d4.mean():.6f} (should be ~0)")
print(f"Std: {signals_d4.std():.6f} (should be ~1)")

# 5. Ready for RQA analysis!
# (signals are normalized and ready for embedding)
```

---

**The SubbandExtractor is ready to use!**
