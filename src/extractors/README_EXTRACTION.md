# PAF Extractor - Extract ALL Patients (n* and p*)

## Overview

The extractor now supports **two extraction modes**:
1. **Extract ALL** - Get all segments from all patients (n* and p*) without balancing
2. **Extract Balanced** - Balance pre-AF and normal counts with controlled n* proportion

## Extraction Modes

### Mode 1: Extract ALL (Recommended for Independent Analysis)
**New default mode** - Extracts all segments from all patients without balancing.

**Strategy:**
1. Extract from ALL p-even records (pre-AF)
2. Extract from ALL n* records (healthy subjects)
3. Extract from ALL p-odd records (PAF patients, normal rhythm)
4. **No balancing** - return everything

**Benefits:**
- Maximum data utilization
- Proper representation of all patient groups
- Supports independent sample analysis
- No artificial balancing

**Output Example:**
```
Pre-AF (p-even)  : 2398
Normal (n*)      : 937
Normal (p-odd)   : 2400
Total Normal     : 3337
Total All        : 5735
```

### Mode 2: Extract Balanced (Legacy Mode)
Balances pre-AF and normal counts with configurable n* proportion.

**Strategy:**
1. Extract from ALL p-even records (pre-AF)
2. Extract n* records (configurable % of normal quota)
3. Fill remaining quota from p-odd records
4. Balance: normal == pre-AF

**Benefits:**
- Equal class sizes for some ML algorithms
- Controlled n* representation
- Backward compatible

## Configuration Parameters

### `balanced` (bool, default=False)
- `False`: Extract ALL mode (recommended)
- `True`: Extract Balanced mode (legacy)

### `segments_per_record` (int, default=100)
- Number of segments to extract from each record
- Higher values = more data per patient

### For Balanced Mode Only:

#### `include_n_records` (bool, default=True)
- Whether to include n* (healthy subject) records
- Set to `False` to use only paired p-odd records

#### `n_record_proportion` (float, default=0.3)
- Proportion of normal quota to reserve for n* records
- Range: 0.0 to 1.0
- Examples:
  - `0.0`: No n* records (all from p-odd)
  - `0.3`: 30% from n*, 70% from p-odd
  - `0.5`: 50/50 split
  - `1.0`: All from n* records (if possible)

## Dataset Structure

### PAF Dataset Records
- **n* records**: Healthy subjects with no AF history (n01, n02, ...)
- **p-odd records**: PAF subjects during normal rhythm (p01, p03, p05, ...)
- **p-even records**: PAF subjects immediately before AF onset (p02, p04, p06, ...)

### Labels
- `label=0`: Normal rhythm
- `label=1`: Pre-AF (immediately precedes AF onset)

## Usage

### Mode 1: Extract ALL (Recommended)
```python
from extractors.paf_extractor import PAFExtractor

extractor = PAFExtractor(random_seed=42, enable_subband=True)

# Extract ALL segments from ALL patients
extractor.export_to_hdf5(
    segments_per_record=100,
    export_csv=True,
    balanced=False  # Extract all, no balancing
)
```

### Mode 2: Extract Balanced
```python
# Extract with balancing and 30% n* records
extractor.export_to_hdf5(
    segments_per_record=100,
    export_csv=True,
    balanced=True,
    include_n_records=True,
    n_record_proportion=0.3
)

# Extract with 50% n* records
extractor.export_to_hdf5(
    segments_per_record=100,
    balanced=True,
    include_n_records=True,
    n_record_proportion=0.5  # Half from n*, half from p-odd
)

# Extract without n* records (only p-odd)
extractor.export_to_hdf5(
    segments_per_record=100,
    balanced=True,
    include_n_records=False
)
```

## Output

### HDF5 File
- Path: `data/extracted/paf_updated/paf_segments_all_clean_new_cleaning.h5`
- Contains:
  - Signals from pre-AF (p-even) and normal (n* + p-odd) records
  - Patient IDs preserved (e.g., `paf_n01`, `paf_p01`)
  - Balanced: equal number of pre-AF and normal segments

### Console Output Example
```
Extraction complete:
  Pre-AF : 1000
  Normal : 1000
    n* records   : 300
    paired p-odd : 700
```

## Patient ID Format

- **n* records**: `paf_n01`, `paf_n02`, ... (healthy subjects)
- **p-pair records**: `paf_p01`, `paf_p02`, ... (PAF patients)

This allows filtering in the analysis pipeline:
```python
# Filter for n* patients only
is_n_patient = '_n' in patient_id.lower()
```

## Benefits

1. **Better Control Group**: n* patients are truly healthy (no AF history)
2. **Independent Samples**: n* patients are independent from pre-AF patients
3. **Proper Statistical Testing**: Enables Mann-Whitney U test (independent samples)
4. **Configurable Mix**: Balance between truly healthy (n*) and paired normal (p-odd)

## Recommended Settings

For **independent sample analysis** (current approach):
- `include_n_records=True`
- `n_record_proportion=0.3` to `0.5`

For **paired sample analysis** (old approach):
- `include_n_records=False`
- Uses only p-odd records paired with p-even

## Quality Control

Both n* and p-odd records use the same quality thresholds:
- `MAX_RR_CV_NORMAL = 0.6`
- Heart rate variability check
- Artifact removal
- Baseline wander correction
