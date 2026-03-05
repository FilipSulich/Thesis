# Three Methods for Wavelet-based Recurrence Plot Analysis

This directory contains visualizations comparing three different approaches for computing recurrence plots from wavelet-decomposed ECG signals.

## Methods Overview

### Method 1: SWT + Reconstruction → RP
**File suffix:** `_method1_swt_recon.png`

- Uses **Stationary Wavelet Transform (SWT)** for decomposition
- Each subband is **reconstructed by zeroing out all other coefficients**
  - For D1: Zero out D2, D3, D4, and A4 coefficients, keep only D1
  - For A4: Zero out all detail coefficients (D1-D4), keep only A4
- RPs are computed from the **reconstructed signals**
- **Advantages:**
  - Each reconstruction contains ONLY that subband's information
  - Isolates frequency band dynamics perfectly
  - Time-domain representation of individual subbands
  - Standard approach in wavelet-based signal analysis
- **File structure:** Shows reconstructed subbands (left) and their RPs (right)

### Method 2: SWT → Direct RP from Coefficients
**File suffix:** `_method2_swt_direct.png`

- Uses **Stationary Wavelet Transform (SWT)** for decomposition
- RPs are computed **directly from wavelet coefficients** (no reconstruction)
- Works with raw coefficient arrays
- **Advantages:**
  - No reconstruction step required
  - Potentially captures coefficient-space dynamics
  - More computationally efficient
- **File structure:** Shows raw coefficients (left) and their RPs (right)

### Method 3: DWT + Reconstruction → RP
**File suffix:** `_method3_dwt_recon.png`

- Uses **Discrete Wavelet Transform (DWT)** for decomposition
- Each subband is **reconstructed by zeroing out all other coefficients**
  - For D1: Zero out D2, D3, D4, and A4 coefficients, keep only D1
  - For A4: Zero out all detail coefficients (D1-D4), keep only A4
- Inverse DWT is applied to get time-domain signals
- RPs are computed from the **reconstructed signals**
- **Advantages:**
  - DWT is more commonly used than SWT in literature
  - Orthogonal decomposition
  - Better energy compaction
  - Downsampled coefficients (more efficient)
- **File structure:** Shows reconstructed subbands (left) and their RPs (right)

## Key Differences

| Aspect | Method 1 (SWT+Recon) | Method 2 (SWT Direct) | Method 3 (DWT+Recon) |
|--------|---------------------|----------------------|---------------------|
| Transform | SWT | SWT | DWT |
| Reconstruction | Yes | No | Yes |
| Input to RP | Time-domain signal | Wavelet coefficients | Time-domain signal |
| Signal length | Same as original | Variable (shorter) | Same as original |
| Translation invariance | Yes (SWT) | Yes (SWT) | No (DWT) |

## Dataset Information

- **Normal segments:** From patients with IDs containing '_n' (e.g., paf_n20)
  - Total available: 3,929 segments from 6,337 normal labels
- **Pre-AF segments:** From patients with pre-AF labels (e.g., paf_p44)
  - Total available: 2,398 segments

## Files Generated

Each signal has 3 files (one per method):
- `normal_X_[segment_id]_method1_swt_recon.png`
- `normal_X_[segment_id]_method2_swt_direct.png`
- `normal_X_[segment_id]_method3_dwt_recon.png`
- `preaf_X_[segment_id]_method1_swt_recon.png`
- `preaf_X_[segment_id]_method2_swt_direct.png`
- `preaf_X_[segment_id]_method3_dwt_recon.png`

## Subbands

All methods decompose into 5 subbands:
- **D1:** Highest frequency details (64-128 Hz at 128 Hz sampling)
- **D2:** High frequency details (32-64 Hz)
- **D3:** Medium frequency details (16-32 Hz)
- **D4:** Low frequency details (8-16 Hz)
- **A4:** Approximation/Low frequency (0-8 Hz)

## RP Parameters

- **Epsilon method:** Percentile (5th percentile)
- **Embedding parameters:** Subband-specific (tau and m from subband_params)
- **Metric:** Euclidean distance

## Fixed Issues

- **A4 reconstruction:** Now properly reconstructed using full wavelet reconstruction instead of raw coefficients
- **Normal segment filtering:** Only includes segments from patients with '_n' in their ID