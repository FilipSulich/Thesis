"""
Example: Extract Subbands from MIT-BIH Dataset

Shows how to use the export_subbands_to_csv() method in MITBIHPipeline.
"""

from pathlib import Path
from pipeline_mitbih import MITBIHPipeline

# =============================================================================
# Configuration
# =============================================================================

LEAD_TIME = 5  # minutes before AF onset

# Cleaning parameters (same as dataset preparation)
APPLY_CLEANING = True
HIGHPASS_CUTOFF = 0.5  # Hz
NOTCH_FREQ = 60.0  # Hz
NOTCH_Q = 30.0
NORMALIZE_METHOD = 'zscore'
SWT_DENOISE = True
SWT_LEVEL = 4
NORMAL_BUFFER_MIN = 60  # Minutes from AF for normal segments

# Subband extraction parameters
WAVELET = 'db4'  # Wavelet type
DECOMPOSITION_LEVEL = 4  # Number of levels

# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    # Initialize pipeline
    pipeline = MITBIHPipeline(
        lead_time=LEAD_TIME,
        normal_buffer_min=NORMAL_BUFFER_MIN,
        apply_cleaning=APPLY_CLEANING,
        highpass_cutoff=HIGHPASS_CUTOFF,
        notch_freq=NOTCH_FREQ,
        notch_q=NOTCH_Q,
        normalize_method=NORMALIZE_METHOD,
        swt_denoise=SWT_DENOISE,
        swt_level=SWT_LEVEL,
    )

    # Option 1: First prepare dataset, then extract subbands
    print("Step 1: Preparing dataset...")
    preaf_segments, normal_segments = pipeline.prepare_dataset()
    pipeline.export_to_csv()

    print("\nStep 2: Extracting subbands...")
    pipeline.export_subbands_to_csv(wavelet=WAVELET, level=DECOMPOSITION_LEVEL)

    # Option 2: Extract subbands from existing CSV
    # (if you already ran prepare_dataset() and have the CSV)
    # pipeline.export_subbands_to_csv(
    #     csv_path=Path('results/dataset_lt5.csv'),
    #     output_dir=Path('results/subbands'),
    #     wavelet=WAVELET,
    #     level=DECOMPOSITION_LEVEL
    # )

    print("\nComplete! Subband CSV files created:")
    print("  - mitbih_lt5_D1.csv (64-128 Hz)")
    print("  - mitbih_lt5_D2.csv (32-64 Hz)")
    print("  - mitbih_lt5_D3.csv (16-32 Hz)")
    print("  - mitbih_lt5_D4.csv (4-8 Hz)")
    print("  - mitbih_lt5_A4.csv (<4 Hz)")
