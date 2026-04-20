"""
Quick visualization of UTRP generation - test on a small subset before full dataset generation
"""

import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import wfdb
from scipy import signal as sp_signal

from src.CNN.dataset.dataset_generator import (
    compute_utrp, 
    extract_subbands_no_denoise,
    denoise_and_extract_subbands,
    _minmax,
    _pad,
    FS_ORIG,
    FS_TARGET,
    SEGMENT_SAMPLES,
    N_LEVELS_SWT,
    SUBBAND_LEVELS,
    WAVELET,
)

# Configuration
DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
N_TEST_RECORDS = 3  # Test on 3 records only
N_SEGMENTS_PER_RECORD = 2  # Only 2 segments per record

print("=" * 70)
print("UTRP VISUALIZATION - QUICK CHECK")
print("=" * 70)
print(f"\nTesting on {N_TEST_RECORDS} records with {N_SEGMENTS_PER_RECORD} segments each")
print(f"Wavelet: {WAVELET}  |  Colormap: JET\n")

# Helper functions
def _load_signal(record_id: str):
    """Load and resample ECG signal."""
    try:
        rec = wfdb.rdrecord(str(DATA_DIR / record_id))
        raw = rec.p_signal[:, 0].astype(np.float64)
        n = int(len(raw) * FS_TARGET / FS_ORIG)
        return sp_signal.resample(raw, n).astype(np.float32)
    except Exception as e:
        print(f"  ✗ Failed to load {record_id}: {e}")
        return None


def _extract_segments(signal: np.ndarray, n_segments: int) -> list:
    """Extract n_segments from signal."""
    max_start = len(signal) - SEGMENT_SAMPLES
    if max_start <= 0:
        return []
    starts = np.linspace(0, max_start, n_segments, dtype=int)
    segs = []
    for s in starts:
        seg = signal[s:s + SEGMENT_SAMPLES]
        if len(seg) < SEGMENT_SAMPLES:
            continue
        segs.append(_minmax(seg.astype(np.float32)))
    return segs


# Get test records (mix of NSR and pre-AF)
test_records = [
    ('n01', 0, 'NSR'),      # Normal sinus rhythm
    ('n02', 0, 'NSR'),      # Normal sinus rhythm
    ('p01', 1, 'Pre-AF'),   # Paroxysmal AF
][:N_TEST_RECORDS]

# Generate UTRPs for visualization
all_images = []

for record_id, label, label_name in test_records:
    print(f"\nProcessing {record_id} ({label_name})...")
    
    try:
        # Load signal
        signal = _load_signal(record_id)
        if signal is None:
            print(f"  ✗ Failed to load signal")
            continue
        
        # Extract segments
        segments = _extract_segments(signal, n_segments=N_SEGMENTS_PER_RECORD)
        if not segments:
            print(f"  ✗ No segments extracted")
            continue
        
        print(f"  ✓ Loaded signal, extracted {len(segments)} segments")
        
        # Process first segment only for preview
        seg = segments[0]
        seg_pad = _pad(seg)
        
        # Single-scale: denoise and get full reconstruction
        # seg_den, _ = denoise_and_extract_subbands(
        #     seg_pad, WAVELET, N_LEVELS_SWT, SUBBAND_LEVELS
        # )
        
        # Generate single-scale UTRP from denoised signal
        utrp_single = compute_utrp(seg, tau=12, m=3)
        if utrp_single is None:
            print(f"  ✗ Failed to compute single-scale UTRP")
            continue
        
        # Multi-scale: extract subbands WITHOUT denoising
        subbands_multi = extract_subbands_no_denoise(
            seg_pad, WAVELET, N_LEVELS_SWT, SUBBAND_LEVELS
        )
        
        # Generate multi-scale UTRPs (D2-D8)
        utrps_multi = []
        for level in SUBBAND_LEVELS:
            subband_name = f'D{level}'
            utrp = compute_utrp(subbands_multi[subband_name], tau=3, m=3)
            if utrp is not None:
                utrps_multi.append(utrp)
        
        print(f"  ✓ Generated single-scale UTRP: {utrp_single.shape}")
        print(f"  ✓ Generated {len(utrps_multi)} multi-scale UTRPs")
        
        all_images.append({
            'record': record_id,
            'label': label_name,
            'single': utrp_single,
            'multi': utrps_multi
        })
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()


print(f"\n✓ Generated {len(all_images)} preview sets\n")

# Create visualization
if all_images:
    # Figure 1: Single-scale UTRPs
    fig1 = plt.figure(figsize=(15, 4 * len(all_images)))
    fig1.suptitle('Single-Scale UTRPs (Single.png)', fontsize=16, fontweight='bold')
    
    for idx, data in enumerate(all_images):
        ax = fig1.add_subplot(len(all_images), 1, idx + 1)
        img = data['single']  # Shape: (3, 256, 256)
        img_display = img.transpose(1, 2, 0)  # Convert to (256, 256, 3) for display
        im = ax.imshow(img_display)
        ax.set_title(f"{data['record']} ({data['label']})", fontsize=12, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Intensity')
    
    plt.tight_layout()
    plt.savefig('utrp_preview_single.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: utrp_preview_single.png")
    plt.close()
    
    # Figure 2: Multi-scale UTRPs (one per record, showing all D2-D8)
    for data_idx, data in enumerate(all_images):
        n_levels = len(data['multi'])
        fig2 = plt.figure(figsize=(16, 3 * ((n_levels + 2) // 3)))
        fig2.suptitle(f"Multi-Scale UTRPs - {data['record']} ({data['label']})", 
                     fontsize=16, fontweight='bold')
        
        for level_idx, utrp in enumerate(data['multi']):
            level = 2 + level_idx
            ax = fig2.add_subplot((n_levels + 2) // 3, 3, level_idx + 1)
            utrp_display = utrp.transpose(1, 2, 0)  # Convert to (256, 256, 3) for display
            im = ax.imshow(utrp_display)
            ax.set_title(f"D{level} Subband UTRP", fontsize=11, fontweight='bold')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            plt.colorbar(im, ax=ax, label='Intensity')
        
        plt.tight_layout()
        filename = f"utrp_preview_multi_{data['record']}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {filename}")
        plt.close()
    
    # Figure 3: Comparison grid
    fig3, axes = plt.subplots(len(all_images), 2, figsize=(12, 5 * len(all_images)))
    if len(all_images) == 1:
        axes = axes.reshape(1, -1)
    
    fig3.suptitle('Quick Comparison: Single vs First Multi-Scale', fontsize=16, fontweight='bold')
    
    for idx, data in enumerate(all_images):
        # Single-scale
        ax = axes[idx, 0]
        single_display = data['single'].transpose(1, 2, 0)  # Convert to (256, 256, 3)
        im1 = ax.imshow(single_display)
        ax.set_title(f"{data['record']} - Single-Scale", fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im1, ax=ax, label='Intensity')
        
        # First multi-scale (D2)
        ax = axes[idx, 1]
        multi_display = data['multi'][0].transpose(1, 2, 0)  # Convert to (256, 256, 3)
        im2 = ax.imshow(multi_display)
        ax.set_title(f"{data['record']} - D2 Multi-Scale", fontsize=11, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im2, ax=ax, label='Intensity')
    
    plt.tight_layout()
    plt.savefig('utrp_preview_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ Saved: utrp_preview_comparison.png")
    plt.close()
    
    print("\n" + "=" * 70)
    print("PREVIEW COMPLETE")
    print("=" * 70)
    print("\nGenerated 3 preview images:")
    print("  1. utrp_preview_single.png       - All single-scale UTRPs")
    print("  2. utrp_preview_multi_*.png      - Multi-scale UTRPs for each record")
    print("  3. utrp_preview_comparison.png   - Quick side-by-side comparison")
    print("\n✓ Images look good? Proceed with full dataset generation!")
    print("✓ Images look bad/black? Check dataset_generator.py compute_utrp() method\n")
    
else:
    print("✗ Failed to generate any preview images")

