#!/usr/bin/env python3
"""Test the modified PAF pipeline with a small sample."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline_paf.pipeline_paf import PAFPipeline

OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated')

if __name__ == '__main__':
    pipeline = PAFPipeline()

    # Test with small sample
    max_samples = 10
    results_dir = OUTPUT_DIR / 'results_test'

    print("\n" + "="*80)
    print(f"Testing pipeline with {max_samples} samples (Pre-AF vs n* Normal patients)")
    print("="*80)

    # Test Method 0: Single-scale
    print("\n### Testing Method 0: Single-scale RQA ###")
    results_single = pipeline.run_batch(
        method='single_scale',
        filter_patients=True,
        max_samples=max_samples
    )
    print(f"Processed {len(results_single)} segments")
    if results_single:
        print(f"Sample result: {results_single[0].to_dict()}")
    pipeline.results_to_csv(results_single, results_dir / 'test_single_scale.csv')

    # Test Method 1: SWT + Reconstruction
    print("\n### Testing Method 1: SWT + Reconstruction ###")
    results_swt_recon = pipeline.run_batch(
        method='swt_recon',
        filter_patients=True,
        max_samples=max_samples
    )
    print(f"Processed {len(results_swt_recon)} segments")
    if results_swt_recon:
        print(f"Sample result keys: {list(results_swt_recon[0].to_dict().keys())[:10]}...")
    pipeline.results_to_csv(results_swt_recon, results_dir / 'test_swt_recon.csv')

    # Test Method 2: SWT Direct
    print("\n### Testing Method 2: SWT Direct ###")
    results_swt_direct = pipeline.run_batch(
        method='swt_direct',
        filter_patients=True,
        max_samples=max_samples
    )
    print(f"Processed {len(results_swt_direct)} segments")
    pipeline.results_to_csv(results_swt_direct, results_dir / 'test_swt_direct.csv')

    # Test Method 3: DWT + Reconstruction
    print("\n### Testing Method 3: DWT + Reconstruction ###")
    results_dwt_recon = pipeline.run_batch(
        method='dwt_recon',
        filter_patients=True,
        max_samples=max_samples
    )
    print(f"Processed {len(results_dwt_recon)} segments")
    pipeline.results_to_csv(results_dwt_recon, results_dir / 'test_dwt_recon.csv')

    print("\n" + "="*80)
    print("All tests completed successfully!")
    print(f"Test results saved to: {results_dir}")
    print("="*80)
