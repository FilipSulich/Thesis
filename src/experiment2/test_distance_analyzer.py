"""
Quick test of SubbandDistanceAnalyzer with a small subset of data.

Tests embedding parameter estimation and single-wavelet analysis.
"""

import sys
import os
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment2.subband_distance_analyzer import SubbandDistanceAnalyzer
from src.experiment2.test import load_segments_from_hdf5


if __name__ == '__main__':
    print("\n" + "="*80)
    print("SUBBAND DISTANCE ANALYZER — QUICK TEST")
    print("="*80)

    # Load segments
    print("\nLoading segments...")
    DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2')
    HDF5_PATH = DATA_DIR / 'paf_with_quality.h5'

    segments_pre_af, segments_nsr, _ = load_segments_from_hdf5(HDF5_PATH)

    # Use small subset for testing
    print("\nCreating test subset:")
    test_pre_af = dict(list(segments_pre_af.items())[:3])
    test_nsr = dict(list(segments_nsr.items())[:3])

    print(f"  Pre-AF: {len(test_pre_af)} patients")
    print(f"  NSR: {len(test_nsr)} patients")

    # Also limit segments per patient for speed
    for pid in test_pre_af:
        test_pre_af[pid] = test_pre_af[pid][:5]
    for pid in test_nsr:
        test_nsr[pid] = test_nsr[pid][:5]

    # Initialize analyzer with just 2 wavelets for testing
    print("\nInitializing analyzer...")
    analyzer = SubbandDistanceAnalyzer(
        wavelets=[
            'coif4', 'coif5', 'coif3'],
        level=4,
        fs=128,
        signal_subbands=['D3', 'D4', 'A4'],
        control_subbands=['D1', 'D2'],
        random_seed=42
    )

    # Test embedding parameter estimation
    print("\nTesting embedding parameter estimation...")
    sample_pre_af = dict(list(test_pre_af.items())[:2])
    sample_nsr = dict(list(test_nsr.items())[:2])

    analyzer.estimate_embedding_params(
        sample_segments_pre_af=sample_pre_af,
        sample_segments_nsr=sample_nsr,
        reference_wavelet='db4',
        n_samples=2
    )

    print("\nEstimated embedding parameters:")
    for subband, params in analyzer.embedding_params.items():
        print(f"  {subband}: tau={params['tau']}, m={params['m']}")

    # Test single segment analysis
    print("\nTesting single segment analysis...")
    test_segment = list(test_pre_af.values())[0][0]
    print(f"  Segment shape: {test_segment.shape}")

    segment_features = analyzer.analyze_segment(test_segment, 'db4')

    print("\nExtracted features from test segment (db4):")
    for subband, features in segment_features.items():
        print(f"\n  {subband}:")
        for feat_name, feat_val in features.items():
            if not np.isnan(feat_val):
                print(f"    {feat_name}: {feat_val:.6f}")
            else:
                print(f"    {feat_name}: NaN")

    # Test patient-level aggregation
    print("\n" + "-"*80)
    print("Testing patient-level aggregation...")
    test_patient_segments = list(test_pre_af.values())[0]
    print(f"  Analyzing {len(test_patient_segments)} segments for one patient...")

    patient_features = analyzer.analyze_patient_segments(test_patient_segments, 'db4')

    print("\nPatient-level features (median across segments):")
    for subband, features in patient_features.items():
        print(f"\n  {subband}:")
        for feat_name, feat_val in features.items():
            if not np.isnan(feat_val):
                print(f"    {feat_name}: {feat_val:.6f}")

    # Test full analysis on small dataset
    print("\n" + "="*80)
    print("Running mini analysis (2 wavelets, 3 patients per cohort, 5 segments each)...")
    print("="*80 + "\n")

    results = analyzer.run_full_analysis(
        segments_pre_af=test_pre_af,
        segments_nsr=test_nsr,
        output_dir=Path('test_utrp_results')
    )

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)

    print("\nResults structure:")
    print(f"  Patient features: {len(results['patient_features'])} rows")
    print(f"  Statistical tests: {len(results['statistical_tests'])} rows")
    print(f"  Selection scores: {len(results['selection_scores'])} rows")

    print("\nSelection scores:")
    print(results['selection_scores'].to_string(index=False))

    print("\nSample statistical test (D3, mean_dist, db4):")
    sample_test = results['statistical_tests'][
        (results['statistical_tests']['wavelet'] == 'db4') &
        (results['statistical_tests']['subband'] == 'D3') &
        (results['statistical_tests']['feature'] == 'mean_dist')
    ]
    if not sample_test.empty:
        print(sample_test.iloc[0].to_string())
    else:
        print("  No results for this combination")

    print("\n" + "="*80)
    print("All tests passed! ✓")
    print("="*80 + "\n")