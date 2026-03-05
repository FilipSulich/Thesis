#!/usr/bin/env python3
"""Test the modified RQA statistics with Mann-Whitney U test."""

import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.rqa_stats_new import RQAStatistics

if __name__ == "__main__":
    # Test with the test results from pipeline
    test_dir = Path("/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf_updated/results_test")

    print("\n" + "="*80)
    print("Testing RQA Statistics with Mann-Whitney U Test")
    print("="*80)

    # Test single-scale
    print("\n### Testing Single-Scale ###")
    stats_single = RQAStatistics(str(test_dir / "test_single_scale.csv"))
    stats_single.print_summary(prefix="signal")

    sig = stats_single.get_significant_results(alpha=0.05, prefixes=["signal"])
    print(f"\nSignificant results (p < 0.05): {len(sig)}")
    for r in sig:
        print(f"  {r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")

    # Test multi-scale (SWT Recon)
    print("\n### Testing Multi-Scale (SWT Recon) ###")
    stats_multi = RQAStatistics(str(test_dir / "test_swt_recon.csv"))
    stats_multi.print_multiscale_summary()

    sig = stats_multi.get_significant_results(
        alpha=0.05,
        prefixes=["D1", "D2", "D3", "D4", "A4"]
    )
    print(f"\nSignificant results (p < 0.05): {len(sig)}")
    for r in sig:
        print(f"  {r['prefix']}_{r['metric']}: p={r['p_value']:.4f}, d={r['cohens_d']:.3f}")

    print("\n" + "="*80)
    print("Test completed successfully!")
    print("="*80)
