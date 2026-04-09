import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment2.utrp_separability import UTRPSeparabilityAnalyzer
from src.experiment2.test import load_segments_from_hdf5


if __name__ == '__main__':

    N_PATIENTS_QUICK = 25
    N_SEGMENTS_QUICK = 5

    DATA_DIR = Path('/data/data_new/exp2')
    HDF5_PATH = DATA_DIR / 'paf_with_quality.h5'

    segments_pre_af, segments_nsr, _ = load_segments_from_hdf5(HDF5_PATH)

    segments_pre_af = dict(list(segments_pre_af.items())[:N_PATIENTS_QUICK])
    segments_nsr = dict(list(segments_nsr.items())[:N_PATIENTS_QUICK])
    for pid in segments_pre_af:
        segments_pre_af[pid] = segments_pre_af[pid][:N_SEGMENTS_QUICK]
    for pid in segments_nsr:
        segments_nsr[pid] = segments_nsr[pid][:N_SEGMENTS_QUICK]
    # wavelets = ['db2', 'db4']
    wavelets = ['db2', 'db4', 'db6', 'sym2', 'sym4', 'sym6', 'coif1', 'coif2', 'coif3']
    mode_desc = f"{N_PATIENTS_QUICK} patients, {N_SEGMENTS_QUICK} segments, 1 wavelet"

    print(f"  Pre-AF: {len(segments_pre_af)} patients")
    print(f"  NSR: {len(segments_nsr)} patients")

    total_pre_af_segs = sum(len(segs) for segs in segments_pre_af.values())
    total_nsr_segs = sum(len(segs) for segs in segments_nsr.values())
    print(f"  Total segments: {total_pre_af_segs} pre-AF, {total_nsr_segs} NSR")

    analyzer = UTRPSeparabilityAnalyzer(
        wavelets=wavelets,
        level=4,
        fs=128,
        target_subbands=['D1', 'D2', 'D3', 'D4', 'A4'],
        random_seed=42,
    )

    output_path = Path('../wavelets/utrp_separability_results.csv')
    results, scores = analyzer.run_full_analysis(
        segments_pre_af=segments_pre_af,
        segments_nsr=segments_nsr,
        output_path=output_path,
    )
    print(scores)
    #

    print("\nResults DataFrame (first 20 rows):")
    print(results.head(20).to_string(index=False))

    # print("\n" + "="*80)
    # print("WEIGHTED SUMMARY (Top wavelets by composite score)")
    # print("="*80)
    # print("\nFrobenius-based scores:")
    # frob_summary = summary[summary['metric'] == 'frob'].head(5)
    # print(frob_summary.to_string(index=False))
    #
    # print("\nCorrelation-based scores:")
    # corr_summary = summary[summary['metric'] == 'corr'].head(5)
    # print(corr_summary.to_string(index=False))
    #
    # # Compute average GSR per wavelet across all subbands (for Frobenius metric)
    # print("\n" + "="*80)
    # print("AVERAGE GSR PER WAVELET (Frobenius, across all subbands)")
    # print("="*80)
    #
    # avg_gsr_per_wavelet = results.groupby('wavelet')['gsr_frob'].mean().sort_values(ascending=False)
    #
    # avg_rqa_per_wavelet = results.groupby('wavelet')['gsr_rqa'].mean().sort_values(ascending=False)
    #
    # print("\nWavelet Rankings (by average Frobenius GSR):")
    # print("-"*80)
    # for i, (wavelet, avg_gsr) in enumerate(avg_gsr_per_wavelet.items(), 1):
    #     print(f"  {i}. {wavelet:10s} → Average GSR = {avg_gsr:.4f}")
    #
    # for i, (wavelet, avg_rqa) in enumerate(avg_rqa_per_wavelet.items(), 1):
    #     print(f"  {i}. {wavelet:10s} → Average RQA = {avg_rqa:.4f}")
    #
    # best_wavelet_frob = avg_gsr_per_wavelet.index[0]
    # best_gsr_frob = avg_gsr_per_wavelet.iloc[0]
    # best_wavelet_rqa = avg_rqa_per_wavelet.index[0]
    # best_gsr_rqa = avg_rqa_per_wavelet.iloc[0]
    #
    # print("\n" + "="*80)
    # print(f">>> BEST WAVELET (Frobenius): {best_wavelet_frob} (Average GSR = {best_gsr_frob:.4f}) <<<")
    # print("="*80)
    # print(f">>> BEST WAVELET (RQA): {best_wavelet_rqa} (Average RQA = {best_gsr_rqa:.4f}) <<<")
    #
    #
    # print(f"\nPer-subband GSR (Frobenius) for {best_wavelet_frob}:")
    # print("-"*80)
    # best_wavelet_results = results[results['wavelet'] == best_wavelet_frob].sort_values('subband')
    # for _, row in best_wavelet_results.iterrows():
    #     print(f"  {row['subband']}: GSR = {row['gsr_frob']:.4f}")