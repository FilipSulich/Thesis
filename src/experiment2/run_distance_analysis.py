import sys
import os
from pathlib import Path
import multiprocessing
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.experiment2.subband_distance_analyzer import SubbandDistanceAnalyzer
from src.experiment2.test import load_segments_from_hdf5

# ------------------------------------------------------------------ #
# Configuration                                                      #
# ------------------------------------------------------------------ #

WAVELETS = ['db2', 'db4', 'db6',
            'sym4', 'sym6', 'sym8',
            'coif1', 'coif2', 'coif3']

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_new/exp2')
HDF5_PATH = DATA_DIR / 'paf_with_quality.h5'
OUTPUT_DIR = Path('utrp_distance_analysis_results')

SIGNAL_SUBBANDS = ['D3', 'D4', 'A4']
CONTROL_SUBBANDS = ['D1', 'D2']

# ------------------------------------------------------------------ #
# Parallel Worker Function                                           #
# ------------------------------------------------------------------ #

def analyze_single_wavelet(wavelet, segments_pre_af, segments_nsr, embedding_params):
    """
    Worker function to process a single wavelet in a separate process.
    """
    print(f"--- Starting analysis for wavelet: {wavelet} ---")

    # Initialize a fresh analyzer for this specific wavelet
    # This prevents state leakage between processes
    worker_analyzer = SubbandDistanceAnalyzer(
        wavelets=[wavelet],
        level=4,
        fs=128,
        signal_subbands=SIGNAL_SUBBANDS,
        control_subbands=CONTROL_SUBBANDS,
        random_seed=42
    )

    # Manually set the embedding parameters calculated in the main process
    worker_analyzer.embedding_params = embedding_params

    # Run analysis for this single wavelet
    # We disable the internal output_dir to prevent file writing conflicts
    # and instead aggregate results in memory first.
    result = worker_analyzer.run_full_analysis(
        segments_pre_af=segments_pre_af,
        segments_nsr=segments_nsr,
        output_dir=None
    )
    return result

# ------------------------------------------------------------------ #
# Main Analysis Pipeline                                             #
# ------------------------------------------------------------------ #

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PARALLEL UNTHRESHOLDED RECURRENCE PLOT GEOMETRY ANALYSIS")
    print("="*80)

    # Step 1: Load segments
    print("\nStep 1: Loading segments from HDF5...")
    segments_pre_af, segments_nsr, segments_af = load_segments_from_hdf5(HDF5_PATH)

    # Step 2: Initialize master analyzer for parameter estimation
    print("\nStep 2: Initializing master analyzer...")
    master_analyzer = SubbandDistanceAnalyzer(
        wavelets=WAVELETS,
        level=4,
        fs=128,
        signal_subbands=SIGNAL_SUBBANDS,
        control_subbands=CONTROL_SUBBANDS,
        random_seed=42
    )

    # Step 3: Estimate embedding parameters (Serial - fast)
    print("\nStep 3: Estimating embedding parameters...")
    sample_pre_af = dict(list(segments_pre_af.items())[:5])
    sample_nsr = dict(list(segments_nsr.items())[:5])

    master_analyzer.estimate_embedding_params(
        sample_segments_pre_af=sample_pre_af,
        sample_segments_nsr=sample_nsr,
        reference_wavelet='db4',
        n_samples=5
    )

    # Extract params to pass to workers
    shared_params = master_analyzer.embedding_params

    # Step 4: Limit segments per patient
    print("\nStep 4: Preparing data for analysis...")
    MAX_SEGMENTS_PER_PATIENT = 10

    def limit_dict(d):
        return {pid: segs[:MAX_SEGMENTS_PER_PATIENT] for pid, segs in d.items()}

    segments_pre_af_limited = limit_dict(segments_pre_af)
    segments_nsr_limited = limit_dict(segments_nsr)

    # Step 5: Run parallel analysis
    n_cores = min(len(WAVELETS), multiprocessing.cpu_count())
    print(f"\nStep 5: Running parallel analysis using {n_cores} cores...")

    # Execute wavelet analyses in parallel
    results_list = Parallel(n_jobs=n_cores)(
        delayed(analyze_single_wavelet)(
            w, segments_pre_af_limited, segments_nsr_limited, shared_params
        ) for w in WAVELETS
    )

    # Step 6: Aggregate and Save Results
    print("\nStep 6: Aggregating results and saving to disk...")
    # Combine results from all workers (assuming they return dicts with dataframes)
    import pandas as pd

    final_results = {
        'patient_features': pd.concat([r['patient_features'] for r in results_list]),
        'statistical_comparison': pd.concat([r['statistical_comparison'] for r in results_list]),
        'selection_scores': pd.concat([r['selection_scores'] for r in results_list]).sort_values(
            'selectivity_ratio', ascending=False
        )
    }

    # Save to the requested directory
    OUTPUT_DIR.mkdir(exist_ok=True)
    final_results['patient_features'].to_csv(OUTPUT_DIR / 'distance_features_per_patient.csv', index=False)
    final_results['statistical_comparison'].to_csv(OUTPUT_DIR / 'statistical_comparison.csv', index=False)
    final_results['selection_scores'].to_csv(OUTPUT_DIR / 'wavelet_selection_scores.csv', index=False)

    # Print best wavelet
    best_wavelet = final_results['selection_scores'].iloc[0]['wavelet']
    best_ratio = final_results['selection_scores'].iloc[0]['selectivity_ratio']

    print("\n" + "="*80)
    print(f">>> SELECTED WAVELET: {best_wavelet} <<<")
    print(f">>> Selectivity Ratio: {best_ratio:.4f} <<<")
    print("="*80 + "\n")