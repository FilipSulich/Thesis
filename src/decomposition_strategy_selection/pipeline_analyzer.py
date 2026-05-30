"""wavelet pipeline decomposition and analysis."""

import numpy as np
import pandas as pd

from src.pipelines.pipeline_wavelets import Pipeline
from src.decomposition_strategy_selection.normalization import min_max_normalize
from src.decomposition_strategy_selection.embedding import estimate_embedding_params
from src.decomposition_strategy_selection.recurrence import (
    compute_distance_matrix,
    compute_rqa_metrics,
    extract_rp_array,
)
from src.decomposition_strategy_selection.metrics import compute_structural_metrics


def decompose_signal(signal, pipeline_name, level, wavelet):
    """
    decompose signal using the named wavelet pipeline.

    args:
        signal: input signal
        pipeline_name: one of 'swt_recon', 'swt_direct', 'dwt_recon'
        level: decomposition level
        wavelet: wavelet name (e.g., 'db4')

    returns:
        dict of subbands {band_name: signal_array}
    """
    pipeline_obj = Pipeline(
        pipeline_name='all',
        level=level,
        wavelet=wavelet
    )
    func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
    return func(signal)


def analyze_pipeline(signal, pipeline_name, config, band_names,ground_truth=None, ground_truth_data=None):
    """
    decompose signal with the named pipeline and compute metrics.

    args:
        signal: input signal to decompose
        pipeline_name: one of 'swt_recon', 'swt_direct', 'dwt_recon'
        config: ExperimentConfig instance
        band_names: list of band names (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])
        ground_truth: optional dict with ground truth metrics
        ground_truth_data: optional dict with ground truth RPs

    returns:
        pandas DataFrame with one row per subband
    """
    subbands = decompose_signal(signal, pipeline_name, config.level, config.wavelet)
    results = []

    for band in band_names:
        freq = config.subband_freqs[band]

        if isinstance(subbands, (list, np.ndarray)) and len(subbands) == 1:
            subband_signal = subbands[0]
        else:
            subband_signal = subbands[band]

        subband_signal = min_max_normalize(subband_signal)

        if ground_truth is not None and band in ground_truth:
            tau = ground_truth[band]['tau']
            m = ground_truth[band]['m']
            epsilon = ground_truth[band]['epsilon']
        else:
            tau, m = estimate_embedding_params(subband_signal)
            dm = compute_distance_matrix(subband_signal, tau, m)
            epsilon = 0.1 * dm.max()

        # compute RQA from subband signal
        metrics = compute_rqa_metrics(
            subband_signal, tau, m, epsilon, config.output_dir
        )

        # RQA deviations vs ground truth
        det_deviation = np.nan
        lam_deviation = np.nan
        tt_deviation = np.nan
        entr_deviation = np.nan

        if ground_truth is not None and band in ground_truth:
            ref = ground_truth[band]
            det_deviation = abs(ref['DET'] - metrics['DET'])
            lam_deviation = abs(ref['LAM'] - metrics['LAM'])
            tt_deviation = abs(ref['TT'] - metrics['TT'])
            entr_deviation = abs(ref['ENTR'] - metrics['ENTR'])

        # structural similarity — zero-pad to ground truth RP size
        hamming_distance = np.nan
        rp_pattern_corr = np.nan

        if (ground_truth_data is not None and band in ground_truth_data and metrics['rp_matrix'] is not None):

            ref_rp = extract_rp_array(ground_truth_data[band]['rp_matrix'])
            pipe_rp = extract_rp_array(metrics['rp_matrix'])

            if ref_rp is not None and pipe_rp is not None:
                try:
                    structural = compute_structural_metrics(ref_rp, pipe_rp)
                    hamming_distance = structural['hamming_distance']
                    rp_pattern_corr = structural['pattern_correlation']
                except Exception as e:
                    print(e)

        results.append({
            'pipeline': pipeline_name,
            'band': band,
            'freq_hz': freq,
            'signal_length': len(subband_signal),
            'tau': tau,
            'm': m,
            'epsilon': float(epsilon),
            'DET': metrics['DET'],
            'LAM': metrics['LAM'],
            'TT': metrics['TT'],
            'ENTR': metrics['ENTR'],
            'L_mean_diagonal': metrics['L_mean_diagonal'],
            'L_longest_diagonal': metrics['L_longest_diagonal'],
            'L_longest_vertical': metrics['L_longest_vertical'],
            'det_deviation': det_deviation,
            'lam_deviation': lam_deviation,
            'tt_deviation': tt_deviation,
            'entr_deviation': entr_deviation,
            'hamming_distance': hamming_distance,
            'rp_pattern_corr': rp_pattern_corr,
        })

    return pd.DataFrame(results)


def analyze_all_pipelines(signal, config, band_names, pipeline_names,ground_truth=None, ground_truth_data=None):
    """
    analyze all pipelines and concatenate results.

    args:
        signal: input signal to decompose
        config: ExperimentConfig instance
        band_names: list of band names
        pipeline_names: list of pipeline names
        ground_truth: optional dict with ground truth metrics
        ground_truth_data: optional dict with ground truth RPs

    returns:
        pandas DataFrame with results from all pipelines
    """
    all_results = []

    for pipeline_name in pipeline_names:
        df = analyze_pipeline(
            signal, pipeline_name, config, band_names,
            ground_truth, ground_truth_data
        )
        all_results.append(df)

    return pd.concat(all_results, ignore_index=True)
