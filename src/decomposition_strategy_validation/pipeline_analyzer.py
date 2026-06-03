import numpy as np
import pandas as pd

from src.pipelines.pipeline_wavelets import Pipeline
from src.decomposition_strategy_validation.normalization import min_max_normalize
from src.decomposition_strategy_validation.embedding import estimate_embedding_params
from src.decomposition_strategy_validation.recurrence import (
    compute_distance_matrix,
    compute_rqa_metrics,
    extract_rp_array,
)
from src.decomposition_strategy_validation.metrics import compute_structural_metrics


def decompose_signal(signal, pipeline_name, level, wavelet):
    """
    decompose input signal using specified wavelet decomposition pipeline strategy

    Args:
        signal (numpy.ndarray): input time series signal to decompose
        pipeline_name (str): pipeline identifier, one of 'swt_recon', 'swt_direct', or 'dwt_recon'
        level (int): number of decomposition levels
        wavelet (str): wavelet family name (e.g., 'db4', 'haar', 'sym5')

    Returns:
        dict: dictionary mapping subband names to their corresponding signal arrays
    """
    pipeline_obj = Pipeline(
        pipeline_name='all',
        level=level,
        wavelet=wavelet
    )
    func = getattr(pipeline_obj, f"pipeline_{pipeline_name}")
    return func(signal)


def analyze_pipeline(signal, pipeline_name, config, band_names, ground_truth=None, ground_truth_data=None):
    """
    decompose signal using specified pipeline and compute RQA metrics and deviation from ground truth for each subband

    Args:
        signal (numpy.ndarray): input time series signal to decompose and analyze
        pipeline_name (str): pipeline identifier, one of 'swt_recon', 'swt_direct', or 'dwt_recon'
        config (ExperimentConfig): experiment configuration instance with decomposition parameters
        band_names (list): list of wavelet subband names to analyze (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])
        ground_truth (dict): optional dictionary with reference RQA metrics for deviation computation (default: None)
        ground_truth_data (dict): optional dictionary with reference recurrence plots for structural comparison (default: None)

    Returns:
        pandas.DataFrame: dataframe with one row per subband containing metrics, deviations, and structural similarities
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

        metrics = compute_rqa_metrics(
            subband_signal, tau, m, epsilon, config.output_dir
        )

        det_deviation = np.nan
        lam_deviation = np.nan
        tt_deviation = np.nan

        if ground_truth is not None and band in ground_truth:
            ref = ground_truth[band]
            det_deviation = abs(ref['DET'] - metrics['DET'])
            lam_deviation = abs(ref['LAM'] - metrics['LAM'])
            tt_deviation = abs(ref['TT'] - metrics['TT'])

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
            'det_deviation': det_deviation,
            'lam_deviation': lam_deviation,
            'tt_deviation': tt_deviation,
            'hamming_distance': hamming_distance,
            'rp_pattern_corr': rp_pattern_corr,
        })

    return pd.DataFrame(results)