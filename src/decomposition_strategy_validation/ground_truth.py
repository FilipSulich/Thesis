import os
import json
import numpy as np

from src.decomposition_strategy_validation.signals import generate_single_sine
from src.decomposition_strategy_validation.normalization import min_max_normalize
from src.decomposition_strategy_validation.embedding import estimate_embedding_params
from src.decomposition_strategy_validation.recurrence import (
    compute_distance_matrix,
    compute_rqa_metrics,
    extract_rp_array,
)


def generate_ground_truth(config, band_names):
    """
    generate reference recurrence plots and RQA metrics from single-frequency sinusoidal signals for each subband

    Args:
        config (ExperimentConfig): experiment configuration instance containing sampling frequency, signal length, subband frequencies, and output directory
        band_names (list): list of wavelet subband names to generate ground truth for (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])

    Returns:
        tuple: tuple containing (ground_truth, ground_truth_data) where ground_truth is a dict with RQA metrics for each band and ground_truth_data is a dict with matrices and signals
    """
    ground_truth = {}
    ground_truth_data = {}
    for band in band_names:
        freq = config.subband_freqs[band]
        pure_sine = generate_single_sine(config.n, config.fs, freq)
        pure_sine = min_max_normalize(pure_sine)

        tau, m = estimate_embedding_params(pure_sine)

        dist_mat = compute_distance_matrix(pure_sine, tau, m)
        max_dist = dist_mat.max()
        epsilon = 0.1 * max_dist

        metrics = compute_rqa_metrics(pure_sine, tau, m, epsilon, config.output_dir)

        ground_truth[band] = {
            'freq_hz': freq,
            'tau': int(tau),
            'm': int(m),
            'epsilon': float(epsilon),
            'DET': float(metrics['DET']),
            'LAM': float(metrics['LAM']),
            'TT': float(metrics['TT']),
        }

        ground_truth_data[band] = {
            'rp_matrix': metrics['rp_matrix'],
            'distance_matrix': dist_mat,
            'pure_sine': pure_sine,
        }

    return ground_truth, ground_truth_data


def save_ground_truth(ground_truth, ground_truth_data, band_names, save_path):
    """
    save ground truth metrics and data to JSON and NPZ files for later reuse

    Args:
        ground_truth (dict): dictionary containing RQA metrics and parameters for each band
        ground_truth_data (dict): dictionary containing recurrence plot matrices, distance matrices, and pure sine signals
        band_names (list): list of wavelet subband names
        save_path (str): file path for saving files
    """
    with open(save_path, 'w') as f:
        json.dump(ground_truth, f, indent=2)

    npz_path = save_path.replace('.json', '_data.npz')
    arrays = {}
    for band in band_names:
        rp = extract_rp_array(ground_truth_data[band]['rp_matrix'])
        arrays[f'{band}_rp'] = rp
        arrays[f'{band}_dist'] = ground_truth_data[band]['distance_matrix']
        arrays[f'{band}_signal'] = ground_truth_data[band]['pure_sine']
    np.savez(npz_path, **arrays)


def load_ground_truth(band_names, json_path, npz_path=None):
    """
    load previously saved ground truth metrics and data from JSON and NPZ files

    Args:
        band_names (list): list of wavelet subband names to load
        json_path (str): file path to ground truth JSON file containing metrics
        npz_path (str): optional file path to NPZ file containing matrices and signals (default: inferred from json_path)

    Returns:
        tuple: tuple containing (ground_truth, ground_truth_data) dictionaries
    """
    if npz_path is None:
        npz_path = json_path.replace('.json', '_data.npz')

    with open(json_path) as f:
        ground_truth = json.load(f)

    data = np.load(npz_path, allow_pickle=True)
    ground_truth_data = {}
    for band in band_names:
        ground_truth_data[band] = {
            'rp_matrix': data[f'{band}_rp'],
            'distance_matrix': data[f'{band}_dist'],
            'pure_sine': data[f'{band}_signal'],
        }

    return ground_truth, ground_truth_data


def get_or_generate_ground_truth(config, band_names, force_regenerate=False):
    """
    load existing ground truth from files or generate new ground truth if files not found

    Args:
        config (ExperimentConfig): experiment configuration instance containing output directory and parameters
        band_names (list): list of wavelet subband names
        force_regenerate (bool): if True, always generate new ground truth regardless of existing files (default: False)

    Returns:
        tuple: tuple containing (ground_truth, ground_truth_data) dictionaries
    """
    gt_path = os.path.join(config.output_dir, 'ground_truth.json')

    if not force_regenerate and os.path.exists(gt_path):
        return load_ground_truth(band_names, gt_path)
    else:
        ground_truth, ground_truth_data = generate_ground_truth(config, band_names)
        save_ground_truth(ground_truth, ground_truth_data, band_names, gt_path)
        return ground_truth, ground_truth_data
