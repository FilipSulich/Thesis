"""ground truth generation and loading for decomposition experiments."""

import os
import json
import numpy as np

from src.decomposition_strategy_selection.signals import generate_single_sine
from src.decomposition_strategy_selection.normalization import min_max_normalize
from src.decomposition_strategy_selection.embedding import estimate_embedding_params
from src.decomposition_strategy_selection.recurrence import (
    compute_distance_matrix,
    compute_rqa_metrics,
    extract_rp_array,
)


def generate_ground_truth(config, band_names):
    """
    generate reference RPs from pure single-frequency sines.

    for each subband:
      1. generate single sine at subband frequency
      2. min-max normalise
      3. estimate tau and m (AMI / FNN)
      4. compute epsilon = 10% of max distance
      5. compute binary RP and RQA metrics

    args:
        config: ExperimentConfig instance with fs, n, subband_freqs, output_dir
        band_names: list of band names (e.g., ['D1', 'D2', 'D3', 'D4', 'A4'])

    returns:
        tuple of (ground_truth, ground_truth_data):
            - ground_truth: dict with metrics for each band
            - ground_truth_data: dict with rp_matrix, distance_matrix, pure_sine
    """
    print("generating ground truth...")
    ground_truth = {}
    ground_truth_data = {}

    for band in band_names:
        freq = config.subband_freqs[band]
        print(f"  {band} ({freq} Hz) …")

        # generate and normalize pure sine
        pure_sine = generate_single_sine(config.n, config.fs, freq)
        pure_sine = min_max_normalize(pure_sine)

        # estimate embedding parameters
        tau, m = estimate_embedding_params(pure_sine)

        # compute distance matrix and epsilon
        dist_mat = compute_distance_matrix(pure_sine, tau, m)
        max_dist = dist_mat.max()
        epsilon = 0.1 * max_dist

        # compute RQA metrics
        metrics = compute_rqa_metrics(pure_sine, tau, m, epsilon, config.output_dir)

        # store metadata
        ground_truth[band] = {
            'freq_hz': freq,
            'tau': int(tau),
            'm': int(m),
            'epsilon': float(epsilon),
            'DET': float(metrics['DET']),
            'LAM': float(metrics['LAM']),
            'TT': float(metrics['TT']),
            'ENTR': float(metrics['ENTR']),
            'L_mean_diagonal': float(metrics['L_mean_diagonal']),
            'L_longest_diagonal': float(metrics['L_longest_diagonal']),
            'L_longest_vertical': float(metrics['L_longest_vertical']),
        }

        # store raw data
        ground_truth_data[band] = {
            'rp_matrix': metrics['rp_matrix'],
            'distance_matrix': dist_mat,
            'pure_sine': pure_sine,
        }

    return ground_truth, ground_truth_data


def save_ground_truth(ground_truth, ground_truth_data, band_names, save_path):
    """
    save ground truth to JSON and NPZ files.

    args:
        ground_truth: dict with metrics for each band
        ground_truth_data: dict with rp_matrix, distance_matrix, pure_sine
        band_names: list of band names
        save_path: path to save JSON file (NPZ will have _data.npz suffix)
    """
    # save JSON metadata
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
    load previously saved ground truth.

    args:
        band_names: list of band names
        json_path: path to ground truth JSON file
        npz_path: optional path to NPZ file (default: json_path with _data.npz)

    returns:
        tuple of (ground_truth, ground_truth_data)
    """
    if npz_path is None:
        npz_path = json_path.replace('.json', '_data.npz')

    # load JSON metadata
    with open(json_path) as f:
        ground_truth = json.load(f)

    # load NPZ raw data
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
    load existing ground truth or generate new one if not found.

    args:
        config: ExperimentConfig instance
        band_names: list of band names
        force_regenerate: if True, always generate new ground truth

    returns:
        tuple of (ground_truth, ground_truth_data)
    """
    gt_path = os.path.join(config.output_dir, 'ground_truth.json')

    if not force_regenerate and os.path.exists(gt_path):
        print(f"loading existing ground truth from {gt_path}")
        return load_ground_truth(band_names, gt_path)
    else:
        ground_truth, ground_truth_data = generate_ground_truth(config, band_names)
        save_ground_truth(ground_truth, ground_truth_data, band_names, gt_path)
        return ground_truth, ground_truth_data
