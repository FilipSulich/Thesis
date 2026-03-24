"""
Robustness Analysis: Pipeline Stability Under Input Perturbation

Ranks three wavelet pipelines by output stability when ECG segments are perturbed:
- Noise injection (SNR = 20 dB and 10 dB)
- Circular shift (±50 samples)

Computes normalized ΔRQA = |RQA_perturbed - RQA_clean| / (|RQA_clean| + ε_floor)
for six RQA metrics: RR, DET, L_mean, ENTR, LAM, TT

Aggregates to patient level using median, then computes grand median across patients.
"""

import numpy as np
import h5py
import csv
import os
import sys
import warnings
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrence_plot.AMI import AMI
from recurrence_plot.FNN import FNN
from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from wavelets.stationary_wavelet_transform import SWT
from wavelets.discrete_wavelet_transform import DWT

# Constants
WAVELET = 'db4'
LEVEL = 4
SAMPLING_FREQ = 128
BANDS = ['D1', 'D2', 'D3', 'D4', 'A4']
PIPELINES = ['single_scale', 'swt_recon', 'swt_direct', 'dwt_recon']
RQA_METRICS = ['RR', 'DET', 'L_mean', 'ENTR', 'LAM', 'TT']

# Perturbation parameters
SNR_LEVELS = [30, 20, 10, 5]  # dB
SHIFT_OFFSETS = [-50, 50]  # samples

# Analysis parameters
EPSILON_FLOOR = 1e-6  # For normalized difference calculation
TARGET_RR = 0.05  # Target recurrence rate for epsilon selection

# Paths
BASE_DIR = '/Users/filipsulich/Developer/uni/Thesis/Thesis'
HDF5_PATH = os.path.join(BASE_DIR, 'data/extracted_clean/paf1/paf_segment_no_quality.h5')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs/experiment_1_robustness')
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib.pyplot as plt

def plot_ecg_noise_perturbation(signal, snr_db, fs=128, segment_id="Segment"):
    """
    Visualizes the clean ECG, the generated Gaussian noise, and the combined signal.
    Matches the logic in add_gaussian_noise().
    """
    # 1. Calculate noise (replicated logic from add_gaussian_noise)
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    noisy_signal = signal + noise

    # Create time axis
    time = np.arange(len(signal)) / fs

    # 2. Plotting
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f'Robustness Analysis: Noise Injection (SNR: {snr_db} dB)\n{segment_id}', fontsize=14)

    # Clean Signal
    axes[0].plot(time, signal, color='#1f77b4', lw=1.5)
    axes[0].set_title("Original Clean ECG")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(alpha=0.3)

    # Noise Component
    axes[1].plot(time, noise, color='#d62728', lw=0.8, alpha=0.7)
    axes[1].set_title(f"Generated Gaussian Noise (Power: {noise_power:.6f})")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(alpha=0.3)

    # Noisy Signal
    axes[2].plot(time, noisy_signal, color='#2ca02c', lw=1.5)
    axes[2].set_title(f"Perturbed Signal (Clean + Noise)")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def load_segments2(segments_per_patient=10):
    with h5py.File(HDF5_PATH, 'r') as f:
        all_group = f['all']
        signals = all_group['signals'][:]
        labels = all_group['labels'][:]
        patient_ids = [pid.decode() for pid in all_group['patient_ids'][:]]

    # Build patient map tracking indices and labels separately
    patient_map = {}
    for idx, pid in enumerate(patient_ids):
        if pid not in patient_map:
            patient_map[pid] = {'indices': [], 'labels': []}
        patient_map[pid]['indices'].append(idx)
        patient_map[pid]['labels'].append(int(labels[idx]))

    # Pre-AF: paf_p* patients that have label=1 segments
    preaf_pids = [pid for pid, info in patient_map.items()
                  if 'paf_p' in pid and any(l == 1 for l in info['labels'])]

    # Normal: paf_n* patients that have label=0 segments
    normal_pids = [pid for pid, info in patient_map.items()
                   if 'paf_n' in pid and any(l == 0 for l in info['labels'])]

    print(f"Pre-AF patients found: {len(preaf_pids)}")
    print(f"Normal patients found: {len(normal_pids)}")

    np.random.seed(42)
    selected_indices = []
    actual_preaf_patient_count = 0

    # Process all pre-AF patients
    for pid in sorted(preaf_pids):
        indices = [i for i, l in zip(patient_map[pid]['indices'],
                                     patient_map[pid]['labels']) if l == 1]
        if len(indices) >= segments_per_patient:
            choice = np.random.choice(indices, size=segments_per_patient, replace=False)
            selected_indices.extend(choice)
            actual_preaf_patient_count += 1
        else:
            print(f"  Skipping {pid}: only {len(indices)} pre-AF segments")

    # Match normal patients to pre-AF count
    if len(normal_pids) < actual_preaf_patient_count:
        raise ValueError(f"Not enough normal patients ({len(normal_pids)}) "
                         f"to match pre-AF ({actual_preaf_patient_count})")

    selected_normal_pids = np.random.choice(
        normal_pids, size=actual_preaf_patient_count, replace=False)

    for pid in selected_normal_pids:
        indices = [i for i, l in zip(patient_map[pid]['indices'],
                                     patient_map[pid]['labels']) if l == 0]
        if len(indices) >= segments_per_patient:
            choice = np.random.choice(indices, size=segments_per_patient, replace=False)
            selected_indices.extend(choice)
        else:
            print(f"  Warning: normal patient {pid} has only {len(indices)} segments")

    selected_indices = np.array(selected_indices)

    data = {
        'signals': signals[selected_indices],
        'labels': labels[selected_indices],
        'patient_ids': np.array(patient_ids)[selected_indices],
        'indices': selected_indices
    }

    print(f"--- Selection Summary ---")
    print(f"Pre-AF patients: {actual_preaf_patient_count}, "
          f"Normal patients: {actual_preaf_patient_count}")
    print(f"Segments per patient: {segments_per_patient}")
    print(f"Total segments: {len(selected_indices)}")

    return data
def load_segments(n_preaf=100, n_normal=100):
    """Load random subset of ECG segments."""
    print(f"Loading segments from {HDF5_PATH}...")

    with h5py.File(HDF5_PATH, 'r') as f:
        all_group = f['all']
        signals = all_group['signals'][:]
        labels = all_group['labels'][:]
        patient_ids = all_group['patient_ids'][:]

    # Identify p-even (pre-AF) and n* (normal) segments
    p_even_mask = np.array([b'_p' in pid and int(pid.decode().split('_p')[1][:2]) % 2 == 0
                            for pid in patient_ids])
    n_star_mask = np.array([b'_n' in pid for pid in patient_ids])

    preaf_indices = np.where(p_even_mask)[0]
    normal_indices = np.where(n_star_mask)[0]

    # Random sample
    np.random.seed(42)
    selected_preaf = np.random.choice(preaf_indices, size=min(n_preaf, len(preaf_indices)),
                                      replace=False)
    selected_normal = np.random.choice(normal_indices, size=min(n_normal, len(normal_indices)),
                                       replace=False)

    selected_indices = np.concatenate([selected_preaf, selected_normal])

    data = {
        'signals': signals[selected_indices],
        'labels': labels[selected_indices],
        'patient_ids': patient_ids[selected_indices],
        'indices': selected_indices
    }

    print(f"Selected: {len(selected_preaf)} pre-AF, {len(selected_normal)} normal")
    return data


def add_gaussian_noise(signal, snr_db):
    """
    Add Gaussian noise at specified SNR level.
    SNR_dB = 10 * log10(signal_power / noise_power)
    """
    signal_power = np.mean(signal ** 2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def circular_shift(signal, offset):
    """Circularly shift signal by offset samples."""
    return np.roll(signal, offset)


def pipeline_single_scale(signal):
    """Single-scale: no decomposition, raw signal."""
    return {'raw': signal}


def pipeline_swt_recon(signal):
    """SWT decomposition -> isolate subband -> reconstruct."""
    swt = SWT(wavelet=WAVELET, level=LEVEL)
    coeffs = swt.decompose(signal, level=LEVEL)
    subbands = {}

    # Extract detail subbands
    for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
        isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
                          for j in range(len(coeffs))]
        detail_idx = len(coeffs) - 1 - i
        isolated_coeffs[detail_idx] = (np.zeros_like(coeffs[detail_idx][0]),
                                       coeffs[detail_idx][1])
        recon = swt.reconstruct(isolated_coeffs)
        subbands[band] = recon[:len(signal)]

    # Extract approximation
    isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
                      for j in range(len(coeffs))]
    isolated_coeffs[0] = (coeffs[0][0], np.zeros_like(coeffs[0][1]))
    recon = swt.reconstruct(isolated_coeffs)
    subbands['A4'] = recon[:len(signal)]

    return subbands


def pipeline_swt_direct(signal):
    """SWT decomposition -> extract coefficients directly."""
    swt = SWT(wavelet=WAVELET, level=LEVEL)
    coeffs = swt.decompose(signal, level=LEVEL)
    subbands = {}

    for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
        detail_idx = len(coeffs) - 1 - i
        subbands[band] = coeffs[detail_idx][1]

    subbands['A4'] = coeffs[0][0]
    return subbands


def pipeline_dwt_recon(signal):
    """DWT decomposition -> isolate subband -> reconstruct."""
    dwt = DWT(wavelet=WAVELET, level=LEVEL)
    coeffs = dwt.decompose(signal, level=LEVEL)
    subbands = {}

    for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
        isolated_coeffs = [np.zeros_like(c) for c in coeffs]
        detail_idx = len(coeffs) - 1 - i
        isolated_coeffs[detail_idx] = coeffs[detail_idx]
        recon = dwt.reconstruct(isolated_coeffs)
        subbands[band] = recon[:len(signal)]

    # Approximation
    isolated_coeffs = [np.zeros_like(c) for c in coeffs]
    isolated_coeffs[0] = coeffs[0]
    recon = dwt.reconstruct(isolated_coeffs)
    subbands['A4'] = recon[:len(signal)]

    return subbands


def compute_epsilon_fixed_rr(signal, tau, m, target_rr=TARGET_RR, max_iters=20, tol=0.01):
    """Binary search for epsilon that yields target recurrence rate."""
    from scipy.spatial.distance import pdist, squareform

    n = len(signal) - (m - 1) * tau
    embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
    distances = squareform(pdist(embedded, metric='euclidean'))

    # Binary search bounds
    all_dists = distances[np.triu_indices_from(distances, k=1)]
    eps_min = np.percentile(all_dists, 1)
    eps_max = np.percentile(all_dists, 30)

    for _ in range(max_iters):
        eps_mid = (eps_min + eps_max) / 2
        rp = (distances <= eps_mid).astype(int)
        np.fill_diagonal(rp, 0)
        rr = np.sum(rp) / (rp.shape[0] * (rp.shape[0] - 1))

        if abs(rr - target_rr) < tol:
            return eps_mid
        elif rr < target_rr:
            eps_min = eps_mid
        else:
            eps_max = eps_mid

    return eps_mid


def compute_embedding_params(signal, max_lag=50, max_dim=10, fnn_threshold=5.0):
    """Estimate tau using AMI and m using FNN."""
    ami = AMI(signal, max_lag=max_lag)
    tau = ami.get_first_minimum()
    if tau is None or tau == 0:
        tau = 1

    fnn = FNN(signal, tau=tau, max_dim=max_dim)
    m = fnn.get_optimal_dimension(threshold=fnn_threshold)
    if m is None or m < 2:
        m = 3

    return tau, m


def compute_rqa_metrics(signal, tau, m, epsilon):
    """Compute RQA metrics using fixed epsilon."""
    try:
        # Suppress OpenCL output
        import os
        old_env = os.environ.get('PYOPENCL_CTX', None)
        os.environ['PYOPENCL_CTX'] = '0'

        rp_gen = RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            path=OUTPUT_DIR,
            epsilon=epsilon,
            unthresholded=False
        )

        rqa_result, _ = rp_gen.compute_rqa_binary()

        # Restore environment
        if old_env is None:
            os.environ.pop('PYOPENCL_CTX', None)
        else:
            os.environ['PYOPENCL_CTX'] = old_env

        # Extract only the 6 metrics we need
        metrics = {
            'RR': rqa_result.recurrence_rate,
            'DET': rqa_result.determinism,
            'L_mean': rqa_result.average_diagonal_line,
            'ENTR': rqa_result.entropy_diagonal_lines,
            'LAM': rqa_result.laminarity,
            'TT': rqa_result.trapping_time
        }

        # Validate
        for key, val in metrics.items():
            if val is None or np.isnan(val):
                metrics[key] = 0.0

        return metrics

    except Exception as e:
        warnings.warn(f"RQA computation failed: {e}")
        return {metric: 0.0 for metric in RQA_METRICS}


def process_segment_clean(signal, pipeline_name):
    """
    Process clean segment through pipeline.
    Returns dict of {band: {tau, m, epsilon, metrics}}
    """
    # Select pipeline
    if pipeline_name == 'single_scale':
        subbands = pipeline_single_scale(signal)
    elif pipeline_name == 'swt_recon':
        subbands = pipeline_swt_recon(signal)
    elif pipeline_name == 'swt_direct':
        subbands = pipeline_swt_direct(signal)
    elif pipeline_name == 'dwt_recon':
        subbands = pipeline_dwt_recon(signal)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    results = {}

    # Get bands to process
    bands_to_process = ['raw'] if pipeline_name == 'single_scale' else BANDS

    for band in bands_to_process:
        subband_signal = subbands[band]

        # Validate
        if len(subband_signal) < 100 or np.std(subband_signal) < 1e-6:
            continue

        # Compute embedding parameters on clean signal
        tau, m = compute_embedding_params(subband_signal)

        # Compute epsilon for target RR on clean signal
        epsilon = compute_epsilon_fixed_rr(subband_signal, tau, m, target_rr=TARGET_RR)

        # Compute RQA on clean signal
        metrics = compute_rqa_metrics(subband_signal, tau, m, epsilon)

        results[band] = {
            'tau': tau,
            'm': m,
            'epsilon': epsilon,
            'metrics': metrics
        }

    return results


def process_segment_perturbed(signal, pipeline_name, clean_params):
    """
    Process perturbed segment using embedding parameters from clean signal.

    Args:
        signal: Perturbed signal
        pipeline_name: Pipeline to use
        clean_params: Dict with tau, m, epsilon from clean processing

    Returns:
        Dict of {band: metrics}
    """
    # Select pipeline
    if pipeline_name == 'single_scale':
        subbands = pipeline_single_scale(signal)
    elif pipeline_name == 'swt_recon':
        subbands = pipeline_swt_recon(signal)
    elif pipeline_name == 'swt_direct':
        subbands = pipeline_swt_direct(signal)
    elif pipeline_name == 'dwt_recon':
        subbands = pipeline_dwt_recon(signal)
    else:
        raise ValueError(f"Unknown pipeline: {pipeline_name}")

    results = {}

    # Get bands to process
    bands_to_process = ['raw'] if pipeline_name == 'single_scale' else BANDS

    for band in bands_to_process:
        if band not in clean_params:
            continue

        subband_signal = subbands[band]

        # Use parameters from clean signal
        tau = clean_params[band]['tau']
        m = clean_params[band]['m']
        epsilon = clean_params[band]['epsilon']

        # Compute RQA on perturbed signal with fixed parameters
        metrics = compute_rqa_metrics(subband_signal, tau, m, epsilon)
        results[band] = metrics

    return results


def compute_delta_rqa(clean_metrics, perturbed_metrics):
    """
    Compute normalized ΔRQA = |M_pert - M_clean| / (|M_clean| + ε_floor)
    """
    delta = {}

    for metric in RQA_METRICS:
        clean_val = clean_metrics.get(metric, 0.0)
        pert_val = perturbed_metrics.get(metric, 0.0)

        delta[metric] = abs(pert_val - clean_val) / (abs(clean_val) + EPSILON_FLOOR)

    return delta


def aggregate_to_patient_level(segment_deltas, patient_ids):
    """
    Aggregate segment-level ΔRQA to patient-level using median.

    Returns:
        Dict {patient_id: {metric: median_delta}}
    """
    patient_data = defaultdict(lambda: defaultdict(list))

    for seg_idx, deltas in segment_deltas.items():
        patient_id = patient_ids[seg_idx]
        for metric, value in deltas.items():
            patient_data[patient_id][metric].append(value)

    # Compute median per patient
    patient_medians = {}
    for patient_id, metrics_dict in patient_data.items():
        patient_medians[patient_id] = {
            metric: np.median(values) for metric, values in metrics_dict.items()
        }

    return patient_medians


def compute_grand_statistics(patient_medians):
    """
    Compute grand median and IQR across all patients.

    Returns:
        Dict {metric: (median, q25, q75)}
    """
    stats = {}

    for metric in RQA_METRICS:
        values = [pm[metric] for pm in patient_medians.values() if metric in pm]

        if len(values) > 0:
            median = np.median(values)
            q25 = np.percentile(values, 25)
            q75 = np.percentile(values, 75)
            stats[metric] = (median, q25, q75)
        else:
            stats[metric] = (0.0, 0.0, 0.0)

    return stats


def main():
    data = load_segments2(segments_per_patient=5)
    n_segments = len(data['signals'])

    clean_results = {}

    perturbed_results = defaultdict(lambda: defaultdict(dict))

    delta_rqa = defaultdict(lambda: defaultdict(dict))

    for seg_idx in range(n_segments):
        print(f"Segment {seg_idx+1}/{n_segments}", end='\r')

        signal = data['signals'][seg_idx]
        clean_results[seg_idx] = {}

        for pipeline_name in PIPELINES:
            try:
                results = process_segment_clean(signal, pipeline_name)
                clean_results[seg_idx][pipeline_name] = results
            except Exception as e:
                warnings.warn(f"Failed clean processing: seg={seg_idx}, pipeline={pipeline_name}: {e}")

    for snr_db in SNR_LEVELS:
        print(f"\n  SNR = {snr_db} dB")
        pert_type = f"noise_{snr_db}dB"

        sample_signal = data['signals'][0]
        plot_ecg_noise_perturbation(sample_signal, snr_db, fs=SAMPLING_FREQ, segment_id=f"Patient")

        for seg_idx in range(n_segments):
            print(f"    Segment {seg_idx+1}/{n_segments}", end='\r')

            if seg_idx not in clean_results:
                continue

            signal = data['signals'][seg_idx]
            noisy_signal = add_gaussian_noise(signal, snr_db)


            for pipeline_name in PIPELINES:
                if pipeline_name not in clean_results[seg_idx]:
                    continue

                try:
                    clean_params = clean_results[seg_idx][pipeline_name]
                    pert_metrics = process_segment_perturbed(noisy_signal, pipeline_name, clean_params)

                    # Compute ΔRQA for each band
                    band_deltas = {}
                    for band in BANDS:
                        if band in clean_params and band in pert_metrics:
                            clean_m = clean_params[band]['metrics']
                            pert_m = pert_metrics[band]
                            delta = compute_delta_rqa(clean_m, pert_m)

                            # Average across bands for this segment
                            for metric, value in delta.items():
                                if metric not in band_deltas:
                                    band_deltas[metric] = []
                                band_deltas[metric].append(value)

                    # Store average ΔRQA across bands
                    if band_deltas:
                        delta_rqa[pert_type][pipeline_name][seg_idx] = {
                            metric: np.mean(values) for metric, values in band_deltas.items()
                        }

                except Exception as e:
                    warnings.warn(f"Failed noise processing: {e}")

    for offset in SHIFT_OFFSETS:
        print(f"\n  Shift offset = {offset:+d} samples")
        pert_type = f"shift_{offset:+d}"

        for seg_idx in range(n_segments):
            print(f"    Segment {seg_idx+1}/{n_segments}", end='\r')

            if seg_idx not in clean_results:
                continue

            signal = data['signals'][seg_idx]
            shifted_signal = circular_shift(signal, offset)

            for pipeline_name in PIPELINES:
                if pipeline_name not in clean_results[seg_idx]:
                    continue

                try:
                    clean_params = clean_results[seg_idx][pipeline_name]
                    pert_metrics = process_segment_perturbed(shifted_signal, pipeline_name, clean_params)

                    # Compute ΔRQA for each band
                    band_deltas = {}
                    for band in BANDS:
                        if band in clean_params and band in pert_metrics:
                            clean_m = clean_params[band]['metrics']
                            pert_m = pert_metrics[band]
                            delta = compute_delta_rqa(clean_m, pert_m)

                            for metric, value in delta.items():
                                if metric not in band_deltas:
                                    band_deltas[metric] = []
                                band_deltas[metric].append(value)

                    # Store average ΔRQA across bands
                    if band_deltas:
                        delta_rqa[pert_type][pipeline_name][seg_idx] = {
                            metric: np.mean(values) for metric, values in band_deltas.items()
                        }

                except Exception as e:
                    warnings.warn(f"Failed shift processing: {e}")

        print(f"\n  Completed shift offset = {offset:+d}")

    patient_ids = [pid.decode() if isinstance(pid, bytes) else pid
                   for pid in data['patient_ids']]

    results_table = []

    for pert_type in delta_rqa.keys():
        print(f"\n  {pert_type}")

        for pipeline_name in PIPELINES:
            if pipeline_name not in delta_rqa[pert_type]:
                continue

            # Aggregate to patient level
            patient_medians = aggregate_to_patient_level(
                delta_rqa[pert_type][pipeline_name],
                patient_ids
            )

            # Compute grand statistics
            stats = compute_grand_statistics(patient_medians)

            # Prepare result row
            row = {
                'perturbation': pert_type,
                'pipeline': pipeline_name,
                'n_patients': len(patient_medians)
            }

            for metric in RQA_METRICS:
                median, q25, q75 = stats[metric]
                row[f'{metric}_median'] = median
                row[f'{metric}_q25'] = q25
                row[f'{metric}_q75'] = q75

            # Compute mean across all metrics
            row['mean_delta'] = np.mean([stats[m][0] for m in RQA_METRICS])

            results_table.append(row)

            print(f"    {pipeline_name}: mean ΔRQA = {row['mean_delta']:.4f}")

    output_path = os.path.join(OUTPUT_DIR, 'robustness_results.csv')

    with open(output_path, 'w', newline='') as f:
        fieldnames = ['perturbation', 'pipeline', 'n_patients']
        for metric in RQA_METRICS:
            fieldnames.extend([f'{metric}_median', f'{metric}_q25', f'{metric}_q75'])
        fieldnames.append('mean_delta')

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_table)


    overall_scores = defaultdict(list)
    for row in results_table:
        overall_scores[row['pipeline']].append(row['mean_delta'])

    ranked = sorted(overall_scores.items(), key=lambda x: np.mean(x[1]))

    for rank, (pipeline, scores) in enumerate(ranked, 1):
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{rank}. {pipeline:<15}: {mean_score:.6f} ± {std_score:.6f}")

if __name__ == '__main__':
    main()