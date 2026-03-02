import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet


class DataCleaner:
    def __init__(self, fs=128, highpass_cutoff=0.5,
                 filter_order=4, notch_freq=60.0, notch_q=30.0,
                 swt_denoise=False, swt_level=4):
        self.fs = fs
        self.highpass_cutoff = highpass_cutoff
        self.filter_order = filter_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q
        self.swt_denoise = swt_denoise
        self.swt_level = swt_level

    def highpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.filter_order, cutoff, btype='highpass')
        return filtfilt(b, a, signal)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = 40.0 / nyquist
        b, a = butter(self.filter_order, cutoff, btype='lowpass')
        return filtfilt(b, a, signal)

    def normalize(self, signal):
        std = np.std(signal)
        if std == 0:
            return signal
        return (signal - np.mean(signal)) / std

    def min_max_normalize(self, signal):
        min = np.int32(np.min(signal))
        max = np.int32(np.max(signal))
        return (signal - min) / (max - min + 1e-8)

    def trim_to_center(self, signal, target_seconds=10):
        target_samples = int(target_seconds * self.fs)
        total_samples = len(signal)
        start = (total_samples - target_samples) // 2
        return signal[start:start + target_samples]

    def _estimate_noise_sigma(self, detail_coeffs):
        """
        Estimate noise standard deviation using Median Absolute Deviation (MAD).

        σ = MAD / 0.6745

        Args:
            detail_coeffs: Detail coefficients from first level (D1)

        Returns:
            Estimated noise standard deviation
        """
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        sigma = mad / 0.6745
        return sigma

    def _soft_threshold(self, coeffs, threshold):
        """
        Apply soft thresholding to coefficients.

        Soft threshold:
        - If |c| < λ → 0
        - Else: sign(c) * (|c| - λ)

        This shrinks coefficients toward zero.

        Args:
            coeffs: Wavelet coefficients
            threshold: Threshold value λ

        Returns:
            Thresholded coefficients
        """
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

    def swt_adaptive_denoise(self, signal, wavelet = "db4"):
        """
        SWT-based adaptive denoising using soft thresholding.

        For each detail subband Dk:
        1. Compute threshold λ = σ * sqrt(2 * log(N))
        2. Apply soft thresholding
        3. Reconstruct denoised signal

        Args:
            signal: Input signal

        Returns:
            Denoised signal
        """
        N = len(signal)

        # Decompose signal using SWT
        swt = stationary_wavelet.SWT(wavelet=wavelet, level=self.swt_level)
        coeffs = swt.decompose(signal, level=self.swt_level)

        # Estimate noise from first detail level (D1)
        # D1 is at index 0 in coeffs, detail coeffs are in the second element of tuple
        d1_coeffs = coeffs[0][1]  # (approximation, detail) -> detail
        sigma = self._estimate_noise_sigma(d1_coeffs)

        # Compute universal threshold
        threshold = sigma * np.sqrt(2 * np.log(N))

        # Apply soft thresholding to all detail coefficients
        denoised_coeffs = []
        for i, (approx, detail) in enumerate(coeffs):
            # Apply soft threshold to detail coefficients
            denoised_detail = self._soft_threshold(detail, threshold)
            # Keep approximation as is
            denoised_coeffs.append((approx, denoised_detail))

        # Reconstruct signal
        denoised_signal = swt.reconstruct(denoised_coeffs)

        return denoised_signal

    def quality_check(self, signal, min_beats=5, max_beats=20, min_snr=5.0, max_amplitude_mv=3.0, max_rr_cv=0.5):

        # 1. Amplitude check
        # if np.max(np.abs(signal)) > max_amplitude_mv:
        #     return False, {"reason": f"Amplitude {np.max(np.abs(signal)):.2f} exceeds {max_amplitude_mv}mV"}

        # 2. R-peak detection with robust threshold
        signal_range = np.percentile(signal, 95) - np.percentile(signal, 5)
        height_threshold = signal_range * 0.4
        min_distance = int(0.3 * self.fs)
        peaks, _ = find_peaks(signal, height=height_threshold, distance=min_distance)

        num_beats = len(peaks)
        if num_beats < min_beats or num_beats > max_beats:
            return False, {"num_beats": num_beats, "snr": None,
                           "reason": f"Beat count {num_beats} outside [{min_beats}, {max_beats}]"}

        # 3. RR interval regularity — noise produces irregular spacing
        rr_intervals = np.diff(peaks)
        rr_cv = np.std(rr_intervals) / np.mean(rr_intervals)
        if rr_cv > max_rr_cv:
            return False, {"num_beats": num_beats, "snr": None,
                           "reason": f"RR interval CV {rr_cv:.2f} too irregular (max {max_rr_cv})"}

        # 4. SNR check
        peak_amplitudes = signal[peaks]
        median_peak = np.median(np.abs(peak_amplitudes))

        peak_mask = np.zeros(len(signal), dtype=bool)
        window = int(0.05 * self.fs)
        for p in peaks:
            start = max(0, p - window)
            end = min(len(signal), p + window)
            peak_mask[start:end] = True

        baseline = signal[~peak_mask]
        baseline_std = np.std(baseline)

        if baseline_std == 0:
            return False, {"reason": "Zero baseline variance"}

        snr = median_peak / baseline_std
        if snr < min_snr:
            return False, {"num_beats": num_beats, "snr": round(snr, 2),
                           "reason": f"SNR {snr:.2f} below threshold {min_snr}"}

        return True, {"num_beats": num_beats, "snr": round(snr, 2),
                      "rr_cv": round(rr_cv, 2), "reason": "passed"}

    def preprocess(self, signal, check_quality=True):
        signal = signal - np.mean(signal)
        signal = self.highpass_filter(signal)
        signal = self.lowpass_filter(signal)
        signal = self.trim_to_center(signal, target_seconds=10)

        if check_quality:
            is_good, info = self.quality_check(signal)
            if not is_good:
                return None, info
        else:
            info = {"reason": "skipped"}

        signal = self.normalize(signal)
        return signal, info