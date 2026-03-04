import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wavelets.stationary_wavelet_transform as stationary_wavelet


class DataCleaner:
    def __init__(self, fs=250, highpass_cutoff=0.5,
                 filter_order=4, swt_denoise=False, swt_level=4):
        self.fs = fs
        self.highpass_cutoff = highpass_cutoff
        self.filter_order = filter_order
        self.swt_denoise = swt_denoise
        self.swt_level = swt_level
        self._lowpass_cutoff_hz = 40.0
        self._boundary_trim_seconds = 0.5

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def highpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.filter_order, cutoff, btype='highpass')
        return filtfilt(b, a, signal)

    def lowpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self._lowpass_cutoff_hz / nyquist
        if cutoff >= 1.0:
            return signal
        b, a = butter(self.filter_order, cutoff, btype='lowpass')
        return filtfilt(b, a, signal)

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def normalize(self, signal):
        std = np.std(signal)
        if std == 0:
            return signal
        return (signal - np.mean(signal)) / std

    # ------------------------------------------------------------------
    # Subband post-processing
    # ------------------------------------------------------------------

    def clip_subband_outliers(self, signal, n_std=3):
        mu = np.mean(signal)
        sigma = np.std(signal)
        if sigma == 0:
            return signal
        return np.clip(signal, mu - n_std * sigma, mu + n_std * sigma)

    def trim_subband_boundaries(self, signal, trim_seconds=None):
        if trim_seconds is None:
            trim_seconds = self._boundary_trim_seconds
        n_trim = int(trim_seconds * self.fs)
        if 2 * n_trim >= len(signal):
            return signal
        return signal[n_trim:-n_trim]

    def preprocess_subband(self, subband_signal, clip_std=3, trim_seconds=None):
        """
        Post-decomposition processing: trim boundaries → clip outliers → z-score.

        Returns None if the subband has insufficient energy (variance < 1e-4),
        indicating the subband contains no useful content (e.g. D1/D2 after
        40 Hz lowpass).
        """
        signal = self.trim_subband_boundaries(subband_signal, trim_seconds)
        signal = self.clip_subband_outliers(signal, clip_std)
        if np.var(signal) < 1e-4:
            return None
        signal = self.normalize(signal)
        return signal

    # ------------------------------------------------------------------
    # Segment extraction
    # ------------------------------------------------------------------

    def trim_to_center(self, signal, target_seconds=10):
        target_samples = int(target_seconds * self.fs)
        total_samples = len(signal)
        if total_samples <= target_samples:
            return signal
        start = (total_samples - target_samples) // 2
        return signal[start:start + target_samples]

    def _detect_r_peaks(self, signal):
        """
        Pan-Tompkins R-peak detection.

        Pipeline:
          1. Bandpass filter (5-15 Hz) — isolates QRS energy
          2. Differentiate — emphasizes steep QRS slopes
          3. Square — makes all values positive, amplifies large slopes
          4. Moving window integration (150ms) — smooths into peaks
          5. Adaptive threshold — handles varying QRS amplitude
          6. Refractory period (200ms) — prevents double detection

        Works for both upright and inverted QRS because squaring
        removes polarity entirely.
        """
        fs = self.fs

        # Step 1: Bandpass 5-15 Hz
        nyquist = 0.5 * fs
        b, a = butter(2, [5 / nyquist, 15 / nyquist], btype='bandpass')
        filtered = filtfilt(b, a, signal)

        # Step 2: Differentiate
        diff = np.diff(filtered, prepend=filtered[0])

        # Step 3: Square
        squared = diff ** 2

        # Step 4: Moving window integration (150ms window)
        window = int(0.15 * fs)
        kernel = np.ones(window) / window
        integrated = np.convolve(squared, kernel, mode='same')

        # Step 5: Adaptive threshold
        # Initialise from first 2 seconds of signal
        init_samples = min(int(2 * fs), len(integrated))
        threshold = 0.5 * np.max(integrated[:init_samples])

        # Step 6: Find peaks above threshold with 200ms refractory period
        min_distance = int(0.2 * fs)
        peaks, properties = find_peaks(integrated,
                                       height=threshold,
                                       distance=min_distance)

        # If nothing found, try lower threshold (handles low-amplitude signals)
        if len(peaks) < 2:
            threshold = 0.2 * np.max(integrated)
            peaks, _ = find_peaks(integrated,
                                  height=threshold,
                                  distance=min_distance)

        # Refine: snap each detected peak to the true maximum of the
        # original signal within a ±100ms window around the integration peak.
        # This corrects the smearing introduced by the moving window integrator.
        refine_window = int(0.1 * fs)
        refined = []
        for p in peaks:
            start = max(0, p - refine_window)
            end = min(len(signal), p + refine_window)
            local = np.abs(signal[start:end])  # abs handles inverted QRS
            refined.append(start + np.argmax(local))

        return np.array(refined, dtype=int)

    def quality_check(self, signal, min_beats=5, max_beats=30,
                      min_snr=1.5, max_rr_cv=0.5):

        peaks = self._detect_r_peaks(signal)
        num_beats = len(peaks)

        if num_beats < min_beats or num_beats > max_beats:
            return False, {
                "num_beats": num_beats, "snr": None, "rr_cv": None,
                "reason": f"Beat count {num_beats} outside [{min_beats}, {max_beats}]"
            }

        # RR intervals
        rr_intervals = np.diff(peaks)
        rr_mean = np.mean(rr_intervals)
        if rr_mean == 0:
            return False, {"num_beats": num_beats, "snr": None,
                           "rr_cv": None, "reason": "Zero mean RR interval"}
        rr_cv = np.std(rr_intervals) / rr_mean
        if rr_cv > max_rr_cv:
            return False, {
                "num_beats": num_beats, "snr": None,
                "rr_cv": round(rr_cv, 3),
                "reason": f"RR CV {rr_cv:.3f} exceeds max {max_rr_cv}"
            }

        # SNR using refined peaks
        median_peak = np.median(np.abs(signal[peaks]))
        peak_mask = np.zeros(len(signal), dtype=bool)
        window = int(0.05 * self.fs)
        for p in peaks:
            peak_mask[max(0, p - window): min(len(signal), p + window)] = True

        baseline = signal[~peak_mask]
        if len(baseline) == 0 or np.std(baseline) == 0:
            return False, {"num_beats": num_beats, "snr": None,
                           "rr_cv": round(rr_cv, 3), "reason": "Degenerate baseline"}

        snr = median_peak / np.std(baseline)
        if snr < min_snr:
            return False, {
                "num_beats": num_beats, "snr": round(snr, 3),
                "rr_cv": round(rr_cv, 3),
                "reason": f"SNR {snr:.3f} below threshold {min_snr}"
            }

        return True, {
            "num_beats": num_beats, "snr": round(snr, 3),
            "rr_cv": round(rr_cv, 3), "reason": "passed"
        }

    # ------------------------------------------------------------------
    # SWT denoising (disabled by default)
    # ------------------------------------------------------------------

    def _estimate_noise_sigma(self, detail_coeffs):
        mad = np.median(np.abs(detail_coeffs - np.median(detail_coeffs)))
        return mad / 0.6745

    def _soft_threshold(self, coeffs, threshold):
        return np.sign(coeffs) * np.maximum(np.abs(coeffs) - threshold, 0)

    def swt_adaptive_denoise(self, signal, wavelet="db4"):
        N = len(signal)
        swt = stationary_wavelet.SWT(wavelet=wavelet, level=self.swt_level)
        coeffs = swt.decompose(signal, level=self.swt_level)
        d1_coeffs = coeffs[0][1]
        sigma = self._estimate_noise_sigma(d1_coeffs)
        threshold = sigma * np.sqrt(2 * np.log(N))
        denoised_coeffs = []
        for approx, detail in coeffs:
            denoised_detail = self._soft_threshold(detail, threshold)
            denoised_coeffs.append((approx, denoised_detail))
        return swt.reconstruct(denoised_coeffs)
----

    def preprocess(self, signal, check_quality=True, max_rr_cv=0.5):
        signal = signal - np.mean(signal)
        signal = self.highpass_filter(signal)
        signal = self.lowpass_filter(signal)

        if check_quality:
            is_good, info = self.quality_check(signal, max_rr_cv=max_rr_cv)
            if not is_good:
                return None, info
        else:
            info = {"reason": "skipped"}

        signal = self.normalize(signal)
        return signal, info