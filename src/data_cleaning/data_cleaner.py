import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks


class DataCleaner:
    def __init__(self, fs=250, highpass_cutoff=0.5,
                 filter_order=4, notch_freq=50.0, notch_q=30.0):
        self.fs = fs
        self.highpass_cutoff = highpass_cutoff
        self.filter_order = filter_order
        self.notch_freq = notch_freq
        self.notch_q = notch_q

    def highpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.filter_order, cutoff, btype='highpass')
        return filtfilt(b, a, signal)

    def notch_filter(self, signal):
        nyquist = 0.5 * self.fs
        w0 = self.notch_freq / nyquist
        b, a = iirnotch(w0, self.notch_q)
        return filtfilt(b, a, signal)

    def normalize(self, signal):
        mean = np.mean(signal)
        std = np.std(signal)
        if std == 0:
            return signal
        return (signal - mean) / std

    def trim_to_center(self, signal, target_seconds=10):
        target_samples = target_seconds * self.fs
        total_samples = len(signal)
        start = (total_samples - target_samples) // 2
        return signal[start:start + target_samples]

    def quality_check(self, signal, min_beats=5, max_beats=20, min_snr=3.0):
        """
        Check signal quality using R-peak detection.

        Returns:
            is_good (bool): True if signal passes quality check
            info (dict): Diagnostic info (num_beats, snr)
        """
        # Minimum distance between R-peaks: ~300ms (200 bpm max)
        min_distance = int(0.3 * self.fs)

        # Detect R-peaks using adaptive threshold
        height_threshold = np.std(signal) * 1.5
        peaks, properties = find_peaks(
            signal,
            height=height_threshold,
            distance=min_distance
        )

        num_beats = len(peaks)

        if num_beats < min_beats or num_beats > max_beats:
            return False, {"num_beats": num_beats, "snr": None,
                           "reason": f"Beat count {num_beats} outside [{min_beats}, {max_beats}]"}

        # Compute SNR: median R-peak amplitude vs baseline noise
        peak_amplitudes = signal[peaks]
        median_peak = np.median(np.abs(peak_amplitudes))

        # Baseline = everything not within 50ms of an R-peak
        peak_mask = np.zeros(len(signal), dtype=bool)
        window = int(0.05 * self.fs)  # 50ms around each peak
        for p in peaks:
            start = max(0, p - window)
            end = min(len(signal), p + window)
            peak_mask[start:end] = True

        baseline = signal[~peak_mask]
        if len(baseline) == 0:
            return False, {"num_beats": num_beats, "snr": None,
                           "reason": "No baseline samples"}

        baseline_std = np.std(baseline)
        if baseline_std == 0:
            return False, {"num_beats": num_beats, "snr": None,
                           "reason": "Zero baseline variance"}

        snr = median_peak / baseline_std

        if snr < min_snr:
            print(num_beats)
            return False, {"num_beats": num_beats, "snr": round(snr, 2),
                           "reason": f"SNR {snr:.2f} below threshold {min_snr}"}

        return True, {"num_beats": num_beats, "snr": round(snr, 2), "reason": "passed"}

    def preprocess(self, signal, check_quality=True):
        signal = signal - np.mean(signal)
        signal = self.highpass_filter(signal)
        signal = self.notch_filter(signal)
        signal = self.trim_to_center(signal, target_seconds=10)

        if check_quality:
            is_good, info = self.quality_check(signal, max_beats=25)
            if not is_good:
                return None, info
        else:
            info = {"reason": "skipped"}

        signal = self.normalize(signal)
        return signal, info

    def preprocess_group(self, signals, check_quality=True):
        results = []
        rejected = 0
        for signal in signals:
            cleaned, info = self.preprocess(signal, check_quality=check_quality)
            if cleaned is not None:
                results.append(cleaned)
            else:
                rejected += 1
        print(f"Processed {len(signals)} segments: {len(results)} passed, {rejected} rejected")
        return results