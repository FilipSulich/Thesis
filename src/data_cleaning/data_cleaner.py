import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        from ecgdetectors import Detectors
        self.detectors = Detectors(fs)

    def notch_filter(self):
        pass

    def highpass_filter(self, signal):
        nyquist = 0.5 * self.fs
        cutoff = self.highpass_cutoff / nyquist
        b, a = butter(self.filter_order, cutoff, btype='highpass')
        return filtfilt(b, a, signal)

    def normalize(self, signal):
        min = float(np.min(signal))
        max = float(np.max(signal))
        return (signal - min) / (max - min + 1e-8)

    def clip_subband_outliers(self, signal, n_std=3):
        mu = np.mean(signal)
        sigma = np.std(signal)
        if sigma == 0:
            return signal
        return np.clip(signal, mu - n_std * sigma, mu + n_std * sigma)

    def _pan_tompkins_(self, signal):
        r_peaks = self.detectors.pan_tompkins_detector(signal)
        return r_peaks

    def quality_check(self, signal, min_beats=5, max_beats=None,
                      min_snr=1.5, max_rr_cv=0.5):

        peaks = self._pan_tompkins_(signal)

        num_beats = len(peaks)

        if num_beats < min_beats:
            return False, {
                "num_beats": num_beats, "snr": None, "rr_cv": None,
                "reason": f"Beat count {num_beats} below minimum {min_beats}"
            }

        if max_beats is not None and num_beats > max_beats:
            return False, {
                "num_beats": num_beats, "snr": None, "rr_cv": None,
                "reason": f"Beat count {num_beats} exceeds maximum {max_beats}"
            }

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

    def preprocess(self, signal, check_quality=False, max_rr_cv=0.5):
        signal = signal - np.mean(signal) # DC artifact removal
        signal = self.highpass_filter(signal) #to remove the baseline wander (0.5Hz)

        if check_quality:
            is_good, info = self.quality_check(signal, max_rr_cv=max_rr_cv)
            if not is_good:
                return None, info
        else:
            info = {"reason": "skipped"}

        # signal = self.normalize(signal)
        return signal, info