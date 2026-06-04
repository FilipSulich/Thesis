import numpy as np

class SignalGenerator:
    def __init__(self, fs=128, n_samples=1280):
        """
        generates synthetic test signals for wavelet leakage experiments

        Args:
            fs (int): sampling frequency in Hz
            n_samples (int): number of samples in generated signal
        """
        self.fs = fs
        self.n_samples = n_samples
        self.t = np.arange(n_samples) / fs

    def pure_sine(self, freq, jitter=0.0):
        """
        generates pure sine wave at specified frequency with random phase and optional random frequency jitter

        Args:
            freq (float): target frequency in Hz
            jitter (float): maximum percentage jitter to apply to frequency (0.0 to 1.0)

        Returns:
            np.array: sine wave signal with random phase offset
        """
        if jitter > 0:
            freq = freq + np.random.uniform(-jitter, jitter) * freq

        phase = np.random.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * self.t + phase).astype(np.float32)