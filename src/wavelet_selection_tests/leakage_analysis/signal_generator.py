import numpy as np

class SignalGenerator:
    def __init__(self, fs=128, n_samples=1280):
        """
        Args:
            fs: Sampling frequency
            n_samples: Signal length
        """
        self.fs = fs
        self.n_samples = n_samples
        self.t = np.arange(n_samples) / fs

    def pure_sine(self, freq, jitter=0.0):
        """
        Generate sine at given frequency with optional jitter.

        Args:
            freq: Target frequency (Hz)
            jitter: Max relative deviation (e.g., 0.05 = 5% jitter)
        """
        if jitter > 0:
            freq = freq + np.random.uniform(-jitter, jitter) * freq

        # Random phase for variety
        phase = np.random.uniform(0, 2 * np.pi)
        return np.sin(2 * np.pi * freq * self.t + phase).astype(np.float32)

    def composite_sine(self, freq_map, random_phase=False):
        """
        Mix multiple frequencies together.

        Args:
            freq_map: Dict mapping band names to frequencies
            random_phase: Randomize phase for each component
        """
        signal = np.zeros(self.n_samples)
        for band, freq in freq_map.items():
            phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
            signal += np.sin(2 * np.pi * freq * self.t + phase)
        return signal.astype(np.float32)
