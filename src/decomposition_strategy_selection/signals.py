"""signal generation functions."""

import numpy as np


def generate_composite_sine(n, fs, freq_dict, random_phase=False):
    """
    generate composite signal as sum of sinusoids.

    args:
        n: signal length (samples)
        fs: sampling frequency (Hz)
        freq_dict: dict mapping band names to frequencies
        random_phase: if True, use random phases for each component

    returns:
        composite signal as numpy array
    """
    t = np.arange(n) / fs
    signal = np.zeros(n)

    for band, freq in freq_dict.items():
        phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
        signal += np.sin(2 * np.pi * freq * t + phase)

    return signal


def generate_single_sine(n, fs, freq, random_phase=False):
    """
    generate single sinusoid at given frequency.

    args:
        n: signal length (samples)
        fs: sampling frequency (Hz)
        freq: frequency (Hz)
        random_phase: if True, use random phase

    returns:
        sine wave as numpy array
    """
    t = np.arange(n) / fs
    phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
    return np.sin(2 * np.pi * freq * t + phase)
