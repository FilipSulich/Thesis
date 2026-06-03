"""signal generation functions."""

import numpy as np


def generate_composite_sine(n, fs, freq_dict):
    """
    generate composite signal as sum of sinusoids with random phases for each frequency component

    Args:
        n (int): signal length in samples
        fs (int): sampling frequency in Hz
        freq_dict (dict): dictionary mapping band names to their frequencies in Hz

    Returns:
        numpy.ndarray: composite signal as sum of all sinusoidal components with random phases
    """
    t = np.arange(n) / fs
    signal = np.zeros(n)

    for band, freq in freq_dict.items():
        phase = np.random.uniform(0, 2 * np.pi)
        signal += np.sin(2 * np.pi * freq * t + phase)

    return signal


def generate_single_sine(n, fs, freq, random_phase=False):
    """
    generate single sinusoid at specified frequency with optional random phase

    Args:
        n (int): signal length in samples
        fs (int): sampling frequency in Hz
        freq (float): frequency of the sinusoid in Hz
        random_phase (bool): if True, use random phase offset, otherwise use zero phase (default: False)

    Returns:
        numpy.ndarray: sine wave signal array
    """
    t = np.arange(n) / fs
    phase = np.random.uniform(0, 2 * np.pi) if random_phase else 0.0
    return np.sin(2 * np.pi * freq * t + phase)
