import pywt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wavelets.wavelet_transform import WaveletTransform


class SWT(WaveletTransform):
    """Stationary Wavelet Transform (SWT) implementation using PyWavelets"""

    def decompose(self, signal: np.ndarray, level: int) -> list:
        """
        performs multi-level SWT decomposition without decimation, pads signal to dyadic length if needed

        Args:
            signal (np.ndarray): input signal to decompose
            level (int): number of decomposition levels

        Returns:
            list: wavelet coefficients [(cA_n, cD_n), (cA_n-1, cD_n-1), ..., (cA_1, cD_1)] where each level has same length
        """
        required_multiple = 2 ** self.level
        remainder = len(signal) % required_multiple
        # we need the signal length to be divisible by 2^level (dyadic length), if not we need to pad
        if remainder != 0:
            pad_width = required_multiple - remainder
            signal = np.pad(signal, (0, pad_width), mode='reflect')
        return pywt.swt(signal, self.wavelet, level=level)

    def reconstruct(self, coefficients: list) -> np.ndarray:
        """
        reconstructs signal from SWT coefficients using inverse transform

        Args:
            coefficients (list): list of (cA, cD) tuples from decompose()

        Returns:
            np.ndarray: reconstructed time-domain signal
        """
        return pywt.iswt(coefficients, self.wavelet)