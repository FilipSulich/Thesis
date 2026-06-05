import pywt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wavelets.wavelet_transform import WaveletTransform


class DWT(WaveletTransform):
    """Discrete Wavelet Transform (DWT) implementation using PyWavelets"""

    def decompose(self, signal: np.ndarray, level) -> list:
        """
        performs multi-level DWT decomposition producing downsampled coefficients

        Args:
            signal (np.ndarray): input signal to decompose
            level (int): number of decomposition levels

        Returns:
            list: wavelet coefficients [cA_n, cD_n, cD_n-1, ..., cD_1] where cA is approximation and cD are details
        """
        return pywt.wavedec(signal, self.wavelet, level=level)

    def reconstruct(self, coeffs: list) -> np.ndarray:
        """
        reconstructs signal from DWT coefficients using inverse transform

        Args:
            coeffs (list): wavelet coefficients from decompose()

        Returns:
            np.ndarray: reconstructed time-domain signal
        """
        return pywt.waverec(coeffs, self.wavelet)