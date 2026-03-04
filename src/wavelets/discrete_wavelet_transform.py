import pywt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelets.wavelet_transform import WaveletTransform


class DWT(WaveletTransform):
    def decompose(self, signal: np.ndarray, level) -> list:
        return pywt.wavedec(signal, self.wavelet, level=level)

    def reconstruct(self, coeffs: list) -> np.ndarray:
        return pywt.waverec(coeffs, self.wavelet)