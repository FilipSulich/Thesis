import pywt
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelets.wavelet_transform import WaveletTransform


class SWT(WaveletTransform):
    def decompose(self, signal: np.ndarray, level: int) -> list:
        required_multiple = 2 ** self.level
        remainder = len(signal) % required_multiple
        # we need the signal length to be divisible by 2^level (dyadic length), if not we need to pad
        if remainder != 0:
            pad_width = required_multiple - remainder
            signal = np.pad(signal, (0, pad_width))
        return pywt.swt(signal, self.wavelet, level=level)

    def reconstruct(self, coefficients: list) -> np.ndarray:
        return pywt.iswt(coefficients, self.wavelet)