import pywt
import numpy as np
from src.wavelets.wavelet_transform import WaveletTransform

class SWT(WaveletTransform):
    def decompose(self, signal: np.ndarray) -> list:
        required_multiple = 2 ** self.level
        remainder = len(signal) % required_multiple
        # we need the signal length to be divisible by 2^level, if not we pad
        if remainder != 0:
            print("length not even - adding padding")
            pad_width = required_multiple - remainder
            signal = np.pad(signal, (0, pad_width))

        return pywt.swt(signal, self.wavelet, level=self.level)

    def reconstruct(self, coeffs: list) -> np.ndarray:
        return pywt.iswt(coeffs, self.wavelet)