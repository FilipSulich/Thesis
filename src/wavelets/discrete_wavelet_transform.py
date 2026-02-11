import pywt
import numpy as np
from src.wavelets.wavelet_transform import WaveletTransform


class DWT(WaveletTransform):
    def decompose(self, signal: np.ndarray) -> list:

        return pywt.wavedec(signal, self.wavelet, level=self.level)

    def reconstruct(self, coeffs: list) -> np.ndarray:
        return pywt.waverec(coeffs, self.wavelet)