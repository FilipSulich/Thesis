import pywt
import numpy as np
from src import WaveletTransform


class DWT(WaveletTransform):
    def decompose(self, signal: np.ndarray, level) -> list:
        return pywt.wavedec(signal, self.wavelet, level=level)

    def reconstruct(self, coeffs: list) -> np.ndarray:
        return pywt.waverec(coeffs, self.wavelet)