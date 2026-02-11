from abc import ABC, abstractmethod
import pywt
import numpy as np


class WaveletTransform(ABC):
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    @abstractmethod
    def decompose(self, signal: np.ndarray) -> list:
        """Decompose signal into wavelet coefficients."""
        pass

    @abstractmethod
    def reconstruct(self, coeffs: list) -> np.ndarray:
        """Reconstruct signal from wavelet coefficients."""
        pass