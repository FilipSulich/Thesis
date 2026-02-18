from abc import ABC, abstractmethod
import numpy as np

class WaveletTransform(ABC):
    def __init__(self, wavelet='db4', level=4):
        self.wavelet = wavelet
        self.level = level

    @abstractmethod
    def decompose(self, signal: np.ndarray, level) -> list:
        pass

    @abstractmethod
    def reconstruct(self, coeffs: list) -> np.ndarray:
        pass