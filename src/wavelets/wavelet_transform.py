from abc import ABC, abstractmethod
import numpy as np

class WaveletTransform(ABC):
    """abstract base class for wavelet transform implementations"""

    def __init__(self, wavelet='db4', level=4):
        """
        initializes wavelet transform with wavelet function and decomposition level

        Args:
            wavelet (str): wavelet function name (e.g., 'db4', 'sym5')
            level (int): decomposition level for multi-level analysis

        Returns:
            None
        """
        self.wavelet = wavelet
        self.level = level

    @abstractmethod
    def decompose(self, signal: np.ndarray, level) -> list:
        """
        decomposes signal into wavelet coefficients at specified level

        Args:
            signal (np.ndarray): input signal to decompose
            level (int): decomposition level

        Returns:
            list: wavelet coefficients
        """
        pass

    @abstractmethod
    def reconstruct(self, coeffs: list) -> np.ndarray:
        """
        reconstructs signal from wavelet coefficients

        Args:
            coeffs (list): wavelet coefficients from decomposition

        Returns:
            np.ndarray: reconstructed signal
        """
        pass