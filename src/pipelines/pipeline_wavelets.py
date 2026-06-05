import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wavelets.discrete_wavelet_transform import DWT
from src.wavelets.stationary_wavelet_transform import SWT

import numpy as np

class Pipeline:
    """wrapper for different wavelet decomposition strategies used in the leakage experiments"""

    def __init__(self, pipeline_name, level, wavelet):
        """
        initializes pipeline with wavelet decomposition configuration

        Args:
            pipeline_name (str): name identifier for the pipeline
            level (int): decomposition level for wavelet transform
            wavelet (str): wavelet function name (e.g., 'db8')

        Returns:
            None
        """
        self.pipeline_name = pipeline_name
        self.level = level
        self.wavelet = wavelet

        self.swt = SWT(wavelet=wavelet, level=level)
        self.dwt = DWT(wavelet=wavelet, level=level)

    def pipeline_swt_recon(self, signal):
        """
        performs SWT decomposition and reconstructs each subband separately in time domain (SWT-R)

        Args:
            signal (np.ndarray): input time series signal

        Returns:
            dict: dictionary mapping subband names (D1-D4, A4) to reconstructed signals
        """
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))for j in range(len(coeffs))]

            detail_idx = len(coeffs) - 1 - i

            isolated_coeffs[detail_idx] = (np.zeros_like(coeffs[detail_idx][0]),coeffs[detail_idx][1])

            recon = self.swt.reconstruct(isolated_coeffs)
            subbands[band] = recon[:len(signal)]

        isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1])) for j in range(len(coeffs))]
        isolated_coeffs[0] = (coeffs[0][0], np.zeros_like(coeffs[0][1]))
        recon = self.swt.reconstruct(isolated_coeffs)
        subbands['A4'] = recon[:len(signal)]

        return subbands

    def pipeline_swt_direct(self, signal):
        """
        extracts wavelet coefficients directly from SWT without reconstruction (SWT-D)

        Args:
            signal (np.ndarray): input time series signal

        Returns:
            dict: dictionary mapping subband names (D1-D4, A4) to coefficient arrays
        """
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            detail_idx = len(coeffs) - 1 - i
            subbands[band] = coeffs[detail_idx][1]

        subbands['A4'] = coeffs[0][0]

        return subbands

    def pipeline_dwt_recon(self, signal):
        """
        performs DWT decomposition and reconstructs each subband separately in time domain (DWT-R)

        Args:
            signal (np.ndarray): input time series signal

        Returns:
            dict: dictionary mapping subband names (D1-D4, A4) to reconstructed signals
        """
        coeffs = self.dwt.decompose(signal, level=self.level)

        subbands = {}

        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            isolated_coeffs = [np.zeros_like(c) for c in coeffs]
            detail_idx = len(coeffs) - 1 - i
            isolated_coeffs[detail_idx] = coeffs[detail_idx]

            recon = self.dwt.reconstruct(isolated_coeffs)
            subbands[band] = recon[:len(signal)]

        isolated_coeffs = [np.zeros_like(c) for c in coeffs]
        isolated_coeffs[0] = coeffs[0]
        recon = self.dwt.reconstruct(isolated_coeffs)
        subbands['A4'] = recon[:len(signal)]

        return subbands


    def pipeline_single(self, signal):
        """
        returns signal without decomposition for full-band baseline comparison

        Args:
            signal (np.ndarray): input time series signal

        Returns:
            np.ndarray: unmodified input signal
        """
        return np.ndarray(signal)