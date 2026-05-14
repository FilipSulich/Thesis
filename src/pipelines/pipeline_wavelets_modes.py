"""Pipeline with configurable SWT boundary modes for testing."""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.wavelets.discrete_wavelet_transform import DWT
from src.wavelets.stationary_wavelet_transform_modes import SWT_Modes

import numpy as np


class Pipeline_Modes:
    """
    Modified Pipeline class that allows testing different boundary modes.

    This creates variants of swt_recon with different iswt() boundary handling
    to test if boundary artifacts are causing the TT deviation outliers.
    """

    def __init__(self, pipeline_name, level, wavelet, mode='symmetric'):
        """
        Initialize pipeline with boundary mode.

        Args:
            pipeline_name: Pipeline type
            level: Decomposition level
            wavelet: Wavelet name
            mode: Boundary mode ('symmetric', 'zero', 'periodic', 'smooth')
        """
        self.pipeline_name = pipeline_name
        self.level = level
        self.wavelet = wavelet
        self.mode = mode

        self.swt = SWT_Modes(wavelet=wavelet, level=level, mode=mode)
        self.dwt = DWT(wavelet=wavelet, level=level)

    def pipeline_swt_recon(self, signal):
        """SWT reconstruction with configurable boundary mode."""
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        # Detail subbands D1-D4
        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            isolated_coeffs = [
                (np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
                for j in range(len(coeffs))
            ]

            detail_idx = len(coeffs) - 1 - i
            isolated_coeffs[detail_idx] = (
                np.zeros_like(coeffs[detail_idx][0]),
                coeffs[detail_idx][1]
            )

            recon = self.swt.reconstruct(isolated_coeffs)
            subbands[band] = recon[:len(signal)]

        # Approximation subband A4
        isolated_coeffs = [
            (np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
            for j in range(len(coeffs))
        ]
        isolated_coeffs[0] = (coeffs[0][0], np.zeros_like(coeffs[0][1]))
        recon = self.swt.reconstruct(isolated_coeffs)
        subbands['A4'] = recon[:len(signal)]

        return subbands

    def pipeline_swt_direct(self, signal):
        """SWT direct (unaffected by boundary mode)."""
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            detail_idx = len(coeffs) - 1 - i
            subbands[band] = coeffs[detail_idx][1]

        subbands['A4'] = coeffs[0][0]

        return subbands

    def pipeline_dwt_recon(self, signal):
        """DWT reconstruction (for comparison)."""
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