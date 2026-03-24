import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wavelets.discrete_wavelet_transform import DWT
from wavelets.stationary_wavelet_transform import SWT

import numpy as np

class Pipeline:
    def __init__(self, pipeline_name, level, wavelet):
        self.pipeline_name = pipeline_name
        self.level = level
        self.wavelet = wavelet

        self.swt = SWT(wavelet=wavelet, level=level)
        self.dwt = DWT(wavelet=wavelet, level=level)

    def pipeline_swt_recon(self, signal):
        coeffs = self.swt.decompose(signal, level=self.level)

        # coeffs is list of (cA_n, cD_n), (cA_n-1, cD_n-1), ..., (cA_1, cD_1)
        # We need to extract D1, D2, D3, D4, A4
        subbands = {}

        # Extract each detail level by zeroing others
        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            # Create zero coefficients
            isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
                              for j in range(len(coeffs))]
            # Set only target detail to non-zero (reversed indexing: D1 is last, D4 is first)
            detail_idx = len(coeffs) - 1 - i
            isolated_coeffs[detail_idx] = (np.zeros_like(coeffs[detail_idx][0]),
                                           coeffs[detail_idx][1])

            # Reconstruct
            recon = self.swt.reconstruct(isolated_coeffs)
            subbands[band] = recon[:len(signal)]  # Trim to original length

        # Extract A4 (approximation at level 4)
        isolated_coeffs = [(np.zeros_like(coeffs[j][0]), np.zeros_like(coeffs[j][1]))
                          for j in range(len(coeffs))]
        isolated_coeffs[0] = (coeffs[0][0], np.zeros_like(coeffs[0][1]))
        recon = self.swt.reconstruct(isolated_coeffs)
        subbands['A4'] = recon[:len(signal)]

        return subbands

    def pipeline_swt_direct(self, signal):
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        # Extract detail coefficients directly
        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            detail_idx = len(coeffs) - 1 - i
            subbands[band] = coeffs[detail_idx][1]  # cD at that level

        # Extract approximation at level 4
        subbands['A4'] = coeffs[0][0]  # cA at level 4

        return subbands

    def pipeline_dwt_recon(self,signal):
        """DWT decomposition -> isolate subband -> reconstruct -> time-domain signals."""
        coeffs = self.dwt.decompose(signal, level=self.level)

        # coeffs is [cA_n, cD_n, cD_n-1, ..., cD_1]
        subbands = {}

        # Extract each detail level
        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            isolated_coeffs = [np.zeros_like(c) for c in coeffs]
            detail_idx = len(coeffs) - 1 - i  # D1 is last, D4 is at index 1
            isolated_coeffs[detail_idx] = coeffs[detail_idx]

            recon = self.dwt.reconstruct(isolated_coeffs)
            subbands[band] = recon[:len(signal)]  # Trim to original length

        # Extract A4
        isolated_coeffs = [np.zeros_like(c) for c in coeffs]
        isolated_coeffs[0] = coeffs[0]
        recon = self.dwt.reconstruct(isolated_coeffs)
        subbands['A4'] = recon[:len(signal)]

        return subbands

    def pipeline_dwt_direct(self, signal):
        """DWT decomposition -> extract coefficients directly (no reconstruction)."""
        coeffs = self.dwt.decompose(signal, level=self.level)
        # coeffs is [cA_n, cD_n, cD_n-1, ..., cD_1] - a flat list of arrays

        subbands = {}

        # Extract detail coefficients directly
        for i, band in enumerate(['D1', 'D2', 'D3', 'D4']):
            detail_idx = len(coeffs) - 1 - i  # D1 is last, D4 is at index 1
            subbands[band] = coeffs[detail_idx]  # Direct coefficient array

        # Extract approximation at level 4
        subbands['A4'] = coeffs[0]  # cA at level 4

        return subbands