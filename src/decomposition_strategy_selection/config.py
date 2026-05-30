"""shared configuration and constants for decomposition strategy selection."""

from dataclasses import dataclass
from typing import Dict


@dataclass
class ExperimentConfig:
    """configuration for decomposition experiments."""

    # signal parameters
    fs: int = 128           # sampling frequency (Hz)
    n: int = 1024           # signal length (samples)

    # wavelet parameters
    level: int = 4          # decomposition level
    wavelet: str = 'db4'    # wavelet type

    # subband frequencies (Hz)
    subband_freqs: Dict[str, float] = None

    # output directory
    output_dir: str = None

    def __post_init__(self):
        # default subband frequencies if not provided
        if self.subband_freqs is None:
            self.subband_freqs = {
                'A4': 2,
                'D4': 6,
                'D3': 12,
                'D2': 24,
                'D1': 48
            }

        # default output directory
        if self.output_dir is None:
            self.output_dir = 'results'

    @property
    def band_names(self):
        """ordered list of band names."""
        return ['D1', 'D2', 'D3', 'D4', 'A4']

    @property
    def pipelines(self):
        """list of pipeline names."""
        return ['swt_recon', 'swt_direct', 'dwt_recon']


# pipeline display labels
PIPELINE_LABELS = {
    'swt_recon': 'SWT-R',
    'swt_direct': 'SWT-D',
    'dwt_recon': 'DWT-R',
}

# pipeline colors for plots
PIPELINE_COLORS = {
    'SWT-R': '#1f77b4',
    'SWT-D': '#d62728',
    'DWT-R': '#2ca02c',
}

# metric display labels
METRIC_LABELS = {
    'det_deviation': 'DET Deviation',
    'lam_deviation': 'LAM Deviation',
    'tt_deviation': 'TT Deviation',
    'entr_deviation': 'ENTR Deviation',
    'hamming_distance': 'Hamming Distance',
    'rp_pattern_corr': 'RP Pattern Correlation'
}

# default metrics to analyze
DEFAULT_METRICS = [
    'det_deviation',
    'lam_deviation',
    'tt_deviation',
    'hamming_distance',
    'rp_pattern_corr'
]
