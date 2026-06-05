from dataclasses import dataclass
from typing import Dict


@dataclass
class ExperimentConfig:
    """
    configuration dataclass for wavelet decomposition validation experiments

    Attributes:
        fs (int): sampling frequency in Hz (default: 128)
        n (int): signal length in samples (default: 1024)
        level (int): wavelet decomposition level (default: 4)
        wavelet (str): wavelet function (default: 'db4')
        subband_freqs (dict): mapping of subband names to their center frequencies in Hz
        output_dir (str): directory path for saving results and outputs
    """

    # signal parameters
    fs: int = 128           # sampling frq
    n: int = 1024           # signal length (dyadic length)

    # wavelet parameters
    level: int = 4          # decomposition level
    wavelet: str = 'db4'    # wavelet fnction

    subband_freqs: Dict[str, float] = None

    output_dir: str = None

    def __post_init__(self):
        if self.subband_freqs is None:
            self.subband_freqs = {
                'A4': 2,
                'D4': 6,
                'D3': 12,
                'D2': 24,
                'D1': 48
            }

        if self.output_dir is None:
            self.output_dir = 'results'

    @property
    def band_names(self):
        """
        get ordered list of wavelet decomposition subband names from highest to lowest frequency

        Returns:
            list: list of subband names ['D1', 'D2', 'D3', 'D4', 'A4']
        """
        return ['D1', 'D2', 'D3', 'D4', 'A4']

    @property
    def pipelines(self):
        """
        get list of available wavelet decomposition pipeline names for comparison

        Returns:
            list: list of pipeline identifiers ['swt_recon', 'swt_direct', 'dwt_recon']
        """
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
