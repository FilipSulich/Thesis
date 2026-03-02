import numpy as np
import pandas as pd
import h5py
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from wavelets import stationary_wavelet_transform as stationary_wavelet


class SubbandExtractor:
    """
    Extracts and saves decomposed signal subbands.

    Pipeline:
    1. Load cleaned signal (already filtered + denoised)
    2. SWT decomposition
    3. Reconstruct each subband
    4. Z-score normalize each subband
    5. Save to CSV
    """

    SUBBANDS = ['D1', 'D2', 'D3', 'D4', 'A4']

    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 4
    ):
        """
        Initialize subband extractor.

        Args:
            wavelet: Wavelet type for SWT
            level: Decomposition level
        """
        self.wavelet = wavelet
        self.level = level
        self.swt = stationary_wavelet.SWT(wavelet=wavelet, level=level)

    def decompose_and_normalize(
        self,
        signal: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Decompose signal into subbands and normalize each.

        Args:
            signal: Input signal (already cleaned)

        Returns:
            Dictionary mapping subband name -> normalized signal
        """
        # SWT decomposition
        coeffs = self.swt.decompose(signal, level=self.level)

        subbands = {}

        # Detail coefficients D1-D4
        for i in range(len(coeffs)):
            # Zero out all other coefficients
            zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
            zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])

            # Reconstruct this subband
            subband = self.swt.reconstruct(zeroed)

            # Z-score normalize
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)

            subbands[f'D{i+1}'] = subband

        # Approximation A4
        zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
        subband = self.swt.reconstruct(zeroed)
        subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
        subbands['A4'] = subband

        return subbands

    def extract_from_hdf5(
        self,
        hdf5_path: Path,
        output_dir: Path,
        dataset_name: str,
        split: str = 'all'
    ):
        """
        Extract subbands from HDF5 file and save to CSV.

        Args:
            hdf5_path: Path to HDF5 file with cleaned signals
            output_dir: Directory to save CSV files
            dataset_name: Name prefix for output files
            split: HDF5 group name ('train', 'val', 'test', 'all')
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Extracting subbands from {hdf5_path}")
        print(f"Split: {split}, Output: {output_dir}")
        print(f"{'='*80}\n")

        # Load data
        with h5py.File(hdf5_path, 'r') as f:
            grp = f[split]
            signals = grp['signals'][:]
            labels = grp['labels'][:]
            segment_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]]

            # Optional fields
            if 'patient_ids' in grp:
                patient_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]]
            else:
                patient_ids = [''] * len(signals)

            if 'record_ids' in grp:
                record_ids = [s.decode() if isinstance(s, bytes) else s for s in grp['record_ids'][:]]
            else:
                record_ids = [''] * len(signals)

        print(f"Loaded {len(signals)} signals")

        # Initialize storage for each subband
        subband_data = {sb: [] for sb in self.SUBBANDS}

        # Process each signal
        for i in tqdm(range(len(signals)), desc="Decomposing signals"):
            signal = signals[i]

            # Decompose and normalize
            subbands = self.decompose_and_normalize(signal)

            # Create row metadata
            metadata = {
                'segment_id': segment_ids[i],
                'patient_id': patient_ids[i],
                'record_id': record_ids[i],
                'label': labels[i]
            }

            # Store each subband
            for sb_name in self.SUBBANDS:
                row = metadata.copy()

                # Add signal values
                sb_signal = subbands[sb_name]
                for j, val in enumerate(sb_signal):
                    row[f's{j}'] = val

                subband_data[sb_name].append(row)

        # Save each subband to CSV
        for sb_name in self.SUBBANDS:
            output_path = output_dir / f'{dataset_name}_{sb_name}.csv'
            df = pd.DataFrame(subband_data[sb_name])
            df.to_csv(output_path, index=False)
            print(f"Saved {sb_name}: {len(df)} segments → {output_path}")

        print(f"\n{'='*80}")
        print(f"Subband extraction complete!")
        print(f"{'='*80}\n")

    def extract_from_csv(
        self,
        csv_path: Path,
        output_dir: Path,
        dataset_name: str
    ):
        """
        Extract subbands from CSV file and save to separate CSV files.

        Args:
            csv_path: Path to CSV file with cleaned signals
            output_dir: Directory to save CSV files
            dataset_name: Name prefix for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Extracting subbands from {csv_path}")
        print(f"Output: {output_dir}")
        print(f"{'='*80}\n")

        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} segments from CSV")

        # Extract signal columns
        signal_cols = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
        signal_cols = sorted(signal_cols, key=lambda x: int(x[1:]))

        metadata_cols = [c for c in df.columns if c not in signal_cols]

        print(f"Signal columns: {len(signal_cols)}")
        print(f"Metadata columns: {metadata_cols}")

        # Initialize storage for each subband
        subband_data = {sb: [] for sb in self.SUBBANDS}

        # Process each row
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Decomposing signals"):
            # Extract signal
            signal = row[signal_cols].values.astype(np.float32)

            # Decompose and normalize
            subbands = self.decompose_and_normalize(signal)

            # Create metadata
            metadata = {col: row[col] for col in metadata_cols}

            # Store each subband
            for sb_name in self.SUBBANDS:
                sb_row = metadata.copy()

                # Add signal values
                sb_signal = subbands[sb_name]
                for j, val in enumerate(sb_signal):
                    sb_row[f's{j}'] = val

                subband_data[sb_name].append(sb_row)

        # Save each subband to CSV
        for sb_name in self.SUBBANDS:
            output_path = output_dir / f'{dataset_name}_{sb_name}.csv'
            df_sb = pd.DataFrame(subband_data[sb_name])
            df_sb.to_csv(output_path, index=False)
            print(f"Saved {sb_name}: {len(df_sb)} segments → {output_path}")

        print(f"\n{'='*80}")
        print(f"Subband extraction complete!")
        print(f"{'='*80}\n")

    def load_subband_csv(
        self,
        csv_path: Path
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], List[str]]:
        """
        Load a subband CSV file.

        Args:
            csv_path: Path to subband CSV

        Returns:
            Tuple of (signals, labels, segment_ids, patient_ids, record_ids)
        """
        df = pd.read_csv(csv_path)

        # Extract signal columns
        signal_cols = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]
        signal_cols = sorted(signal_cols, key=lambda x: int(x[1:]))

        signals = df[signal_cols].values.astype(np.float32)
        labels = df['label'].values.astype(np.int8)
        segment_ids = df['segment_id'].values.tolist()

        patient_ids = df['patient_id'].values.tolist() if 'patient_id' in df.columns else [''] * len(df)
        record_ids = df['record_id'].values.tolist() if 'record_id' in df.columns else [''] * len(df)

        return signals, labels, segment_ids, patient_ids, record_ids


# =============================================================================
# Example Usage Functions
# =============================================================================

def extract_paf_subbands():
    """Extract subbands from PAF dataset."""
    from extractors.paf_extractor import OUTPUT_DIR as PAF_OUTPUT_DIR

    hdf5_path = PAF_OUTPUT_DIR / 'paf_segments_all_clean_new_cleaning_min_max_with_upper.h5'
    output_dir = PAF_OUTPUT_DIR / 'subbands'

    extractor = SubbandExtractor(wavelet='db4', level=4)
    extractor.extract_from_hdf5(
        hdf5_path=hdf5_path,
        output_dir=output_dir,
        dataset_name='paf',
        split='all'
    )


def extract_mitbih_subbands(lead_time: int = 5):
    """Extract subbands from MIT-BIH dataset for a specific lead time."""
    from pipeline_mitbih.pipeline_mitbih import OUTPUT_DIR as MITBIH_OUTPUT_DIR

    csv_path = MITBIH_OUTPUT_DIR / f'mitbih_cleaned_lt{lead_time}.csv'
    output_dir = MITBIH_OUTPUT_DIR / 'subbands'

    extractor = SubbandExtractor(wavelet='db4', level=4)
    extractor.extract_from_csv(
        csv_path=csv_path,
        output_dir=output_dir,
        dataset_name=f'mitbih_lt{lead_time}'
    )


def extract_all_mitbih_subbands():
    """Extract subbands for all MIT-BIH lead times."""
    lead_times = [5, 10, 15, 30, 45]

    for lt in lead_times:
        print(f"\n{'#'*80}")
        print(f"Processing Lead Time: {lt} minutes")
        print(f"{'#'*80}")
        extract_mitbih_subbands(lead_time=lt)


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Extract decomposed signal subbands')
    parser.add_argument('--dataset', type=str, choices=['paf', 'mitbih', 'mitbih_all'],
                        default='mitbih', help='Dataset to process')
    parser.add_argument('--lead_time', type=int, default=5,
                        help='Lead time for MIT-BIH (only used if dataset=mitbih)')

    args = parser.parse_args()

    if args.dataset == 'paf':
        extract_paf_subbands()
    elif args.dataset == 'mitbih':
        extract_mitbih_subbands(lead_time=args.lead_time)
    elif args.dataset == 'mitbih_all':
        extract_all_mitbih_subbands()
