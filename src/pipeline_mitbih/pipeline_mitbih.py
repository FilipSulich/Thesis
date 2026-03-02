"""
MIT-BIH Lead-Time-Based RQA Pipeline

Patient-wise analysis of RQA metrics at specific lead times before AF onset.
For each patient at each lead time, compares pre-AF segments with matched
normal segments from the same patient (far from any AF episode).

Usage:
    pipeline = MITBIHPipeline(lead_time=5, ...)
    pipeline.prepare_dataset()
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import csv

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning.data_cleaner import DataCleaner
from wavelets import stationary_wavelet_transform as stationary_wavelet
import wfdb
import pandas as pd

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
RAW_DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/MIT')
HDF5_PATH = DATA_DIR / 'mitbih_af_segments.h5'
OUTPUT_DIR = DATA_DIR / 'results'

SAMPLING_FREQ = 250  # MIT-BIH sampling frequency
SEGMENT_DURATION_SEC = 10  # 10-second segments


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class SegmentInfo:
    """Information about a single segment."""
    segment_id: str
    patient_id: str
    label: int  # 0=normal, 1=pre-AF
    lead_time: int  # minutes before AF onset (0 for normal segments)
    signal: np.ndarray
    episode_index: int
    lead_index: int
    lead_name: str

    def to_dict(self, include_signal: bool = False) -> Dict[str, Any]:
        """Convert to dictionary for CSV export."""
        d = {
            'segment_id': self.segment_id,
            'patient_id': self.patient_id,
            'label': self.label,
            'lead_time': self.lead_time,
            'episode_index': self.episode_index,
            'lead_index': self.lead_index,
            'lead_name': self.lead_name,
        }
        if include_signal:
            for i, val in enumerate(self.signal):
                d[f's{i}'] = val
        return d


@dataclass
class PatientData:
    """Patient-level data for a specific lead time."""
    patient_id: str
    lead_time: int
    preaf_segments: List[SegmentInfo] = field(default_factory=list)
    normal_segments: List[SegmentInfo] = field(default_factory=list)

    def get_min_count(self) -> int:
        """Get minimum count between pre-AF and normal segments."""
        return min(len(self.preaf_segments), len(self.normal_segments))


# =============================================================================
# Pipeline Class
# =============================================================================

class MITBIHPipeline:
    """
    MIT-BIH Lead-Time-Based RQA Pipeline.

    Processes ECG segments at specific lead times before AF onset,
    matching pre-AF segments with normal segments from the same patient.
    """

    def __init__(
        self,
        lead_time: int = 5,
        hdf5_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        normal_buffer_min: int = 60,
        apply_cleaning: bool = True,
        highpass_cutoff: float = 0.5,
        notch_freq: float = 60.0,
        notch_q: float = 30.0,
        normalize_method: str = 'minmax',
        swt_denoise: bool = False,
        swt_level: int = 4,
    ):
        """
        Initialize MIT-BIH pipeline.

        Args:
            lead_time: Lead time in minutes before AF onset (5, 10, 15, 30, or 45)
            hdf5_path: Path to HDF5 file with segments
            output_dir: Directory for output files
            normal_buffer_min: Minimum distance from AF for normal segments
            apply_cleaning: Whether to apply data cleaning
            highpass_cutoff: Highpass filter cutoff (Hz)
            notch_freq: Notch filter frequency (Hz)
            notch_q: Notch filter Q factor
            normalize_method: Normalization method ('zscore' or 'minmax')
            swt_denoise: Whether to apply SWT-based adaptive denoising
            swt_level: SWT decomposition level for denoising
        """
        self.lead_time = lead_time
        self.hdf5_path = hdf5_path or HDF5_PATH
        self.output_dir = output_dir or OUTPUT_DIR
        self.normal_buffer_min = normal_buffer_min
        self.apply_cleaning = apply_cleaning
        self.normalize_method = normalize_method
        self.swt_denoise = swt_denoise

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data cleaner
        if apply_cleaning:
            self.cleaner = DataCleaner(
                fs=SAMPLING_FREQ,
                highpass_cutoff=highpass_cutoff,
                notch_freq=notch_freq,
                notch_q=notch_q,
                swt_denoise=swt_denoise,
                swt_level=swt_level,
            )
        else:
            self.cleaner = None

        # Data storage
        self.all_segments: List[SegmentInfo] = []
        self.patient_data: Dict[str, PatientData] = {}

        print(f"Initialized MIT-BIH Pipeline for lead time: {lead_time} min")
        print(f"  HDF5 path: {self.hdf5_path}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Normal buffer: {normal_buffer_min} min from AF")
        print(f"  Apply cleaning: {apply_cleaning}")

    def load_data(self) -> Tuple[np.ndarray, np.ndarray, List[str], List[str], np.ndarray]:
        """Load all data from HDF5 file."""
        print(f"\nLoading data from {self.hdf5_path}...")

        with h5py.File(self.hdf5_path, 'r') as f:
            signals = f['signals'][:]
            labels = f['labels'][:]
            segment_ids = [s.decode() if isinstance(s, bytes) else s for s in f['segment_ids'][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f['patient_ids'][:]]
            lead_times = f['lead_times'][:]
            episode_indices = f['episode_indices'][:]
            lead_indices = f['lead_indices'][:]
            lead_names = [s.decode() if isinstance(s, bytes) else s for s in f['lead_names'][:]]

        print(f"Loaded {len(signals)} segments from {len(np.unique(patient_ids))} patients")
        print(f"  Pre-AF: {np.sum(labels == 1)}, Normal: {np.sum(labels == 0)}")

        return signals, labels, segment_ids, patient_ids, lead_times, episode_indices, lead_indices, lead_names

    def _clean_signal(self, signal: np.ndarray, wavelet="db4") -> np.ndarray:
        """
        Apply data cleaning to a signal for DATASET PREPARATION.

        Pipeline for single-scale preparation:
        1. Raw ECG
        2. Basic filtering (highpass/lowpass)
        3. SWT decomposition
        4. Soft thresholding on ALL detail coefficients
        5. Reconstruction to ONE signal
        6. Z-score normalization

        This creates a cleaned signal ready for embedding/RP/RQA.
        """
        if self.cleaner is None:
            return signal

        # Step 1-2: Basic filtering
        signal = signal - np.mean(signal)
        signal = self.cleaner.highpass_filter(signal)
        signal = self.cleaner.lowpass_filter(signal)

        # Step 3-5: SWT-based adaptive denoising (if enabled)
        # This does: decompose → threshold details → reconstruct to one signal
        if self.swt_denoise:
            signal = self.cleaner.swt_adaptive_denoise(signal, wavelet=wavelet)

        # Step 6: Z-score normalization
        signal = self.cleaner.normalize(signal)

        return signal

    def _extract_normal_segments_for_patient(
        self,
        patient_id: str,
        record_id: str,
        n_segments: int
    ) -> List[SegmentInfo]:
        """
        Extract normal segments from raw MIT-BIH data for a specific patient.

        Extracts segments from Normal rhythm periods that are far from any AF episode.

        Args:
            patient_id: Patient identifier
            record_id: MIT-BIH record ID
            n_segments: Number of normal segments to extract

        Returns:
            List of SegmentInfo objects with normal segments
        """
        try:
            # Load record
            record_path = str(RAW_DATA_DIR / record_id)
            record = wfdb.rdrecord(record_path)
            annotation = wfdb.rdann(record_path, 'atr')

            # Find Normal rhythm periods far from AF
            normal_periods = []
            af_samples = []

            # First, identify all AF episodes
            for i, aux_note in enumerate(annotation.aux_note):
                if aux_note and ('AFIB' in aux_note or 'AFL' in aux_note):
                    af_samples.append(annotation.sample[i])

            # Find Normal rhythm periods
            for i, aux_note in enumerate(annotation.aux_note):
                if aux_note and aux_note.strip('\x00') == '(N':
                    start_sample = annotation.sample[i]

                    # Find end of this Normal period
                    end_sample = annotation.sample[i + 1] if i + 1 < len(annotation.sample) else len(record.p_signal)

                    # Check if far enough from AF
                    min_distance_to_af = float('inf')
                    for af_sample in af_samples:
                        distance = abs(start_sample - af_sample)
                        min_distance_to_af = min(min_distance_to_af, distance)

                    # Convert distance to minutes
                    distance_min = min_distance_to_af / (SAMPLING_FREQ * 60)

                    # Only keep if far enough from AF (> normal_buffer_min)
                    if distance_min > self.normal_buffer_min:
                        normal_periods.append((start_sample, end_sample))

            # Extract random segments from normal periods
            segment_length = int(SEGMENT_DURATION_SEC * SAMPLING_FREQ)
            extracted_segments = []

            for start, end in normal_periods:
                period_length = end - start
                if period_length < segment_length:
                    continue

                # Calculate how many segments fit in this period
                max_segments = (period_length - segment_length) // segment_length

                # Extract random segments
                for _ in range(min(max_segments, max(1, n_segments // len(normal_periods)))):
                    if len(extracted_segments) >= n_segments:
                        break

                    # Random start within this normal period
                    random_start = start + np.random.randint(0, period_length - segment_length + 1)
                    signal = record.p_signal[random_start:random_start + segment_length, 0]  # Lead 0

                    # Apply cleaning
                    if self.apply_cleaning:
                        signal = self._clean_signal(signal.astype(np.float32))
                    else:
                        signal = signal.astype(np.float32)

                    segment_id = f"mitbihafdb_{record_id}_normal_{len(extracted_segments)}_l0"
                    seg_info = SegmentInfo(
                        segment_id=segment_id,
                        patient_id=patient_id,
                        label=0,  # Normal
                        lead_time=0,
                        signal=signal,
                        episode_index=-1,  # Not from AF episode
                        lead_index=0,
                        lead_name='ECG1',
                    )
                    extracted_segments.append(seg_info)

                if len(extracted_segments) >= n_segments:
                    break

            return extracted_segments[:n_segments]

        except Exception as e:
            print(f"Error extracting normal segments for {patient_id}: {e}")
            return []

    def prepare_dataset(self) -> Tuple[List[SegmentInfo], List[SegmentInfo]]:
        """
        Prepare dataset for the specified lead time.

        Returns:
            Tuple of (preaf_segments, normal_segments) with matched counts per patient
        """
        # Load raw data
        signals, labels, segment_ids, patient_ids, lead_times, episode_indices, lead_indices, lead_names = self.load_data()

        # Process all segments
        print(f"\nProcessing segments for lead time {self.lead_time} min...")

        all_preaf = []
        all_normal_candidates = []

        for i in tqdm(range(len(signals)), desc="Loading segments"):
            signal = signals[i]
            label = int(labels[i])
            segment_id = segment_ids[i]
            patient_id = patient_ids[i]
            lead_time = int(lead_times[i])
            episode_idx = int(episode_indices[i])
            lead_idx = int(lead_indices[i])
            lead_name = lead_names[i]

            # Apply cleaning if enabled
            if self.apply_cleaning:
                signal = self._clean_signal(signal)

            seg_info = SegmentInfo(
                segment_id=segment_id,
                patient_id=patient_id,
                label=label,
                lead_time=lead_time,
                signal=signal,
                episode_index=episode_idx,
                lead_index=lead_idx,
                lead_name=lead_name,
            )

            # Collect pre-AF segments for this lead time
            if label == 1 and lead_time == self.lead_time:
                all_preaf.append(seg_info)

            # Note: We don't collect normal segments from HDF5 since they don't exist
            # We'll extract them from raw data later
            pass

        print(f"\nFound {len(all_preaf)} pre-AF segments at {self.lead_time} min lead time")
        print(f"Found {len(all_normal_candidates)} normal segments (candidates)")

        # Group by patient
        print("\nGrouping segments by patient...")
        patient_data: Dict[str, PatientData] = {}

        # Add pre-AF segments
        for seg in all_preaf:
            if seg.patient_id not in patient_data:
                patient_data[seg.patient_id] = PatientData(
                    patient_id=seg.patient_id,
                    lead_time=self.lead_time,
                )
            patient_data[seg.patient_id].preaf_segments.append(seg)

        # Extract normal segments for patients with pre-AF segments
        print("\nExtracting normal segments from raw MIT-BIH data...")
        for patient_id, pdata in patient_data.items():
            n_preaf = len(pdata.preaf_segments)

            # Extract record ID from patient ID
            # patient_id format: mitbih_04015 -> record_id: 04015
            record_id = patient_id.replace('mitbih_', '')

            # Extract normal segments
            normal_segs = self._extract_normal_segments_for_patient(
                patient_id=patient_id,
                record_id=record_id,
                n_segments=n_preaf  # Match the count of pre-AF segments
            )

            pdata.normal_segments = normal_segs
            print(f"  {patient_id}: Extracted {len(normal_segs)} normal segments (needed: {n_preaf})")

        # Match counts per patient and sample
        print("\nMatching segments per patient...")
        final_preaf = []
        final_normal = []
        skipped_patients = []

        for patient_id, pdata in patient_data.items():
            n_preaf = len(pdata.preaf_segments)
            n_normal = len(pdata.normal_segments)
            min_count = pdata.get_min_count()

            if min_count == 0:
                skipped_patients.append((patient_id, n_preaf, n_normal))
                continue

            # Sample min_count from each group
            # For pre-AF: random sample
            preaf_selected = np.random.choice(pdata.preaf_segments, size=min_count, replace=False).tolist()

            # For normal: random sample
            normal_selected = np.random.choice(pdata.normal_segments, size=min_count, replace=False).tolist()

            final_preaf.extend(preaf_selected)
            final_normal.extend(normal_selected)

            print(f"  {patient_id}: {min_count} segments per group (pre-AF: {n_preaf}, normal: {n_normal})")

        if skipped_patients:
            print(f"\nSkipped {len(skipped_patients)} patients with insufficient segments:")
            for pid, n_preaf, n_normal in skipped_patients:
                print(f"  {pid}: pre-AF={n_preaf}, normal={n_normal}")

        print(f"\nFinal dataset:")
        print(f"  Pre-AF segments: {len(final_preaf)}")
        print(f"  Normal segments: {len(final_normal)}")
        print(f"  Patients: {len(patient_data) - len(skipped_patients)}")

        self.all_segments = final_preaf + final_normal

        return final_preaf, final_normal

    def export_to_csv(self, output_path: Optional[Path] = None):
        """
        Export prepared dataset to CSV file.

        Args:
            output_path: Output CSV path (default: output_dir/dataset_lt{lead_time}.csv)
        """
        if output_path is None:
            output_path = self.output_dir / f'dataset_lt{self.lead_time}.csv'

        if not self.all_segments:
            print("No segments to export. Run prepare_dataset() first.")
            return

        print(f"\nExporting dataset to {output_path}...")

        # Get signal length
        n_samples = len(self.all_segments[0].signal)

        # Prepare header
        header = ['segment_id', 'patient_id', 'label', 'lead_time', 'episode_index', 'lead_index', 'lead_name']
        header.extend([f's{i}' for i in range(n_samples)])

        # Write CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()

            for seg in tqdm(self.all_segments, desc="Writing CSV"):
                writer.writerow(seg.to_dict(include_signal=True))

        print(f"Exported {len(self.all_segments)} segments to {output_path}")
        print(f"  Signal length: {n_samples} samples ({n_samples/SAMPLING_FREQ:.1f}s at {SAMPLING_FREQ}Hz)")

    def export_subbands_to_csv(
        self,
        csv_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        wavelet: str = 'db4',
        level: int = 4
    ):
        """
        Extract decomposed subbands and save to separate CSV files.

        Pipeline:
        1. Load cleaned signal from CSV (already filtered + denoised)
        2. SWT decomposition
        3. Reconstruct each subband independently
        4. Z-score normalize each subband
        5. Save each subband to separate CSV file

        Args:
            csv_path: Path to cleaned signal CSV (default: output_dir/dataset_lt{lead_time}.csv)
            output_dir: Directory for subband CSV files (default: output_dir/subbands)
            wavelet: Wavelet type for SWT
            level: Decomposition level

        Output files:
            - {output_dir}/mitbih_lt{lead_time}_D1.csv
            - {output_dir}/mitbih_lt{lead_time}_D2.csv
            - {output_dir}/mitbih_lt{lead_time}_D3.csv
            - {output_dir}/mitbih_lt{lead_time}_D4.csv
            - {output_dir}/mitbih_lt{lead_time}_A4.csv
        """
        # Set defaults
        if csv_path is None:
            csv_path = self.output_dir / f'dataset_lt{self.lead_time}.csv'
        if output_dir is None:
            output_dir = self.output_dir / 'subbands'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*80}")
        print(f"Extracting subbands from {csv_path}")
        print(f"Output: {output_dir}")
        print(f"Wavelet: {wavelet}, Level: {level}")
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

        # Initialize SWT
        swt = stationary_wavelet.SWT(wavelet=wavelet, level=level)

        # Subband names
        subbands_names = ['D1', 'D2', 'D3', 'D4', 'A4']
        subband_data = {sb: [] for sb in subbands_names}

        # Process each signal
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Decomposing signals"):
            # Extract signal
            signal = row[signal_cols].values.astype(np.float32)

            # SWT decomposition
            coeffs = swt.decompose(signal, level=level)

            # Reconstruct and normalize each subband
            subbands = {}

            # Detail coefficients D1-D4
            for i in range(len(coeffs)):
                # Zero out all other coefficients
                zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
                zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])

                # Reconstruct this subband
                subband = swt.reconstruct(zeroed)

                # Z-score normalize
                subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
                subbands[f'D{i+1}'] = subband

            # Approximation A4
            zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
            subband = swt.reconstruct(zeroed)
            subband = (subband - np.mean(subband)) / (np.std(subband) + 1e-8)
            subbands['A4'] = subband

            # Create metadata
            metadata = {col: row[col] for col in metadata_cols}

            # Store each subband
            for sb_name in subbands_names:
                sb_row = metadata.copy()

                # Add signal values
                sb_signal = subbands[sb_name]
                for j, val in enumerate(sb_signal):
                    sb_row[f's{j}'] = val

                subband_data[sb_name].append(sb_row)

        # Save each subband to CSV
        for sb_name in subbands_names:
            output_path = output_dir / f'mitbih_lt{self.lead_time}_{sb_name}.csv'
            df_sb = pd.DataFrame(subband_data[sb_name])
            df_sb.to_csv(output_path, index=False)
            print(f"Saved {sb_name}: {len(df_sb)} segments → {output_path}")

        print(f"\n{'='*80}")
        print(f"Subband extraction complete!")
        print(f"{'='*80}\n")


if __name__ == '__main__':
    # Configuration
    LEAD_TIME = 10  # minutes before AF onset

    # Cleaning parameters
    APPLY_CLEANING = True
    HIGHPASS_CUTOFF = 0.5  # Hz
    NOTCH_FREQ = 60.0  # Hz (for powerline noise)
    NOTCH_Q = 30.0
    NORMALIZE_METHOD = 'zscore'  # Always use z-score for consistency
    SWT_DENOISE = True  # Apply SWT-based adaptive denoising
    SWT_LEVEL = 4  # SWT decomposition level

    # Pipeline parameters
    NORMAL_BUFFER_MIN = 60  # Minutes from AF for normal segments

    # Initialize pipeline
    pipeline = MITBIHPipeline(
        lead_time=LEAD_TIME,
        normal_buffer_min=NORMAL_BUFFER_MIN,
        apply_cleaning=APPLY_CLEANING,
        highpass_cutoff=HIGHPASS_CUTOFF,
        notch_freq=NOTCH_FREQ,
        notch_q=NOTCH_Q,
        normalize_method=NORMALIZE_METHOD,
        swt_denoise=SWT_DENOISE,
        swt_level=SWT_LEVEL,
    )

    # Prepare dataset
    preaf_segments, normal_segments = pipeline.prepare_dataset()

    # Export to CSV
    pipeline.export_to_csv()
    # pipeline.export_subbands_to_csv()

    print("\nDataset preparation complete!")
