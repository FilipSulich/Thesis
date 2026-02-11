"""
Base AF Segment Extractor

Abstract base class for extracting pre-AF segments from various databases.
Provides unified interface for:
- MIT-BIH AFDB
- LTAFDB (Long Term AF Database)
- PTB-XL

Designed for:
- Patient-wise data splitting
- Wavelet transform processing
- Multi-database merging
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import json
from enum import Enum


class DatabaseSource(Enum):
    """Supported database sources."""
    MIT_BIH_AFDB = "MIT-BIH-AFDB"
    LTAFDB = "LTAFDB"
    PTB_XL = "PTB-XL"


@dataclass
class UnifiedSegment:
    """
    Unified segment representation across all databases.

    This structure ensures consistent handling regardless of source database.
    """
    # Core identification
    segment_id: str  # Globally unique segment ID
    patient_id: str  # Globally unique patient ID (for patient-wise split)
    source_database: str

    # Signal data
    signal: np.ndarray  # (segment_length,) float32
    sampling_freq: int  # Original sampling frequency
    lead_name: str

    # Temporal information
    lead_time_min: int  # Minutes before AF onset
    segment_duration_sec: float
    segment_start_sec: float  # Time within original recording

    # Labels and flags
    label: int  # 1 = pre-AF, 0 = normal/control
    is_normal_to_af: bool  # True if direct Normal→AF transition

    # Optional metadata
    episode_index: Optional[int] = None
    record_id: Optional[str] = None
    preceding_rhythm: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary (excluding signal for metadata-only ops)."""
        return {
            'segment_id': self.segment_id,
            'patient_id': self.patient_id,
            'source_database': self.source_database,
            'sampling_freq': self.sampling_freq,
            'lead_name': self.lead_name,
            'lead_time_min': self.lead_time_min,
            'segment_duration_sec': self.segment_duration_sec,
            'segment_start_sec': self.segment_start_sec,
            'label': self.label,
            'is_normal_to_af': self.is_normal_to_af,
            'episode_index': self.episode_index,
            'record_id': self.record_id,
            'preceding_rhythm': self.preceding_rhythm,
        }


@dataclass
class ExtractionConfig:
    """Configuration for segment extraction."""
    segment_duration_sec: int = 10
    lead_times_min: Tuple[int, ...] = (5, 10, 30, 45)
    target_fs: Optional[int] = 250  # Resample to this frequency (None = keep original)
    lead_indices: Tuple[int, ...] = (0,)
    require_normal_preceding: bool = True


@dataclass
class DatasetSplit:
    """Train/val/test split with patient IDs."""
    train_patients: Set[str]
    val_patients: Set[str]
    test_patients: Set[str]

    @property
    def all_patients(self) -> Set[str]:
        return self.train_patients | self.val_patients | self.test_patients

    def get_split_for_patient(self, patient_id: str) -> str:
        if patient_id in self.train_patients:
            return 'train'
        elif patient_id in self.val_patients:
            return 'val'
        elif patient_id in self.test_patients:
            return 'test'
        return 'unknown'


class BaseAFExtractor(ABC):
    """
    Abstract base class for AF segment extraction.

    Subclasses must implement database-specific logic:
    - get_patient_id(): Map record to patient
    - find_af_episodes(): Identify AF episodes in a record
    - extract_segment(): Extract signal data
    """

    def __init__(
        self,
        data_dir: Path,
        config: ExtractionConfig,
        database_source: DatabaseSource
    ):
        self.data_dir = Path(data_dir)
        self.config = config
        self.database_source = database_source

    @abstractmethod
    def get_record_ids(self) -> List[str]:
        """Get list of all record IDs in the database."""
        pass

    @abstractmethod
    def get_patient_id(self, record_id: str) -> str:
        """
        Get globally unique patient ID from record ID.

        Must include database prefix for cross-database uniqueness.
        """
        pass

    @abstractmethod
    def find_af_episodes(self, record_id: str) -> List[Dict]:
        """
        Find all AF episodes in a record.

        Returns list of dicts with at minimum:
        - onset_sample: Sample index of AF onset
        - onset_time_sec: Time in seconds of AF onset
        - preceding_rhythm: Rhythm before AF (for transition validation)
        - preceding_duration_sec: Duration of preceding rhythm
        - is_valid_transition: True if Normal→AF
        """
        pass

    @abstractmethod
    def extract_segment_signal(
        self,
        record_id: str,
        start_sample: int,
        end_sample: int,
        lead_index: int
    ) -> Tuple[np.ndarray, int, str]:
        """
        Extract signal data for a segment.

        Returns:
        - signal: np.ndarray of shape (n_samples,)
        - sampling_freq: Original sampling frequency
        - lead_name: Name of the lead
        """
        pass

    def get_extractable_segments(
        self,
        record_id: str
    ) -> List[UnifiedSegment]:
        """
        Find and extract all valid segments from a record.

        This method handles the common logic of:
        1. Finding AF episodes
        2. Calculating segment boundaries
        3. Validating segments
        4. Extracting signals
        """
        patient_id = self.get_patient_id(record_id)
        af_episodes = self.find_af_episodes(record_id)
        segments = []

        for ep_idx, episode in enumerate(af_episodes):
            # Skip invalid transitions if required
            if self.config.require_normal_preceding and not episode.get('is_valid_transition', False):
                continue

            for lead_time_min in self.config.lead_times_min:
                lead_time_sec = lead_time_min * 60

                # Calculate segment boundaries
                segment_end_sec = episode['onset_time_sec'] - lead_time_sec
                segment_start_sec = segment_end_sec - self.config.segment_duration_sec

                # Validation: segment must be within bounds
                if segment_start_sec < 0:
                    continue

                # Validation: segment must be within preceding rhythm
                min_required = lead_time_sec + self.config.segment_duration_sec
                if episode.get('preceding_duration_sec', 0) < min_required:
                    continue

                # Extract for each lead
                for lead_idx in self.config.lead_indices:
                    try:
                        # Get sampling frequency to calculate samples
                        # First get a small sample to determine fs
                        test_signal, fs, lead_name = self.extract_segment_signal(
                            record_id, 0, 10, lead_idx
                        )

                        start_sample = int(segment_start_sec * fs)
                        end_sample = int(segment_end_sec * fs)

                        signal, fs, lead_name = self.extract_segment_signal(
                            record_id, start_sample, end_sample, lead_idx
                        )

                        segment_id = (
                            f"{self.database_source.value.lower().replace('-', '')}_{record_id}_"
                            f"ep{ep_idx}_lt{lead_time_min}_l{lead_idx}"
                        )

                        segments.append(UnifiedSegment(
                            segment_id=segment_id,
                            patient_id=patient_id,
                            source_database=self.database_source.value,
                            signal=signal.astype(np.float32),
                            sampling_freq=fs,
                            lead_name=lead_name,
                            lead_time_min=lead_time_min,
                            segment_duration_sec=self.config.segment_duration_sec,
                            segment_start_sec=segment_start_sec,
                            label=1,  # Pre-AF
                            is_normal_to_af=episode.get('is_valid_transition', False),
                            episode_index=ep_idx,
                            record_id=record_id,
                            preceding_rhythm=episode.get('preceding_rhythm', 'UNKNOWN')
                        ))

                    except Exception as e:
                        print(f"Error extracting segment from {record_id}: {e}")
                        continue

        return segments

    def extract_all(self, verbose: bool = True) -> List[UnifiedSegment]:
        """Extract segments from all records in the database."""
        from tqdm import tqdm

        record_ids = self.get_record_ids()
        all_segments = []

        iterator = tqdm(record_ids, desc=f"Extracting from {self.database_source.value}") if verbose else record_ids

        for record_id in iterator:
            try:
                segments = self.get_extractable_segments(record_id)
                all_segments.extend(segments)
            except Exception as e:
                if verbose:
                    print(f"\nError processing {record_id}: {e}")

        return all_segments


def create_patient_wise_split(
    segments: List[UnifiedSegment],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> DatasetSplit:
    """
    Create patient-wise train/val/test split.

    Ensures no data leakage by splitting on patient IDs, not segments.

    Parameters:
    -----------
    segments : List[UnifiedSegment]
        All extracted segments
    train_ratio, val_ratio, test_ratio : float
        Split ratios (must sum to 1.0)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    DatasetSplit object
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    # Get unique patients
    patient_ids = list(set(seg.patient_id for seg in segments))
    np.random.seed(random_seed)
    np.random.shuffle(patient_ids)

    n_patients = len(patient_ids)
    n_train = int(n_patients * train_ratio)
    n_val = int(n_patients * val_ratio)

    train_patients = set(patient_ids[:n_train])
    val_patients = set(patient_ids[n_train:n_train + n_val])
    test_patients = set(patient_ids[n_train + n_val:])

    return DatasetSplit(
        train_patients=train_patients,
        val_patients=val_patients,
        test_patients=test_patients
    )


def resample_signal(
    signal: np.ndarray,
    original_fs: int,
    target_fs: int
) -> np.ndarray:
    """
    Resample signal to target frequency.

    Uses scipy's resample function for high-quality resampling.
    """
    from scipy.signal import resample

    if original_fs == target_fs:
        return signal

    n_samples_target = int(len(signal) * target_fs / original_fs)
    return resample(signal, n_samples_target).astype(np.float32)


def merge_datasets(
    *segment_lists: List[UnifiedSegment],
    target_fs: Optional[int] = 250
) -> List[UnifiedSegment]:
    """
    Merge segments from multiple databases.

    Optionally resamples all signals to a common frequency.

    Parameters:
    -----------
    *segment_lists : List[UnifiedSegment]
        Variable number of segment lists to merge
    target_fs : Optional[int]
        Target sampling frequency (None = keep original)

    Returns:
    --------
    Merged list of segments
    """
    merged = []

    for segments in segment_lists:
        for seg in segments:
            if target_fs and seg.sampling_freq != target_fs:
                # Resample signal
                resampled = resample_signal(seg.signal, seg.sampling_freq, target_fs)
                # Create new segment with resampled signal
                merged.append(UnifiedSegment(
                    segment_id=seg.segment_id,
                    patient_id=seg.patient_id,
                    source_database=seg.source_database,
                    signal=resampled,
                    sampling_freq=target_fs,
                    lead_name=seg.lead_name,
                    lead_time_min=seg.lead_time_min,
                    segment_duration_sec=seg.segment_duration_sec,
                    segment_start_sec=seg.segment_start_sec,
                    label=seg.label,
                    is_normal_to_af=seg.is_normal_to_af,
                    episode_index=seg.episode_index,
                    record_id=seg.record_id,
                    preceding_rhythm=seg.preceding_rhythm
                ))
            else:
                merged.append(seg)

    return merged


def save_unified_hdf5(
    segments: List[UnifiedSegment],
    split: DatasetSplit,
    output_path: Path,
    config: ExtractionConfig
) -> None:
    """
    Save unified dataset to HDF5 with patient-wise splits.

    Structure:
    - train/signals, train/labels, train/metadata
    - val/signals, val/labels, val/metadata
    - test/signals, test/labels, test/metadata
    - global metadata as attributes
    """
    # Group segments by split
    train_segs = [s for s in segments if split.get_split_for_patient(s.patient_id) == 'train']
    val_segs = [s for s in segments if split.get_split_for_patient(s.patient_id) == 'val']
    test_segs = [s for s in segments if split.get_split_for_patient(s.patient_id) == 'test']

    def save_split(f: h5py.File, name: str, segs: List[UnifiedSegment]):
        if not segs:
            return

        grp = f.create_group(name)

        # Signals
        signals = np.stack([s.signal for s in segs])
        grp.create_dataset('signals', data=signals, dtype='float32',
                          compression='gzip', compression_opts=4)

        # Labels
        grp.create_dataset('labels', data=np.array([s.label for s in segs], dtype=np.int8))
        grp.create_dataset('lead_times', data=np.array([s.lead_time_min for s in segs], dtype=np.int16))

        # String metadata
        dt_str = h5py.special_dtype(vlen=str)
        grp.create_dataset('segment_ids', data=[s.segment_id for s in segs], dtype=dt_str)
        grp.create_dataset('patient_ids', data=[s.patient_id for s in segs], dtype=dt_str)
        grp.create_dataset('source_databases', data=[s.source_database for s in segs], dtype=dt_str)

        grp.attrs['n_segments'] = len(segs)
        grp.attrs['n_patients'] = len(set(s.patient_id for s in segs))

    with h5py.File(output_path, 'w') as f:
        save_split(f, 'train', train_segs)
        save_split(f, 'val', val_segs)
        save_split(f, 'test', test_segs)

        # Global attributes
        f.attrs['total_segments'] = len(segments)
        f.attrs['total_patients'] = len(set(s.patient_id for s in segments))
        f.attrs['segment_duration_sec'] = config.segment_duration_sec
        f.attrs['lead_times_min'] = list(config.lead_times_min)
        f.attrs['databases'] = list(set(s.source_database for s in segments))

        # Split info
        f.attrs['train_patients'] = list(split.train_patients)
        f.attrs['val_patients'] = list(split.val_patients)
        f.attrs['test_patients'] = list(split.test_patients)

    print(f"Saved unified dataset to {output_path}")
    print(f"  Train: {len(train_segs)} segments from {len(set(s.patient_id for s in train_segs))} patients")
    print(f"  Val: {len(val_segs)} segments from {len(set(s.patient_id for s in val_segs))} patients")
    print(f"  Test: {len(test_segs)} segments from {len(set(s.patient_id for s in test_segs))} patients")


def load_unified_hdf5(path: Path, split: str = 'train') -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Load unified dataset from HDF5.

    Parameters:
    -----------
    path : Path
        Path to HDF5 file
    split : str
        Which split to load ('train', 'val', 'test')

    Returns:
    --------
    Tuple of (signals, labels, metadata_df)
    """
    with h5py.File(path, 'r') as f:
        grp = f[split]

        signals = grp['signals'][:]
        labels = grp['labels'][:]

        metadata = pd.DataFrame({
            'segment_id': [s.decode() if isinstance(s, bytes) else s for s in grp['segment_ids'][:]],
            'patient_id': [s.decode() if isinstance(s, bytes) else s for s in grp['patient_ids'][:]],
            'source_database': [s.decode() if isinstance(s, bytes) else s for s in grp['source_databases'][:]],
            'lead_time_min': grp['lead_times'][:],
            'label': labels
        })

    return signals, labels, metadata