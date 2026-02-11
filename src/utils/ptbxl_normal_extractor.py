"""
PTB-XL Normal ECG Segment Extractor

Extracts 10-second normal ECG segments from PTB-XL dataset to balance
pre-AF segments from MIT-BIH AFDB and LTAFDB.

PTB-XL characteristics:
- 21,799 ECG recordings (10 seconds each)
- 9,514 recordings with NORM diagnostic class
- 18,869 unique patients
- Available at 100 Hz and 500 Hz

This script:
1. Filters for NORM diagnostic class
2. Samples segments ensuring patient diversity
3. Sets lead_time to -1 (normal rhythm has no arrhythmia event)
4. Bucket assignment for train/test splits is handled at training time

Usage:
    python ptbxl_normal_extractor.py --data_dir /path/to/ptb-xl --output_dir /path/to/output
"""

import wfdb
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import json
import h5py
from tqdm import tqdm
import ast


@dataclass
class SegmentConfig:
    """Configuration for segment extraction."""
    segment_duration_sec: int = 10  # PTB-XL recordings are 10 seconds
    target_fs: int = 100  # Use 100 Hz version (matching common target)
    lead_index: int = 1  # Lead II (index 1) - standard for rhythm analysis
    min_norm_confidence: float = 50.0  # Minimum NORM confidence score


@dataclass
class NormalSegment:
    """Metadata for an extracted normal segment."""
    segment_id: str
    patient_id: str
    source_database: str
    ecg_id: int
    lead_time_min: int  # -1 for normal segments (no arrhythmia event)
    segment_duration_sec: float
    original_fs: int
    lead_index: int
    lead_name: str
    norm_confidence: float
    age: Optional[float]
    sex: Optional[int]
    strat_fold: int
    label: int = 0  # Normal class


@dataclass
class ExtractedSegment:
    """A fully extracted segment with signal data."""
    signal: np.ndarray
    metadata: NormalSegment


def load_ptbxl_metadata(data_dir: Path) -> pd.DataFrame:
    """
    Load and parse PTB-XL metadata.

    Parameters:
    -----------
    data_dir : Path
        Path to PTB-XL directory

    Returns:
    --------
    pd.DataFrame with parsed metadata
    """
    csv_path = data_dir / 'ptbxl_database.csv'
    df = pd.read_csv(csv_path, index_col='ecg_id')

    # Parse scp_codes from string to dict
    df['scp_codes'] = df['scp_codes'].apply(lambda x: ast.literal_eval(x))

    return df


def load_scp_statements(data_dir: Path) -> pd.DataFrame:
    """Load SCP diagnostic code statements."""
    scp_path = data_dir / 'scp_statements.csv'
    return pd.read_csv(scp_path, index_col=0)


def filter_normal_records(
    df: pd.DataFrame,
    min_confidence: float = 50.0
) -> pd.DataFrame:
    """
    Filter for records with NORM diagnostic class.

    Parameters:
    -----------
    df : pd.DataFrame
        PTB-XL metadata DataFrame
    min_confidence : float
        Minimum NORM confidence score (0-100)

    Returns:
    --------
    pd.DataFrame with only NORM records
    """
    def has_norm(scp_codes: dict) -> bool:
        return 'NORM' in scp_codes and scp_codes['NORM'] >= min_confidence

    mask = df['scp_codes'].apply(has_norm)
    return df[mask].copy()


def get_norm_confidence(scp_codes: dict) -> float:
    """Extract NORM confidence score from scp_codes dict."""
    return scp_codes.get('NORM', 0.0)


def sample_segments(
    df: pd.DataFrame,
    n_samples: int,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Sample normal segments ensuring patient diversity.

    Parameters:
    -----------
    df : pd.DataFrame
        Filtered NORM records
    n_samples : int
        Number of segments to sample
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame with sampled records
    """
    np.random.seed(random_seed)

    # Get unique patients
    patient_ids = df['patient_id'].unique()

    if len(df) < n_samples:
        print(f"Warning: Only {len(df)} NORM records available, need {n_samples}")
        return df.copy()

    # Sample ensuring patient diversity
    np.random.shuffle(patient_ids)

    # Sample records, prioritizing unique patients
    sampled_records = []

    # First pass: one record per patient
    for pid in patient_ids:
        if len(sampled_records) >= n_samples:
            break
        patient_records = df[df['patient_id'] == pid]
        if len(patient_records) > 0:
            record = patient_records.sample(n=1, random_state=random_seed)
            sampled_records.append(record)

    # Second pass: if still need more, allow repeats
    if len(sampled_records) < n_samples:
        remaining = n_samples - len(sampled_records)
        available = df[~df.index.isin([r.index[0] for r in sampled_records])]
        if len(available) >= remaining:
            extra = available.sample(n=remaining, random_state=random_seed)
            sampled_records.append(extra)
        else:
            sampled_records.append(available)

    return pd.concat(sampled_records).iloc[:n_samples]


def extract_signal(
    data_dir: Path,
    filename: str,
    lead_index: int = 1
) -> Tuple[np.ndarray, int, str]:
    """
    Extract ECG signal from PTB-XL record.

    Parameters:
    -----------
    data_dir : Path
        Path to PTB-XL directory
    filename : str
        Relative path to record (e.g., 'records100/00000/00001_lr')
    lead_index : int
        Which lead to extract (default: 1 = Lead II)

    Returns:
    --------
    Tuple of (signal, sampling_freq, lead_name)
    """
    record_path = str(data_dir / filename)
    record = wfdb.rdrecord(record_path)

    signal = record.p_signal[:, lead_index].astype(np.float32)
    fs = record.fs
    lead_name = record.sig_name[lead_index]

    return signal, fs, lead_name


def extract_all_segments(
    data_dir: Path,
    sampled_df: pd.DataFrame,
    config: SegmentConfig,
    verbose: bool = True
) -> List[ExtractedSegment]:
    """
    Extract all sampled segments.

    Parameters:
    -----------
    data_dir : Path
        Path to PTB-XL directory
    sampled_df : pd.DataFrame
        Sampled records with assigned lead times
    config : SegmentConfig
        Extraction configuration
    verbose : bool
        Print progress

    Returns:
    --------
    List of ExtractedSegment objects
    """
    extracted = []
    errors = []

    # Choose filename column based on target frequency
    if config.target_fs == 100:
        filename_col = 'filename_lr'
    else:
        filename_col = 'filename_hr'

    iterator = tqdm(sampled_df.iterrows(), total=len(sampled_df),
                    desc="Extracting signals") if verbose else sampled_df.iterrows()

    for ecg_id, row in iterator:
        try:
            signal, fs, lead_name = extract_signal(
                data_dir,
                row[filename_col],
                config.lead_index
            )

            # Verify signal length
            expected_samples = config.segment_duration_sec * fs
            if len(signal) != expected_samples:
                # Pad or trim
                if len(signal) > expected_samples:
                    signal = signal[:expected_samples]
                else:
                    signal = np.pad(signal, (0, expected_samples - len(signal)), mode='edge')

            metadata = NormalSegment(
                segment_id=f"ptbxl_{ecg_id}_l{config.lead_index}",
                patient_id=f"ptbxl_{int(row['patient_id'])}",
                source_database="PTB-XL",
                ecg_id=ecg_id,
                lead_time_min=-1,  # Normal segments have no arrhythmia event
                segment_duration_sec=config.segment_duration_sec,
                original_fs=fs,
                lead_index=config.lead_index,
                lead_name=lead_name,
                norm_confidence=get_norm_confidence(row['scp_codes']),
                age=row['age'] if pd.notna(row['age']) else None,
                sex=int(row['sex']) if pd.notna(row['sex']) else None,
                strat_fold=int(row['strat_fold']),
                label=0  # Normal class
            )

            extracted.append(ExtractedSegment(signal=signal, metadata=metadata))

        except Exception as e:
            errors.append({'ecg_id': ecg_id, 'error': str(e)})

    if verbose and errors:
        print(f"\nEncountered {len(errors)} errors during extraction")
        for err in errors[:5]:
            print(f"  {err}")

    return extracted


def save_to_hdf5(
    extracted: List[ExtractedSegment],
    output_path: Path,
    config: SegmentConfig
) -> None:
    """Save extracted segments to HDF5 format."""
    if not extracted:
        print("No segments to save!")
        return

    n_segments = len(extracted)
    expected_length = config.segment_duration_sec * config.target_fs

    signals = np.zeros((n_segments, expected_length), dtype=np.float32)

    segment_ids = []
    patient_ids = []
    ecg_ids = []
    lead_times = []
    lead_indices = []
    lead_names = []
    labels = []
    original_fs = []
    norm_confidences = []
    ages = []
    sexes = []
    strat_folds = []

    for i, seg in enumerate(extracted):
        sig = seg.signal
        if len(sig) > expected_length:
            sig = sig[:expected_length]
        elif len(sig) < expected_length:
            sig = np.pad(sig, (0, expected_length - len(sig)), mode='edge')
        signals[i, :] = sig

        m = seg.metadata
        segment_ids.append(m.segment_id)
        patient_ids.append(m.patient_id)
        ecg_ids.append(m.ecg_id)
        lead_times.append(m.lead_time_min)
        lead_indices.append(m.lead_index)
        lead_names.append(m.lead_name)
        labels.append(m.label)
        original_fs.append(m.original_fs)
        norm_confidences.append(m.norm_confidence)
        ages.append(m.age if m.age is not None else -1)
        sexes.append(m.sex if m.sex is not None else -1)
        strat_folds.append(m.strat_fold)

    with h5py.File(output_path, 'w') as f:
        f.create_dataset(
            'signals', data=signals, dtype='float32',
            compression='gzip', compression_opts=4
        )

        f.create_dataset('labels', data=np.array(labels, dtype=np.int8))
        f.create_dataset('lead_times', data=np.array(lead_times, dtype=np.int16))
        f.create_dataset('ecg_ids', data=np.array(ecg_ids, dtype=np.int32))
        f.create_dataset('lead_indices', data=np.array(lead_indices, dtype=np.int8))
        f.create_dataset('original_fs', data=np.array(original_fs, dtype=np.int16))
        f.create_dataset('norm_confidences', data=np.array(norm_confidences, dtype=np.float32))
        f.create_dataset('ages', data=np.array(ages, dtype=np.float32))
        f.create_dataset('sexes', data=np.array(sexes, dtype=np.int8))
        f.create_dataset('strat_folds', data=np.array(strat_folds, dtype=np.int8))

        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset('segment_ids', data=segment_ids, dtype=dt_str)
        f.create_dataset('patient_ids', data=patient_ids, dtype=dt_str)
        f.create_dataset('lead_names', data=lead_names, dtype=dt_str)

        f.attrs['n_segments'] = n_segments
        f.attrs['segment_length'] = expected_length
        f.attrs['segment_duration_sec'] = config.segment_duration_sec
        f.attrs['target_fs'] = config.target_fs
        f.attrs['source_database'] = 'PTB-XL'
        f.attrs['unique_patients'] = len(set(patient_ids))

    print(f"Saved {n_segments} segments to {output_path}")
    print(f"  Shape: ({n_segments}, {expected_length})")
    print(f"  Unique patients: {len(set(patient_ids))}")


def save_metadata_csv(extracted: List[ExtractedSegment], output_path: Path) -> None:
    """Save segment metadata to CSV."""
    metadata_records = [asdict(seg.metadata) for seg in extracted]
    df = pd.DataFrame(metadata_records)
    df.to_csv(output_path, index=False)
    print(f"Saved metadata to {output_path}")


def print_summary(extracted: List[ExtractedSegment]) -> None:
    """Print extraction summary."""
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)

    df = pd.DataFrame([asdict(seg.metadata) for seg in extracted])

    print(f"\nTotal segments extracted: {len(df)}")
    print(f"Unique patients: {df['patient_id'].nunique()}")
    print(f"Lead time: {df['lead_time_min'].iloc[0]} (normal rhythm)")

    print(f"\nNORM confidence stats:")
    print(f"  Mean: {df['norm_confidence'].mean():.1f}%")
    print(f"  Min:  {df['norm_confidence'].min():.1f}%")
    print(f"  Max:  {df['norm_confidence'].max():.1f}%")

    if 'age' in df.columns:
        valid_ages = df[df['age'] > 0]['age']
        if len(valid_ages) > 0:
            print(f"\nAge distribution:")
            print(f"  Mean: {valid_ages.mean():.1f} years")
            print(f"  Range: {valid_ages.min():.0f} - {valid_ages.max():.0f} years")


def main(
    data_dir: str,
    output_dir: str,
    n_samples: int = 535,
    target_fs: int = 100,
    lead_index: int = 1,
    min_norm_confidence: float = 50.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Main extraction pipeline for PTB-XL normal segments.

    Parameters:
    -----------
    data_dir : str
        Path to PTB-XL directory
    output_dir : str
        Path to output directory
    n_samples : int
        Number of normal segments to extract
    target_fs : int
        Target sampling frequency (100 or 500)
    lead_index : int
        Which lead to extract (default: 1 = Lead II)
    min_norm_confidence : float
        Minimum NORM confidence score
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame : Segment metadata
    """
    data_path = Path(data_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = SegmentConfig(
        segment_duration_sec=10,
        target_fs=target_fs,
        lead_index=lead_index,
        min_norm_confidence=min_norm_confidence
    )

    print("PTB-XL Normal Segment Extractor")
    print("=" * 70)
    print(f"Data directory: {data_path}")
    print(f"Output directory: {output_path}")
    print(f"Segment duration: {config.segment_duration_sec}s")
    print(f"Target frequency: {config.target_fs} Hz")
    print(f"Lead index: {config.lead_index}")
    print(f"Min NORM confidence: {config.min_norm_confidence}%")
    print(f"Target segments: {n_samples}")
    print("=" * 70)

    # Step 1: Load metadata
    print("\n[1/4] Loading PTB-XL metadata...")
    df = load_ptbxl_metadata(data_path)
    print(f"  Total records: {len(df)}")

    # Step 2: Filter for NORM records
    print("\n[2/4] Filtering for NORM records...")
    norm_df = filter_normal_records(df, config.min_norm_confidence)
    print(f"  NORM records (confidence >= {config.min_norm_confidence}%): {len(norm_df)}")
    print(f"  Unique patients: {norm_df['patient_id'].nunique()}")

    # Step 3: Sample segments
    print("\n[3/4] Sampling segments...")
    sampled_df = sample_segments(norm_df, n_samples, random_seed)
    print(f"  Sampled records: {len(sampled_df)}")

    # Step 4: Extract signals
    print("\n[4/4] Extracting signals...")
    extracted = extract_all_segments(data_path, sampled_df, config)

    # Save outputs
    print("\nSaving outputs...")
    save_to_hdf5(extracted, output_path / 'ptbxl_normal_segments.h5', config)
    save_metadata_csv(extracted, output_path / 'ptbxl_normal_segments_metadata.csv')

    # Save summary
    summary = {
        'total_segments': len(extracted),
        'unique_patients': len(set(seg.metadata.patient_id for seg in extracted)),
        'config': {
            'target_fs': config.target_fs,
            'lead_index': config.lead_index,
            'min_norm_confidence': config.min_norm_confidence
        }
    }
    with open(output_path / 'ptbxl_normal_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print_summary(extracted)

    return sampled_df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Extract normal ECG segments from PTB-XL to balance AF datasets'
    )
    parser.add_argument(
        '--data_dir',
        default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/ptb-xl',
        help='Path to PTB-XL directory'
    )
    parser.add_argument(
        '--output_dir',
        default='/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/ptbxl',
        help='Path to output directory'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=535,
        help='Number of normal segments to extract'
    )
    parser.add_argument(
        '--target_fs',
        type=int,
        default=100,
        choices=[100, 500],
        help='Target sampling frequency (100 or 500 Hz)'
    )
    parser.add_argument(
        '--lead',
        type=int,
        default=1,
        help='Lead index to extract (default: 1 = Lead II)'
    )
    parser.add_argument(
        '--min_confidence',
        type=float,
        default=50.0,
        help='Minimum NORM confidence score (0-100)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility'
    )

    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        n_samples=args.n_samples,
        target_fs=args.target_fs,
        lead_index=args.lead,
        min_norm_confidence=args.min_confidence,
        random_seed=args.seed
    )