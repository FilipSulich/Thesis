"""
MIT-BIH Lead-Time-Based RQA Pipeline

Patient-wise analysis of RQA metrics at specific lead times before AF onset.
For each patient at each lead time, compares pre-AF segments with matched
normal segments from the same patient (far from any AF episode).

Usage:
    pipeline = MITBIHPipeline(lead_time=5)
    pipeline.prepare_dataset()
    pipeline.export_to_csv()
    pipeline.export_subbands_to_csv()
"""

from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
import csv
import pandas as pd
import wfdb

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_cleaning.data_cleaner import DataCleaner
from wavelets import stationary_wavelet_transform as stationary_wavelet

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR        = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih')
RAW_DATA_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/MIT')
HDF5_PATH       = DATA_DIR / 'mitbih_af_segments.h5'
OUTPUT_DIR      = DATA_DIR / 'results'          # renamed from results_min_max

SAMPLING_FREQ         = 250   # MIT-BIH AFDB sampling frequency
SEGMENT_DURATION_SEC  = 10    # 10-second segments
RANDOM_SEED           = 42    # for reproducible normal segment selection


# =============================================================================
# Data container
# =============================================================================

@dataclass
class SegmentInfo:
    segment_id: str
    patient_id: str
    label: int          # 0=normal, 1=pre-AF
    lead_time: int      # minutes before AF onset (0 for normal segments)
    signal: np.ndarray

    def to_dict(self, include_signal: bool = False) -> Dict:
        d = {
            'segment_id': self.segment_id,
            'patient_id': self.patient_id,
            'label': self.label,
            'lead_time': self.lead_time,
        }
        if include_signal:
            for i, val in enumerate(self.signal):
                d[f's{i}'] = val
        return d


# =============================================================================
# Pipeline
# =============================================================================

class MITBIHPipeline:
    def __init__(
        self,
        lead_time: int = 5,
        normal_buffer_min: int = 60,
        apply_cleaning: bool = True,
        random_seed: int = RANDOM_SEED,
    ):
        """
        Parameters
        ----------
        lead_time : int
            Minutes before AF onset to extract pre-AF segment.
        normal_buffer_min : int
            Minimum distance in minutes from any AF episode for a normal segment.
        apply_cleaning : bool
            Whether to apply DataCleaner preprocessing.
        random_seed : int
            Seed for reproducible normal segment selection.
        """
        self.lead_time        = lead_time
        self.normal_buffer_min = normal_buffer_min
        self.apply_cleaning   = apply_cleaning
        self.rng              = np.random.default_rng(random_seed)

        self.output_dir = OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # DataCleaner must be instantiated with correct fs
        self.cleaner = DataCleaner(fs=SAMPLING_FREQ) if apply_cleaning else None

        self.all_segments: List[SegmentInfo] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _clean_signal(self, signal: np.ndarray) -> Optional[np.ndarray]:
        """Run full preprocessing pipeline. Returns None if quality check fails."""
        if self.cleaner is None:
            return signal
        cleaned, info = self.cleaner.preprocess(signal, check_quality=True)
        return cleaned  # None if rejected

    # ------------------------------------------------------------------
    # Segment loading
    # ------------------------------------------------------------------

    def _load_preaf_segments(self) -> List[SegmentInfo]:
        """Load pre-AF segments at self.lead_time from the HDF5 file."""
        with h5py.File(HDF5_PATH, 'r') as f:
            signals     = f['signals'][:]
            labels      = f['labels'][:]
            segment_ids = [s.decode() if isinstance(s, bytes) else s for s in f['segment_ids'][:]]
            patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f['patient_ids'][:]]
            lead_times  = f['lead_times'][:]

        preaf_segments = []
        rejected = 0

        for i in tqdm(range(len(signals)), desc="Loading pre-AF segments"):
            if labels[i] != 1 or lead_times[i] != self.lead_time:
                continue

            signal = signals[i].astype(np.float32)

            if self.apply_cleaning:
                signal = self._clean_signal(signal)
                if signal is None:
                    rejected += 1
                    continue

            preaf_segments.append(SegmentInfo(
                segment_id=segment_ids[i],
                patient_id=patient_ids[i],
                label=1,
                lead_time=self.lead_time,
                signal=signal,
            ))

        print(f"  Loaded:   {len(preaf_segments)} pre-AF segments")
        print(f"  Rejected: {rejected} segments (failed quality check)")
        return preaf_segments

    def _extract_normal_segments(
        self,
        patient_id: str,
        n_segments: int,
    ) -> List[SegmentInfo]:
        """
        Extract n_segments normal segments from a patient's record.

        Normal segments must be at least normal_buffer_min minutes from
        any AF episode. Segments are spread across all available normal
        periods rather than always taking the earliest ones.

        Parameters
        ----------
        patient_id : str
            Patient identifier (e.g. 'mitbih_05091').
        n_segments : int
            Number of segments to extract.

        Returns
        -------
        List[SegmentInfo]
        """
        record_id = patient_id.replace('mitbih_', '')

        try:
            record_path = str(RAW_DATA_DIR / record_id)
            record      = wfdb.rdrecord(record_path)
            annotation  = wfdb.rdann(record_path, 'atr')

            # Collect all AF onset sample positions
            af_samples = [
                annotation.sample[i]
                for i, note in enumerate(annotation.aux_note)
                if note and ('AFIB' in note or 'AFL' in note)
            ]

            # Collect normal rhythm periods sufficiently far from AF
            segment_length = int(SEGMENT_DURATION_SEC * SAMPLING_FREQ)
            candidate_periods = []

            for i, note in enumerate(annotation.aux_note):
                if not (note and note.strip('\x00') == '(N'):
                    continue

                start_sample = annotation.sample[i]
                end_sample   = (annotation.sample[i + 1]
                                if i + 1 < len(annotation.sample)
                                else len(record.p_signal))

                # Check distance from all AF episodes
                if af_samples:
                    min_distance_samples = min(abs(start_sample - af) for af in af_samples)
                    distance_min = min_distance_samples / (SAMPLING_FREQ * 60)
                else:
                    distance_min = float('inf')

                if distance_min <= self.normal_buffer_min:
                    continue

                period_length = end_sample - start_sample
                if period_length < segment_length:
                    continue

                candidate_periods.append((start_sample, end_sample))

            if not candidate_periods:
                print(f"  {patient_id}: No valid normal periods found")
                return []

            # Spread extraction evenly across candidate periods
            # Each period contributes at most ceil(n_segments / n_periods) segments
            extracted_segments = []
            rejected = 0
            period_indices = self.rng.permutation(len(candidate_periods))

            for period_idx in period_indices:
                if len(extracted_segments) >= n_segments:
                    break

                start, end = candidate_periods[period_idx]
                period_length = end - start

                # Take one segment from the center of each period
                # (deterministic given seed — avoids bias toward period start)
                center = start + period_length // 2
                seg_start = max(start, center - segment_length // 2)
                seg_start = min(seg_start, end - segment_length)
                signal = record.p_signal[seg_start:seg_start + segment_length, 0].astype(np.float32)

                if self.apply_cleaning:
                    signal = self._clean_signal(signal)
                    if signal is None:
                        rejected += 1
                        continue

                extracted_segments.append(SegmentInfo(
                    segment_id=f"mitbihafdb_{record_id}_normal_{len(extracted_segments)}_l0",
                    patient_id=patient_id,
                    label=0,
                    lead_time=0,
                    signal=signal,
                ))

            if rejected > 0:
                print(f"  {patient_id}: Rejected {rejected} normal segments (quality check)")

            return extracted_segments[:n_segments]

        except Exception as e:
            print(f"  Error loading {patient_id}: {e}")
            return []

    # ------------------------------------------------------------------
    # Dataset preparation
    # ------------------------------------------------------------------

    def prepare_dataset(self) -> Tuple[List[SegmentInfo], List[SegmentInfo]]:
        """
        Build matched pre-AF / normal dataset.

        For each patient, extracts the same number of normal segments as
        pre-AF segments so classes are balanced within each patient.

        Returns
        -------
        all_preaf : List[SegmentInfo]
        all_normal : List[SegmentInfo]
        """
        print(f"\n{'='*60}")
        print(f"Lead time: {self.lead_time} min | Normal buffer: {self.normal_buffer_min} min")
        print(f"{'='*60}\n")

        preaf_segments = self._load_preaf_segments()

        # Group by patient
        patient_preaf: Dict[str, List[SegmentInfo]] = {}
        for seg in preaf_segments:
            patient_preaf.setdefault(seg.patient_id, []).append(seg)

        print(f"\nPatients with pre-AF segments: {len(patient_preaf)}")
        print("\nExtracting matched normal segments...")

        all_preaf:  List[SegmentInfo] = []
        all_normal: List[SegmentInfo] = []

        for patient_id, preaf_segs in patient_preaf.items():
            n_preaf     = len(preaf_segs)
            normal_segs = self._extract_normal_segments(patient_id, n_segments=n_preaf)
            min_count   = min(n_preaf, len(normal_segs))

            if min_count == 0:
                print(f"  {patient_id}: Skipped (no normal segments available)")
                continue

            # Use rng for reproducible subsampling
            preaf_idx  = self.rng.choice(len(preaf_segs),  size=min_count, replace=False)
            normal_idx = self.rng.choice(len(normal_segs), size=min_count, replace=False)

            all_preaf.extend([preaf_segs[i]  for i in preaf_idx])
            all_normal.extend([normal_segs[i] for i in normal_idx])

            print(f"  {patient_id}: {min_count} segments/group")

        self.all_segments = all_preaf + all_normal

        print(f"\nFinal Dataset:")
        print(f"  Pre-AF:   {len(all_preaf)} segments")
        print(f"  Normal:   {len(all_normal)} segments")
        print(f"  Patients: {len(patient_preaf)}")
        print(f"{'='*60}\n")

        return all_preaf, all_normal

    # ------------------------------------------------------------------
    # Export: raw cleaned signals
    # ------------------------------------------------------------------

    def export_to_csv(self, output_path: Optional[Path] = None):
        """Export cleaned raw signals to CSV."""
        if output_path is None:
            output_path = self.output_dir / f'dataset_lt{self.lead_time}.csv'

        if not self.all_segments:
            print("No segments to export. Run prepare_dataset() first.")
            return

        n_samples = len(self.all_segments[0].signal)
        header = ['segment_id', 'patient_id', 'label', 'lead_time'] + [f's{i}' for i in range(n_samples)]

        print(f"Exporting raw signals → {output_path}")
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for seg in tqdm(self.all_segments, desc="Writing"):
                writer.writerow(seg.to_dict(include_signal=True))

        print(f"  {len(self.all_segments)} segments | {n_samples} samples ({n_samples/SAMPLING_FREQ:.1f}s)\n")

    # ------------------------------------------------------------------
    # Export: subband decompositions
    # ------------------------------------------------------------------

    def export_subbands_to_csv(
        self,
        csv_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        wavelet: str = 'db4',
        level: int = 4,
    ):
        """
        Decompose cleaned signals into SWT subbands and save each to CSV.

        Post-decomposition pipeline per subband (via DataCleaner.preprocess_subband):
          1. Trim 0.5s boundary artifacts from each end
          2. Clip outliers at ±3 std (handles artifact spikes)
          3. Z-score normalize

        NOTE: After boundary trimming, subband signals are shorter than the
        raw signal (2000 samples instead of 2500 at 250 Hz for 10s segments).
        This is expected and consistent across all subbands.

        Subbands saved: D1, D2, D3, D4, A4
        Analysis should focus on D3, D4, A4 — D1 and D2 contain near-zero
        energy after the 40 Hz lowpass filter applied in preprocessing.

        Parameters
        ----------
        csv_path : Path or None
            Input CSV (output of export_to_csv). Default: dataset_lt{n}.csv
        output_dir : Path or None
            Output directory for subband CSVs. Default: results/subbands/
        wavelet : str
            Mother wavelet. Default 'db4' (shape matches ECG morphology).
        level : int
            Decomposition level. Default 4 → subbands D1–D4, A4.
        """
        if csv_path is None:
            csv_path = self.output_dir / f'dataset_lt{self.lead_time}.csv'
        if output_dir is None:
            output_dir = self.output_dir / 'subbands'

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.cleaner is None:
            raise RuntimeError("DataCleaner required for subband preprocessing. "
                               "Set apply_cleaning=True.")

        df = pd.read_csv(csv_path)
        signal_cols  = sorted([c for c in df.columns if c.startswith('s') and c[1:].isdigit()],
                               key=lambda x: int(x[1:]))
        metadata_cols = [c for c in df.columns if c not in signal_cols]

        print(f"Loaded {len(df)} segments from {csv_path}")
        print(f"Decomposing with SWT (wavelet={wavelet}, level={level})...")

        swt           = stationary_wavelet.SWT(wavelet=wavelet, level=level)
        subband_names = [f'D{i+1}' for i in range(level)] + [f'A{level}']
        subband_rows  = {sb: [] for sb in subband_names}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Decomposing"):
            signal = row[signal_cols].values.astype(np.float32)
            coeffs = swt.decompose(signal, level=level)
            metadata = {col: row[col] for col in metadata_cols}

            # --- Detail subbands D1 ... D{level} ---
            for i in range(level):
                # Reconstruct subband by zeroing all other coefficients
                zeroed = [(np.zeros_like(c[0]), np.zeros_like(c[1])) for c in coeffs]
                zeroed[i] = (np.zeros_like(coeffs[i][0]), coeffs[i][1])
                subband_raw = swt.reconstruct(zeroed)

                # Apply boundary trim + outlier clip + z-score normalization
                subband = self.cleaner.preprocess_subband(subband_raw)

                sb_row = metadata.copy()
                for j, val in enumerate(subband):
                    sb_row[f's{j}'] = val
                subband_rows[f'D{i+1}'].append(sb_row)

            # --- Approximation subband A{level} ---
            zeroed = [(c[0], np.zeros_like(c[1])) for c in coeffs]
            subband_raw = swt.reconstruct(zeroed)
            subband = self.cleaner.preprocess_subband(subband_raw)

            sb_row = metadata.copy()
            for j, val in enumerate(subband):
                sb_row[f's{j}'] = val
            subband_rows[f'A{level}'].append(sb_row)

        # Save each subband to its own CSV
        for sb_name in subband_names:
            out_path = output_dir / f'mitbih_lt{self.lead_time}_{sb_name}.csv'
            pd.DataFrame(subband_rows[sb_name]).to_csv(out_path, index=False)
            n_cols = len(subband_rows[sb_name][0]) - len(metadata_cols)
            print(f"  Saved {sb_name}: {n_cols} samples/segment → {out_path}")

        print(f"\nNote: Subband signals are {n_cols} samples "
              f"({n_cols/SAMPLING_FREQ:.2f}s) after boundary trimming.\n")


if __name__ == '__main__':
    LEAD_TIME         = 45
    NORMAL_BUFFER_MIN = 60

    pipeline = MITBIHPipeline(
        lead_time=LEAD_TIME,
        normal_buffer_min=NORMAL_BUFFER_MIN,
        apply_cleaning=True,
        random_seed=RANDOM_SEED,
    )

    preaf_segments, normal_segments = pipeline.prepare_dataset()
    pipeline.export_to_csv()
    pipeline.export_subbands_to_csv()