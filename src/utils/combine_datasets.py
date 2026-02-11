import h5py
import numpy as np
from scipy.signal import resample
from pathlib import Path


def resample_to_250hz(signals: np.ndarray, original_fs: int) -> np.ndarray:
    """Resample signals to 250 Hz."""
    if original_fs == 250:
        return signals

    target_samples = int(signals.shape[1] * 250 / original_fs)
    resampled = np.zeros((signals.shape[0], target_samples), dtype=np.float32)

    for i in range(signals.shape[0]):
        resampled[i] = resample(signals[i], target_samples)

    return resampled


def load_h5(path: str) -> dict:
    """Load essential data from HDF5 file."""
    with h5py.File(path, 'r') as f:
        data = {
            'signals': f['signals'][:],
            'labels': f['labels'][:],
            'lead_times': f['lead_times'][:],
            'patient_ids': [s.decode() if isinstance(s, bytes) else s for s in f['patient_ids'][:]],
            'segment_ids': [s.decode() if isinstance(s, bytes) else s for s in f['segment_ids'][:]],
            'original_fs': f['original_fs'][:] if 'original_fs' in f else np.full(len(f['signals']), f.attrs.get('target_fs', 250)),
        }
        # Get source database
        if 'source_databases' in f:
            data['source'] = [s.decode() if isinstance(s, bytes) else s for s in f['source_databases'][:]]
        else:
            data['source'] = [f.attrs.get('source_database', 'unknown')] * len(data['signals'])
    return data


def main():
    base_dir = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted')
    output_path = base_dir / 'combined_250hz.h5'

    print("Loading datasets...")

    # Load each dataset
    mitbih = load_h5(base_dir / 'mitbih/mitbih_af_segments.h5')
    ltafdb = load_h5(base_dir / 'ltafdb/ltafdb_af_segments.h5')
    ptbxl = load_h5(base_dir / 'ptbxl/ptbxl_normal_segments.h5')

    print(f"  MIT-BIH: {mitbih['signals'].shape} @ 250 Hz")
    print(f"  LTAFDB:  {ltafdb['signals'].shape} @ 128 Hz")
    print(f"  PTB-XL:  {ptbxl['signals'].shape} @ 100 Hz")

    # Resample to 250 Hz
    print("\nResampling to 250 Hz...")
    mitbih_resampled = mitbih['signals']  # Already 250 Hz
    ltafdb_resampled = resample_to_250hz(ltafdb['signals'], 128)
    ptbxl_resampled = resample_to_250hz(ptbxl['signals'], 100)

    print(f"  MIT-BIH: {mitbih_resampled.shape}")
    print(f"  LTAFDB:  {ltafdb_resampled.shape}")
    print(f"  PTB-XL:  {ptbxl_resampled.shape}")

    # Combine
    print("\nCombining...")
    signals = np.vstack([mitbih_resampled, ltafdb_resampled, ptbxl_resampled])
    labels = np.concatenate([mitbih['labels'], ltafdb['labels'], ptbxl['labels']])
    lead_times = np.concatenate([mitbih['lead_times'], ltafdb['lead_times'], ptbxl['lead_times']])
    patient_ids = mitbih['patient_ids'] + ltafdb['patient_ids'] + ptbxl['patient_ids']
    segment_ids = mitbih['segment_ids'] + ltafdb['segment_ids'] + ptbxl['segment_ids']
    sources = mitbih['source'] + ltafdb['source'] + ptbxl['source']

    print(f"  Combined: {signals.shape}")

    # Save
    print(f"\nSaving to {output_path}...")
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('signals', data=signals, dtype='float32', compression='gzip')
        f.create_dataset('labels', data=labels, dtype='int8')
        f.create_dataset('lead_times', data=lead_times, dtype='int16')

        dt_str = h5py.special_dtype(vlen=str)
        f.create_dataset('patient_ids', data=patient_ids, dtype=dt_str)
        f.create_dataset('segment_ids', data=segment_ids, dtype=dt_str)
        f.create_dataset('sources', data=sources, dtype=dt_str)

        f.attrs['n_segments'] = len(signals)
        f.attrs['sampling_rate'] = 250
        f.attrs['segment_duration_sec'] = 10
        f.attrs['segment_length'] = 2500

    # Summary
    print("\n" + "="*50)
    print("COMBINED DATASET SUMMARY")
    print("="*50)
    print(f"Total segments: {len(signals)}")
    print(f"Signal shape: {signals.shape}")
    print(f"Sampling rate: 250 Hz")
    print(f"\nBy label:")
    print(f"  Pre-AF (1): {np.sum(labels == 1)}")
    print(f"  Normal (0): {np.sum(labels == 0)}")
    print(f"\nBy lead time:")
    for lt in [5, 10, 30, 45]:
        print(f"  {lt} min: {np.sum(lead_times == lt)}")
    print(f"\nUnique patients: {len(set(patient_ids))}")
    print(f"\nSaved to: {output_path}")


if __name__ == '__main__':
    main()