"""
dataset classes for loading UTRP data from HDF5 files.
"""

import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    """Extracts data into raw NumPy matrices."""

    def __init__(self, h5_path: Path, mode='multi', subband_keys=None):
        """
        Args:
            h5_path: Path to H5 file
            mode: 'single' (loads single UTRP) or 'multi' (loads subbands)
            subband_keys: List of subband keys to load (e.g., ['D2', 'D3', 'D4'])
                         If None and mode='multi', loads all subbands D2-D8
        """
        self.h5_path = h5_path
        self.mode = mode
        self.subband_keys = subband_keys

        with h5py.File(h5_path, 'r') as f:
            # Read metadata
            meta_group = f.get('metadata', f)
            self.labels = meta_group['labels'][:].astype(np.int64)
            self.record_ids = np.array([r.decode() if isinstance(r, bytes) else r
                                       for r in meta_group['record_ids'][:]])
            self.n_samples = len(self.labels)

            if mode == 'single':
                # Load single-scale UTRP: (N_SAMPLES, 3, 256, 256)
                self.data = f['single'][:].astype(np.float32)
            else:  # multi
                # Load specified subbands
                if subband_keys is None:
                    subband_keys = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

                n_subbands = len(subband_keys)
                n_channels = n_subbands * 3

                self.data = np.zeros((self.n_samples, n_channels, 256, 256), dtype=np.float32)

                for ch_idx, key in enumerate(subband_keys):
                    if key in f:
                        start_ch = ch_idx * 3
                        end_ch = start_ch + 3
                        self.data[:, start_ch:end_ch, :, :] = f[key][:]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        x = self.data[idx]
        y = int(self.labels[idx])
        return torch.from_numpy(x), y


class CombinedH5Dataset(Dataset):
    def __init__(self, h5_dir: Path, mode: str = 'multi', subband_keys: list = None):
        """
        args:
            h5_dir: directory containing train.h5, val.h5, test.h5
            mode: 'single' or 'multi'
            subband_keys: optional list of specific subbands to load (e.g., ['D2', 'D3', 'D4'])
        """
        self.h5_dir = Path(h5_dir)
        self.mode = mode
        self.subband_keys = subband_keys

        all_labels, all_rids, all_maps = [], [], []
        self.split_datasets = {}

        for split in ('train', 'val'):
            h5_path = self.h5_dir / f'{split}.h5'
            if not h5_path.exists():
                continue

            ds = H5Dataset(h5_path, mode=mode, subband_keys=subband_keys)
            self.split_datasets[split] = ds
            all_labels.extend(ds.labels.tolist())
            all_rids.extend(ds.record_ids.tolist())
            for i in range(len(ds)):
                all_maps.append((split, i))

        self.labels = np.array(all_labels, dtype=np.int64)
        self.record_ids = np.array(all_rids)
        self.sample_map = all_maps

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        split, local_idx = self.sample_map[idx]
        return self.split_datasets[split][local_idx]
