import numpy as np
import torch
import h5py
from pathlib import Path
from torch.utils.data import Dataset


class H5Dataset(Dataset):
    """
    extracts URP data from HDF5 files into raw NumPy matrices for PyTorch consumption
    """

    def __init__(self, h5_path: Path, mode='multi', subband_keys=None):
        """
        initializes the dataset by loading URP data and metadata from HDF5 file

        Args:
            h5_path (Path): path to HDF5 file containing URP data
            mode (str): loading mode, 'single' loads full-band URP or 'multi' loads multiple subbands, defaults to 'multi'
            subband_keys (list): list of subband keys to load, defaults to None which loads D2-D8
        """
        self.h5_path = h5_path
        self.mode = mode
        self.subband_keys = subband_keys

        with h5py.File(h5_path, 'r') as f:
            meta_group = f.get('metadata', f)
            self.labels = meta_group['labels'][:].astype(np.int64)
            self.record_ids = np.array([r.decode() if isinstance(r, bytes) else r
                                       for r in meta_group['record_ids'][:]])
            self.n_samples = len(self.labels)

            if mode == 'single':
                self.data = f['single'][:].astype(np.float32)
            else:  # multi
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
        """
        returns the total number of samples in the dataset

        Returns:
            int: number of URP samples loaded from the HDF5 file
        """
        return self.n_samples

    def __getitem__(self, idx):
        """
        retrieves a single URP sample and its label as PyTorch tensors

        Args:
            idx (int): index of the sample to retrieve

        Returns:
            tuple: (torch.Tensor, int) containing the URP image tensor and classification label
        """
        x = self.data[idx]
        y = int(self.labels[idx])
        return torch.from_numpy(x), y


class CombinedH5Dataset(Dataset):
    """
    combines multiple HDF5 dataset splits into a single unified dataset for cross-validation
    """
    def __init__(self, h5_dir: Path, mode: str = 'multi', subband_keys: list = None):
        """
        initializes the combined dataset by loading train, validation, and test splits

        Args:
            h5_dir (Path): directory containing train.h5, val.h5, and test.h5 files
            mode (str): loading mode, either 'single' or 'multi', defaults to 'multi'
            subband_keys (list): optional list of specific subbands to load, defaults to None which loads all available
        """
        self.h5_dir = Path(h5_dir)
        self.mode = mode
        self.subband_keys = subband_keys

        all_labels, all_rids, all_maps = [], [], []
        self.split_datasets = {}

        for split in ('train', 'val', 'test'):
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
        """
        returns the total number of samples across all combined splits

        Returns:
            int: total number of samples in the combined dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        retrieves a sample from the appropriate split using the unified index mapping

        Args:
            idx (int): global index across all splits

        Returns:
            tuple: (torch.Tensor, int) containing the URP image tensor and classification label
        """
        split, local_idx = self.sample_map[idx]
        return self.split_datasets[split][local_idx]
