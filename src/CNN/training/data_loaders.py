import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from src.CNN.utils.datasets import H5Dataset

SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

class FullDataset(Dataset):
    """
    Combines train + val HDF5 files into one pool for patient-level 10-fold CV.
    """
    def __init__(self, h5_dir: Path, mode: str = 'multi_denoise', cache: bool = True):
        self.mode = mode
        self.h5_dir = Path(h5_dir)

        all_labels, all_rids, all_maps = [], [], []
        self.split_datasets = {}

        for split in ('train', 'val'):
            ds = H5Dataset(
                h5_path=h5_dir / f'{split}.h5',
                mode=mode,
                subband_keys=SUBBAND_KEYS if mode == 'multi' else None
            )
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


class AugmentedSubset(Dataset):
    """Wrapper that applies data augmentation to a subset of data"""

    def __init__(self, dataset, indices, augment: bool = False):
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]

        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])

        return x, y
