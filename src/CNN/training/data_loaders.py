import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from src.CNN.utils.datasets import H5Dataset

SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']

class FullDataset(Dataset):
    """
    combines data from different sets (training, validation and test) into one pool for subject-level 10-fold CV
    """

    def __init__(self, h5_dir: Path, mode: str = 'multi_denoise', subband_keys=None):
        """
        initializes the combined dataset by loading all splits

        Args:
            h5_dir (Path): directory containing train.h5, val.h5, and test.h5 files
            mode (str): loading mode, either 'multi', 'multi_denoise', or 'single', defaults to 'multi_denoise'
            subband_keys (list): list of subband keys to load for multi modes, defaults to None which uses all D2-D8
        """
        self.mode = mode
        self.h5_dir = Path(h5_dir)

        all_labels, all_rids, all_maps = [], [], []
        self.split_datasets = {}

        if subband_keys is None and mode in ('multi', 'multi_denoise'):
            subband_keys = SUBBAND_KEYS

        for split in ('train', 'val', 'test'):
            ds = H5Dataset(
                h5_path=h5_dir / f'{split}.h5',
                mode=mode,
                subband_keys=subband_keys if mode in ('multi', 'multi_denoise') else None
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
        """
        returns the total number of samples across all splits

        Returns:
            int: total number of samples in the combined dataset
        """
        return len(self.labels)

    def __getitem__(self, idx: int):
        """
        retrieves a sample from the appropriate split dataset using the unified index

        Args:
            idx (int): global index across all splits

        Returns:
            tuple: (image_tensor, label) where image_tensor is the URP data and label is the classification target
        """
        split, local_idx = self.sample_map[idx]
        return self.split_datasets[split][local_idx]


class AugmentedSubset(Dataset):
    """
    wrapper that applies data augmentation to a subset of URPs for training set augmentation
    """
    def __init__(self, dataset, indices, augment: bool = False):
        """
        initializes the augmented subset wrapper around a base dataset

        Args:
            dataset (Dataset): base dataset to wrap
            indices (list): list of indices to include in this subset
            augment (bool): whether to apply horizontal flip augmentation, defaults to False
        """
        self.dataset = dataset
        self.indices = indices
        self.augment = augment

    def __len__(self):
        """
        returns the number of samples in this subset

        Returns:
            int: number of indices in the subset
        """
        return len(self.indices)

    def __getitem__(self, idx):
        """
        retrieves a sample from the base dataset with optional horizontal flip augmentation

        Args:
            idx (int): index within the subset

        Returns:
            tuple: (image_tensor, label) with augmentation applied if enabled
        """
        x, y = self.dataset[self.indices[idx]]

        if self.augment and torch.rand(1).item() < 0.5:
            x = torch.flip(x, dims=[-1])

        return x, y
