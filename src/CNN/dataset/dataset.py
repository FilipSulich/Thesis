import json
from pathlib import Path
from typing import List, Optional, Dict

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ZhuDataset(Dataset):
    """
    Simple, clear Dataset for Zhu et al. UTRP replication.

    Args:
        hdf5_path: Path to HDF5 file with binary UTRPs
        split: 'train', 'val', or 'test'
        mode: 'single', 'multi', or 'hybrid'
        augment: Whether to apply augmentation (train only)
    """

    def __init__(self,
                 hdf5_path: Path,
                 split: str = 'train',
                 mode: str = 'single',
                 augment: bool = False,
                 json_path: str = None,
                 # New options for inspection / preprocessing
                 as_grayscale: bool = False,
                 per_sample_standardize: bool = False,
                 per_channel_standardize: bool = False,
                 ):
        assert split in ('train', 'val', 'test')
        assert mode in ('single', 'multi', 'hybrid')

        self.hdf5_path = Path(hdf5_path)
        self.split = split
        self.mode = mode
        self.augment = augment
        # preprocessing flags (do not change defaults to preserve existing behavior)
        self.as_grayscale = as_grayscale
        self.per_sample_standardize = per_sample_standardize
        self.per_channel_standardize = per_channel_standardize

        # Load split JSON
        if json_path is not None:
            split_path = Path(json_path)
        else :
            split_path = self.hdf5_path.with_suffix('.split.json')

        with open(split_path, 'r') as f:
            split_info = json.load(f)

        split_records = set(split_info[split])

        # Load all metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            all_record_ids = [rid.decode('utf-8') for rid in f['metadata/record_ids'][:]]
            all_labels = f['metadata/labels'][:]

        # Filter indices by split
        self.indices = [
            i for i, rid in enumerate(all_record_ids)
            if rid in split_records
        ]

        self.record_ids = [all_record_ids[i] for i in self.indices]
        self.labels = all_labels[self.indices]

        # Cache for UTRPs (loaded on first access)
        self._cache_single = None
        self._cache_multi = None

        # Print statistics
        n_preaf = int((self.labels == 1).sum())
        n_nsr = int((self.labels == 0).sum())
        n_records = len(set(self.record_ids))

        print(f"ZhuDataset [{mode}-{split}{'+ aug' if augment else ''}]: "
              f"{len(self.indices)} segments, "
              f"{n_records} records, "
              f"Pre-AF={n_preaf} NSR={n_nsr}")

    def _load_cache(self):
        """Lazy load UTRPs from HDF5."""
        with h5py.File(self.hdf5_path, 'r') as f:
            if self.mode in ('single', 'hybrid') and self._cache_single is None:
                self._cache_single = f['single_scale/utrps'][self.indices]
            if self.mode in ('multi', 'hybrid') and self._cache_multi is None:
                data = f['multi_scale/utrps'][self.indices]
                # Reshape from (N, n_subbands, 3, H, W) to (N, n_subbands*3, H, W)
                # e.g., (N, 9, 3, 256, 256) -> (N, 27, 256, 256)
                N = data.shape[0]
                n_subbands = data.shape[1]
                H, W = data.shape[-2:]
                self._cache_multi = data.reshape(N, n_subbands * 3, H, W)

    def __len__(self):
        return len(self.indices)

    def distance_matrix_to_rgb(self, dist: np.ndarray) -> np.ndarray:
        """
        Convert raw distance matrix directly to RGB.
        Normalises by actual min/max of this specific matrix — no p95 clipping.
        The full colour range is used regardless of the absolute distance scale.

        Args:
            dist: (N, N) float32 raw pairwise distances
        Returns:
            (3, H, W) float32 RGB image
        """
        d_min = dist.min()
        d_max = dist.max()
        if d_max - d_min < 1e-9:
            return np.zeros((3, dist.shape[0], dist.shape[1]), dtype=np.float32)

        # Normalise to [0, 255] using full range
        d_norm = ((dist - d_min) / (d_max - d_min) * 255).astype(np.uint8)

        # Apply JET colourmap
        bgr = cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)  # (3, H, W)

    # def __getitem__(self, idx: int):
    #     self._load_cache()
    #
    #     label = int(self.labels[idx])
    #
    #     if self.mode == 'single':
    #         # Single-scale: (256, 256) → (3, 256, 256)
    #         utrp_2d = self._cache_single[idx].copy()
    #
    #
    #         # tensor = torch.from_numpy(self.distance_matrix_to_rgb(utrp_2d)).float()
    #         return utrp_2d, label
    #
    #     elif self.mode == 'multi':
    #         utrps = self._cache_multi[idx].copy()  # (5, 256, 256)
    #
    #         if self.augment:
    #             utrps = self._augment_multichannel(utrps)
    #
    #         channels = []
    #         for c in range(utrps.shape[0]):
    #             # dist = self.distance_matrix_to_rgb(utrps[c])  # (3, H, W)
    #             channels.append(utrps[c])
    #
    #         rgb = np.concatenate(channels, axis=0)
    #         return torch.from_numpy(rgb), label
    #
    #     else:  # hybrid
    #         single = self._cache_single[idx].copy()  # (256, 256)
    #         multi = self._cache_multi[idx].copy()    # (5, 256, 256)
    #
    #         # Stack into (6, 256, 256)
    #         combined = np.concatenate([single[np.newaxis, :, :], multi], axis=0)
    #
    #         # Apply augmentation consistently
    #         if self.augment:
    #             combined = self._augment_multichannel(combined)
    #
    #         # Apply JET colormap to each of the 6 RPs
    #
    #         channels = []
    #         for c in range(combined.shape[0]):
    #             channels.append(combined[c])
    #
    #         # Concatenate: 6 × (3, H, W) → (18, H, W)
    #         rgb = np.concatenate(channels, axis=0)
    #         return torch.from_numpy(rgb), label
    def __getitem__(self, idx):
        self._load_cache()
        label = int(self.labels[idx])

        if self.mode == 'single':
            x = self._cache_single[idx].copy()  # (3, 256, 256) float32

            # Optional augmentation (keeps channel-first layout)
            if self.augment:
                x = self._augment_multichannel(x)

            # Optional per-channel standardization
            if self.per_channel_standardize:
                # standardize each channel independently to zero mean, unit std
                for c in range(x.shape[0]):
                    ch = x[c]
                    m = ch.mean()
                    s = ch.std()
                    x[c] = (ch - m) / (s if s > 0 else 1.0)

            # Optional per-sample standardization (across all channels)
            if self.per_sample_standardize:
                m = x.mean()
                s = x.std()
                x = (x - m) / (s if s > 0 else 1.0)

            # Optional grayscale conversion: average across color channels
            if self.as_grayscale:
                # convert (3,H,W) -> (1,H,W) by mean and then replicate to 3 channels
                gray = x.mean(axis=0, keepdims=True)
                # replicate into 3 channels so models expecting 3 channels still work
                x = np.repeat(gray, 3, axis=0)

            return torch.from_numpy(x), label

        elif self.mode == 'multi':
            x = self._cache_multi[idx].copy()  # (n_sub*3, 256, 256) float32
            if self.augment:
                x = self._augment_multichannel(x)

            # For multi, we optionally support per-sample / per-channel standardization
            if self.per_channel_standardize:
                # assume channels as independent images
                for c in range(x.shape[0]):
                    ch = x[c]
                    m = ch.mean()
                    s = ch.std()
                    x[c] = (ch - m) / (s if s > 0 else 1.0)

            if self.per_sample_standardize:
                m = x.mean()
                s = x.std()
                x = (x - m) / (s if s > 0 else 1.0)

            return torch.from_numpy(x), label

        else:  # hybrid
            ss = self._cache_single[idx].copy()  # (3, 256, 256)
            ms = self._cache_multi[idx].copy()  # (n_sub*3, 256, 256)
            if self.augment:
                combined = np.concatenate([ss, ms], axis=0)
                combined = self._augment_multichannel(combined)
                ss = combined[:3]
                ms = combined[3:]
            return torch.from_numpy(ss), torch.from_numpy(ms), label

    @staticmethod
    def _augment(utrp: np.ndarray) -> np.ndarray:
        """
        Augment binary UTRP (H, W).

        Valid augmentations for symmetric distance matrices:
          - Transpose (time reversal)
          - Random crop + resize
          - Gaussian noise

        Args:
            utrp: (H, W) float32 [0, 1]

        Returns:
            Augmented (H, W) float32 [0, 1]
        """
        h = utrp.shape[0]

        # Transpose — time reversal, preserves all RQA measures
        if np.random.rand() > 0.5:
            utrp = utrp.T.copy()

        # Random crop + resize (75-100% scale)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.75, 1.0)
            crop = int(h * scale)
            x = np.random.randint(0, h - crop + 1)
            y = np.random.randint(0, h - crop + 1)
            utrp = cv2.resize(utrp[x:x + crop, y:y + crop], (h, h),
                              interpolation=cv2.INTER_LINEAR)
            np.fill_diagonal(utrp, 0.0)

        # Gaussian noise
        if np.random.rand() > 0.7:
            utrp = np.clip(utrp + np.random.normal(0, 0.01, utrp.shape)
                           .astype(np.float32), 0., 1.)

        return utrp



    @staticmethod
    def _augment_multichannel(utrps: np.ndarray) -> np.ndarray:
        """
        Augment multi-channel UTRPs (C, H, W).
        Same augmentation applied consistently across all channels.

        Args:
            utrps: (C, H, W) float32 [0, 1]

        Returns:
            Augmented (C, H, W) float32 [0, 1]
        """
        h = utrps.shape[-1]

        # Decide augmentations once
        do_transpose = np.random.rand() > 0.5
        do_crop = np.random.rand() > 0.5
        do_noise = np.random.rand() > 0.7

        if do_crop:
            scale = np.random.uniform(0.75, 1.0)
            crop = int(h * scale)
            x = np.random.randint(0, h - crop + 1)
            y = np.random.randint(0, h - crop + 1)

        # Apply same augmentation to all channels
        channels = []
        for c in range(utrps.shape[0]):
            ch = utrps[c].copy()

            if do_transpose:
                ch = ch.T.copy()

            if do_crop:
                ch = cv2.resize(ch[x:x + crop, y:y + crop], (h, h),
                                interpolation=cv2.INTER_LINEAR)
                np.fill_diagonal(ch, 0.0)

            if do_noise:
                ch = np.clip(ch + np.random.normal(0, 0.01, ch.shape)
                             .astype(np.float32), 0., 1.)

            channels.append(ch)

        return np.stack(channels, axis=0)

    def get_class_distribution(self) -> Dict[int, int]:
        """Return class distribution for this split."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
