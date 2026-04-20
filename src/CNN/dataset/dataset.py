from pathlib import Path
from typing import Dict, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class UTRPDataset(Dataset):
    SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]
    KEYS = ['single'] + [f'D{l}' for l in SUBBAND_LEVELS]

    def __init__(self, data_dir: Path, split: str = 'train',
                 mode: str = 'both', augment: bool = False):
        assert split in ('train', 'val', 'test'), f"Invalid split: {split}"
        assert mode in ('single', 'multi', 'both'), f"Invalid mode: {mode}"

        self.data_dir  = Path(data_dir)
        self.split     = split
        self.mode      = mode
        self.augment   = augment and (split == 'train')  # never augment val/test
        self.h5_path   = self.data_dir / f'{split}.h5'

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        with h5py.File(self.h5_path, 'r') as f:
            meta_group = f.get('metadata', f)
            self.labels          = meta_group['labels'][:].astype(np.int64)
            self.record_ids      = np.array([r.decode() for r in meta_group['record_ids'][:]])
            self.segment_indices = meta_group['segment_indices'][:]
            self.available_keys  = [k for k in self.KEYS if k in f]

        self.n_samples = len(self.labels)
        n_preaf = int(np.sum(self.labels == 1))
        n_nsr   = int(np.sum(self.labels == 0))
        aug_str = ' +aug' if self.augment else ''
        print(f"UTRPDataset [{mode}-{split}{aug_str}]: {self.n_samples} samples "
              f"(NSR={n_nsr}, Pre-AF={n_preaf})")

    # ── Augmentation ───────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """
        Minimal augmentation for recurrence plots.

        Valid operations:
          - Transpose along main diagonal: equivalent to time-reversal of the
            signal. The RP of a reversed signal is the transpose of the original
            RP, so this is an exact symmetry — all RQA measures are preserved.
          - Low-amplitude Gaussian noise: adds small perturbations to the
            distance values, simulating measurement noise. Kept very small
            (σ=0.01) to avoid distorting the recurrence structure.

        NOT applied:
          - Random crops / zooms: would distort the time axis non-uniformly,
            breaking the correspondence between distance and trajectory position.
          - Horizontal/vertical flips: equivalent to transpose, redundant.
          - Colour jitter: the JET colourmap encodes distance monotonically;
            shifting colours would corrupt the distance information.
          - Rotations other than 90°: RPs are symmetric along the main diagonal
            only; arbitrary rotations destroy that symmetry.

        Args:
            img: (C, H, W) float32 in [0, 1]

        Returns:
            Augmented (C, H, W) float32 in [0, 1]
        """
        # Transpose (time reversal) — 50% probability
        # Swap spatial axes H and W; channel axis unchanged
        if np.random.rand() < 0.5:
            img = img.transpose(0, 2, 1).copy()

        # Gaussian noise — 30% probability
        if np.random.rand() < 0.3:
            noise = np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
            img   = np.clip(img + noise, 0.0, 1.0)

        return img

    def _augment_consistent(self, imgs: list) -> list:
        """
        Apply the SAME augmentation to a list of images.

        For multi/both mode, all subbands and the single-scale image must
        receive identical spatial transforms so the time axis is consistent
        across channels. Noise is applied independently per image.

        Args:
            imgs: list of (C, H, W) float32 arrays

        Returns:
            List of augmented arrays with same transforms applied
        """
        do_transpose = np.random.rand() < 0.5
        do_noise     = np.random.rand() < 0.3

        out = []
        for img in imgs:
            if do_transpose:
                img = img.transpose(0, 2, 1).copy()
            if do_noise:
                noise = np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
                img   = np.clip(img + noise, 0.0, 1.0)
            out.append(img)
        return out

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, int],
                                              Tuple[torch.Tensor, torch.Tensor, int]]:
        label = int(self.labels[idx])

        with h5py.File(self.h5_path, 'r') as f:

            if self.mode == 'single':
                single = f['single'][idx].astype(np.float32)   # (3, 256, 256)
                if self.augment:
                    single = self._augment(single)
                return torch.from_numpy(single), label

            elif self.mode == 'multi':
                subbands = [f[f'D{l}'][idx].astype(np.float32)
                            for l in self.SUBBAND_LEVELS if f'D{l}' in f]
                if self.augment:
                    subbands = self._augment_consistent(subbands)
                multi = np.concatenate(subbands, axis=0)        # (21, 256, 256)
                return torch.from_numpy(multi), label

            else:  # 'both'
                single   = f['single'][idx].astype(np.float32)
                subbands = [f[f'D{l}'][idx].astype(np.float32)
                            for l in self.SUBBAND_LEVELS if f'D{l}' in f]
                if self.augment:
                    all_imgs = self._augment_consistent([single] + subbands)
                    single   = all_imgs[0]
                    subbands = all_imgs[1:]
                multi = np.concatenate(subbands, axis=0)        # (21, 256, 256)
                return (torch.from_numpy(single),
                        torch.from_numpy(multi),
                        label)

    def get_metadata(self, idx: int) -> Dict:
        return {
            'record_id':     self.record_ids[idx],
            'label':         int(self.labels[idx]),
            'segment_index': int(self.segment_indices[idx]),
        }

    def get_class_distribution(self) -> Dict[int, int]:
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))