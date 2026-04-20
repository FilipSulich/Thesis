import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import pywt
import wfdb
from scipy import signal as sp_signal
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR        = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/paf')
OUTPUT_DIR      = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8_no_denoise')

FS_ORIG         = 128
FS_TARGET       = 500
SEGMENT_SAMPLES = FS_TARGET * 10
N_SEGMENTS      = 15
UTRP_SIZE       = 256
PAD_SAMPLES     = 8192             # 2^13 — nearest power of 2 above 5000, covers 2^10 SWT

WAVELET         = 'db8'
SUBBAND_LEVELS  = [2, 3, 4, 5, 6, 7, 8]
N_LEVELS_SWT    = 10
EPSILON_FRAC    = 0.10             # ε = 10% of max pairwise distance

class UTRPVisualiser:
    """
    Display a small subset of UTRPs from the generated HDF5 dataset.

    Shows a grid: rows = samples, columns = image keys (single, D2-D8).
    """

    KEYS = ['single'] + [f'D{l}' for l in SUBBAND_LEVELS]

    def __init__(self, dataset_dir: Path = OUTPUT_DIR):
        self.dataset_dir = Path(dataset_dir)

    def _load_split(self, split: str) -> Tuple[Dict[str, np.ndarray],
                                                np.ndarray, np.ndarray]:
        """Load all image arrays and metadata for a split."""
        path = self.dataset_dir / f'{split}.h5'
        if not path.exists():
            raise FileNotFoundError(f"No HDF5 found at {path}")

        images = {}
        with h5py.File(path, 'r') as f:
            for key in self.KEYS:
                if key in f:
                    images[key] = f[key][:]          # (N, 3, H, W)
            labels = f['metadata/labels'][:]
            rids   = np.array([r.decode() for r in f['metadata/record_ids'][:]])
        return images, labels, rids

    def show_grid(self,
                  split: str = 'train',
                  n_samples: int = 6,
                  keys: Optional[List[str]] = None,
                  seed: int = 42,
                  save_path: Optional[Path] = None) -> None:
        """
        Display a grid of UTRP images.

        Rows = randomly selected samples (balanced between classes).
        Columns = image keys (single-scale + subbands).

        Args:
            split:     'train', 'val', or 'test'
            n_samples: total samples to show (split equally between NSR/PreAF)
            keys:      which image types to show (default: all)
            seed:      random seed for sample selection
            save_path: if provided, save figure instead of showing
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches

        images, labels, rids = self._load_split(split)
        keys = keys or self.KEYS

        # Balanced sample selection
        rng      = np.random.default_rng(seed)
        n_each   = n_samples // 2
        nsr_idx  = np.where(labels == 0)[0]
        paf_idx  = np.where(labels == 1)[0]
        sel_nsr  = rng.choice(nsr_idx, min(n_each, len(nsr_idx)), replace=False)
        sel_paf  = rng.choice(paf_idx, min(n_each, len(paf_idx)), replace=False)
        sel_idx  = np.concatenate([sel_nsr, sel_paf])
        sel_labels = labels[sel_idx]
        sel_rids   = rids[sel_idx]

        n_rows = len(sel_idx)
        n_cols = len(keys)

        fig, axes = plt.subplots(n_rows, n_cols,
                                  figsize=(n_cols * 2.2, n_rows * 2.2))
        if n_rows == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        # Column headers
        for col, key in enumerate(keys):
            axes[0, col].set_title(key, fontsize=10, fontweight='bold', pad=6)

        # Row labels + images
        for row, (idx, label, rid) in enumerate(zip(sel_idx, sel_labels, sel_rids)):
            class_name = 'Pre-AF' if label == 1 else 'NSR'
            colour     = '#d62728' if label == 1 else '#1f77b4'

            # Row label on left
            axes[row, 0].set_ylabel(
                f'{rid}\n{class_name}',
                fontsize=8, color=colour, rotation=0,
                labelpad=60, va='center'
            )

            for col, key in enumerate(keys):
                ax  = axes[row, col]
                img = images[key][idx]           # (3, H, W)
                ax.imshow(img.transpose(1, 2, 0))  # (H, W, 3)
                ax.set_xticks([]); ax.set_yticks([])

                # Coloured border per class
                for spine in ax.spines.values():
                    spine.set_edgecolor(colour)
                    spine.set_linewidth(2)

        # Legend
        patches = [
            mpatches.Patch(color='#1f77b4', label='NSR'),
            mpatches.Patch(color='#d62728', label='Pre-AF'),
        ]
        fig.legend(handles=patches, loc='lower center',
                   ncol=2, fontsize=10, frameon=True,
                   bbox_to_anchor=(0.5, -0.01))

        fig.suptitle(
            f'UTRP Sample Grid — {split.capitalize()} split\n'
            f'Wavelet: {WAVELET} ',
            fontsize=11, y=1.01
        )
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved → {save_path}')
        else:
            plt.show()
        plt.close()

    def show_single_sample(self, split: str = 'train',
                            idx: int = 0,
                            save_path: Optional[Path] = None) -> None:
        """
        Show all image types for a single sample side by side.
        Useful for sanity-checking a specific record/segment.
        """
        import matplotlib.pyplot as plt

        images, labels, rids = self._load_split(split)
        label     = labels[idx]
        rid       = rids[idx]
        class_name = 'Pre-AF' if label == 1 else 'NSR'

        n_keys = len(self.KEYS)
        fig, axes = plt.subplots(1, n_keys, figsize=(n_keys * 2.5, 3))

        for ax, key in zip(axes, self.KEYS):
            img = images[key][idx].transpose(1, 2, 0)
            ax.imshow(img)
            ax.set_title(key, fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])

        fig.suptitle(f'{rid}  [{class_name}]  — idx={idx}', fontsize=11)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f'Saved → {save_path}')
        else:
            plt.show()
        plt.close()

    def class_distribution(self) -> None:
        """Print class distribution across all splits."""
        for split in ('train', 'val', 'test'):
            path = self.dataset_dir / f'{split}.h5'
            if not path.exists():
                continue
            with h5py.File(path, 'r') as f:
                labels = f['metadata/labels'][:]
            n_nsr = int((labels == 0).sum())
            n_paf = int((labels == 1).sum())
            print(f'{split:5s}: {len(labels):5d} segments  '
                  f'NSR={n_nsr:4d}  Pre-AF={n_paf:4d}  '
                  f'ratio={n_nsr/max(1,n_paf):.2f}:1')