"""
H5ImageExtractor - Extract images from HDF5 files and save to folder structure
Supports parallel extraction for faster processing.
"""

import json
import os
import sys
from pathlib import Path
from typing import Tuple
from multiprocessing import Pool, cpu_count, get_context

import h5py
import numpy as np
from tqdm import tqdm

# Set multiprocessing start method for macOS compatibility
# This prevents "RuntimeError: spawn not allowed" errors
if sys.platform == 'darwin':  # macOS
    try:
        if 'fork' in get_context().get_start_method():
            import multiprocessing

            multiprocessing.set_start_method('fork', force=True)
    except RuntimeError:
        pass  # Already set


# ── Worker functions for parallel extraction ──────────────────────────────────

def _extract_single_image(args):
    """Worker function for parallel extraction of single-scale images."""
    h5_path, idx, output_dir = args

    with h5py.File(h5_path, 'r') as f:
        if 'single' in f:
            img = f['single'][idx].astype(np.float32)
            np.save(output_dir / f'{idx}.npy', img)
            return (idx, True, None)

    return (idx, False, "single key not found")


def _extract_multi_images(args):
    """Worker function for parallel extraction of multi-scale images."""
    h5_path, idx, output_dir, subband_levels = args

    with h5py.File(h5_path, 'r') as f:
        subbands = []
        for level in subband_levels:
            key = f'D{level}'
            if key in f:
                subbands.append(f[key][idx].astype(np.float32))

        if subbands:
            multi = np.concatenate(subbands, axis=0)
            np.save(output_dir / f'{idx}.npy', multi)
            return (idx, True, None)

    return (idx, False, "No subbands found")


class H5ImageExtractor:
    """
    Extract images from HDF5 dataset files and save them to disk for faster loading.

    Supports parallel extraction for significant speedup on multi-core systems.

    Folder structure created:
        output_dir/
        ├── split1/
        │   ├── single/
        │   │   ├── 0.npy
        │   │   ├── 1.npy
        │   │   └── ...
        │   └── multi/
        │       ├── 0.npy
        │       ├── 1.npy
        │       └── ...
        ├── split2/
        └── metadata.json
    """

    SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]  # D2-D8

    def __init__(self, h5_dir: Path, output_dir: Path, parallel: bool = True, num_workers: int = None):
        """
        Initialize extractor.

        Args:
            h5_dir: Directory containing train.h5, val.h5, test.h5 files
            output_dir: Directory where extracted images will be saved
            parallel: Whether to use parallel processing (default: True)
            num_workers: Number of worker processes (default: number of CPU cores)
        """
        self.h5_dir = Path(h5_dir)
        self.output_dir = Path(output_dir)
        self.parallel = parallel
        self.num_workers = num_workers or cpu_count()

        if not self.h5_dir.exists():
            raise FileNotFoundError(f"HDF5 directory not found: {self.h5_dir}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}

        if self.parallel:
            print(f"✓ Parallel extraction enabled ({self.num_workers} workers)")
        else:
            print(f"✓ Sequential extraction mode")

    def extract_all(self, splits: list = None) -> dict:
        """
        Extract all images from specified splits.

        Args:
            splits: List of split names (e.g., ['train', 'val', 'test']).
                   If None, extracts from all available.

        Returns:
            Dictionary with extraction summary
        """
        if splits is None:
            splits = ['train', 'val', 'test']

        summary = {}

        for split in splits:
            h5_path = self.h5_dir / f'{split}.h5'

            if not h5_path.exists():
                print(f"⚠ Skipping {split}: file not found ({h5_path})")
                continue

            print(f"\n{'=' * 70}")
            print(f"Extracting {split.upper()} split")
            print(f"{'=' * 70}")

            result = self._extract_split(h5_path, split)
            summary[split] = result

        # Save metadata
        self._save_metadata(summary)

        return summary

    def _extract_split(self, h5_path: Path, split: str) -> dict:
        """
        Extract all images from a single split.

        Args:
            h5_path: Path to .h5 file
            split: Name of split (train/val/test)

        Returns:
            Extraction summary for this split
        """
        split_dir = self.output_dir / split
        single_dir = split_dir / 'single'
        multi_dir = split_dir / 'multi'

        single_dir.mkdir(parents=True, exist_ok=True)
        multi_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, 'r') as f:
            # Get metadata
            meta_group = f.get('metadata', f)
            labels = meta_group['labels'][:]
            record_ids = np_array_to_list(meta_group['record_ids'][:])
            segment_indices = meta_group.get('segment_indices', np.arange(len(labels)))[:]

            n_samples = len(labels)

            # Create metadata for this split
            split_metadata = {
                'split': split,
                'n_samples': n_samples,
                'n_nsr': int(np.sum(labels == 0)),
                'n_preaf': int(np.sum(labels == 1)),
                'record_ids': record_ids.tolist() if isinstance(record_ids, np.ndarray) else record_ids,
                'labels': labels.tolist(),
                'segment_indices': segment_indices.tolist(),
                'files': {
                    'single': [f'{i}.npy' for i in range(n_samples)],
                    'multi': [f'{i}.npy' for i in range(n_samples)],
                }
            }

            # Get available subbands
            available_subbands = [k for k in self.SUBBAND_LEVELS if f'D{k}' in f]

            # Extract single-scale images
            print(f"\n  Extracting single-scale images ({n_samples} samples)...")
            if 'single' in f:
                if self.parallel:
                    self._extract_parallel(
                        _extract_single_image,
                        [(h5_path, i, single_dir) for i in range(n_samples)],
                        desc='Single'
                    )
                else:
                    self._extract_sequential(
                        _extract_single_image,
                        [(h5_path, i, single_dir) for i in range(n_samples)],
                        desc='Single'
                    )
            else:
                print(f"    ⚠ 'single' key not found in {split}.h5")

            # Extract multi-scale images (concatenate all subbands)
            print(f"  Extracting multi-scale images ({n_samples} samples, {len(available_subbands)} subbands)...")
            if available_subbands:
                if self.parallel:
                    self._extract_parallel(
                        _extract_multi_images,
                        [(h5_path, i, multi_dir, available_subbands) for i in range(n_samples)],
                        desc='Multi'
                    )
                else:
                    self._extract_sequential(
                        _extract_multi_images,
                        [(h5_path, i, multi_dir, available_subbands) for i in range(n_samples)],
                        desc='Multi'
                    )
                split_metadata['n_subbands'] = len(available_subbands)
                split_metadata['subband_levels'] = available_subbands
            else:
                print(f"    ⚠ No subband keys found in {split}.h5")

            print(
                f"  ✓ Extracted {n_samples} samples ({split_metadata['n_nsr']} NSR, {split_metadata['n_preaf']} Pre-AF)")

            return split_metadata

    def _extract_parallel(self, worker_fn, tasks: list, desc: str = 'Extracting') -> None:
        """
        Extract images in parallel using multiprocessing.

        Args:
            worker_fn: Worker function to apply to each task
            tasks: List of arguments to pass to worker_fn
            desc: Description for progress bar
        """
        with Pool(processes=self.num_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_fn, tasks, chunksize=32),
                total=len(tasks),
                desc=desc,
                leave=False
            ))

        # Check for failures
        failures = [r for r in results if not r[1]]
        if failures:
            print(f"    ⚠ {len(failures)} samples failed to extract")

    def _extract_sequential(self, worker_fn, tasks: list, desc: str = 'Extracting') -> None:
        """
        Extract images sequentially (fallback).

        Args:
            worker_fn: Worker function to apply to each task
            tasks: List of arguments to pass to worker_fn
            desc: Description for progress bar
        """
        for task in tqdm(tasks, desc=desc, leave=False):
            worker_fn(task)

    def _save_metadata(self, summary: dict) -> None:
        """Save extraction metadata."""
        metadata_path = self.output_dir / 'metadata.json'

        # Convert numpy arrays to lists for JSON serialization
        metadata = {
            'extraction_info': {
                'source_h5_dir': str(self.h5_dir),
                'output_dir': str(self.output_dir),
            },
            'splits': summary,
        }

        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"\n✓ Metadata saved to {metadata_path}")


class FolderBasedUTRPDataset:
    """
    Load UTRP images directly from folder structure (created by H5ImageExtractor).

    Mimics the interface of UTRPDataset but loads from numpy files instead of HDF5.
    """

    SUBBAND_LEVELS = [2, 3, 4, 5, 6, 7, 8]

    def __init__(self, data_dir: Path, split: str = 'train', mode: str = 'multi', augment: bool = False):
        """
        Initialize folder-based dataset.

        Args:
            data_dir: Root directory with extracted images (from H5ImageExtractor)
            split: 'train', 'val', or 'test'
            mode: 'single', 'multi', or 'both'
            augment: Whether to apply augmentation (train only)
        """
        assert split in ('train', 'val', 'test'), f"Invalid split: {split}"
        assert mode in ('single', 'multi', 'both'), f"Invalid mode: {mode}"

        self.data_dir = Path(data_dir)
        self.split = split
        self.mode = mode
        self.augment = augment and (split == 'train')

        # Load metadata
        metadata_path = self.data_dir / 'metadata.json'
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        if split not in self.metadata['splits']:
            raise ValueError(f"Split '{split}' not found in metadata")

        split_meta = self.metadata['splits'][split]
        self.labels = np.array(split_meta['labels'], dtype=np.int64)
        self.record_ids = np.array(split_meta['record_ids'])
        self.segment_indices = np.array(split_meta['segment_indices'])

        self.single_dir = self.data_dir / split / 'single'
        self.multi_dir = self.data_dir / split / 'multi'

        # Verify directories exist
        if self.mode in ('single', 'both') and not self.single_dir.exists():
            raise FileNotFoundError(f"Single-scale directory not found: {self.single_dir}")
        if self.mode in ('multi', 'both') and not self.multi_dir.exists():
            raise FileNotFoundError(f"Multi-scale directory not found: {self.multi_dir}")

        self.n_samples = len(self.labels)
        n_preaf = int(np.sum(self.labels == 1))
        n_nsr = self.n_samples - n_preaf
        aug_str = ' +aug' if self.augment else ''

        print(f"FolderBasedUTRPDataset [{mode}-{split}{aug_str}]: {self.n_samples} samples "
              f"(NSR={n_nsr}, Pre-AF={n_preaf})")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int):
        """Load single or both single/multi scale images."""
        import torch

        label = int(self.labels[idx])

        if self.mode == 'single':
            single = np.load(self.single_dir / f'{idx}.npy').astype(np.float32)
            if self.augment:
                single = self._augment(single)
            return torch.from_numpy(single), label

        elif self.mode == 'multi':
            multi = np.load(self.multi_dir / f'{idx}.npy').astype(np.float32)
            if self.augment:
                multi = self._augment(multi)
            return torch.from_numpy(multi), label

        else:  # 'both'
            single = np.load(self.single_dir / f'{idx}.npy').astype(np.float32)
            multi = np.load(self.multi_dir / f'{idx}.npy').astype(np.float32)
            if self.augment:
                single, multi = self._augment_consistent([single, multi])
            return (torch.from_numpy(single), torch.from_numpy(multi), label)

    def _augment(self, img: np.ndarray) -> np.ndarray:
        """Apply single augmentation."""
        # Transpose (time reversal) — 50% probability
        if np.random.rand() < 0.5:
            img = img.transpose(0, 2, 1).copy()

        # Gaussian noise — 30% probability
        if np.random.rand() < 0.3:
            noise = np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
            img = np.clip(img + noise, 0.0, 1.0)

        return img

    def _augment_consistent(self, imgs: list) -> list:
        """Apply same augmentation to multiple images."""
        do_transpose = np.random.rand() < 0.5
        do_noise = np.random.rand() < 0.3

        out = []
        for img in imgs:
            if do_transpose:
                img = img.transpose(0, 2, 1).copy()
            if do_noise:
                noise = np.random.normal(0.0, 0.01, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0.0, 1.0)
            out.append(img)
        return out

    def get_class_distribution(self) -> dict:
        """Get number of samples per class."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))

    def get_metadata(self, idx: int) -> dict:
        """Get metadata for a sample."""
        return {
            'record_id': self.record_ids[idx],
            'label': int(self.labels[idx]),
            'segment_index': int(self.segment_indices[idx]),
        }


def np_array_to_list(arr: np.ndarray):
    """Convert numpy array to list (handles bytes)."""
    if len(arr) == 0:
        return []
    if isinstance(arr[0], bytes):
        return [x.decode() if isinstance(x, bytes) else str(x) for x in arr]
    return arr.tolist()


# ── Convenience function ────────────────────────────────────────────────────

def extract_h5_to_folders(h5_dir: str, output_dir: str, splits: list = None, parallel: bool = True,
                          num_workers: int = None):
    """
    Convenience function to extract HDF5 to folder structure.

    Usage:
        extract_h5_to_folders(
            h5_dir='/path/to/h5/files',
            output_dir='/path/to/output/folder',
            parallel=True,
            num_workers=8
        )
    """
    extractor = H5ImageExtractor(Path(h5_dir), Path(output_dir), parallel=parallel, num_workers=num_workers)
    return extractor.extract_all(splits=splits)


if __name__ == '__main__':
    # Example usage with parallel extraction
    H5_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_final/utrp_dataset_db8')
    OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/utrp_extracted')

    print("Extracting HDF5 images to folders (PARALLEL MODE)...")
    print("=" * 70)

    # Use all available CPU cores for parallel extraction
    extractor = H5ImageExtractor(H5_DIR, OUTPUT_DIR, parallel=True, num_workers=None)
    summary = extractor.extract_all()

    print("\n✓ Extraction complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Summary: {summary}")

    # Example of sequential extraction (if needed):
    # extractor_seq = H5ImageExtractor(H5_DIR, OUTPUT_DIR, parallel=False)
    # summary_seq = extractor_seq.extract_all()