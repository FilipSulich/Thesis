import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Optional
from pathlib import Path
from collections import defaultdict
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import re

def get_subject_id(record_id: str) -> str:
    """
    Maps record IDs to unique human subjects for cross-validation splitting.

    Dataset structure:
    - n01-n50: Healthy controls (each is a unique subject)
    - p01/p02 are from same PAF patient (p01=Pre-AF, p02=NSR)
    - p03/p04 are from same PAF patient (p03=Pre-AF, p04=NSR), etc.

    Returns subject ID for grouping in CV splits.
    """
    match = re.search(r'([pn])(\d+)', record_id)
    if match:
        prefix = match.group(1)
        num = int(match.group(2))

        if prefix == 'p':
            # PAF patients: group odd/even pairs (p01/p02 -> subject_p01)
            subject_num = ((num + 1) // 2) * 2 - 1  # Maps to odd: 1,2->1, 3,4->3, etc.
            return f"subject_p{subject_num:02d}"
        else:
            # Healthy controls: each n* is a unique subject
            return f"subject_n{num:02d}"

    return str(record_id).strip().lower()

class UTRPDataset(Dataset):
    """
    PyTorch dataset for UTRP images.

    Loads from HDF5 file and supports:
      - Single-scale or multi-scale mode
      - Optional data augmentation (training only)
      - Patient-level indexing for stratified splits
    """

    def __init__(
            self,
            hdf5_path: Path,
            mode: str = 'single',
            augment: bool = False,
            patient_ids: Optional[List[str]] = None  # Patient IDs to filter (e.g., ['p01', 'n01'])
    ):
        assert mode in ['single', 'multi'], "mode must be 'single' or 'multi'"

        self.hdf5_path = Path(hdf5_path)
        self.mode = mode
        self.augment = augment

        # Load metadata
        with h5py.File(self.hdf5_path, 'r') as f:
            all_labels = f['metadata/labels'][:]

            # Check if using old or new dataset structure
            if 'patient_ids' in f['metadata']:
                # New simplified dataset
                all_patient_ids = [pid.decode('utf-8') if isinstance(pid, bytes) else pid
                                   for pid in f['metadata/patient_ids'][:]]

                # Filter by patient IDs
                if patient_ids is not None:
                    patient_set = set(patient_ids)
                    self.indices = [i for i, pid in enumerate(all_patient_ids) if pid in patient_set]
                else:
                    self.indices = list(range(len(all_labels)))

                self.patient_ids = [all_patient_ids[i] for i in self.indices]
                self.record_ids = self.patient_ids  # Same in new dataset
                self.subject_ids = [get_subject_id(pid) for pid in self.patient_ids]

            else:
                # Old dataset with record_ids
                raw_record_ids = [rid.decode('utf-8') for rid in f['metadata/record_ids'][:]]
                all_subject_ids = [get_subject_id(rid) for rid in raw_record_ids]

                # Filter by record IDs
                if patient_ids is not None:
                    patient_set = set(patient_ids)
                    self.indices = [i for i, rid in enumerate(raw_record_ids) if rid in patient_set]
                else:
                    self.indices = list(range(len(all_labels)))

                self.record_ids = [raw_record_ids[i] for i in self.indices]
                self.subject_ids = [all_subject_ids[i] for i in self.indices]
                self.patient_ids = self.record_ids

        # Store filtered labels
        self.labels = all_labels[self.indices]

        print(f"Loaded {len(self.indices)} segments from {len(set(self.patient_ids))} unique patients.")

        # Cache for data (lazy loading)
        self._data_cache = None

    def _load_data(self):
        if self._data_cache is None:
            with h5py.File(self.hdf5_path, 'r') as f:
                dataset_key = 'single_scale/utrps' if self.mode == 'single' else 'multi_scale/utrps'
                # Accessing only the needed indices saves memory
                self._data_cache = f[dataset_key][self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Returns: (utrp_tensor, label, record_id)
        Adding record_id here helps with patient-level evaluation later.
        """
        self._load_data()

        utrp = self._data_cache[idx].astype(np.float32)
        label = int(self.labels[idx])
        record_id = self.record_ids[idx]

        if self.mode == 'single' and utrp.ndim == 2:
            utrp = utrp[np.newaxis, :, :]  # Ensure (1, H, W)

        if self.augment:
            utrp = self._augment(utrp)

        return torch.from_numpy(utrp), label, record_id

    # def _augment(self, utrp: np.ndarray) -> np.ndarray:
    #     # Rotations are also great for UTRPs since they are symmetric
    #     if np.random.rand() > 0.5:
    #         utrp = np.flip(utrp, axis=2).copy()  # Horizontal
    #     if np.random.rand() > 0.5:
    #         utrp = np.flip(utrp, axis=1).copy()  # Vertical
    #     if np.random.rand() > 0.5:
    #         noise = np.random.normal(0, 0.01, utrp.shape).astype(np.float32)
    #         utrp = np.clip(utrp + noise, 0, 1)
    #     return utrp

    def _augment(self, utrp: np.ndarray) -> np.ndarray:
        # Transpose: swap axes -1 and -2
        # Valid because UTRP is a distance matrix — transposing
        # is equivalent to reversing the time direction of the signal
        # which should not change the pre-AF label
        if np.random.rand() > 0.5:
            utrp = utrp.transpose(0, 2, 1).copy()  # (C, H, W) -> (C, W, H)

        # Small Gaussian noise on the distance values
        # Simulates measurement noise in the phase space embedding
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.02, utrp.shape).astype(np.float32)
            utrp = np.clip(utrp + noise, 0, 1)

        return utrp

    def get_patient_ids(self) -> List[str]:
        """Get list of patient IDs in this dataset."""
        return list(set(self.patient_ids))

    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution."""
        unique, counts = np.unique(self.labels, return_counts=True)
        return dict(zip(unique, counts))


def compute_class_weights(dataset: UTRPDataset) -> torch.Tensor:
    """
    Compute class weights for handling imbalance.

    Args:
        dataset: UTRPDataset

    Returns:
        Tensor with pos_weight for BCEWithLogitsLoss
    """
    dist = dataset.get_class_distribution()
    n_negative = dist.get(0, 0)
    n_positive = dist.get(1, 0)

    if n_positive == 0:
        return torch.tensor([1.0])

    pos_weight = n_negative / n_positive
    return torch.tensor([pos_weight])


def create_dataloaders(
    train_dataset: UTRPDataset,
    val_dataset: UTRPDataset,
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        batch_size: Batch size
        num_workers: Number of worker processes

    Returns:
        (train_loader, val_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )

    return train_loader, val_loader


def get_all_patient_ids(hdf5_path: Path) -> Tuple[List[str], np.ndarray]:
    """
    Returns all unique RECORD IDs and their corresponding labels.

    Important: Records keep their individual labels.
    - p01: label=1 (Pre-AF)
    - p02: label=0 (NSR from same patient)
    - n01: label=0 (NSR from healthy control)

    Returns:
        (record_ids, labels): 100 records with their individual labels
    """
    with h5py.File(hdf5_path, 'r') as f:
        all_record_ids = [rid.decode('utf-8') for rid in f['metadata/record_ids'][:]]
        all_labels = f['metadata/labels'][:]

    # Get unique records and their labels
    record_to_label = {}
    for rid, label in zip(all_record_ids, all_labels):
        if rid not in record_to_label:
            record_to_label[rid] = label

    unique_records = list(record_to_label.keys())
    record_labels = np.array([record_to_label[rid] for rid in unique_records])

    return unique_records, record_labels