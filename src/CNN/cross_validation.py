import numpy as np
from typing import List, Tuple
from sklearn.model_selection import StratifiedKFold
import random
import re


def get_subject_id(record_id: str) -> str:
    """
    Maps record IDs to unique human subjects.

    With p-even excluded, each record is its own independent subject:
      - p01, p03, ..., p49 → each is a unique Pre-AF subject
      - n01, n02, ..., n50 → each is a unique NSR subject

    The old paired logic (p01/p02 → same subject) is removed because
    p-even records are no longer in the dataset.
    """
    return str(record_id).strip().lower()


def get_label_from_record_id(record_id: str) -> int:
    """
    Infer label from record ID alone (no HDF5 lookup needed).
      - p-odd  → 1 (Pre-AF)
      - n*     → 0 (NSR)
    """
    match = re.search(r'([pn])(\d+)', record_id)
    if match:
        prefix = match.group(1)
        num = int(match.group(2))
        if prefix == 'p':
            return 1 if num % 2 == 1 else 0  # p-odd = Pre-AF
        else:
            return 0  # n* = NSR
    raise ValueError(f"Cannot infer label from record_id: {record_id!r}")


class PatientStratifiedKFold:
    """
    Stratified K-fold at the PATIENT level.

    Each patient_id is treated as an independent unit — no paired logic.
    Stratification ensures each fold has balanced Pre-AF / NSR patients.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits = n_splits
        self.random_state = random_state
        self.skf = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    def split(self, patient_ids: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Create stratified K-fold splits at the patient level.

        Args:
            patient_ids: List of record IDs (e.g. ['p01', 'p03', 'n01', ...])
                         Must contain only p-odd and n* records.

        Returns:
            List of (train_patient_ids, val_patient_ids) tuples
        """
        patient_labels = np.array([get_label_from_record_id(pid) for pid in patient_ids])

        n_preaf = patient_labels.sum()
        n_nsr = len(patient_labels) - n_preaf
        print(f"CV split over {len(patient_ids)} patients: "
              f"{n_preaf} Pre-AF, {n_nsr} NSR")

        splits = []
        for train_idx, val_idx in self.skf.split(patient_ids, patient_labels):
            train_patients = [patient_ids[i] for i in train_idx]
            val_patients   = [patient_ids[i] for i in val_idx]
            splits.append((train_patients, val_patients))

        return splits


def create_holdout_test_set(
    patient_ids: List[str],
    n_test_per_class: int = 5,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Hold out a balanced test set at the patient level.

    Args:
        patient_ids: All record IDs (p-odd + n* only).
        n_test_per_class: Number of patients per class for the test set.
        random_state: RNG seed.

    Returns:
        (test_patient_ids, train_val_patient_ids)
    """
    rng = random.Random(random_state)

    preaf_patients = [pid for pid in patient_ids if get_label_from_record_id(pid) == 1]
    nsr_patients   = [pid for pid in patient_ids if get_label_from_record_id(pid) == 0]

    rng.shuffle(preaf_patients)
    rng.shuffle(nsr_patients)

    if n_test_per_class > min(len(preaf_patients), len(nsr_patients)):
        raise ValueError(
            f"n_test_per_class={n_test_per_class} exceeds available patients "
            f"(Pre-AF: {len(preaf_patients)}, NSR: {len(nsr_patients)})"
        )

    test_patients     = preaf_patients[:n_test_per_class] + nsr_patients[:n_test_per_class]
    train_val_patients = preaf_patients[n_test_per_class:] + nsr_patients[n_test_per_class:]

    print(f"\nHold-out test set:")
    print(f"  Pre-AF: {n_test_per_class} patients")
    print(f"  NSR:    {n_test_per_class} patients")
    print(f"  Total test patients:     {len(test_patients)}")
    print(f"  Remaining (train+val):   {len(train_val_patients)}")

    return test_patients, train_val_patients


def get_remaining_labels(patient_ids: List[str]) -> np.ndarray:
    """
    Return labels for a list of patient IDs without any HDF5 lookup.
    Works purely from the record ID naming convention.
    """
    return np.array([get_label_from_record_id(pid) for pid in patient_ids])