"""
shared utility functions for cnn training and evaluation.
"""

import re
import numpy as np
import torch


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()

def get_subject_group(record_id: str) -> int:
    """
    extract patient group from record id.

    ensures paired recordings (p01/p02, p03/p04, n01/n01c) stay together.

    """
    match = re.search(r'(\d+)', record_id)
    if not match:
        raise ValueError(f"cannot extract patient number from: {record_id}")

    num = int(match.group(1))

    # pre-af patients (p*)
    if record_id.startswith('p'):
        return 100 + (num + 1) // 2

    # nsr patients (n*)
    else:
        return num


def set_seed(seed: int = 42):
    """set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


N_FOLDS = 10
N_SUBBANDS = 7
SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
BATCH_SIZE = 32
SEED = 42