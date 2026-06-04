import re
import numpy as np
import torch


def get_device():
    """
    gets the available device for PyTorch (GPU, MPS, or CPU).
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


DEVICE = get_device()

def get_subject_group(record_id: str) -> int:
    """
    extract patient group from record ID for cross-validation.

    grouping strategy:
    - pre-AF (p*): odd-even pairs from same patient
      p01 & p02 → group 101
      p03 & p04 → group 102
      p05 & p06 → group 103, etc.

    - NSR (n*): all variants of same base patient number
      n01, n01c, n02, n02c → group 201 (all from patient 1)
      n03, n03c, n04, n04c → group 202 (all from patient 2)
      n05, n05c, n06, n06c → group 203, etc.

    this prevents data leakage between sets by keeping all recordings from the same patient in the same fold.
    """
    match = re.search(r'(\d+)', record_id)
    if not match:
        raise ValueError(f"cannot extract patient number from: {record_id}")

    num = int(match.group(1))

    if record_id.startswith('p'):
        return 100 + (num + 1) // 2

    elif record_id.startswith('n'):
        return 200 + (num + 1) // 2

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