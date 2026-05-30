import re
import pingouin as pg

def get_subject_group(record_id):
    """
    Map record ID to subject group for CV splits (to avoid data leakage between sets).
    Paired records (p01/p02, p03/p04) go in same group.
    """
    m = re.match(r'n(\d+)', record_id)
    if m:
        return int(m.group(1))

    m = re.match(r'p(\d+)', record_id)
    if m:
        num = int(m.group(1))
        return 100 + num // 2

    return -1


def minmax_normalize(x):
    """Normalize array to [0, 1] range."""
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo + 1e-9)

def _compute_rank_biserial(x, y):
    """
    Rank-biserial correlation (non-parametric effect size).

    Args:
        x, y: Arrays to compare

    Returns:
        Rank-biserial correlation (-1 to 1)
    """
    mwu_result = pg.mwu(x, y)
    rb = mwu_result['RBC'].values[0]
    return float(rb)
