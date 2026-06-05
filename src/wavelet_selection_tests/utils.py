import re
import pingouin as pg

def get_subject_group(record_id):
    """
    maps record ID to subject group for cross-validation splits to prevent data leakage

    Args:
        record_id (str): patient record identifier (e.g., 'p01', 'n03')

    Returns:
        int: group ID for grouping
    """
    m = re.match(r'n(\d+)', record_id)
    if m:
        num = int(m.group(1))
        return 200 + (num + 1) // 2

    m = re.match(r'p(\d+)', record_id)
    if m:
        num = int(m.group(1))
        return 100 + (num + 1) // 2

    return -1

def _compute_rank_biserial(x, y):
    """
    computes rank-biserial correlation as non-parametric effect size

    Args:
        x (np.array): first sample array
        y (np.array): second sample array

    Returns:
        float: rank-biserial correlation value between -1 and 1
    """
    mwu_result = pg.mwu(x, y)
    rb = mwu_result['RBC'].values[0]
    return float(rb)
