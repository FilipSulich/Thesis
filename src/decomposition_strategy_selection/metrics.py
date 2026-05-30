"""structural similarity metrics for recurrence plots."""

import numpy as np


def pad_matrix(matrix, target_size):
    """
    zero-pad a square matrix to target size.

    used only for structural comparison, never for RQA computation.

    args:
        matrix: input square matrix
        target_size: desired size

    returns:
        padded or cropped matrix of size target_size x target_size
    """
    s = matrix.shape[0]

    if s == target_size:
        return matrix

    if s > target_size:
        # crop if larger
        return matrix[:target_size, :target_size]

    # pad with zeros if smaller
    padded = np.zeros((target_size, target_size), dtype=matrix.dtype)
    padded[:s, :s] = matrix
    return padded


def hamming_distance(rp_a, rp_b):
    """
    normalized hamming distance between two binary RPs.

    args:
        rp_a: first recurrence plot
        rp_b: second recurrence plot (same size as rp_a)

    returns:
        normalized hamming distance in [0, 1]
    """
    # binarize
    a = (rp_a > 0).astype(np.int8)
    b = (rp_b > 0).astype(np.int8)

    # normalized hamming distance
    return float(np.sum(np.abs(a - b))) / (rp_a.shape[0] ** 2)


def pattern_correlation(rp_a, rp_b):
    """
    pearson correlation between two RPs.

    args:
        rp_a: first recurrence plot
        rp_b: second recurrence plot (same size as rp_a)

    returns:
        correlation coefficient
    """
    return float(np.corrcoef(rp_a.ravel(), rp_b.ravel())[0, 1])


def compute_structural_metrics(ref_rp, test_rp):
    """
    compute both hamming distance and pattern correlation.

    automatically pads test_rp to match ref_rp size if needed.

    args:
        ref_rp: reference recurrence plot
        test_rp: test recurrence plot to compare

    returns:
        dict with 'hamming_distance' and 'pattern_correlation'
    """
    try:
        # pad test RP to match reference size
        ref_size = ref_rp.shape[0]
        test_rp_padded = pad_matrix(test_rp, ref_size)

        return {
            'hamming_distance': hamming_distance(ref_rp, test_rp_padded),
            'pattern_correlation': pattern_correlation(ref_rp, test_rp_padded)
        }

    except Exception as e:
        return {
            'hamming_distance': np.nan,
            'pattern_correlation': np.nan
        }
