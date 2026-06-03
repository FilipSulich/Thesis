import numpy as np


def pad_matrix(matrix, target_size):
    """
    zero-pad or crop a square matrix to match the specified target size

    Args:
        matrix (numpy.ndarray): input square matrix to be resized
        target_size (int): desired dimension for output square matrix

    Returns:
        numpy.ndarray: padded or cropped matrix of size target_size x target_size
    """
    s = matrix.shape[0]

    if s == target_size:
        return matrix

    if s > target_size:
        return matrix[:target_size, :target_size]

    # pad with zeros if smaller
    padded = np.zeros((target_size, target_size), dtype=matrix.dtype)
    padded[:s, :s] = matrix
    return padded


def hamming_distance(rp_a, rp_b):
    """
    compute normalized hamming distance between two binary recurrence plots

    Args:
        rp_a (numpy.ndarray): first recurrence plot matrix
        rp_b (numpy.ndarray): second recurrence plot matrix with same size as rp_a

    Returns:
        float: normalized hamming distance in range [0, 1] where 0 means identical and 1 means completely different
    """
    a = (rp_a > 0).astype(np.int8)
    b = (rp_b > 0).astype(np.int8)

    return float(np.sum(np.abs(a - b))) / (rp_a.shape[0] ** 2)


def pearson_correlation(rp_a, rp_b):
    """
    compute pearson correlation coefficient between two recurrence plot matrices

    Args:
        rp_a (numpy.ndarray): first recurrence plot matrix
        rp_b (numpy.ndarray): second recurrence plot matrix with same size as rp_a

    Returns:
        float: pearson correlation coefficient measuring linear relationship between the two recurrence plots
    """
    return float(np.corrcoef(rp_a.ravel(), rp_b.ravel())[0, 1])


def compute_structural_metrics(ref_rp, test_rp):
    """
    compute both hamming distance and pearson correlation to compare two recurrence plots

    Args:
        ref_rp (numpy.ndarray): reference recurrence plot matrix
        test_rp (numpy.ndarray): test recurrence plot matrix to compare against reference

    Returns:
        dict: dictionary containing 'hamming_distance' and 'pattern_correlation' keys with their computed values, or NaN values if computation fails
    """
    try:
        # pad test RP to match reference size
        ref_size = ref_rp.shape[0]
        test_rp_padded = pad_matrix(test_rp, ref_size)

        return {
            'hamming_distance': hamming_distance(ref_rp, test_rp_padded),
            'pattern_correlation': pearson_correlation(ref_rp, test_rp_padded)
        }

    except Exception as e:
        return {
            'hamming_distance': np.nan,
            'pattern_correlation': np.nan
        }
