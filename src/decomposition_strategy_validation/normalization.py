def min_max_normalize(signal):
    """
    normalize signal to [0, 1] range using min-max scaling transformation

    Args:
        signal (numpy.ndarray): input signal array to be normalized

    Returns:
        numpy.ndarray: normalized signal scaled to [0, 1] range, or copy of original if range is zero
    """
    lo = signal.min()
    hi = signal.max()
    rng = hi - lo

    if rng > 0:
        return (signal - lo) / rng
    else:
        return signal.copy()
