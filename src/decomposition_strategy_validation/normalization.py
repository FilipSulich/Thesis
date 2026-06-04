def min_max_normalize(signal):
    """
    normalize signal to [0, 1] range using min-max scaling.

    Args:
        signal (numpy.ndarray): input signal array to be normalized

    Returns:
        numpy.ndarray: normalized signal scaled to [0, 1] range
    """

    return (signal - signal.min()) / (signal.max() - signal.min())
