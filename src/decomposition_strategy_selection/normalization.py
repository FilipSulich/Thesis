"""signal normalization functions."""

def min_max_normalize(signal):
    """
    normalize signal to [0, 1] range using min-max scaling.

    args:
        signal: input signal as numpy array

    returns:
        normalized signal
    """
    lo = signal.min()
    hi = signal.max()
    rng = hi - lo

    if rng > 0:
        return (signal - lo) / rng
    else:
        # avoid division by zero for constant signals
        return signal.copy()
