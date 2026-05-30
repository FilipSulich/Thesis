"""embedding parameter estimation using AMI and FNN."""

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN


def estimate_tau(signal, max_lag=100):
    """
    estimate time delay (tau) using first minimum of average mutual information.

    args:
        signal: input signal
        max_lag: maximum lag to search

    returns:
        optimal tau value
    """
    ami = AMI(signal)
    tau = ami.get_first_minimum()

    if tau is None:
        # fallback if no minimum found
        tau = max(1, len(signal) // 100)

    return tau


def estimate_embedding_dim(signal, tau, max_dim=15):
    """
    estimate embedding dimension (m) using false nearest neighbors.

    args:
        signal: input signal
        tau: time delay
        max_dim: maximum dimension to test

    returns:
        optimal embedding dimension
    """
    fnn = FNN(signal=signal, tau=tau)
    m = fnn.get_optimal_dimension()
    return m


def estimate_embedding_params(signal):
    """
    estimate both tau and m for a signal.

    args:
        signal: input signal

    returns:
        tuple of (tau, m)
    """
    tau = estimate_tau(signal)
    m = estimate_embedding_dim(signal, tau)
    return tau, m
