"""embedding parameter estimation using AMI and FNN."""

from src.recurrence_plot.AMI import AMI
from src.recurrence_plot.FNN import FNN


def estimate_tau(signal):
    """
    estimate optimal time delay (tau) using the first minimum of average mutual information (AMI)

    Args:
        signal (numpy.ndarray): input time series signal

    Returns:
        int: optimal time delay value for phase space reconstruction
    """
    ami = AMI(signal)
    tau = ami.get_first_minimum()

    if tau is None:
        # fallback if no minimum found
        tau = max(1, len(signal) // 100)

    return tau


def estimate_embedding_dim(signal, tau):
    """
    estimate optimal embedding dimension (m) using false nearest neighbors (FNN) method

    Args:
        signal (numpy.ndarray): input time series signal
        tau (int): time delay for phase space reconstruction

    Returns:
        int: optimal embedding dimension for phase space reconstruction
    """
    fnn = FNN(signal=signal, tau=tau)
    m = fnn.get_optimal_dimension()
    return m


def estimate_embedding_params(signal):
    """
    estimate both optimal time delay and embedding dimension for phase space reconstruction of a signal

    Args:
        signal (numpy.ndarray): input time series signal

    Returns:
        tuple: tuple containing (tau, m) where tau is time delay and m is embedding dimension
    """
    tau = estimate_tau(signal)
    m = estimate_embedding_dim(signal, tau)
    return tau, m
