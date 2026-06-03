"""recurrence plot and RQA computation."""

import os
import numpy as np
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import Unthresholded
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation
from pyrqa.time_series import TimeSeries

from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator


def compute_distance_matrix(signal, tau, m):
    """
    compute unthresholded recurrence plot (URP) for a time series signal

    Args:
        signal (numpy.ndarray): input time series signal
        tau (int): time delay for phase space reconstruction
        m (int): embedding dimension for phase space reconstruction

    Returns:
        numpy.ndarray: URP with continuous values representing pairwise distances
    """
    ts = TimeSeries(signal, embedding_dimension=m, time_delay=tau)
    settings = Settings(
        ts,
        analysis_type=Cross,
        neighbourhood=Unthresholded(),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1,
    )
    result = RPComputation.create(settings, verbose=False).run()
    return result.recurrence_matrix_reverse_normalized


def compute_rqa_metrics(signal, tau, m, epsilon, output_dir='/tmp'):
    """
    compute binary recurrence plot and recurrence quantification analysis (RQA) metrics for a time series

    Args:
        signal (numpy.ndarray): input time series signal
        tau (int): time delay for phase space reconstruction
        m (int): embedding dimension for phase space reconstruction
        epsilon (float): threshold value for binarizing the recurrence plot
        output_dir (str): temporary directory path for intermediate computation files (default: '/tmp')

    Returns:
        dict: dictionary containing RQA metrics including DET (determinism), LAM (laminarity), TT (trapping time), and rp_matrix (recurrence plot matrix)
    """
    try:
        rp_gen = RecurrencePlotGenerator(
            signal=signal,
            time_delay=tau,
            embedding_dim=m,
            path=output_dir,
            epsilon=epsilon,
            unthresholded=False,
        )

        rqa_result, rp_matrix = rp_gen.compute_rqa_binary()

        def safe_float(v):
            return float(v) if not np.isnan(v) else 0.0

        return {
            'DET': safe_float(rqa_result.determinism),
            'LAM': safe_float(rqa_result.laminarity),
            'TT': safe_float(rqa_result.trapping_time),
            'rp_matrix': rp_matrix,
        }

    except Exception as e:
        return {
            'DET': 0.0,
            'LAM': 0.0,
            'TT': 0.0,
            'rp_matrix': None,
        }


def extract_rp_array(rp_obj):
    """
    extract numpy array from pyrqa result object or pass through if already an array

    Args:
        rp_obj (object or numpy.ndarray): pyrqa computation result object or numpy array

    Returns:
        numpy.ndarray: recurrence plot as numpy array
    """
    if hasattr(rp_obj, 'recurrence_matrix_reverse_normalized'):
        return rp_obj.recurrence_matrix_reverse_normalized
    return rp_obj
