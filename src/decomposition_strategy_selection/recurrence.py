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
    compute unthresholded (continuous) distance matrix.

    args:
        signal: input signal
        tau: time delay
        m: embedding dimension

    returns:
        distance matrix as numpy array
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
    compute binary recurrence plot and RQA metrics.

    args:
        signal: input signal
        tau: time delay
        m: embedding dimension
        epsilon: threshold
        output_dir: temporary directory for computation

    returns:
        dict with DET, LAM, TT, ENTR, and other RQA metrics
    """
    try:
        os.environ['PYOPENCL_CTX'] = '0'

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
            'ENTR': safe_float(rqa_result.entropy_diagonal_lines),
            'L_mean_diagonal': safe_float(rqa_result.average_diagonal_line),
            'L_longest_diagonal': safe_float(rqa_result.longest_diagonal_line),
            'L_longest_vertical': safe_float(rqa_result.longest_vertical_line),
            'rp_matrix': rp_matrix,
        }

    except Exception as e:
        print(f"  warning: RP computation failed: {e}")
        return {
            'DET': 0.0,
            'LAM': 0.0,
            'TT': 0.0,
            'ENTR': 0.0,
            'L_mean_diagonal': 0.0,
            'L_longest_diagonal': 0.0,
            'L_longest_vertical': 0.0,
            'rp_matrix': None,
        }


def extract_rp_array(rp_obj):
    """
    extract numpy array from pyrqa result object.

    args:
        rp_obj: pyrqa result or numpy array

    returns:
        numpy array
    """
    if hasattr(rp_obj, 'recurrence_matrix_reverse_normalized'):
        return rp_obj.recurrence_matrix_reverse_normalized
    return rp_obj
