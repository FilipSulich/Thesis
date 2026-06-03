from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross, Classic
from pyrqa.neighbourhood import Unthresholded, FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation, RQAComputation
import numpy as np

class RecurrencePlotGenerator:
    """generates binary and unthresholded recurrence plots from time series using pyrqa"""

    def __init__(self, signal, time_delay, embedding_dim, epsilon, unthresholded=False, path = ""):
        """
        initializes recurrence plot generator with embedding parameters

        Args:
            signal (np.ndarray): input time series signal
            time_delay (int): time delay tau for phase space reconstruction
            embedding_dim (int): embedding dimension m for phase space reconstruction
            epsilon (float): distance threshold for binary recurrence plot
            unthresholded (bool): whether to compute distance matrix instead of binary RP
            path (str): optional path for saving results

        Returns:
            None
        """

        self.time_series = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
        self.unthresholded = unthresholded
        self.epsilon = epsilon
        self.settings_unthreshold = Settings(time_series=self.time_series, analysis_type=Cross, neighbourhood=Unthresholded(),similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.settings = Settings(time_series=self.time_series, analysis_type=Classic, neighbourhood=FixedRadius(epsilon), similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.path = path

    def compute_rqa_binary(self):
        """
        computes binary recurrence plot and extracts recurrence quantification analysis metrics

        Returns:
            tuple: (RQA metrics object, binary recurrence plot)
        """
        rp_computation = RPComputation.create(self.settings, verbose=False)
        rqa_computation = RQAComputation.create(self.settings, verbose=False)
        rqa = rqa_computation.run()
        rp = rp_computation.run()
        return rqa, rp

    def compute_rp_unthresholded(self) -> np.ndarray:
        """
        generates continuous distance matrix (URP) for CNN input normalized to [0,1]

        Returns:
            np.ndarray: unthresholded recurrence plot as normalized distance matrix
        """
        result = RPComputation.create(self.settings_unthreshold, verbose=False).run()
        return result.recurrence_matrix_reverse_normalized
