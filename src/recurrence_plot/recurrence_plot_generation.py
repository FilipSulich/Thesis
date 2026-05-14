from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross, Classic
from pyrqa.neighbourhood import Unthresholded, FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation, RQAComputation
import numpy as np

class RecurrencePlotGenerator:
    def __init__(self, signal, time_delay, embedding_dim, epsilon, unthresholded=False, path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/outputs/"):
        self.time_series = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
        self.unthresholded = unthresholded
        self.epsilon = epsilon
        self.settings_unthreshold = Settings(time_series=self.time_series, analysis_type=Cross, neighbourhood=Unthresholded(),similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.settings = Settings(time_series=self.time_series, analysis_type=Classic, neighbourhood=FixedRadius(epsilon), similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.path = path

    def compute_rqa_binary(self):
        rp_computation = RPComputation.create(self.settings, verbose=False)
        rqa_computation = RQAComputation.create(self.settings, verbose=False)
        rqa = rqa_computation.run()
        rp = rp_computation.run()
        return rqa, rp

    def compute_rp_unthresholded(self) -> np.ndarray:
        result = RPComputation.create(self.settings_unthreshold, verbose=False).run()
        return result.recurrence_matrix_reverse_normalized
