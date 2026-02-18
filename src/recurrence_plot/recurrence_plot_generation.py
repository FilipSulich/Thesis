from typing import Optional

from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross, Classic
from pyrqa.neighbourhood import Unthresholded, FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation, RQAComputation
from pyrqa.image_generator import ImageGenerator
import os

class RecurrencePlotGenerator:
    def __init__(self, signal, time_delay, embedding_dim, path, unthresholded=False):
        self.time_series = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
        self.unthresholded = unthresholded
        self.settings_unthreshold = Settings(time_series=self.time_series, analysis_type=Cross, neighbourhood=Unthresholded(),similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.settings = Settings(time_series=self.time_series, analysis_type=Classic, neighbourhood=FixedRadius(0.65), similarity_measure=EuclideanMetric, theiler_corrector=1)
        self.path = path

    def compute_rqa_binary(self):
        rp_computation = RPComputation.create(self.settings, verbose=True)
        rqa_computation = RQAComputation.create(self.settings, verbose=True)
        rqa = rqa_computation.run()
        rp = rp_computation.run()
        return rqa, rp

    def plot_recurrence(self, id: str, index: int):
        result_rqa, result_rp = self.compute_rqa_binary()
        if os.path.isdir(self.path):
            if self.unthresholded:
                string = "unthresholded"
            else :
                string = "thresholded"

            path = os.path.join(self.path, f"rp_{id}_{index}_{string}.png")

            if self.unthresholded:
                computation = RPComputation.create(self.settings_unthreshold, verbose=True)
                result = computation.run()
                ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized, path)
            else :
                ImageGenerator.save_recurrence_plot(result_rp.recurrence_matrix_reverse_normalized, path)