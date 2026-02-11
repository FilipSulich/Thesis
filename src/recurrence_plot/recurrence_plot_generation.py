from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross
from pyrqa.neighbourhood import Unthresholded
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
from pyrqa.image_generator import ImageGenerator

class RecurrencePlotGenerator:
    def __init__(self, signal, time_delay, embedding_dim):
        self.time_series = TimeSeries(signal, embedding_dim, time_delay)
        self.settings = Settings(time_series=self.time_series, analysis_type=Cross, neighbourhood=Unthresholded(), similarity_measure=EuclideanMetric, theiler_corrector=1)

    def compute_rqa(self):
        computation = RQAComputation.create(self.settings, verbose=True)
        return computation.run()

    def plot_recurrence(self, path):
        result = self.compute_rqa()
        ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized, path)