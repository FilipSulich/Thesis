from matplotlib import pyplot as plt
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Cross, Classic
from pyrqa.neighbourhood import Unthresholded, FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RPComputation, RQAComputation
from pyrqa.image_generator import ImageGenerator
import os
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

    def compute_rqa(self, signal, epsilon):
        settings_new = Settings(time_series=signal, analysis_type=Classic,neighbourhood=FixedRadius(epsilon), similarity_measure=EuclideanMetric,theiler_corrector=1)

        rp_computation = RPComputation.create(settings_new, verbose=False)
        rqa_computation = RQAComputation.create(settings_new, verbose=False)

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
                computation = RPComputation.create(self.settings_unthreshold, verbose=False)
                result = computation.run()
                ImageGenerator.save_unthresholded_recurrence_plot(result.recurrence_matrix_reverse_normalized, path)
            else :
                ImageGenerator.save_recurrence_plot(result_rp.recurrence_matrix_reverse_normalized, path)


    @staticmethod
    def plot_rp(rp_matrix, tau, m, epsilon=None, det=None, title_prefix="Recurrence Plot",
                ax=None, show_labels=True):
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        else:
            fig = ax.figure

        ax.imshow(rp_matrix, cmap='binary', origin='upper', aspect='equal')

        title_parts = [title_prefix]
        title_parts.append(f'τ={tau}, m={m}')
        if epsilon is not None:
            title_parts.append(f'ε={epsilon:.4f}')
        if det is not None:
            title_parts.append(f'DET={det:.3f}')

        ax.set_title('\n'.join(title_parts), fontsize=10, fontweight='bold')

        if show_labels:
            ax.set_xlabel('Time i', fontsize=9)
            ax.set_ylabel('Time j', fontsize=9)
        else:
            ax.set_xticks([])
            ax.set_yticks([])

        return fig, ax

    @staticmethod
    def find_epsilon_for_target_rr(signal, tau, m, target_rr=0.1):
        from scipy.spatial.distance import pdist, squareform

        n = len(signal) - (m - 1) * tau
        if n < 10:
            return 0.1

        embedded = np.array([signal[i:i + m * tau:tau] for i in range(n)])
        distances = squareform(pdist(embedded, metric='euclidean'))

        all_dists = distances[np.triu_indices_from(distances, k=1)]
        eps_min, eps_max = np.percentile(all_dists, 1), np.percentile(all_dists, 20)

        # Bisection search
        for _ in range(20):
            eps_mid = (eps_min + eps_max) / 2
            rp_test = (distances <= eps_mid).astype(int)
            np.fill_diagonal(rp_test, 0)
            rr = np.sum(rp_test) / (rp_test.shape[0] * (rp_test.shape[0] - 1))

            if abs(rr - target_rr) < 0.01:
                break

            if rr < target_rr:
                eps_min = eps_mid
            else:
                eps_max = eps_mid

        return eps_mid

