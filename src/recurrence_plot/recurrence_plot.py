import numpy as np
from scipy.spatial.distance import pdist, squareform


class RecurrencePlotGenerator:
    def __init__(self, signals):
        self.signals = signals

    def compute_rp(self, vectors):
        distances = pdist(vectors, metric='euclidean')
        rp = squareform(distances)
        return rp

    def embed(self, signal, tau, m):
        N = len(signal)
        num_vectors = N - (m-1) * tau
        embedded = np.zeros((num_vectors, m))
        for i in range(num_vectors):
            embedded[i] = signal[i * tau:(i + 1) * tau: tau]
        return embedded

    def embed_group(self, tau_dict, m_dict):
        rps = np.zeros((len(self.signals), len(self.signals)))
        for signal in self.signals:
            embedded = self.embed(signal, tau_dict, m_dict)
            rps[signal] = self.compute_rp(embedded)

