import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors


class FNN:
    """computes false nearest neighbors (FNN) to determine optimal embedding dimension for phase space reconstruction"""

    def __init__(self, signal, tau, max_dim=15, r_tol=15.0, a_tol=2.0):
        """
        initializes FNN class with signal and embedding parameters

        Args:
            signal (np.ndarray): input time series signal to reconstruct
            tau (int): time delay for phase space embedding
            max_dim (int): maximum embedding dimension to test
            r_tol (float): relative tolerance threshold for distance ratio test
            a_tol (float): absolute tolerance threshold for escape criterion

        Returns:
            None
        """

        self.signal = signal
        self.tau = tau
        self.max_dim = max_dim
        self.r_tol = r_tol
        self.a_tol = a_tol
        self.fnn_percentages = None

    def embed(self, m):
        """
        constructs m-dimensional phase space vectors using time delay embedding

        Args:
            m (int): embedding dimension

        Returns:
            np.ndarray: embedded vectors of shape (n_vectors, m)
        """
        n = len(self.signal) - (m - 1) * self.tau
        embedded = np.array([self.signal[i:i + m * self.tau:self.tau] for i in range(n)])
        return embedded

    def compute_fnn(self):
        """
        calculates percentage of false nearest neighbors across all dimensions from 1 to max_dim

        Returns:
            np.ndarray: FNN percentages for each dimension tested
        """
        fnn_percentages = []

        for m in range(1, self.max_dim + 1):
            embedded_m = self.embed(m)
            n_points = len(embedded_m)

            if n_points < 2:
                fnn_percentages.append(0.0)
                continue

            embedded_m1 = self.embed(m + 1)
            n_points_m1 = len(embedded_m1)

            embedded_m_trunc = embedded_m[:n_points_m1]

            nbrs = NearestNeighbors(n_neighbors=2, metric='euclidean').fit(embedded_m_trunc)
            distances, indices = nbrs.kneighbors(embedded_m_trunc)

            nn_dist_m = distances[:, 1]
            nn_indices = indices[:, 1]

            nn_dist_m1 = np.linalg.norm(embedded_m1 - embedded_m1[nn_indices], axis=1)

            ratio = np.divide(np.abs(nn_dist_m1 - nn_dist_m), nn_dist_m, out=np.zeros_like(nn_dist_m), where=nn_dist_m!=0)

            Ra = np.std(self.signal)

            # count false nearest neighbors
            false_neighbors = 0
            for i in range(len(ratio)):
                if ratio[i] > self.r_tol:
                    false_neighbors += 1
                elif nn_dist_m1[i] / Ra > self.a_tol:
                    false_neighbors += 1

            fnn_percent = (false_neighbors / len(ratio)) * 100
            fnn_percentages.append(fnn_percent)

        self.fnn_percentages = np.array(fnn_percentages)
        return self.fnn_percentages

    def get_optimal_dimension(self, threshold=5.0):
        """
        returns optimal embedding dimension where FNN percentage first drops below treshold

        Args:
            threshold (float): FNN percentage threshold (default 5.0%)

        Returns:
            int: optimal embedding dimension, or max_dim if threshold never reached
        """
        if self.fnn_percentages is None:
            self.compute_fnn()

        optimal = np.where(self.fnn_percentages < threshold)[0]
        return optimal[0] + 1 if len(optimal) > 0 else self.max_dim