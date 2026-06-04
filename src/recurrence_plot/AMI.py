import numpy as np
from scipy.signal import argrelextrema


class AMI:
    """computes average mutual information to estimate optimal time delay for phase space reconstruction"""

    def __init__(self, signal, max_lag=100):
        """
        initializes AMI class with signal and maximum lag to test

        Args:
            signal (np.ndarray): input time series signal
            max_lag (int): maximum time lag to compute AMI for

        Returns:
            None
        """
        self.signal = signal
        self.max_lag = max_lag
        self.ami_values = None
        self.local_minima = None

    def compute_ami(self):
        """
        calculates average mutual information across all time lags from 1 to max_lag

        Returns:
            np.ndarray: AMI values for each lag
        """
        ami_values = []

        for lag in range(1, self.max_lag + 1):
            ami = self._mutual_information(self.signal[:-lag], self.signal[lag:])
            ami_values.append(ami)

        self.ami_values = np.array(ami_values)
        return self.ami_values

    def _mutual_information(self, x, y, bins=50):
        """
        computes mutual information between two time series using histogram-based estimation

        Args:
            x (np.ndarray): first time series
            y (np.ndarray): second time series (time-delayed version)
            bins (int): number of bins for histogram estimation

        Returns:
            float: mutual information value
        """

        hist_xy, _, _ = np.histogram2d(x, y, bins=bins)
        hist_x, _ = np.histogram(x, bins=bins)
        hist_y, _ = np.histogram(y, bins=bins)

        p_xy = hist_xy / np.sum(hist_xy)
        p_x = hist_x / np.sum(hist_x)
        p_y = hist_y / np.sum(hist_y)

        mi = 0.0
        for i in range(bins):
            for j in range(bins):
                if p_xy[i, j] > 0:
                    mi += p_xy[i, j] * np.log(p_xy[i, j] / (p_x[i] * p_y[j] + 1e-10))

        return mi

    def find_local_minima(self):
        """
        finds all local minima in the AMI curve

        Returns:
            np.ndarray: array of lag values where local minima occur
        """
        if self.ami_values is None:
            self.compute_ami()

        minima_indices = argrelextrema(self.ami_values, np.less)[0]
        self.local_minima = minima_indices + 1  # +1 because lag starts at 1
        return self.local_minima

    def get_first_minimum(self):
        """
        returns the first local minimum lag value as optimal time delay

        Returns:
            int: first local minimum lag value, or None if no minima found
        """
        if self.local_minima is None:
            self.find_local_minima()
        return self.local_minima[0] if len(self.local_minima) > 0 else None
