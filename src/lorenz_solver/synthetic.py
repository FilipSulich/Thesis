import numpy as np

from lorenz_solver.lorenz import LorenzSolver


class SyntheticDatasetGenerator:
    def __init__(self, length=1000, fs=100, samples_per_class=100, seed=42):
        self.length = length
        self.fs = fs
        self.samples_per_class = samples_per_class
        self.seed = seed
        np.random.seed(seed)
        self.t = np.linspace(0, length/fs, length)

    def generate_sine(self):
        freq = np.random.uniform(1, 10)
        return np.sin(2 * np.pi * freq * self.t)


    def generate_multiple_sines(self):
        y = np.sin(2 * np.pi * 2 * self.t) + np.sin(2 * np.pi * 6 * self.t) + np.sin(2 * np.pi * 12 * self.t) + np.sin(2 * np.pi * 24 * self.t) + np.sin(2 * np.pi * 48 * self.t)
        return y

    def generate_lorenz_signal(self):
        sigma, rho, beta = np.random.uniform(8, 12), np.random.uniform(24, 32), np.random.uniform(2.5, 3.0)
        solver = LorenzSolver(sigma=sigma, rho=rho, beta=beta)
        signal = solver.generate_lorenz_signal(n_points=self.length * 10, dt=0.01, target_fs=self.fs)
        if len(signal) > self.length:
            start = np.random.randint(0, len(signal) - self.length)
            return signal[start:start + self.length]
        return signal[:self.length]

    def generate_noise(self):
        return np.random.randn(self.length)

    def generate_dataset(self):
        X_list, y_list = [], []
        # for _ in range(self.samples_per_class):
        #     X_list.append(self.generate_sine())
        #     y_list.append(0)
        # for _ in range(self.samples_per_class):
        #     X_list.append(self.generate_lorenz_signal())
        #     y_list.append(1)
        # for _ in range(self.samples_per_class):
        #     X_list.append(self.generate_noise())
        #     y_list.append(2)
        X_list.append(self.generate_multiple_sines())
        y_list.append(1)
        return np.array(X_list), np.array(y_list)
