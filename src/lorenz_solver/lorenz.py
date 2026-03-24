import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

class LorenzSolver:
    def __init__(self, sigma, rho, beta):
        self.sigma = sigma
        self.rho = rho
        self.beta = beta

    def lorenz_system(self, state, t):
        x, y, z = state
        dx_dt = self.sigma * (y - x)
        dy_dt = x * (self.rho - z) - y
        dz_dt = x * y - self.beta * z
        return [dx_dt, dy_dt, dz_dt]

    def plot(self, signal):
        fig, ax = plt.subplots()
        plt.title('x axis')
        plt.xlabel('time')
        plt.ylabel('x')
        ax.plot(signal[0:5000])
        plt.show()

    def generate_lorenz_signal(self, n_points=10000, dt=0.01, target_fs=128):
        t = np.arange(0, n_points * dt, dt)

        initial_state = [1.0, 1.0, 1.0]

        solution = odeint(self.lorenz_system, initial_state, t)
        x, y, z = solution.T

        x_original = x

        original_fs = 1.0 / dt
        n_samples_target = int(len(x_original) * target_fs / original_fs)
        x_resampled = np.interp(
            np.linspace(0, len(x_original) - 1, n_samples_target),
            np.arange(len(x_original)),
            x_original
        )

        x_resampled = x_resampled - np.mean(x_resampled)
        x_resampled = x_resampled / np.std(x_resampled, ddof=1)
        return x_resampled


    def plot_lorenz_attractor(self, n_points=10000, dt=0.01, save_path=None):
        t = np.arange(0, n_points * dt, dt)

        initial_state = [1.0, 1.0, 1.0]

        solution = odeint(self.lorenz_system, initial_state, t)
        x, y, z = solution.T

        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')

        ax.plot(x, y, z, linewidth=0.5, alpha=0.8)
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Z', fontsize=12)
        ax.set_title('Lorenz Attractor', fontsize=14)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()


if __name__ == "__main__":
    solver = LorenzSolver(10.0, 28.0, 8.0/3.0)

    signal = solver.generate_lorenz_signal()

    solver.plot_lorenz_attractor()
