import h5py
import numpy as np
import sys
import matplotlib.pyplot as plt
import os

from src.data_cleaning.data_cleaner import DataCleaner
from src.recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator
from src.wavelets.stationary_wavelet_transform import SWT

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def plot_swt_results(decomposed_signal, original_signal=None):
    num_levels = len(decomposed_signal)

    total_plots = num_levels + 1
    fig, axes = plt.subplots(total_plots, 1, figsize=(10, 2 * total_plots), sharex=True)

    if total_plots == 1: axes = [axes]  # Handle level=0 edge case

    axes[0].plot(decomposed_signal[0][0], color='blue')
    axes[0].set_title(f"Approximation Coefficient (Level {num_levels})")

    for i in range(num_levels):
        detail_coeff = decomposed_signal[i][1]
        level_label = num_levels - i

        axes[i + 1].plot(detail_coeff, color='red')
        axes[i + 1].set_title(f"Detail Coefficient - Level {level_label}")

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt


def plot_dwt_results(coeffs):
    num_plots = len(coeffs)
    level = num_plots - 1

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 2 * num_plots))

    if num_plots == 1: axes = [axes]

    axes[0].plot(coeffs[0], color='blue')
    axes[0].set_title(f"Approximation (cA{level})")
    axes[0].set_ylabel("Amplitude")

    for i in range(1, num_plots):
        current_level = level - i + 1
        axes[i].plot(coeffs[i], color='red')
        axes[i].set_title(f"Detail (cD{current_level})")
        axes[i].set_ylabel("Amplitude")

    plt.tight_layout()
    plt.show()


def plot_signal(file_path, split="learning", index=0, sampling_rate=128):
    with h5py.File(file_path, "r") as f:
        signals = f[f"{split}/signals"]
        labels = f[f"{split}/labels"]
        patient_ids = f[f"{split}/patient_ids"]
        segment_ids = f[f"{split}/segment_ids"]

        if index >= len(signals):
            print("Index out of range.")
            return

        signal = signals[index]
        label = labels[index]
        patient_id = patient_ids[index]
        segment_id = segment_ids[index]

        # data_cleaner = DataCleaner()
        # clean_signal, info = data_cleaner.preprocess(signal)

        # if clean_signal is None:
        #     print(f"Signal at index {index} failed quality check: {info}")
        #     return

        swt = SWT()
        # dwt = DWT()
        (cD1,cA1), (cD2,cA2), (cD3,cA3), (cD4,cA4) = swt.decompose(signal, level=4)
        # decomposed_dwt = dwt.decompose(clean_signal, level=4)

        # plot_dwt_results(decomposed_dwt)
        coeffs = (cD1,cA1), (cD2,cA2), (cD3,cA3), (cD4,cA4)
        plot_swt_results(coeffs)
        # print(len(decomposed_swt))

        print(cD1)

        # recurrence_plot_generator = RecurrencePlotGenerator(signal, 3,3, epsilon=0.3,path="/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted", unthresholded=False)
        # recurrence_plot_generator.plot_recurrence(patient_id, segment_id)

        time = np.arange(len(signal)) / sampling_rate

        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=False, sharey=False)

        # plt.figure()
        ax[0].plot(time, signal)
        ax[0].set_title(f"Signal Raw Label: {label} Index {index} Patient ID {patient_id} Segment ID {segment_id}")
        time_clean = np.arange(len(signal)) / sampling_rate
        ax[1].plot(time_clean, signal)
        ax[1].set_title("Cleaned Signal")
        ax[1].set_xlabel("Time (s)")
        ax[0].set_ylabel("Amplitude")
        ax[1].set_ylabel("Amplitude (normalized)")
        plt.show()


if __name__ == "__main__":
    split = "learning"
    #279
    index = np.random.randint(20)
    file_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/paf/paf_segments_all_clean.h5"
    #650,714 index good
    plot_signal(file_path, split, 500)
