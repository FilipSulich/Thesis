import h5py
import numpy as np
import matplotlib.pyplot as plt

# path = '/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/mitbih/mitbih_af_segments.h5'
# path = '/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/ptbxl/ptbxl_normal_segments.h5'
path = '/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/combined_250hz.h5'

with h5py.File(path, 'r') as f:
    signals = f['signals'][:]
    labels = f['labels'][:]
    lead_times = f['lead_times'][:]
    sources = f['sources'][:]
    patient_ids = [s.decode() if isinstance(s, bytes) else s for s in f['patient_ids'][:]]

fig, axes = plt.subplots(5, 1, figsize=(12, 8))
fs = 250
t = np.arange(signals.shape[1]) / fs
for ax in axes:
    i = np.random.randint(0, len(signals))
    ax.plot(t, signals[i], 'b-', linewidth=0.5)
    ax.set_ylabel('Amplitude')
    ax.set_title(f'Segment {i}: Patient {patient_ids[i]}, Lead time: {lead_times[i]} min, label: {labels[i]}, source: {sources[i]}')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Time (seconds)')
plt.tight_layout()
plt.savefig('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/ecg_samples_combined.png', dpi=150)