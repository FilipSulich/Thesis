"""
Utility script to inspect the UTRP dataset.

Usage:
    python src/CNN/inspect_dataset.py
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def inspect_hdf5(hdf5_path: Path):
    """Print dataset structure and statistics."""

    print("\n" + "="*80)
    print("UTRP DATASET INSPECTION")
    print("="*80)
    print(f"\nFile: {hdf5_path}")

    with h5py.File(hdf5_path, 'r') as f:
        print(f"\nGlobal Attributes:")
        for key, value in f.attrs.items():
            print(f"  {key}: {value}")

        print(f"\nDataset Structure:")

        # Single-scale
        if 'single_scale/utrps' in f:
            single_scale = f['single_scale/utrps']
            print(f"\n  single_scale/utrps:")
            print(f"    Shape: {single_scale.shape}")
            print(f"    Dtype: {single_scale.dtype}")
            print(f"    Size: {single_scale.nbytes / (1024**2):.2f} MB")

        # Multi-scale
        if 'multi_scale/utrps' in f:
            multi_scale = f['multi_scale/utrps']
            print(f"\n  multi_scale/utrps:")
            print(f"    Shape: {multi_scale.shape}")
            print(f"    Dtype: {multi_scale.dtype}")
            print(f"    Size: {multi_scale.nbytes / (1024**2):.2f} MB")

        # Metadata
        print(f"\n  metadata/:")
        labels = f['metadata/labels'][:]
        patient_ids = [pid.decode('utf-8') for pid in f['metadata/record_ids'][:]]
        time_to_onset = f['metadata/time_to_onset'][:]
        # embedding_tau = f['metadata/embedding_tau'][:]
        # embedding_m = f['metadata/embedding_m'][:]

        print(f"    labels: {labels.shape}, {labels.dtype}")
        print(f"    patient_ids: {len(set(patient_ids))} unique patients")
        print(f"    time_to_onset: {time_to_onset.shape}, {time_to_onset.dtype}")
        # print(f"    embedding_tau: {embedding_tau.shape}, {embedding_tau.dtype}")
        # print(f"    embedding_m: {embedding_m.shape}, {embedding_m.dtype}")

        # Class distribution
        print(f"\nClass Distribution:")
        unique, counts = np.unique(labels, return_counts=True)
        for cls, cnt in zip(unique, counts):
            cls_name = 'NSR' if cls == 0 else 'Pre-AF'
            print(f"  {cls_name} (label={cls}): {cnt} samples")

        # Patient distribution
        print(f"\nPatient Distribution:")
        patient_to_label = {}
        for pid, label in zip(patient_ids, labels):
            if pid not in patient_to_label:
                patient_to_label[pid] = label

        nsr_patients = sum(1 for label in patient_to_label.values() if label == 0)
        preaf_patients = sum(1 for label in patient_to_label.values() if label == 1)
        print(f"  NSR patients: {nsr_patients}")
        print(f"  Pre-AF patients: {preaf_patients}")

        # Time to onset statistics (pre-AF only)
        preaf_time_to_onset = time_to_onset[labels == 1]
        preaf_time_to_onset = preaf_time_to_onset[~np.isnan(preaf_time_to_onset)]

        if len(preaf_time_to_onset) > 0:
            print(f"\nTime to AF Onset (Pre-AF segments):")
            print(f"  Min: {preaf_time_to_onset.min()/60:.2f} minutes")
            print(f"  Max: {preaf_time_to_onset.max()/60:.2f} minutes")
            print(f"  Mean: {preaf_time_to_onset.mean()/60:.2f} ± {preaf_time_to_onset.std()/60:.2f} minutes")
            print(f"  Median: {np.median(preaf_time_to_onset)/60:.2f} minutes")



def plot_examples(hdf5_path: Path, n_examples: int = 4):
    """Plot example UTRPs from each class."""

    print(f"\n{'='*80}")
    print("PLOTTING EXAMPLE UTRPs")
    print(f"{'='*80}")

    with h5py.File(hdf5_path, 'r') as f:
        single_scale_utrps = f['single_scale/utrps'][:]
        multi_scale_utrps = f['multi_scale/utrps'][:]
        labels = f['metadata/labels'][:]

        # Find examples from each class
        nsr_indices = np.where(labels == 0)[0]
        preaf_indices = np.where(labels == 1)[0]

        # Sample random examples
        np.random.seed(42)
        nsr_samples = np.random.choice(nsr_indices, min(n_examples, len(nsr_indices)), replace=False)
        preaf_samples = np.random.choice(preaf_indices, min(n_examples, len(preaf_indices)), replace=False)

        # Plot single-scale
        fig, axes = plt.subplots(2, n_examples, figsize=(3*n_examples, 6))
        fig.suptitle('Single-Scale UTRPs', fontsize=14, y=1.02)

        for i, idx in enumerate(nsr_samples):
            # single_scale_utrps stored as (3, H, W) -> transpose to (H, W, 3)
            img = single_scale_utrps[idx].transpose(1, 2, 0)
            axes[0, i].imshow(img, cmap='viridis')
            axes[0, i].set_title(f'NSR #{idx}')
            axes[0, i].axis('off')

        for i, idx in enumerate(preaf_samples):
            img = single_scale_utrps[idx].transpose(1, 2, 0)
            axes[1, i].imshow(img, cmap='viridis')
            axes[1, i].set_title(f'Pre-AF #{idx}')
            axes[1, i].axis('off')

        plt.tight_layout()
        single_path = save_path / 'single.png'
        plt.savefig(single_path, dpi=150, bbox_inches='tight')
        print(f"Single-scale examples saved to {single_path}")
        plt.close()

        # Plot multi-scale (show all subbands for one example from each class)
        # Support both (N, n_subbands, 3, H, W) and (N, C, H, W) layouts
        if multi_scale_utrps.ndim == 5:
            n_subbands = multi_scale_utrps.shape[1]
            get_subband_img = lambda arr, idx, j: arr[idx, j].transpose(1, 2, 0)
        else:
            # (N, C, H, W)
            channels = multi_scale_utrps.shape[1]
            if channels % 3 == 0:
                n_subbands = channels // 3
                get_subband_img = lambda arr, idx, j: arr[idx, j*3:(j+1)*3].transpose(1, 2, 0)
            else:
                # Fall back: treat each channel as a grayscale subband
                n_subbands = channels
                get_subband_img = lambda arr, idx, j: arr[idx, j]

        fig, axes = plt.subplots(2, n_subbands, figsize=(3*n_subbands, 6))
        fig.suptitle('Multi-Scale UTRPs (All Subbands)', fontsize=14, y=1.02)

        # Subband names (fallback to D1..Dn)
        subband_names = [f'D{i+1}' for i in range(n_subbands)]

        # NSR example
        nsr_idx = nsr_samples[0]
        for j in range(n_subbands):
            img = get_subband_img(multi_scale_utrps, nsr_idx, j)
            # img shape may be (H,W,3) or (H,W) depending on layout
            if img.ndim == 3:
                axes[0, j].imshow(img, cmap='viridis')
            else:
                axes[0, j].imshow(img, cmap='viridis')
            axes[0, j].set_title(f'NSR - {subband_names[j]}')
            axes[0, j].axis('off')

        # Pre-AF example
        preaf_idx = preaf_samples[0]
        for j in range(n_subbands):
            img = get_subband_img(multi_scale_utrps, preaf_idx, j)
            if img.ndim == 3:
                axes[1, j].imshow(img, cmap='viridis')
            else:
                axes[1, j].imshow(img, cmap='viridis')
            axes[1, j].set_title(f'Pre-AF - {subband_names[j]}')
            axes[1, j].axis('off')

        plt.tight_layout()
        multi_path =save_path / 'multi.png'
        plt.savefig(multi_path, dpi=150, bbox_inches='tight')
        print(f"Multi-scale examples saved to {multi_path}")
        plt.close()


if __name__ == '__main__':
    hdf5_path = Path('/data/data_new/attention/utrp_new_denoise2.h5')
    save_path = Path('/data/data_new/plots')
    if not hdf5_path.exists():
        print(f"\nError: Dataset not found at {hdf5_path}")
        print("Run dataset_generator.py first to create the dataset.")
    else:
        inspect_hdf5(hdf5_path)
        plot_examples(hdf5_path, n_examples=5)

    print(f"\n{'='*80}")
    print("INSPECTION COMPLETE")
    print(f"{'='*80}\n")

    # Check if mean UTRP value correlates with heart rate
    # (which would confirm the shortcut learning hypothesis)
    import h5py
    import numpy as np
    from scipy import stats


    #
    # with h5py.File(hdf5_path, 'r') as f:
    #     ss_utrps = f['single_scale/utrps'][:]
    #     labels = f['metadata/labels'][:]
    #
    # # Mean value of each UTRP
    # means = ss_utrps.reshape(len(ss_utrps), -1).mean(axis=1)
    #
    # # Variance of each UTRP (proxy for recurrence density variability)
    # stds = ss_utrps.reshape(len(ss_utrps), -1).std(axis=1)
    #
    # # How much does mean value vary within the same patient?
    # with h5py.File(hdf5_path, 'r') as f:
    #     record_ids = [r.decode() for r in f['metadata/record_ids'][:]]
    #
    # # Compute within-patient std of UTRP mean
    # patient_means = {}
    # for i, rid in enumerate(record_ids):
    #     if rid not in patient_means:
    #         patient_means[rid] = []
    #     patient_means[rid].append(means[i])
    #
    # within_patient_std = np.mean([np.std(v) for v in patient_means.values()])
    # between_patient_std = np.std([np.mean(v) for v in patient_means.values()])
    #
    # print(f"Within-patient std of UTRP mean:   {within_patient_std:.4f}")
    # print(f"Between-patient std of UTRP mean:  {between_patient_std:.4f}")
    # print(f"Ratio (>1 means between >> within): {between_patient_std / within_patient_std:.2f}")