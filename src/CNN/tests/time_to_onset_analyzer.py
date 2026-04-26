"""
Time-to-Onset Analyzer - Evaluates model performance across different time windows before AF onset
"""

import json
from pathlib import Path
from typing import Dict, List

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from src.CNN.tests.model_evaluator import ModelEvaluator


class TimeToOnsetAnalyzer:
    """
    Analyzes model performance based on time-to-onset windows.

    Groups segments into 5-minute windows based on time_to_onset_sec field:
    - If time_to_onset == -1: NSR (excluded from analysis)
    - If time_to_onset == -2: Distant AF (labeled separately)
    - Otherwise: Grouped into 5-minute bins before AF onset
    """

    def __init__(self, evaluator: ModelEvaluator, window_size_sec: int = 300):
        """
        Initialize analyzer.

        Args:
            evaluator: Trained ModelEvaluator instance (already loaded with model)
            window_size_sec: Window size in seconds (default: 300 = 5 minutes)
        """
        self.evaluator = evaluator
        self.window_size = window_size_sec
        self.device = evaluator.device

        # Load time_to_onset data_old from HDF5
        self.time_to_onset = None
        self._load_time_to_onset()

        print(f"✓ TimeToOnsetAnalyzer initialized")
        print(f"  Window size: {window_size_sec}s ({window_size_sec/60:.1f} min)")

    def _load_time_to_onset(self) -> None:
        """Load time_to_onset_sec from test dataset metadata."""
        h5_path = self.evaluator.data_dir / 'test.h5'
        if not h5_path.exists():
            raise FileNotFoundError(f"Test dataset not found: {h5_path}")

        with h5py.File(h5_path, 'r') as f:
            meta_group = f.get('metadata', f)
            self.time_to_onset = meta_group['time_to_onset_sec'][:].astype(np.float32)
            self.labels = meta_group['labels'][:].astype(np.int64)

        print(f"  Loaded {len(self.time_to_onset)} time-to-onset values")

    def _create_windows(self) -> Dict[str, List[int]]:
        """
        Create time windows and assign sample indices to each window.

        For each time window, we include:
        - PreAF samples from that specific time window
        - A balanced number of randomly selected NSR samples
        - The same NSR subset is used for all windows

        Returns:
            Dictionary mapping window name to list of sample indices
        """
        windows = {}

        # Collect all NSR indices (time_to_onset == -1)
        nsr_indices = []
        for idx, t2o in enumerate(self.time_to_onset):
            if t2o == -1:
                nsr_indices.append(idx)

        # Create windows for PreAF samples
        for idx, (t2o, label) in enumerate(zip(self.time_to_onset, self.labels)):
            if t2o == -1:  # NSR - handled separately
                continue
            elif t2o == -2:  # Distant AF
                if 'Distant AF' not in windows:
                    windows['Distant AF'] = []
                windows['Distant AF'].append(idx)
            else:  # Actual time to onset
                # Group into 5-minute windows
                # e.g., [0-5), [5-10), [10-15), ..., [25-30)
                window_idx = int(t2o // self.window_size)
                window_start = window_idx * self.window_size
                window_end = (window_idx + 1) * self.window_size
                window_name = f"{window_start//60}-{window_end//60} min"

                if window_name not in windows:
                    windows[window_name] = []
                windows[window_name].append(idx)

        # Find max number of PreAF samples across all windows
        max_preaf = max(len(indices) for indices in windows.values())

        # Randomly select NSR samples (same count as max PreAF)
        # Use the same NSR subset for all windows
        np.random.seed(42)  # For reproducibility
        if len(nsr_indices) >= max_preaf:
            selected_nsr = list(np.random.choice(nsr_indices, size=max_preaf, replace=False))
        else:
            # If not enough NSR samples, use all of them
            selected_nsr = nsr_indices

        # Add selected NSR samples to each window
        for window_name in windows.keys():
            windows[window_name].extend(selected_nsr)

        # Sort windows by time (ascending)
        sorted_windows = {}

        # First add time-based windows in order
        time_windows = [(k, v) for k, v in windows.items() if k != 'Distant AF']
        time_windows.sort(key=lambda x: int(x[0].split('-')[0]))

        for name, indices in time_windows:
            sorted_windows[name] = indices

        # Add distant AF at the end
        if 'Distant AF' in windows:
            sorted_windows['Distant AF'] = windows['Distant AF']

        # Print summary
        print(f"\n✓ Created {len(sorted_windows)} windows:")
        for window_name, indices in sorted_windows.items():
            n_samples = len(indices)
            n_positive = sum(self.labels[idx] for idx in indices)
            print(f"  {window_name:15s}: {n_samples:4d} samples (PreAF={n_positive})")

        return sorted_windows

    def _evaluate_window(self, window_indices: List[int]) -> Dict[str, float]:
        """
        Evaluate model on a specific window.

        Args:
            window_indices: List of sample indices for this window

        Returns:
            Dictionary with metrics (auc, auprc)
        """
        # Create a subset of the test dataset
        subset = Subset(self.evaluator.test_dataset, window_indices)
        loader = DataLoader(subset, batch_size=32, shuffle=False, num_workers=0)

        all_probs = []
        all_labels = []

        self.evaluator.model.eval()
        with torch.no_grad():
            for batch in loader:
                if self.evaluator.model_arch == 'hybrid':
                    x_single, x_multi, batch_y = batch
                    logits = self.evaluator.model(
                        x_single.to(self.device),
                        x_multi.to(self.device)
                    )
                else:
                    batch_x, batch_y = batch
                    logits = self.evaluator.model(batch_x.to(self.device))

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch_y.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute metrics
        metrics = {}
        if len(np.unique(all_labels)) > 1:  # Need both classes for AUC
            metrics['auc'] = roc_auc_score(all_labels, all_probs)
            metrics['auprc'] = average_precision_score(all_labels, all_probs)
        else:
            metrics['auc'] = 0.5
            metrics['auprc'] = 0.5

        metrics['n_samples'] = len(all_labels)
        metrics['n_positive'] = int(np.sum(all_labels))

        return metrics

    def analyze(self) -> Dict[str, Dict[str, float]]:
        """
        Run analysis across all time windows.

        Returns:
            Dictionary mapping window name to metrics
        """
        print(f"\n{'='*70}")
        print("RUNNING TIME-TO-ONSET ANALYSIS")
        print(f"{'='*70}\n")

        windows = self._create_windows()
        results = {}

        print(f"\n{'='*70}")
        print("EVALUATING WINDOWS")
        print(f"{'='*70}\n")

        for window_name, indices in tqdm(windows.items(), desc='Windows'):
            metrics = self._evaluate_window(indices)
            results[window_name] = metrics

            print(f"\n{window_name}:")
            print(f"  Samples: {metrics['n_samples']} (PreAF={metrics['n_positive']})")
            print(f"  AUC:     {metrics['auc']:.4f}")
            print(f"  AUPRC:   {metrics['auprc']:.4f}")

        return results

    def plot_results(self, results: Dict[str, Dict[str, float]],
                     save_path: Path = None) -> None:
        """
        Plot AUC and AUPRC vs time window.

        Args:
            results: Results from analyze()
            save_path: Optional path to save the figure
        """
        # Separate time-based windows from distant AF
        time_windows = [(k, v) for k, v in results.items() if k != 'Distant AF']

        # Extract window midpoints for x-axis (in minutes)
        window_times = []
        auc_scores = []
        auprc_scores = []

        for window_name, metrics in time_windows:
            # Parse window name like "0-5 min" -> midpoint = 2.5
            parts = window_name.replace(' min', '').split('-')
            start_min = int(parts[0])
            end_min = int(parts[1])
            midpoint = (start_min + end_min) / 2.0

            window_times.append(midpoint)
            auc_scores.append(metrics['auc'])
            auprc_scores.append(metrics['auprc'])

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot 1: AUC vs Time
        ax1.plot(window_times, auc_scores, 'o-', linewidth=2.5, markersize=8,
                 color='#2ca02c', label='AUC-ROC')
        ax1.axhline(y=0.5, color='k', linestyle='--', linewidth=1.5, alpha=0.5,
                    label='Random Classifier')
        ax1.set_xlabel('Time Before AF Onset (minutes)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
        ax1.set_title('Model Performance vs Time to AF Onset', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0.4, 1.0])

        # Invert x-axis so time flows right-to-left (countdown to onset)
        ax1.invert_xaxis()

        # Plot 2: AUPRC vs Time
        ax2.plot(window_times, auprc_scores, 'o-', linewidth=2.5, markersize=8,
                 color='#ff7f0e', label='AUPRC')

        # Baseline for AUPRC (proportion of positive class)
        if time_windows:
            all_positive = sum(metrics['n_positive'] for _, metrics in time_windows)
            all_samples = sum(metrics['n_samples'] for _, metrics in time_windows)
            baseline = all_positive / all_samples
            ax2.axhline(y=baseline, color='k', linestyle='--', linewidth=1.5,
                       alpha=0.5, label=f'Random Classifier ({baseline:.3f})')

        ax2.set_xlabel('Time Before AF Onset (minutes)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('AUPRC', fontsize=12, fontweight='bold')
        ax2.set_title('Precision-Recall Performance vs Time to AF Onset',
                     fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0.4, 1.0])

        # Invert x-axis
        ax2.invert_xaxis()

        plt.tight_layout()

        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n✓ Plot saved to {save_path}")

        plt.show()

    def save_results(self, results: Dict[str, Dict[str, float]],
                     save_path: Path) -> None:
        """
        Save analysis results to JSON.

        Args:
            results: Results from analyze()
            save_path: Path to save JSON file
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        output = {
            'model_arch': self.evaluator.model_arch,
            'window_size_sec': self.window_size,
            'window_size_min': self.window_size / 60,
            'results': results
        }

        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Results saved to {save_path}")


def run_time_to_onset_analysis(
    model_path: Path,
    data_dir: Path,
    model_arch: str = 'multi',
    window_size_sec: int = 300,
    save_results_path: Path = None,
    save_plot_path: Path = None,
    seed: int = 42,
    attention_mode: str = 'trained'
) -> Dict[str, Dict[str, float]]:
    """
    Convenience function to run complete time-to-onset analysis.

    Args:
        model_path: Path to trained model checkpoint
        data_dir: Path to dataset directory
        model_arch: Model architecture ('single', 'multi', or 'hybrid')
        window_size_sec: Window size in seconds (default: 300 = 5 min)
        save_results_path: Optional path to save results JSON
        save_plot_path: Optional path to save plot
        seed: Random seed
        attention_mode: 'trained' or 'uniform'

    Returns:
        Dictionary with analysis results
    """
    # Set seed
    ModelEvaluator.set_seed(seed)

    # Create evaluator and load model
    evaluator = ModelEvaluator(model_path, data_dir, model_arch)
    evaluator.load_model(strict=True, attention_mode=attention_mode)
    evaluator.load_test_data(batch_size=32)

    # Create analyzer
    analyzer = TimeToOnsetAnalyzer(evaluator, window_size_sec=window_size_sec)

    # Run analysis
    results = analyzer.analyze()

    # Save results
    if save_results_path:
        analyzer.save_results(results, save_results_path)

    # Plot results
    if save_plot_path:
        analyzer.plot_results(results, save_plot_path)
    else:
        analyzer.plot_results(results)

    return results


if __name__ == '__main__':
    # Example usage with multi-scale model
    model_path = Path('/results_new/multi/final_checkpoint.pt')
    data_dir = Path('/data_final/utrp_dataset_db8_no_denoise_d2_d8')

    results = run_time_to_onset_analysis(
        model_path=model_path,
        data_dir=data_dir,
        model_arch='multi',
        window_size_sec=300,  # 5-minute windows
        save_results_path=Path('/results_new/time_to_onset_analysis/time_to_onset_analysis.json'),
        save_plot_path=Path('/results_new/time_to_onset_analysis/time_to_onset_plot.png'),
        attention_mode='uniform'
    )

    print("\n✓ Time-to-onset analysis complete!")