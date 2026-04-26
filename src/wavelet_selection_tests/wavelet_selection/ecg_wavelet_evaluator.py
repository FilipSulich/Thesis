import warnings
from pathlib import Path
from typing import Dict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ── Constants ──────────────────────────────────────────────────────────────────
WAVELETS = (
    [f'db{i}'   for i in range(2, 9)] +   # db2-db10
    [f'sym{i}'  for i in range(2, 9)] +   # sym2-sym10
    [f'coif{i}' for i in range(1, 4)]      # coif1-coif5
)

LEVELS = list(range(2, 9))  # D2-D8


class ECGWaveletEvaluator:
    """Simplified wavelet evaluation for mean/median AUC analysis."""

    def __init__(self, signals_train: np.ndarray, labels_train: np.ndarray,
                 signals_test: np.ndarray, labels_test: np.ndarray,
                 sampling_rate: int = 500):
        """
        Args:
            signals_train: (N_train, n_samples) float array
            labels_train: (N_train,) int array (0=NSR, 1=pre-AF)
            signals_test: (N_test, n_samples) float array
            labels_test: (N_test,) int array (0=NSR, 1=pre-AF)
            sampling_rate: ECG sampling rate in Hz
        """
        self.signals_train = signals_train.astype(np.float32)
        self.labels_train = labels_train.astype(np.int32)
        self.signals_test = signals_test.astype(np.float32)
        self.labels_test = labels_test.astype(np.int32)
        self.fs = sampling_rate
        self.wavelets = WAVELETS
        self.levels = LEVELS

        # Results: {wavelet: {level_name: auc}}
        self.auc_matrix: Dict[str, Dict[str, float]] = {}

    def evaluate_from_cache(self, h5_path: Path) -> Dict[str, Dict[str, float]]:
        """Load precomputed RQA features and evaluate AUC."""
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"Cache file not found: {h5_path}")

        print(f"\n✓ Loading and evaluating from cache: {h5_path}\n")

        lr = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        scaler = StandardScaler()

        with h5py.File(h5_path, 'r') as f:
            for i, wavelet in enumerate(self.wavelets):
                print(f"  [{i+1:2d}/{len(self.wavelets)}] {wavelet:<8}", flush=True)

                level_aucs = {}
                for level in self.levels:
                    name = f'D{level}'
                    print(f"      {name}...", end=' ', flush=True)

                    try:
                        # Load precomputed features
                        X_train = f[f'train/{wavelet}/level_{level}/features'][:]
                        X_test = f[f'test/{wavelet}/level_{level}/features'][:]

                        # Skip degenerate cases
                        if np.all(X_train == 0) or np.all(X_test == 0):
                            level_aucs[name] = 0.5
                            print(f"(degenerate)")
                            continue

                        # Train and evaluate
                        X_train_scaled = scaler.fit_transform(X_train)
                        lr.fit(X_train_scaled, self.labels_train[:len(X_train)])

                        X_test_scaled = scaler.transform(X_test)
                        y_pred_proba = lr.predict_proba(X_test_scaled)[:, 1]
                        auc = roc_auc_score(self.labels_test[:len(X_test)], y_pred_proba)

                        level_aucs[name] = float(auc)
                        print(f"AUC={auc:.4f}")

                    except Exception as e:
                        print(f"FAILED: {e}")
                        level_aucs[name] = 0.5

                self.auc_matrix[wavelet] = level_aucs
                print()

        return self.auc_matrix

    def auc_dataframe(self) -> pd.DataFrame:
        """Return (wavelets × levels) AUC matrix as DataFrame."""
        level_names = [f'D{l}' for l in self.levels]
        rows = {}
        for wavelet in self.wavelets:
            if wavelet in self.auc_matrix:
                rows[wavelet] = [self.auc_matrix[wavelet].get(l, np.nan)
                                for l in level_names]
        return pd.DataFrame(rows, index=level_names).T

    def stats_per_wavelet(self) -> pd.DataFrame:
        """Compute mean, median, std AUC across all levels for each wavelet."""
        stats_data = []

        for wavelet in self.wavelets:
            if wavelet in self.auc_matrix:
                aucs = list(self.auc_matrix[wavelet].values())
                aucs = [a for a in aucs if not np.isnan(a)]

                if aucs:
                    stats_data.append({
                        'Wavelet': wavelet,
                        'Mean AUC': float(np.mean(aucs)),
                        'Median AUC': float(np.median(aucs)),
                        'Std AUC': float(np.std(aucs))
                    })

        df = pd.DataFrame(stats_data)
        df = df.sort_values('Mean AUC', ascending=False).reset_index(drop=True)
        return df

    def save_stats_json(self, save_path: Path) -> None:
        """Save statistics to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.stats_per_wavelet()
        
        output = {
            'summary': {
                'num_wavelets': len(df),
                'num_levels': len(self.levels),
                'levels': [f'D{l}' for l in self.levels]
            },
            'results': df.to_dict('records')
        }

        import json
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Saved statistics to {save_path}")

    def plot_mean_auc_bars(self, save_path: Path = None) -> plt.Figure:
        """Bar chart: Mean AUC per wavelet with Standard Deviation error bars."""
        df = self.stats_per_wavelet()
        if df.empty:
            print("No results to plot.")
            return None

        # Increased height slightly to accommodate error bar caps
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.4)))

        # Plotting with xerr for Standard Deviation
        bars = ax.barh(df['Wavelet'], df['Mean AUC'],
                       # xerr=df['Std AUC'],
                       color='steelblue',
                       edgecolor='black',
                       linewidth=0.5,
                       capsize=4,
                       error_kw={'elinewidth': 1.2, 'ecolor': 'black', 'alpha': 0.7})

        ax.axvline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5, label='Chance')

        # Scale starting at 0
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Mean AUC', fontsize=11, fontweight='bold')
        ax.set_ylabel('Wavelet', fontsize=11, fontweight='bold')
        ax.set_title('Mean AUC per Wavelet (D2–D8)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')

        # Add value labels (positioned after the error bar)
        for i, val in enumerate(df['Mean AUC']):
            ax.text(val+ 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved mean AUC bars → {save_path}")
        return fig

    def plot_auc_violin(self, save_path: Path = None) -> plt.Figure:
        """
        Generates a violin plot showing the distribution of AUC scores
        across all decomposition levels for each wavelet.
        """
        if not self.auc_matrix:
            print("No results to plot.")
            return None

        # 1. Convert nested dict {wavelet: {level: auc}} to long-form DataFrame
        records = []
        for wavelet, levels in self.auc_matrix.items():
            for level_name, auc in levels.items():
                records.append({'Wavelet': wavelet, 'Level': level_name, 'AUC': auc})
        df = pd.DataFrame(records)

        # 2. Determine sort order based on mean AUC
        order = df.groupby('Wavelet')['AUC'].mean().sort_values(ascending=False).index

        # 3. Create the plot
        fig, ax = plt.subplots(figsize=(12, 7))

        # 'inner="box"' shows the median and quartiles inside the violin
        sns.violinplot(data=df, x='Wavelet', y='AUC', order=order,
                       palette='coolwarm', inner='box', ax=ax, cut=0)

        # 4. Aesthetics and Constraints
        ax.axhline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.6, label='Chance')

        ax.set_ylim(0, 1.0)  # Start at 0 as requested
        ax.set_title('AUC Distribution across Decomposition Levels (D2–D8)',
                     fontsize=14, fontweight='bold')
        ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('Wavelet Type', fontsize=12, fontweight='bold')

        ax.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=45)
        ax.legend(loc='lower right')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved AUC violin plot → {save_path}")
        return fig

    def plot_median_auc_bars(self, save_path: Path = None) -> plt.Figure:
        """Bar chart: Median AUC per wavelet starting at 0."""
        df = self.stats_per_wavelet()
        df = df.sort_values('Median AUC', ascending=False).reset_index(drop=True)

        if df.empty:
            print("No results to plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.4)))

        bars = ax.barh(df['Wavelet'], df['Median AUC'], color='coral', edgecolor='black', linewidth=0.5)
        ax.axvline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5, label='Chance')

        # Scale starting at 0
        ax.set_xlim(0, 1.0)
        ax.set_xlabel('Median AUC', fontsize=11, fontweight='bold')
        ax.set_ylabel('Wavelet', fontsize=11, fontweight='bold')
        ax.set_title('Median AUC per Wavelet (D2–D8)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        ax.legend(loc='lower right')

        # Add value labels
        for i, val in enumerate(df['Median AUC']):
            ax.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved median AUC bars → {save_path}")
        return fig

    def plot_level_wavelet_heatmap(self, save_path: Path = None) -> plt.Figure:
        """Simple heatmap: wavelet × decomposition level."""
        df = self.auc_dataframe()
        if df.empty:
            print("No results to plot.")
            return None

        fig, ax = plt.subplots(figsize=(10, max(6, len(self.wavelets) * 0.35)))

        sns.heatmap(df, ax=ax, annot=True, fmt='.3f', cmap='RdYlGn',
                   vmin=0.5, vmax=0.85, linewidths=0.5,
                   cbar_kws={'label': 'AUC'})

        ax.set_title('Wavelet vs. Decomposition Level AUC', fontsize=13, fontweight='bold')
        ax.set_xlabel('Decomposition Level', fontsize=11)
        ax.set_ylabel('Wavelet', fontsize=11)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved heatmap → {save_path}")
        return fig

    # def plot_mean_auc_bars(self, save_path: Path = None) -> plt.Figure:
    #     """Bar chart: Mean AUC per wavelet."""
    #     df = self.stats_per_wavelet()
    #     if df.empty:
    #         print("No results to plot.")
    #         return None
    #
    #     fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3)))
    #
    #     bars = ax.barh(df['Wavelet'], df['Mean AUC'], color='steelblue', edgecolor='black', linewidth=0.5)
    #     ax.set_xlim(0, 1.0)
    #     ax.axvline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5, label='Chance')
    #     ax.set_xlim(0.45, 0.85)
    #     ax.set_xlabel('Mean AUC', fontsize=11, fontweight='bold')
    #     ax.set_ylabel('Wavelet', fontsize=11, fontweight='bold')
    #     ax.set_title('Mean AUC per Wavelet (D2–D8)', fontsize=13, fontweight='bold')
    #     ax.grid(axis='x', alpha=0.3)
    #     ax.legend()
    #
    #     # Add value labels
    #     for i, (wav, val) in enumerate(zip(df['Wavelet'], df['Mean AUC'])):
    #         ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)
    #
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"✓ Saved mean AUC bars → {save_path}")
    #     return fig

    # def plot_median_auc_bars(self, save_path: Path = None) -> plt.Figure:
    #     """Bar chart: Median AUC per wavelet."""
    #     df = self.stats_per_wavelet()
    #     df = df.sort_values('Median AUC', ascending=False).reset_index(drop=True)
    #
    #     if df.empty:
    #         print("No results to plot.")
    #         return None
    #
    #     fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3)))
    #
    #     bars = ax.barh(df['Wavelet'], df['Median AUC'], color='coral', edgecolor='black', linewidth=0.5)
    #     ax.axvline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5, label='Chance')
    #     ax.set_xlim(0.45, 0.85)
    #     ax.set_xlabel('Median AUC', fontsize=11, fontweight='bold')
    #     ax.set_ylabel('Wavelet', fontsize=11, fontweight='bold')
    #     ax.set_title('Median AUC per Wavelet (D2–D8)', fontsize=13, fontweight='bold')
    #     ax.grid(axis='x', alpha=0.3)
    #     ax.legend()
    #
    #     # Add value labels
    #     for i, (wav, val) in enumerate(zip(df['Wavelet'], df['Median AUC'])):
    #         ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)
    #
    #     plt.tight_layout()
    #     if save_path:
    #         plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #         print(f"✓ Saved median AUC bars → {save_path}")
    #     return fig

    def plot_d5_auc_bars(self, save_path: Path = None) -> plt.Figure:
        """Bar chart: AUC per wavelet for D5 subband only."""
        d5_aucs = []
        
        for wavelet in self.wavelets:
            if wavelet in self.auc_matrix and 'D5' in self.auc_matrix[wavelet]:
                auc = self.auc_matrix[wavelet]['D5']
                d5_aucs.append({'Wavelet': wavelet, 'AUC': auc})
        
        if not d5_aucs:
            print("No D5 data_old found.")
            return None
        
        df = pd.DataFrame(d5_aucs)
        df = df.sort_values('AUC', ascending=False).reset_index(drop=True)
        
        fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.3)))
        
        bars = ax.barh(df['Wavelet'], df['AUC'], color='mediumseagreen', edgecolor='black', linewidth=0.5)
        ax.axvline(0.5, color='red', linewidth=1.5, linestyle='--', alpha=0.5, label='Chance')
        ax.set_xlim(0.45, 0.85)
        ax.set_xlabel('AUC', fontsize=11, fontweight='bold')
        ax.set_ylabel('Wavelet', fontsize=11, fontweight='bold')
        ax.set_title('AUC per Wavelet (D5 Subband Only)', fontsize=13, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        ax.legend()
        
        # Add value labels
        for i, (wav, val) in enumerate(zip(df['Wavelet'], df['AUC'])):
            ax.text(val + 0.005, i, f'{val:.4f}', va='center', fontsize=9)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved D5 AUC bars → {save_path}")
        return fig

    def compute_rqa_metrics_by_class(self, h5_path: Path) -> pd.DataFrame:
        """
        Compute average RQA metrics for Pre-AF and NSR groups.
        
        Analyzes db2 and db8 wavelets only, across decomposition levels D2-D8.
        The 9 RQA metrics are: DET, LAM, REC, ENTR, TT, DEN, L, Lmax, DIV
        
        Returns:
            DataFrame with structure:
            - Index: (wavelet, level)
            - Columns: metric (DET, LAM, REC, ...) with subcolumns (NSR_mean, PreAF_mean, Diff)
        """
        h5_path = Path(h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {h5_path}")
        
        print(f"\n✓ Computing RQA metrics by class from: {h5_path}\n")
        
        # 9 standard RQA metrics
        RQA_METRICS = ['DET', 'LAM', 'REC', 'ENTR', 'TT', 'DEN', 'L', 'Lmax', 'DIV']
        wavelets_to_analyze = ['db2', 'db8']
        
        results = []
        
        with h5py.File(h5_path, 'r') as f:
            for wavelet in wavelets_to_analyze:
                if wavelet not in f['train'].keys():
                    print(f"  ⚠ {wavelet} not found in HDF5, skipping...")
                    continue
                
                print(f"  Processing {wavelet}...")
                
                for level in self.levels:
                    level_name = f'D{level}'
                    try:
                        # Load test features (used as main analysis)
                        # The HDF5 file contains test set features with corresponding labels
                        X_test = f[f'test/{wavelet}/level_{level}/features'][:]
                        labels_test_level = self.labels_test[:len(X_test)]
                        
                        # Skip degenerate cases
                        if X_test.shape[0] == 0 or np.all(X_test == 0):
                            print(f"    {level_name}: degenerate data_old, skipping")
                            continue
                        
                        # Ensure we have enough metrics
                        n_metrics = X_test.shape[1]
                        if n_metrics < 9:
                            print(f"    {level_name}: only {n_metrics} metrics (need 9), skipping")
                            continue
                        
                        # Use first 9 columns for the 9 RQA metrics
                        X_test = X_test[:, :9]
                        
                        # Split by class
                        nsr_mask = labels_test_level == 0
                        preaf_mask = labels_test_level == 1
                        
                        X_nsr = X_test[nsr_mask]
                        X_preaf = X_test[preaf_mask]
                        
                        if X_nsr.shape[0] == 0 or X_preaf.shape[0] == 0:
                            print(f"    {level_name}: empty class, skipping")
                            continue
                        
                        # Compute means for each metric
                        nsr_means = np.mean(X_nsr, axis=0)
                        preaf_means = np.mean(X_preaf, axis=0)
                        diff = preaf_means - nsr_means
                        
                        # Create result row
                        row_dict = {
                            'Wavelet': wavelet,
                            'Level': level_name,
                            'NSR_samples': X_nsr.shape[0],
                            'PreAF_samples': X_preaf.shape[0],
                        }
                        
                        # Add metrics
                        for i, metric_name in enumerate(RQA_METRICS):
                            row_dict[f'{metric_name}_NSR'] = nsr_means[i]
                            row_dict[f'{metric_name}_PreAF'] = preaf_means[i]
                            row_dict[f'{metric_name}_Diff'] = diff[i]
                        
                        results.append(row_dict)
                        print(f"    {level_name}: NSR={X_nsr.shape[0]}, PreAF={X_preaf.shape[0]} ✓")
                        
                    except Exception as e:
                        print(f"    {level_name}: Error - {e}")
                        continue
        
        if not results:
            print("  ✗ No results computed!")
            return pd.DataFrame()
        
        df = pd.DataFrame(results)
        print(f"\n✓ Successfully computed metrics for {len(df)} level-wavelet combinations")
        return df

    def save_rqa_metrics_json(self, df: pd.DataFrame, save_path: Path) -> None:
        """Save RQA metrics DataFrame to JSON."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dict with better structure
        output = {
            'summary': {
                'total_levels': len(df),
                'wavelets': sorted(df['Wavelet'].unique().tolist()),
                'levels': sorted(df['Level'].unique().tolist()),
                'rqa_metrics': ['DET', 'LAM', 'REC', 'ENTR', 'TT', 'DEN', 'L', 'Lmax', 'DIV']
            },
            'data_old': df.to_dict('records')
        }
        
        import json
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"✓ Saved RQA metrics to {save_path}")

    def plot_rqa_metrics_comparison(self, df: pd.DataFrame, save_path: Path = None) -> plt.Figure:
        """
        Plot comparison of RQA metrics between Pre-AF and NSR for each level.
        
        Creates subplots for db2 and db8, showing all 9 metrics.
        """
        if df.empty:
            print("No data_old to plot")
            return None
        
        RQA_METRICS = ['DET', 'LAM', 'REC', 'ENTR', 'TT', 'DEN', 'L', 'Lmax', 'DIV']
        wavelets = sorted(df['Wavelet'].unique())
        
        fig, axes = plt.subplots(len(wavelets), 1, figsize=(14, 5 * len(wavelets)))
        if len(wavelets) == 1:
            axes = [axes]
        
        for ax_idx, wavelet in enumerate(wavelets):
            df_wav = df[df['Wavelet'] == wavelet].sort_values('Level')
            levels = df_wav['Level'].tolist()
            
            x = np.arange(len(levels))
            width = 0.08
            
            for metric_idx, metric in enumerate(RQA_METRICS):
                offset = (metric_idx - len(RQA_METRICS) / 2) * width
                nsr_vals = df_wav[f'{metric}_NSR'].values
                preaf_vals = df_wav[f'{metric}_PreAF'].values
                
                # Plot as lines to show trend across levels
                ax = axes[ax_idx]
                ax.plot(x + offset, nsr_vals, marker='o', linestyle='-', linewidth=1.5,
                       label=f'{metric}_NSR', alpha=0.7)
                ax.plot(x + offset, preaf_vals, marker='s', linestyle='--', linewidth=1.5,
                       label=f'{metric}_PreAF', alpha=0.7)
            
            axes[ax_idx].set_xlabel('Decomposition Level', fontsize=11, fontweight='bold')
            axes[ax_idx].set_ylabel('Metric Value', fontsize=11, fontweight='bold')
            axes[ax_idx].set_title(f'RQA Metrics Comparison - {wavelet}', fontsize=12, fontweight='bold')
            axes[ax_idx].set_xticks(x)
            axes[ax_idx].set_xticklabels(levels)
            axes[ax_idx].grid(True, alpha=0.3)
            axes[ax_idx].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved RQA metrics comparison → {save_path}")
        return fig


if __name__ == '__main__':
    import json
    import wfdb
    from scipy import signal as sp_signal

    DATA_DIR = Path('/data_old/paf')
    SPLIT_JSON = Path('/data_old/data_new/attention/utrp_new_denoise2.split.json')
    OUT_DIR = Path('/src/wavelet_selection_tests/wavelet_selection/results')
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    FS_ORIG = 128
    FS_TARGET = 500
    SEGMENT_SAMPLES = FS_TARGET * 10
    N_SEGMENTS = 5

    # Load records
    with open(SPLIT_JSON) as f:
        splits = json.load(f)
    train_records = set(splits['train'])
    test_records = set(splits['test'])

    def load_segments(record_id: str, label: int):
        try:
            rec = wfdb.rdrecord(str(DATA_DIR / record_id))
            raw = rec.p_signal[:, 0].astype(np.float64)
            n_tgt = int(len(raw) * FS_TARGET / FS_ORIG)
            res = sp_signal.resample(raw, n_tgt)
            max_start = len(res) - SEGMENT_SAMPLES
            if max_start <= 0:
                return [], []
            starts = np.linspace(0, max_start, N_SEGMENTS, dtype=int)
            segs, lbls = [], []
            for s in starts:
                seg = res[s:s + SEGMENT_SAMPLES]
                if len(seg) < SEGMENT_SAMPLES:
                    continue
                seg = seg.astype(np.float32)
                seg = (seg - seg.min()) / (seg.max() - seg.min() + 1e-9)
                segs.append(seg)
                lbls.append(label)
            return segs, lbls
        except Exception as e:
            print(f"  [WARN] {record_id}: {e}")
            return [], []

    nsr_records = ([f'n{i:02d}' for i in range(1, 51)] +
                   [f'n{i:02d}c' for i in range(1, 51)])
    preaf_records = [f'p{i:02d}' for i in range(1, 51)]

    # Load training segments
    signals_train, labels_train = [], []
    print("Loading training segments...")
    for rec_id in nsr_records:
        if rec_id not in train_records:
            continue
        segs, lbls = load_segments(rec_id, 0)
        signals_train.extend(segs); labels_train.extend(lbls)

    for rec_id in preaf_records:
        if rec_id not in train_records:
            continue
        segs, lbls = load_segments(rec_id, 1)
        signals_train.extend(segs); labels_train.extend(lbls)

    # Load test segments
    signals_test, labels_test = [], []
    print("Loading test segments...")
    for rec_id in nsr_records:
        if rec_id not in test_records:
            continue
        segs, lbls = load_segments(rec_id, 0)
        signals_test.extend(segs); labels_test.extend(lbls)

    for rec_id in preaf_records:
        if rec_id not in test_records:
            continue
        segs, lbls = load_segments(rec_id, 1)
        signals_test.extend(segs); labels_test.extend(lbls)

    X_train = np.array(signals_train, dtype=np.float32)
    y_train = np.array(labels_train, dtype=np.int32)
    X_test = np.array(signals_test, dtype=np.float32)
    y_test = np.array(labels_test, dtype=np.int32)

    print(f"Train: {len(X_train)} segments (NSR={int((y_train==0).sum())}  Pre-AF={int((y_train==1).sum())})")
    print(f"Test:  {len(X_test)} segments (NSR={int((y_test==0).sum())}  Pre-AF={int((y_test==1).sum())})")

    # Evaluate
    evaluator = ECGWaveletEvaluator(X_train, y_train, X_test, y_test, sampling_rate=FS_TARGET)
    h5_path = OUT_DIR / 'data_old.h5'

    evaluator.evaluate_from_cache(h5_path)

    # Generate outputs
    print("\n" + "="*70)
    print("GENERATING RESULTS")
    print("="*70)

    # Statistics DataFrame
    stats_df = evaluator.stats_per_wavelet()
    print("\n✓ Statistics (Mean, Median, Std AUC per Wavelet):")
    print(stats_df.to_string(index=False))

    # Save JSON
    # evaluator.save_stats_json(OUT_DIR / 'wavelet_statistics.json')
    # Generate plots
    # evaluator.plot_level_wavelet_heatmap(save_path=OUT_DIR / 'heatmap_wavelet_level.png')

    evaluator.plot_mean_auc_bars(save_path=OUT_DIR / 'bars_mean_auc.png')

    evaluator.plot_auc_violin(save_path=OUT_DIR / 'auc_violin.png')

    evaluator.plot_median_auc_bars(save_path=OUT_DIR / 'bars_median_auc.png')

    # evaluator.plot_d5_auc_bars(save_path=OUT_DIR / 'bars_d5_auc.png')

    # Compute and save RQA metrics
    # rqa_df = evaluator.compute_rqa_metrics_by_class(h5_path)
    # if not rqa_df.empty:
    #     evaluator.save_rqa_metrics_json(rqa_df, OUT_DIR / 'rqa_metrics.json')
    #     evaluator.plot_rqa_metrics_comparison(rqa_df, save_path=OUT_DIR / 'rqa_metrics_comparison.png')
    #
    # print(f"\n✓ All results saved → {OUT_DIR}\n")

