import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec
import seaborn as sns
import os
import numpy as np


class ValidatorVisualizer:
    def __init__(self, output_dir, band_names, subband_freqs, fs):
        self.output_dir = output_dir
        self.BAND_NAMES = band_names
        self.SUBBAND_FREQS = subband_freqs
        self.FS = fs
        sns.set_theme(style="whitegrid", font_scale=1.1)
        self.colors = ['#2ecc71', '#3498db', '#e74c3c', '#34495e']
        self.pipe_labels = {'swt_recon': 'SWT-R', 'swt_direct': 'SWT-D', 'dwt_recon': 'DWT-R', 'dwt_direct': 'DWT-D'}

    def plot_leakage_matrices(self, df):
        pipelines = df['pipeline'].unique()
        fig, axes = plt.subplots(1, len(pipelines), figsize=(20, 5))
        order = ['D1', 'D2', 'D3', 'D4', 'A4']
        pipeline_mapping = {
            'swt_recon': 'SWT-R',
            'swt_direct': 'SWT-D',
            'dwt_recon': 'DWT-R',
            'dwt_direct': 'DWT-D',
        }
        for i, pipe in enumerate(pipelines):
            matrix = df[df['pipeline'] == pipe].groupby(['input_sine', 'output_subband'])[
                'energy_fraction'].mean().unstack().reindex(index=order, columns=order)
            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[i], cbar=(i == len(pipelines) - 1))
            axes[i].set_title(f'{pipeline_mapping[pipe]}')
            axes[i].set_xlabel('Output Subband')
            axes[i].set_ylabel('Input Sine')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'leakage_matrices.png'))
        plt.show()

    def visualize_absolute_metrics(self, master_df):
        metrics = ['hamming_distance', 'rp_pattern_corr']
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        width = 0.2
        df_filtered = master_df[master_df['pipeline'] != 'dwt_direct'].copy()
        valid_pipes = ['swt_recon', 'swt_direct', 'dwt_recon']

        # print(f"DEBUG: master_df shape = {master_df.shape}")
        # print(f"DEBUG: df_filtered shape = {df_filtered.shape}")
        # print(f"DEBUG: hamming_distance non-null count = {df_filtered['hamming_distance'].notna().sum()}")
        # print(f"DEBUG: rp_pattern_corr non-null count = {df_filtered['rp_pattern_corr'].notna().sum()}")

        for idx, m in enumerate(metrics):
            for i, pipe in enumerate(valid_pipes):
                pipe_data = df_filtered[df_filtered['pipeline'] == pipe].copy()
                grouped = pipe_data.groupby('band')[m].agg(['mean', 'std']).reindex(self.BAND_NAMES)

                # print(f"DEBUG: {pipe} - {m}")
                print(grouped)

                # Replace NaN sem with 0 for plotting
                yerr = grouped['std'].fillna(0)

                axes[idx].bar(np.arange(len(self.BAND_NAMES)) + (i - 1) * width, grouped['mean'], width, yerr=yerr,
                              label=self.pipe_labels[pipe], color=self.colors[i], edgecolor='black', capsize=3)
            axes[idx].set_title(f'Global Average {m.replace("_", " ").title()}')
            axes[idx].set_xticks(np.arange(len(self.BAND_NAMES)))
            axes[idx].set_xticklabels(self.BAND_NAMES)
            axes[idx].legend(loc='upper right')
            axes[idx].set_ylim(bottom=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'structural_metrics_filtered.png'), dpi=300)
        plt.show()

    def visualize_bar_comparison_pro(self, results, reference_rqa):
        pipes = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        # Plot 1: Absolute values with reference lines
        absolute_metrics = ['DET', 'LAM', 'TT']
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        for idx, m in enumerate(absolute_metrics):
            data = [results[p][m] for p in pipes]
            parts = axes[idx].violinplot(data, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(self.colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.6)

            # Add reference value line (average across all bands)
            ref_values = [reference_rqa[band][m] for band in self.BAND_NAMES]
            ref_mean = np.mean(ref_values)
            axes[idx].axhline(y=ref_mean, color='red', linestyle='--', linewidth=2, label=f'Reference Mean: {ref_mean:.3f}')

            axes[idx].set_xticks(range(1, len(pipes) + 1))
            axes[idx].set_xticklabels([self.pipe_labels[p] for p in pipes])
            axes[idx].set_title(f'{m} Distribution', fontweight='bold')
            axes[idx].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'violin_group_0.png'), dpi=300)
        plt.show()

        # Plot 2: Deviation metrics
        deviation_metrics = ['det_deviation', 'lam_deviation', 'tt_deviation']
        metric_labels = ['Determinism Deviation', 'Laminarity Deviation', 'Trapping Time Deviation']
        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        for idx, (m, label) in enumerate(zip(deviation_metrics, metric_labels)):
            data = [results[p][m] for p in pipes]
            parts = axes[idx].violinplot(data, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(self.colors[i])
                pc.set_edgecolor('black')
                pc.set_alpha(0.6)
            axes[idx].set_xticks(range(1, len(pipes) + 1))
            axes[idx].set_xticklabels([self.pipe_labels[p] for p in pipes])
            axes[idx].set_title(label, fontweight='bold')
            axes[idx].set_ylabel('Relative Deviation')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pro_violin_deviations.png'), dpi=300)
        plt.show()

    def visualize_frobenius_distances(self, master_df):
        fig, ax = plt.subplots(figsize=(12, 7))
        sns.barplot(data=master_df, x='band', y='frobenius_distance', hue='pipeline', palette=self.colors, ax=ax,
                    capsize=.1)
        ax.set_title('Frobenius Distance Across Subbands', fontweight='bold')
        plt.savefig(os.path.join(self.output_dir, 'frobenius_subbands.png'))
        plt.show()

    def visualize_single_scale_rp(self, signal_clean, noisy_signal, utrp, rp, rqa, tau, m, eps, snr_label):
        fig = plt.figure(figsize=(20, 7), constrained_layout=True)
        gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1])
        fig.suptitle(f'Single-Scale Analysis: {snr_label}', fontsize=18, fontweight='bold')

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(np.arange(len(noisy_signal)) / self.FS, noisy_signal, linewidth=1.0)
        ax1.set_title('Composite Time Series', fontweight='bold')
        ax1.set_xlabel('Time (s)');
        ax1.set_ylabel('Amplitude')

        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(utrp, cmap='magma', aspect='equal')
        ax2.set_title('UTRP (Distance Matrix)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(rp, cmap='binary', aspect='equal')
        ax3.set_title(f'Binary RP (DET: {rqa.determinism:.3f})' if rqa else 'Binary RP', fontweight='bold')

        for ax in [ax2, ax3]: ax.set_xlabel('Time index $j$'); ax.set_ylabel('Time index $i$')
        plt.savefig(os.path.join(self.output_dir, f'single_scale_{snr_label}.png'))
        plt.show()

    def plot_cross_band_summary(self, master_df):
        """Plot average cross-band correlation from master_df"""
        fig, ax = plt.subplots(figsize=(10, 6))

        sns.boxplot(data=master_df, x='pipeline', y='avg_offdiag_corr',
                    palette=self.colors, ax=ax)

        ax.set_title("Average Cross-Band Correlation (Lower = Better Isolation)", fontweight='bold')
        ax.set_ylabel("Average Off-Diagonal Correlation")
        ax.set_xlabel("Pipeline")
        ax.set_xticklabels([self.pipe_labels[p] for p in master_df['pipeline'].unique()])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_band_summary.png'), dpi=300)
        plt.show()

    def plot_shift_invariance(self, df):
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        sns.boxplot(data=df, x='pipeline', y='hamming_shift_diff',
                    palette=self.colors, ax=axes[0])
        axes[0].set_title("Shift Invariance (Hamming)")

        sns.boxplot(data=df, x='pipeline', y='frobenius_shift_diff',
                    palette=self.colors, ax=axes[1])
        axes[1].set_title("Shift Invariance (Frobenius)")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'shift_invariance.png'))
        plt.show()

    def visualize_decomposition_with_rps(self, pipeline_name, subbands, utrps, rps, rqas, ref_params, snr_label):
        n = len(self.BAND_NAMES)
        fig = plt.figure(figsize=(22, 4 * n), constrained_layout=True)
        gs = GridSpec(n, 3, figure=fig, width_ratios=[1.2, 1, 1])
        fig.suptitle(f'Decomposition: {pipeline_name} | {snr_label}', fontsize=20, fontweight='bold')

        for i, band in enumerate(self.BAND_NAMES):
            ax1 = fig.add_subplot(gs[i, 0])
            ax1.plot(subbands[band], linewidth=0.8)
            ax1.set_title(f'Subband {band} ({self.SUBBAND_FREQS[band]} Hz)', fontweight='bold')

            ax2 = fig.add_subplot(gs[i, 1])
            ax2.imshow(utrps[band], cmap='magma', aspect='equal')
            ax2.set_title(f'UTRP {band}', fontweight='bold')

            ax3 = fig.add_subplot(gs[i, 2])
            ax3.imshow(rps[band], cmap='binary', aspect='equal')
            det = rqas[band].determinism if rqas[band] else 0.0
            ax3.set_title(f'Binary RP (DET: {det:.3f})', fontweight='bold')

        plt.savefig(os.path.join(self.output_dir, f'decomp_{pipeline_name}_{snr_label}.png'))
        plt.show()

    def visualize_snr_degradation(self, snr_results):
        fig, axes = plt.subplots(1, 3, figsize=(22, 5))
        metrics = ['det_deviation', 'lam_deviation', 'tt_deviation']
        snr_levels = sorted(snr_results['swt_recon']['det_deviation']['levels'])

        for idx, m in enumerate(metrics):
            for pipe, color in zip(self.pipe_labels.keys(), self.colors):
                means = snr_results[pipe][m]['mean']
                stds = snr_results[pipe][m]['std']
                axes[idx].plot(snr_levels, means, marker='o', label=self.pipe_labels[pipe], color=color)
                axes[idx].fill_between(snr_levels, np.array(means) - np.array(stds), np.array(means) + np.array(stds),
                                       color=color, alpha=0.1)
            axes[idx].set_title(f'{m.replace("_", " ").title()} vs SNR')
            axes[idx].invert_xaxis()
            axes[idx].legend()
        plt.savefig(os.path.join(self.output_dir, 'snr_degradation.png'))
        plt.show()

    def visualize_invariance(self, df_alias, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.2)

        df_alias_summary = df_alias.groupby(['pipeline', 'trial'])[['hamming_shift_diff', 'frobenius_shift_diff']].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_alias_summary, x='pipeline', y='hamming_shift_diff')
        plt.title("Shift Invariance Tests - Hamming Distance (lower = better)")
        plt.ylabel("Hamming Distance")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aliasing_hamming.png"))
        plt.show()

        # --- Plot: Frobenius ---
        sns.boxplot(data=df_alias_summary, x='pipeline', y='frobenius_shift_diff')
        plt.title("Shift Invariance Tests - Frobenius Distance (lower = better)")
        plt.ylabel("Frobenius Distance")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "aliasing_frobenius.png"))
        plt.show()

    def plot_deviation_leakage(self, df_der, output_dir):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_der, x='pipeline', y='DER')
        plt.title("Diagonal Energy Ratio (higher = better)")
        plt.ylabel("DER")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "leakage_der.png"))
        plt.show()

    def plot_leakage(self, df_leakage, output_dir):
        band_order = ['D1', 'D2', 'D3', 'D4', 'A4']
        pipelines = df_leakage['pipeline'].unique()

        fig, axes = plt.subplots(1, len(pipelines), figsize=(20, 5))

        for i, pipe in enumerate(pipelines):
            sub = df_leakage[df_leakage['pipeline'] == pipe]

            matrix = sub.groupby(['input_sine', 'output_subband'])['energy_fraction'] \
                .mean().unstack().reindex(index=band_order, columns=band_order)

            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu",
                        vmin=0, vmax=1,
                        ax=axes[i],
                        cbar=(i == len(pipelines) - 1))

            axes[i].set_title(pipe)

        plt.suptitle("Leakage Heatmaps")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "leakage_heatmaps.png"))
        plt.show()

