import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd


class SubbandPlotter:
    """
    Plotter class containing all visualization methods for subband validation.
    No computation logic - pure visualization only.
    """

    def __init__(self, output_dir, band_names, subband_freqs, fs):
        self.output_dir = output_dir
        self.BAND_NAMES = band_names
        self.SUBBAND_FREQS = subband_freqs
        self.FS = fs
        sns.set_theme(style="whitegrid", font_scale=1.1)

    def plot_leakage_matrices(self, df):
        """Plot leakage heatmaps for all pipelines."""
        import seaborn as sns
        import matplotlib.pyplot as plt

        pipelines = df['pipeline'].unique()
        fig, axes = plt.subplots(1, len(pipelines), figsize=(18, 5))

        for i, pipe in enumerate(pipelines):
            pipe_df = df[df['pipeline'] == pipe]
            matrix = pipe_df.groupby(['input_sine', 'output_subband'])['energy_fraction'].mean().unstack()
            order = ['D1', 'D2', 'D3', 'D4', 'A4']
            matrix = matrix.reindex(index=order, columns=order)

            sns.heatmap(matrix, annot=True, fmt=".3f", cmap="YlGnBu", ax=axes[i], cbar=(i==len(pipelines)-1))
            axes[i].set_title(f'Leakage Matrix: {pipe}')
            axes[i].set_xlabel('Output Subband')
            axes[i].set_ylabel('Input Sine Frequency')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'leakage_matrices.png'), dpi=300)
        plt.show()
        print(f"Saved: {os.path.join(self.output_dir, 'leakage_matrices.png')}")

    def visualize_absolute_metrics(self, master_df):
        """Plot absolute structural metrics (Hamming distance and RP correlation)."""
        import matplotlib.pyplot as plt

        struct_df = master_df[master_df['pipeline'] != 'dwt_direct']

        metrics = ['hamming_distance', 'rp_pattern_corr']
        pipe_colors = {
            'swt_recon': '#2ecc71',
            'swt_direct': '#3498db',
            'dwt_recon': '#e74c3c',
        }

        name_map = {
            'swt_direct': 'SWT-D',
            'swt_recon': 'SWT-R',
            'dwt_recon': 'DWT-R',
        }
        bands = ['D1', 'D2', 'D3', 'D4', 'A4']

        metrics_map = {
            'hamming_distance' : "Average Normalized Hamming Distance (smaller is better)",
            'rp_pattern_corr' : "Average RP Pearson Correlation Coefficient (bigger is better)",
        }

        metrics_x_map = {
            'hamming_distance' : "Subband levels",
            'rp_pattern_corr' : "Subband levels",
        }

        metrics_y_map = {
            'hamming_distance' : "Average Normalized Hamming Distance",
            'rp_pattern_corr' : "Average RP Pearson Correlation Coefficient",
        }

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        width = 0.25

        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            x = np.arange(len(bands))

            for i, pipe in enumerate(pipe_colors.keys()):
                data = struct_df[struct_df['pipeline'] == pipe].groupby('band')[metric].agg(['mean', 'sem']).reindex(bands)
                ax.bar(x + (i - 1) * width, data['mean'], width, yerr=data['sem'],
                       label=pipe, color=pipe_colors[pipe], edgecolor='black', capsize=3)

            ax.set_title(f' {metrics_map[metric]}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xlabel(f'{metrics_x_map[metric]}')
            ax.set_ylabel(metrics_y_map[metric])
            ax.set_xticklabels(bands)
            ax.legend(loc='upper right', frameon=True, fontsize='small')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'structural_metrics_filtered.png'), dpi=300)
        plt.show()
        print(f"Saved: {os.path.join(self.output_dir, 'structural_metrics_filtered.png')}")

    def visualize_bar_comparison_pro(self, results, master_df, reference_rqa):
        """Create professional violin plots for RQA deviations and absolute metrics."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        import os
        import numpy as np

        sns.set_theme(style="whitegrid", font_scale=1.1)

        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
        deviation_metrics = ['det_deviation', 'lam_deviation', 'tt_deviation']
        absolute_metrics = ['DET', 'LAM', 'TT']

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#34495e']
        labels = {'swt_recon': 'SWT-R', 'swt_direct': 'SWT-D', 'dwt_recon': 'DWT-R', 'dwt_direct': 'DWT-D'}
        metrics_mapping = {
            'det_deviation': 'Determinism',
            'lam_deviation': 'Laminarity',
            'tt_deviation': 'Trapping Time'
        }

        fig1, axes1 = plt.subplots(1, 3, figsize=(20, 7))

        for idx, metric in enumerate(deviation_metrics):
            ax = axes1[idx]
            deviation_data = [results[p][metric] for p in pipelines]

            parts = ax.violinplot(deviation_data, positions=range(len(pipelines)),showmeans=True, showmedians=True, widths=0.6)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)

            parts['cmedians'].set_edgecolor('#2c3e50')
            parts['cmedians'].set_linewidth(2)

            ax.set_xticks(range(len(pipelines)))
            ax.set_xticklabels([labels[p] for p in pipelines])
            ax.set_title(f'{metrics_mapping[metric]} Deviation', fontweight='bold', pad=15)
            ax.set_ylabel('Relative Deviation')

            sns.despine(ax=ax)
            ax.grid(axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pro_violin_deviations.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {os.path.join(self.output_dir, 'pro_violin_deviations.png')}")

        fig2, axes2 = plt.subplots(1, 3, figsize=(20, 7))

        abs_metrics_mapping = {'DET': 'Determinism', 'LAM': 'Laminarity', 'TT': 'Trapping Time'}

        for idx, metric in enumerate(absolute_metrics):
            ax = axes2[idx]

            # Extract data for plotting
            plot_data = [master_df[master_df['pipeline'] == p][metric].values for p in pipelines]

            # Create sleek boxplot
            bplot = ax.boxplot(plot_data, positions=range(len(pipelines)),
                               patch_artist=True, widths=0.5,
                               medianprops=dict(color="black", linewidth=1.5),
                               flierprops=dict(marker='d', markerfacecolor='gray', markersize=4, alpha=0.5))

            for i, box in enumerate(bplot['boxes']):
                box.set_facecolor(colors[i])
                box.set_alpha(0.6)
                box.set_edgecolor('black')

            # Ground Truth Reference Line
            if reference_rqa is not None:
                avg_ref = np.mean([reference_rqa[band][metric] for band in reference_rqa.keys()])
                ax.axhline(y=avg_ref, color='#e67e22', linestyle='--', linewidth=1.5,
                           label='Reference Ground Truth', alpha=0.9)
                if idx == 2:  # Only show legend on the last subplot to avoid clutter
                    ax.legend(frameon=False, loc='upper right', fontsize=10)

            ax.set_xticks(range(len(pipelines)))
            ax.set_xticklabels([labels[p] for p in pipelines])
            ax.set_title(f'Absolute {abs_metrics_mapping[metric]}', fontweight='bold', pad=15)
            ax.set_ylabel('Metric Value')

            sns.despine(ax=ax)
            ax.grid(axis='x')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'pro_boxplot_absolute.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {os.path.join(self.output_dir, 'pro_boxplot_absolute.png')}")

    def visualize_statistical_distributions(self, results):
        """Create histogram distributions for each metric."""
        import matplotlib.pyplot as plt

        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
        metrics = ['DET', 'LAM', 'ENTR', 'TT']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        colors = {
            'swt_recon': '#2ecc71',
            'swt_direct': '#3498db',
            'dwt_recon': '#e74c3c',
            'dwt_direct': '#34495e',
        }

        labels = {
            'swt_recon': 'SWT-R',
            'swt_direct': 'SWT-D',
            'dwt_recon': 'DWT-R',
            'dwt_direct': 'DWT-D'
        }

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for pipeline in pipelines:
                data = results[pipeline][metric]
                ax.hist(data, bins=15, alpha=0.5, color=colors[pipeline],label=labels[pipeline], edgecolor='black')

            ax.set_xlabel(f'{metric} Deviation', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{metric} Distribution', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'statistical_distributions.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved distribution plot: {filename}")
        plt.show()

    def visualize_snr_degradation(self, calculator, n_realizations=50, use_reference_rqa=True):
        """Visualize SNR degradation curves for RQA metrics."""
        import matplotlib.pyplot as plt
        from recurrence_plot.recurrence_plot_generation import RecurrencePlotGenerator

        reference_rqa = None
        if use_reference_rqa:
            ref_path = os.path.join(calculator.OUTPUT_DIR, 'reference_rqa_values.json')
            if os.path.exists(ref_path):
                reference_rqa = calculator.load_reference_rqa_values(ref_path)

        signal_clean = calculator.generate_composite_sine()
        snr_levels = [20, 15, 10, 5, 1, 0]
        metrics = ['det_deviation', 'lam_deviation', 'tt_deviation']

        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']

        snr_results = {p: {m: {'mean': [], 'std': []} for m in metrics} for p in pipelines}

        raw_data_for_stats = []

        ref_epsilons = {}
        for pipeline in pipelines:
            subbands_clean = calculator.decompose_signal(signal_clean, pipeline)
            ref_epsilons[pipeline] = {}

            for band in calculator.BAND_NAMES:
                freq = calculator.SUBBAND_FREQS[band]
                subband_signal = subbands_clean[band]

                if subband_signal.std() > 0:
                    subband_signal = (subband_signal - subband_signal.mean()) / subband_signal.std()

                tau, m = calculator.get_analytic_embedding_params(freq=freq, signal=subband_signal)

                ref_epsilons[pipeline][band] = RecurrencePlotGenerator.find_epsilon_for_target_rr(signal=subband_signal, tau=tau, m=m,target_rr=0.1)

        for snr_db in snr_levels:
            for pipeline in pipelines:
                mean_df, std_df = calculator.validate_with_noise(
                    signal_clean, pipeline, ref_epsilons[pipeline], snr_db, n_realizations, reference_rqa)

                for metric in metrics:
                    snr_results[pipeline][metric]['mean'].append(mean_df[metric].mean())
                    snr_results[pipeline][metric]['std'].append(std_df[metric].mean())

                for _, row in mean_df.iterrows():
                    for metric in metrics:
                        raw_data_for_stats.append({
                            'pipeline': pipeline,
                            'snr_db': snr_db,
                            'metric': metric,
                            'deviation': row[metric]
                        })

        colors = {
            'swt_recon': '#2ecc71',
            'swt_direct': '#3498db',
            'dwt_recon': '#e74c3c',
            'dwt_direct': '#34495e',
        }

        labels = {
            'swt_recon': 'SWT-R',
            'swt_direct': 'SWT-D',
            'dwt_recon': 'DWT-R',
            'dwt_direct': 'DWT-D'
        }

        fig, axes = plt.subplots(1, 5, figsize=(22, 5))

        for idx, metric in enumerate(metrics):
            ax = axes[idx]

            for pipeline in pipelines:
                means = np.array(snr_results[pipeline][metric]['mean'])
                stds = np.array(snr_results[pipeline][metric]['std'])

                ax.plot(snr_levels, means, marker='o', linewidth=2, markersize=8,color=colors[pipeline], label=labels[pipeline])

                ax.fill_between(snr_levels, means - stds, means + stds,color=colors[pipeline], alpha=0.2)

            ax.set_xlabel('SNR (dB)', fontsize=12, fontweight='bold')
            ax.set_ylabel(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            ax.set_title(f'{metric.replace("_", " ").title()} vs SNR', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)
            ax.invert_xaxis()

        plt.tight_layout()
        filename = os.path.join(self.output_dir, 'snr_degradation_curves.png')
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved SNR degradation curves: {filename}")
        plt.show()

        return pd.DataFrame(raw_data_for_stats)

    def visualize_frobenius_distances(self, master_df):
        """Create comprehensive visualizations for Frobenius distances."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_theme(style="whitegrid", font_scale=1.1)

        pipelines = ['swt_recon', 'swt_direct', 'dwt_recon', 'dwt_direct']
        bands = ['D1', 'D2', 'D3', 'D4', 'A4']

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#34495e']
        labels = {'swt_recon': 'SWT-R', 'swt_direct': 'SWT-D', 'dwt_recon': 'DWT-R', 'dwt_direct': 'DWT-D'}

        # Filter out NaN values
        df_filtered = master_df[master_df['frobenius_distance'].notna()].copy()

        # Plot 1: Frobenius distances across subbands for each pipeline
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 7))

        width = 0.2
        x = np.arange(len(bands))

        for i, pipe in enumerate(pipelines):
            pipe_df = df_filtered[df_filtered['pipeline'] == pipe]
            means = []
            sems = []
            for band in bands:
                band_data = pipe_df[pipe_df['band'] == band]['frobenius_distance']
                if len(band_data) > 0:
                    means.append(band_data.mean())
                    sems.append(band_data.sem())
                else:
                    means.append(0)
                    sems.append(0)

            ax1.bar(x + i * width, means, width, yerr=sems,
                   label=labels[pipe], color=colors[i], alpha=0.7,
                   edgecolor='black', capsize=4)

        ax1.set_xlabel('Subband', fontweight='bold')
        ax1.set_ylabel('Frobenius Distance', fontweight='bold')
        ax1.set_title('Frobenius Distance Across Subbands', fontweight='bold', pad=15)
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(bands)
        ax1.legend(frameon=True)
        ax1.grid(axis='y', alpha=0.3)
        sns.despine(ax=ax1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frobenius_subbands.png'), dpi=300, bbox_inches='tight')
        plt.close(fig1)
        print(f"Saved: {os.path.join(self.output_dir, 'frobenius_subbands.png')}")

        # Plot 2: Violin plot showing distribution of Frobenius distances for each pipeline
        fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))

        plot_data = []
        plot_labels = []
        for pipe in pipelines:
            pipe_data = df_filtered[df_filtered['pipeline'] == pipe]['frobenius_distance'].values
            if len(pipe_data) > 0:
                plot_data.append(pipe_data)
                plot_labels.append(labels[pipe])

        if len(plot_data) > 0:
            parts = ax2.violinplot(plot_data, positions=range(len(plot_labels)),
                                  showmeans=True, showmedians=True, widths=0.6)

            for i, pc in enumerate(parts['bodies']):
                pc.set_facecolor(colors[i])
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.2)

            parts['cmedians'].set_edgecolor('#2c3e50')
            parts['cmedians'].set_linewidth(2)
            parts['cmeans'].set_edgecolor('red')
            parts['cmeans'].set_linewidth(2)

            ax2.set_xticks(range(len(plot_labels)))
            ax2.set_xticklabels(plot_labels)
            ax2.set_ylabel('Frobenius Distance', fontweight='bold')
            ax2.set_title('Frobenius Distance Distribution by Pipeline', fontweight='bold', pad=15)
            ax2.grid(axis='y', alpha=0.3)
            ax2.set_ylim(bottom=0)
            sns.despine(ax=ax2)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frobenius_distributions.png'), dpi=300, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved: {os.path.join(self.output_dir, 'frobenius_distributions.png')}")

        # Plot 3: Heatmap showing average Frobenius distances
        fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

        heatmap_data = np.zeros((len(pipelines), len(bands)))
        for i, pipe in enumerate(pipelines):
            for j, band in enumerate(bands):
                band_data = df_filtered[(df_filtered['pipeline'] == pipe) & (df_filtered['band'] == band)]['frobenius_distance']
                if len(band_data) > 0:
                    heatmap_data[i, j] = band_data.mean()
                else:
                    heatmap_data[i, j] = np.nan

        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='YlOrRd',
                    xticklabels=bands, yticklabels=[labels[p] for p in pipelines],
                    ax=ax3, cbar_kws={'label': 'Frobenius Distance'})
        ax3.set_title('Average Frobenius Distance Heatmap', fontweight='bold', pad=15)
        ax3.set_xlabel('Subband', fontweight='bold')
        ax3.set_ylabel('Pipeline', fontweight='bold')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'frobenius_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved: {os.path.join(self.output_dir, 'frobenius_heatmap.png')}")

    def plot_deviation_leakage(self, df_der, output_dir):
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_der, x='pipeline', y='DER')
        plt.title("Diagonal Energy Ratio (higher = better)")
        plt.ylabel("DER")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "leakage_der.png"))
        plt.show()

    def visualize_shift_invariance(self, df_alias):
        """Visualize shift invariance test results."""
        os.makedirs(self.output_dir, exist_ok=True)
        sns.set_theme(style="whitegrid", font_scale=1.2)

        df_alias_summary = df_alias.groupby(['pipeline', 'trial'])[['hamming_shift_diff', 'frobenius_shift_diff']].mean().reset_index()

        # Plot: Hamming Distance
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_alias_summary, x='pipeline', y='hamming_shift_diff')
        plt.title("Shift Invariance Tests - Hamming Distance (lower = better)")
        plt.ylabel("Hamming Distance")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "aliasing_hamming.png"), dpi=300)
        plt.show()
        print(f"Saved: {os.path.join(self.output_dir, 'aliasing_hamming.png')}")

        # Plot: Frobenius
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df_alias_summary, x='pipeline', y='frobenius_shift_diff')
        plt.title("Shift Invariance Tests - Frobenius Distance (lower = better)")
        plt.ylabel("Frobenius Distance")
        plt.xlabel("Pipeline")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "aliasing_frobenius.png"), dpi=300)
        plt.show()
        print(f"Saved: {os.path.join(self.output_dir, 'aliasing_frobenius.png')}")

    def plot_cross_band_summary(self, master_df):
        """Plot average cross-band correlation from master_df."""
        fig, ax = plt.subplots(figsize=(10, 6))

        colors = ['#2ecc71', '#3498db', '#e74c3c', '#34495e']
        pipe_labels = {'swt_recon': 'SWT-R', 'swt_direct': 'SWT-D', 'dwt_recon': 'DWT-R', 'dwt_direct': 'DWT-D'}

        sns.boxplot(data=master_df, x='pipeline', y='avg_offdiag_corr',
                    palette=colors, ax=ax)

        ax.set_title("Average Cross-Band Correlation (Lower = Better Isolation)", fontweight='bold')
        ax.set_ylabel("Average Off-Diagonal Correlation")
        ax.set_xlabel("Pipeline")
        ax.set_xticklabels([pipe_labels[p] for p in master_df['pipeline'].unique()])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cross_band_summary.png'), dpi=300)
        plt.show()
        print(f"Saved: {os.path.join(self.output_dir, 'cross_band_summary.png')}")