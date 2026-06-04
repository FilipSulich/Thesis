from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


class LeakagePlotter:
    """creates visualizations for wavelet leakage analysis results"""

    def __init__(self, output_dir):
        """
        initializes plotter with output directory for saving figures

        Args:
            output_dir (Path): path to directory for saving plots

        Returns:
            None
        """
        self.output_dir = Path(output_dir)

    def plot_der_comparison(self, df: pd.DataFrame) -> None:
        """
        creates side-by-side boxplot comparing energy ratios for harmonic vs non-harmonic conditions

        Args:
            df (pd.DataFrame): results dataframe with columns [wavelet, condition, DER]

        Returns:
            None
        """

        def wavelet_key(w):
            family = "".join([c for c in w if not c.isdigit()])
            order = int("".join([c for c in w if c.isdigit()]))
            family_map = {'db': 0, 'sym': 1, 'coif': 2}
            return (family_map.get(family, 3), order)

        sorted_wavelets = sorted(df['wavelet'].unique(), key=wavelet_key)

        colors = {
            'Harmonic': '#1f77b4',
            'Non-harmonic': '#d62728'
        }

        positions = []
        data = []
        box_colors = []

        for i, wavelet in enumerate(sorted_wavelets):
            base_pos = i * 3

            for j, condition in enumerate(['Harmonic', 'Non-harmonic']):
                condition_data = df[
                    (df['wavelet'] == wavelet) &
                    (df['condition'] == condition)
                ]['DER'].values

                if len(condition_data) > 0:
                    positions.append(base_pos + j * 0.6)
                    data.append(condition_data)
                    box_colors.append(colors[condition])

        fig, ax = plt.subplots(figsize=(16, 7))

        box = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.4,
            showfliers=False
        )

        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        x_ticks = [i * 3 + 0.3 for i in range(len(sorted_wavelets))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            sorted_wavelets,
            rotation=45,
            ha='right',
            fontsize=11
        )

        ax.set_xlabel('Wavelet Function', fontsize=14, fontweight='bold')
        ax.set_ylabel('Energy Ratio (ER)', fontsize=14, fontweight='bold')
        ax.set_title('Energy Leakage Analysis', fontsize=16, fontweight='bold', pad=20)
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(alpha=0.3, linestyle='--')

        legend_elements = [
            Patch(facecolor=colors['Harmonic'], alpha=0.35, edgecolor='black', label='Harmonic'),
            Patch(facecolor=colors['Non-harmonic'], alpha=0.35,edgecolor='black', label='Non-harmonic')
        ]

        ax.legend(handles=legend_elements, loc='lower right', fontsize=12)

        plt.tight_layout()
        plot_path = self.output_dir / 'der_boxplot.pdf'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.close()