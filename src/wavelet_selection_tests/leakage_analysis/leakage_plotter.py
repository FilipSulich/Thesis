from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


class LeakagePlotter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)

    def plot_der_comparison(self, df: pd.DataFrame) -> None:
        """
        Side-by-side boxplots for harmonic vs non-harmonic.

        Args:
            df: DataFrame with columns [wavelet, condition, DER]
        """

        # Sort wavelets by family (db, sym, coif) then order
        def wavelet_key(w):
            family = "".join([c for c in w if not c.isdigit()])
            order = int("".join([c for c in w if c.isdigit()]))
            family_map = {'db': 0, 'sym': 1, 'coif': 2}
            return (family_map.get(family, 3), order)

        sorted_wavelets = sorted(df['wavelet'].unique(), key=wavelet_key)

        # Colors for conditions
        colors = {
            'Harmonic': '#1f77b4',
            'Non-harmonic': '#d62728'
        }

        # Build boxplot data
        positions = []
        data = []
        box_colors = []

        for i, wavelet in enumerate(sorted_wavelets):
            base_pos = i * 3  # spacing between wavelets

            for j, condition in enumerate(['Harmonic', 'Non-harmonic']):
                condition_data = df[
                    (df['wavelet'] == wavelet) &
                    (df['condition'] == condition)
                ]['DER'].values

                if len(condition_data) > 0:
                    positions.append(base_pos + j * 0.6)
                    data.append(condition_data)
                    box_colors.append(colors[condition])

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 7))

        box = ax.boxplot(
            data,
            positions=positions,
            patch_artist=True,
            widths=0.4,
            showfliers=False
        )

        # Color boxes
        for patch, color in zip(box['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.35)

        # X-axis labels (centered between two boxes)
        x_ticks = [i * 3 + 0.3 for i in range(len(sorted_wavelets))]
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(
            sorted_wavelets,
            rotation=45,
            ha='right',
            fontsize=11
        )

        ax.set_xlabel('Wavelet Function', fontsize=12, fontweight='bold')
        ax.set_ylabel('Energy Ratio (ER)', fontsize=12, fontweight='bold')
        ax.set_title('Energy Leakage Analysis', fontsize=13, fontweight='bold', pad=20)
        ax.grid(alpha=0.3, linestyle='--')

        # Legend
        legend_elements = [
            Patch(facecolor=colors['Harmonic'], alpha=0.35,
                  edgecolor='black', label='Harmonic'),
            Patch(facecolor=colors['Non-harmonic'], alpha=0.35,
                  edgecolor='black', label='Non-harmonic')
        ]

        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)

        plt.tight_layout()

        plot_path = self.output_dir / 'der_boxplot.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')

        plot_path_pdf = self.output_dir / 'der_boxplot.pdf'
        plt.savefig(plot_path_pdf, bbox_inches='tight')

        plt.close()