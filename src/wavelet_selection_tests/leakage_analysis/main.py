import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.wavelet_selection_tests.leakage_analysis.leakage_tester import LeakageTester
from src.wavelet_selection_tests.leakage_analysis.leakage_plotter import LeakagePlotter
from src.wavelet_selection_tests.leakage_analysis.leakage_stats import LeakageStats


if __name__ == '__main__':
    output_dir = Path(__file__).parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    tester = LeakageTester(
        fs=128,
        n_samples=1280,
        level=4,
        output_dir=output_dir
    )

    print("running leakage tests")

    df = tester.run_all(
        wavelets=[
            # Daubechies
            'db2', 'db3', 'db4', 'db5', 'db6', 'db7', 'db8',
            # Symlets
            'sym2', 'sym3', 'sym4', 'sym5', 'sym6', 'sym7', 'sym8',
            # Coiflets
            'coif1', 'coif2', 'coif3'
        ],
        n_trials=100
    )
    print("plotting")
    plotter = LeakagePlotter(output_dir)
    plotter.plot_der_comparison(df)

    print(f"saving stats to {output_dir}")
    stats = LeakageStats(output_dir)
    stats.save_all_stats(df)
    stats.create_db8_comparison_table(df)