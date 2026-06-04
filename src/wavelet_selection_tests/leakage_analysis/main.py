import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.wavelet_selection_tests.leakage_analysis.leakage_tester import LeakageTester
from src.wavelet_selection_tests.leakage_analysis.leakage_plotter import LeakagePlotter
from src.wavelet_selection_tests.leakage_analysis.leakage_stats import LeakageStats

DATA_ROOT = Path(__file__).parent.parent.parent.parent

output_dir = DATA_ROOT / 'results' / 'leakage_analysis'
output_dir.mkdir(parents=True, exist_ok=True)


def main():
    """
    main function to run the leakage analysis
    """
    print("running leakage tests...")

    tester = LeakageTester(fs=128, n_samples=1280, level=4, output_dir=output_dir)
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


    #plots
    plotter = LeakagePlotter(output_dir)
    plotter.plot_der_comparison(df)

    #stats
    stats = LeakageStats(output_dir)
    stats.save_all_stats(df)
    print(f"saving data to {output_dir}")

if __name__ == '__main__':
   main()