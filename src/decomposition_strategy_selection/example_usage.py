from src.decomposition_strategy_selection import (
    ExperimentConfig,
    run_experiment,
)


def main():

    method = "HARMONIC"
    # method = "NON_HARMONIC"

    subband_freq = {}

    if method == "HARMONIC":
        subband_freq =  {
            'A4': 2,
            'D4': 6,    # 4-8 Hz
            'D3': 12,   # 8-16 Hz
            'D2': 24,   # 16-32 Hz
            'D1': 48    # 32-64 Hz
        }
    elif method == "NON_HARMONIC":
        subband_freq = {
            'A4': 3,
            'D4': 7,    # 4-8 Hz
            'D3': 13,   # 8-16 Hz
            'D2': 31,   # 16-32 Hz
            'D1': 61    # 32-64 Hz
        }

    config = ExperimentConfig(
        fs=128,
        n=1024,
        level=4,
        wavelet='db4',
        subband_freqs=subband_freq,
        output_dir=f'results_{method}'
    )

    # define band names and pipelines
    band_names = ['D1', 'D2', 'D3', 'D4', 'A4']
    pipeline_names = ['swt_recon', 'swt_direct', 'dwt_recon']

    # run experiment
    agg_df, stats_df = run_experiment(
        config=config,
        band_names=band_names,
        pipeline_names=pipeline_names,
        n_trials=100
    )
    print("experiment finished")

if __name__ == '__main__':
    main()
