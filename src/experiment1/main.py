from src.experiment1.subband_validation import SubbandValidator
from src.experiment1.visualizer import ValidatorVisualizer
import os
import numpy as np
import pandas as pd


def run_full_experiment(is_sine=True, n_trials=100):
    calc = SubbandValidator(is_sine=is_sine, is_lorenz=not is_sine)
    viz = ValidatorVisualizer(calc.OUTPUT_DIR, calc.BAND_NAMES, calc.SUBBAND_FREQS, calc.FS)

    ref_path = os.path.join(calc.OUTPUT_DIR, 'reference_rqa_values.json')
    if not os.path.exists(ref_path):
        ref_rqa = calc.generate_reference_rqa_values(ref_path)
    else:
        ref_rqa = calc.load_reference_rqa_values(ref_path)
    ref_data = calc.load_reference_data(ref_path.replace('.json', '_data.npz'))


    results = {p: {m: [] for m in ['DET', 'LAM', 'TT', 'hamming_distance', 'ibf', 'rp_pattern_corr', 'frobenius_distance']} for p in calc.PIPELINES}
    all_trial_data = []

    for trial in range(n_trials):
        print('Trial {}'.format(trial))
        sig = calc.generate_composite_sine(random_phase=True)
        for pipe in calc.PIPELINES:
            df, _ = calc.validate_pipeline(sig, pipe, ref_rqa, ref_data)
            all_trial_data.append(df)
            for m in results[pipe]:
                results[pipe][m].extend(df[m].dropna().tolist())

    master_df = pd.concat(all_trial_data, ignore_index=True)
    for p in calc.PIPELINES:
        for m in results[p]: results[p][m] = np.array(results[p][m])

    # leakage_df = calc.run_leakage_analysis(n_trials=20)
    # viz.plot_leakage_matrices(leakage_df)
    viz.visualize_absolute_metrics(master_df)
    viz.visualize_bar_comparison_pro(results, ref_rqa)
    viz.visualize_frobenius_distances(master_df)

if __name__ == '__main__':
    n_trials = 100
    run_full_experiment(is_sine=True, n_trials=n_trials)