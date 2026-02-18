import numpy as np
from scipy.spatial.distance import pdist
from pyrqa.time_series import TimeSeries
from pyrqa.settings import Settings
from pyrqa.analysis_type import Classic
from pyrqa.neighbourhood import FixedRadius
from pyrqa.metric import EuclideanMetric
from pyrqa.computation import RQAComputation
import matplotlib.pyplot as plt

def get_epsilon_for_rr(signal, embedding_dim, time_delay, target_rr):
    """Find epsilon that achieves target recurrence rate for a signal."""
    N = len(signal) - (embedding_dim - 1) * time_delay
    embedded = np.array([
        signal[i:i + embedding_dim * time_delay:time_delay]
        for i in range(N)
    ])
    distances = pdist(embedded, metric='euclidean')
    return np.quantile(distances, target_rr)


def compute_rqa_with_epsilon(signal, embedding_dim, time_delay, epsilon):
    ts = TimeSeries(signal, embedding_dimension=embedding_dim, time_delay=time_delay)
    settings = Settings(
        time_series=ts,
        analysis_type=Classic,
        neighbourhood=FixedRadius(epsilon),
        similarity_measure=EuclideanMetric,
        theiler_corrector=1
    )
    result = RQAComputation.create(settings, verbose=False).run()
    return {
        'RR':  result.recurrence_rate,
        'DET': result.determinism,
        'LAM': result.laminarity,
        'L':   result.average_diagonal_line,
        'TT':  result.trapping_time,
    }


def run_sensitivity_analysis(af_segments, normal_segments, embedding_dim, time_delay,
                              target_rrs=(0.02, 0.05, 0.10), n_segments=10):
    """
    For each target RR level:
      1. Compute median epsilon across training AF+normal segments
      2. Compute RQA metrics for all segments
      3. Report mean metrics and AF vs normal separation

    af_segments, normal_segments: lists of 1D numpy arrays
    n_segments: how many segments to use from each class (keep small for speed)
    """
    af_segs    = af_segments[:n_segments]
    norm_segs  = normal_segments[:n_segments]
    all_segs   = af_segs + norm_segs
    labels     = ['AF'] * len(af_segs) + ['Normal'] * len(norm_segs)

    results = {}

    for target_rr in target_rrs:
        print(f"\n--- Target RR = {target_rr*100:.0f}% ---")

        # Step 1: find median epsilon across all training segments
        epsilons = [get_epsilon_for_rr(s, embedding_dim, time_delay, target_rr)
                    for s in all_segs]
        epsilon = np.median(epsilons)
        print(f"  Median epsilon: {epsilon:.4f}")

        # Step 2: compute RQA metrics for every segment with this fixed epsilon
        metrics_af, metrics_norm = [], []
        for seg, label in zip(all_segs, labels):
            m = compute_rqa_with_epsilon(seg, embedding_dim, time_delay, epsilon)
            if label == 'AF':
                metrics_af.append(m)
            else:
                metrics_norm.append(m)

        # Step 3: summarise
        metric_names = ['RR', 'DET', 'LAM', 'L', 'TT']
        print(f"  {'Metric':<8} {'AF mean':>10} {'Normal mean':>12} {'Difference':>12}")
        print(f"  {'-'*44}")

        rr_summary = {}
        for mn in metric_names:
            af_vals   = [m[mn] for m in metrics_af]
            norm_vals = [m[mn] for m in metrics_norm]
            af_mean   = np.mean(af_vals)
            norm_mean = np.mean(norm_vals)
            diff      = abs(af_mean - norm_mean)
            print(f"  {mn:<8} {af_mean:>10.4f} {norm_mean:>12.4f} {diff:>12.4f}")
            rr_summary[mn] = {'af': af_mean, 'norm': norm_mean, 'diff': diff}

        results[target_rr] = {'epsilon': epsilon, 'metrics': rr_summary}

    return results


def plot_sensitivity_results(results, metric='DET'):
    rr_levels = sorted(results.keys())
    af_vals   = [results[rr]['metrics'][metric]['af']   for rr in rr_levels]
    norm_vals = [results[rr]['metrics'][metric]['norm'] for rr in rr_levels]
    diffs     = [results[rr]['metrics'][metric]['diff'] for rr in rr_levels]

    x = [f"{rr*100:.0f}%" for rr in rr_levels]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(x, af_vals,   'o-', label='AF',     color='red')
    ax1.plot(x, norm_vals, 'o-', label='Normal', color='blue')
    ax1.set_title(f'{metric} by RR level')
    ax1.set_xlabel('Target Recurrence Rate')
    ax1.set_ylabel(metric)
    ax1.legend()

    ax2.bar(x, diffs, color='steelblue')
    ax2.set_title(f'|AF - Normal| {metric} separation')
    ax2.set_xlabel('Target Recurrence Rate')
    ax2.set_ylabel('Absolute difference')

    plt.tight_layout()
    plt.savefig(f'sensitivity_{metric}.png', dpi=150)
    plt.show()


import pandas as pd
df = pd.read_csv('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/extracted/matched_5min_50.csv')
signal_cols = [c for c in df.columns if c.startswith('s') and c[1:].isdigit()]

af_segments   = [df[df.is_normal == False].iloc[i][signal_cols].values.astype(np.float64)
                 for i in range(len(df[df.is_normal == False]))]

norm_segments = [df[df.is_normal == True].iloc[i][signal_cols].values.astype(np.float64)
                 for i in range(len(df[df.is_normal == True]))]

results = run_sensitivity_analysis(
    af_segments=af_segments,
    normal_segments=norm_segments,
    embedding_dim=3,
    time_delay=19,
    target_rrs=(0.02, 0.05, 0.10),
    n_segments=10
)

plot_sensitivity_results(results, metric='DET')
plot_sensitivity_results(results, metric='LAM')