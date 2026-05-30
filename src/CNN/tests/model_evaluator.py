import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.CNN.models.models import FullBandCNN, MultiScaleCNN
from src.CNN.utils.datasets import CombinedH5Dataset
from src.CNN.utils.evaluation import evaluate_model_folds
from src.CNN.utils.statistics import compare_models
from src.CNN.utils.plotting import plot_roc_and_pr_combined, plot_metric_boxplots
from src.CNN.utils.utils import DEVICE


DATA_ROOT = Path(__file__).parent.parent.parent.parent

H5_DIR = DATA_ROOT / 'data' / 'processed_data' / 'urp_no_denoise'
H5_DIR_DENOISED= DATA_ROOT / 'data' / 'processed_data' / 'urp_denoise'

RESULTS_BASE = DATA_ROOT / 'results_final' / 'models_final_weights'

RESULTS_SINGLE = RESULTS_BASE / 'single'

RESULTS_MULTI = RESULTS_BASE / 'multi'
RESULTS_MULTI_DENOISED = RESULTS_BASE / 'multi_denoise'

RESULTS_HIGH_FREQ = RESULTS_BASE / 'ablation' / 'high_freq'
RESULTS_LOW_FREQ = RESULTS_BASE / 'ablation' / 'low_freq'

OUTPUT_DIR = DATA_ROOT / 'results_final' / 'cv_evaluation_h5'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

SUBBAND_CONFIGS = {
    'single': None,
    'multi': ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
}


def main():

    print("Loading datasets: ")
    dataset_single = CombinedH5Dataset(H5_DIR, mode='single')
    dataset_multi = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    dataset_multi_denoised = CombinedH5Dataset(H5_DIR_DENOISED, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    print("Datasets loaded")

    results = {}

    print("Evaluating models: ")

    results['Full-band'] = evaluate_model_folds(
        model_class=FullBandCNN,
        model_params={'dropout': 0.5},
        checkpoint_dir=RESULTS_SINGLE,
        dataset=dataset_single,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    results['Multi-scale'] = evaluate_model_folds(
        model_class=MultiScaleCNN,
        model_params={'n_subbands': 7, 'dropout': 0.5},
        checkpoint_dir=RESULTS_MULTI,
        dataset=dataset_multi,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    results['Multi-scale denoised'] = evaluate_model_folds(
        model_class=MultiScaleCNN,
        model_params={'n_subbands': 7, 'dropout': 0.5},
        checkpoint_dir=RESULTS_MULTI_DENOISED,
        dataset=dataset_multi_denoised,
        device=DEVICE,
        batch_size=BATCH_SIZE,
    )

    print("Statistical analysis:")

    stats = compare_models(
        results,
        metrics=['auc', 'auprc', 'sensitivity', 'specificity'],
        alpha=0.05
    )

    print(f"\n{'metric':<15s} {'Full-band':>15s} {'Multi-scale':>15s} {'Multi-scale denoised':>15s}")
    print("-" * 80)
    for metric in ['auc', 'auprc', 'sensitivity', 'specificity']:
        single_stats = stats['summary']['Full-band'][metric]
        multi_stats = stats['summary']['Multi-scale'][metric]
        multi_denoise_stats = stats['summary']['Multi-scale denoised'][metric]
        print(f"{metric:<15s} "
            f"{single_stats['mean']:>6.4f} ± {single_stats['std']:<6.4f} "
            f"{multi_stats['mean']:>6.4f} ± {multi_stats['std']:<6.4f}"
            f"{multi_denoise_stats['mean']:>6.4f} ± {multi_denoise_stats['std']:<6.4f}")

    # friedman test results
    if 'friedman' in stats:
        print(f"\nfriedman test results:")
        for metric, result in stats['friedman'].items():
            print(f"  {metric}: test statistic={result['statistic']:.4f}, p={result['p_value']:.4f}")

    # pairwise comparisons
    print(f"\npairwise wilcoxon tests (holm-bonferroni corrected):")
    for metric in ['auc', 'auprc']:
        print(f"\n  {metric.upper()}:")
        for comp in stats['pairwise'][metric]:
            if comp['significant_corrected']:
                print(f"    {comp['comparison']:25s}: p={comp['p_value']:.4f}, "
                      f"r={comp['effect_size']:+.3f} (α={comp['adj_alpha']:.4f})")

    # roc and pr curves
    plot_roc_and_pr_combined(
        results,
        output_path=OUTPUT_DIR / 'roc_pr_comparison.png',
        title='Model Comparison across 10-fold CV\n ROC and Precision-Recall Curves'
    )

    # boxplots
    plot_metric_boxplots(
        results,
        metrics=['auc', 'auprc'],
        output_path=OUTPUT_DIR / 'metric_boxplots.png',
        title='Model Performance Comparison across 10-fold CV'
    )

    import json
    output_json = OUTPUT_DIR / 'comparison_results.json'
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)

if __name__ == '__main__':
    main()
