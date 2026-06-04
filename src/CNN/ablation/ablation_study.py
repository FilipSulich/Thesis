import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.CNN.models.multi_scale_cnn import MultiScaleCNN
from src.CNN.utils.datasets import CombinedH5Dataset
from src.CNN.utils.evaluation import evaluate_model_folds
from src.CNN.utils.statistics import compare_models
from src.CNN.utils.plotting import plot_metric_box_plots, plot_roc_and_pr_combined
from src.CNN.utils.utils import DEVICE

DATA_ROOT = Path(__file__).parent.parent.parent.parent

H5_DIR = DATA_ROOT / 'data_container' / 'data' /'processed_data' / 'urp_no_denoise'

ABLATION_BASE = DATA_ROOT / 'data_container' / 'data' /'model_weights' / 'ablation'

RESULTS_HIGH_FREQ = ABLATION_BASE / 'high_freq'
RESULTS_LOW_FREQ = ABLATION_BASE / 'low_freq'
RESULTS_MID_FREQ = ABLATION_BASE / 'mid_freq'
RESULTS_FULL = DATA_ROOT / 'data_container' / 'data' /'model_weights' / 'multi'

OUTPUT_DIR = DATA_ROOT / 'results' / 'ablation_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

SUBBANDS_HIGH_FREQ = ['D2', 'D3', 'D4']
SUBBANDS_LOW_FREQ = ['D5', 'D6', 'D7', 'D8']
SUBBANDS_MID_FREQ = ['D3', 'D4']
SUBBANDS_FULL = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']


def main():
    """
    main function to run the ablation study to evaluate different frequency band combinations and compare them to the full multi-scale model
    """
    print("loading datasets...")
    dataset_high = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBANDS_HIGH_FREQ)
    dataset_low = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBANDS_LOW_FREQ)
    dataset_full = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBANDS_FULL)
    dataset_mid= CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBANDS_MID_FREQ)
    print("datasets loaded")

    results = {}

    print("evaluating high-frequency model...")
    results['High-frequency'] = evaluate_model_folds(model_class=MultiScaleCNN,model_params={'n_subbands': len(SUBBANDS_HIGH_FREQ),'dropout': 0.5}, checkpoint_dir=RESULTS_HIGH_FREQ, dataset=dataset_high, device=DEVICE, batch_size=BATCH_SIZE)
    print("\nevaluating low-frequency model...")
    results['Low-frequency'] = evaluate_model_folds(model_class=MultiScaleCNN, model_params={'n_subbands': len(SUBBANDS_LOW_FREQ), 'dropout': 0.5}, checkpoint_dir=RESULTS_LOW_FREQ, dataset=dataset_low, device=DEVICE,batch_size=BATCH_SIZE)
    print("\nevaluating mid-frequency model...")
    results['Mid-frequency'] = evaluate_model_folds(model_class=MultiScaleCNN, model_params={'n_subbands': len(SUBBANDS_MID_FREQ), 'dropout': 0.5}, checkpoint_dir=RESULTS_MID_FREQ, dataset=dataset_mid, device=DEVICE, batch_size=BATCH_SIZE)
    print("\nevaluating full multi-frequency model...")
    results['Multi-scale'] = evaluate_model_folds(model_class=MultiScaleCNN, model_params={'n_subbands': len(SUBBANDS_FULL), 'dropout': 0.5}, checkpoint_dir=RESULTS_FULL, dataset=dataset_full, device=DEVICE, batch_size=BATCH_SIZE)

    print("models evaluated")

    stats = compare_models(
        results,
        metrics=['auc', 'auprc', 'sensitivity', 'specificity'],
        alpha=0.05
    )

    plot_roc_and_pr_combined(
        results,
        output_path=OUTPUT_DIR / 'roc_pr_comparison_ablation.pdf',
        title='Ablation Study Comparison across 10-fold CV\n ROC and Precision-Recall Curves'
    )

    output_json = OUTPUT_DIR / 'ablation_comparison_stats.json'
    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"analysis done and results saved to {output_json}")


if __name__ == '__main__':
    main()