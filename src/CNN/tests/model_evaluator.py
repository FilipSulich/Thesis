import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.CNN.models.full_band_cnn import FullBandCNN
from src.CNN.models.multi_scale_cnn import MultiScaleCNN
from src.CNN.utils.datasets import CombinedH5Dataset
from src.CNN.utils.evaluation import evaluate_model_folds
from src.CNN.utils.statistics import compare_models
from src.CNN.utils.plotting import plot_roc_and_pr_combined, plot_metric_box_plots
from src.CNN.utils.utils import DEVICE
import json

DATA_ROOT = Path(__file__).parent.parent.parent.parent

H5_DIR = DATA_ROOT / 'data_container' / 'data' /'processed_data' / 'urp_no_denoise'
H5_DIR_DENOISED= DATA_ROOT / 'data_container' / 'data'/ 'processed_data' / 'urp_full_denoise'

RESULTS_BASE = DATA_ROOT / 'data_container' / 'data' /'model_weights'
RESULTS_SINGLE = RESULTS_BASE / 'single'
RESULTS_MULTI = RESULTS_BASE / 'multi'
RESULTS_MULTI_DENOISED = RESULTS_BASE / 'multi_denoise'

OUTPUT_DIR = DATA_ROOT / 'results' / 'models_evaluation_results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32

output_json = OUTPUT_DIR / 'comparison_results.json'

SUBBAND_CONFIGS = {
    'single': None,
    'multi': ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
}

def main():
    """
    main function to run the complete model evaluation
    """

    print("loading datasets...")
    dataset_single = CombinedH5Dataset(H5_DIR, mode='single')
    dataset_multi = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    dataset_multi_denoised = CombinedH5Dataset(H5_DIR_DENOISED, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    print("datasets loaded")

    results = {}


    print("\nevaluating the full-band model... ")
    results['Full-band'] = evaluate_model_folds(model_class=FullBandCNN, model_params={'dropout': 0.5}, checkpoint_dir=RESULTS_SINGLE, dataset=dataset_single, device=DEVICE, batch_size=BATCH_SIZE)

    print("\nevaluating the multi-scale model... ")
    results['Multi-scale'] = evaluate_model_folds(model_class=MultiScaleCNN,model_params={'n_subbands': 7, 'dropout': 0.5}, checkpoint_dir=RESULTS_MULTI, dataset=dataset_multi, device=DEVICE, batch_size=BATCH_SIZE)

    print("\nevaluating the multi-scale denoised model... ")
    results['Multi-scale denoised'] = evaluate_model_folds(model_class=MultiScaleCNN, model_params={'n_subbands': 7, 'dropout': 0.5}, checkpoint_dir=RESULTS_MULTI_DENOISED, dataset=dataset_multi_denoised, device=DEVICE, batch_size=BATCH_SIZE)

    print("models successfully evaluated")

    stats = compare_models(
        results,
        metrics=['auc', 'auprc', 'sensitivity', 'specificity'],
        alpha=0.05
    )
    plot_roc_and_pr_combined(
        results,
        output_path=OUTPUT_DIR / 'roc_pr_comparison.png',
        title='Model Comparison across 10-fold CV\n ROC and Precision-Recall Curves'
    )

    with open(output_json, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"analysis done and results saved to {output_json}")

if __name__ == '__main__':
    main()
