from pathlib import Path
from src.CNN.models.model_evaluator import ModelEvaluator

# Paths to models and data
SINGLE_MODEL_PATH = Path('/results_new/multi/final_checkpoint.pt')
MULTI_MODEL_PATH = Path('/results_new/multi/final_checkpoint.pt')
DATA_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8_no_denoise')
SAVE_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_new/comparison_plots')

# Create output directory
SAVE_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("LOADING AND EVALUATING SINGLE-SCALE MODEL")
print("="*70)
evaluator_single = ModelEvaluator(SINGLE_MODEL_PATH, DATA_DIR, model_arch='multi')
evaluator_single.load_model(strict=False, attention_mode="trained")
evaluator_single.load_test_data(batch_size=32)
metrics_single = evaluator_single.evaluate()
evaluator_single.print_metrics(metrics_single)

print("\n" + "="*70)
print("LOADING AND EVALUATING MULTI-SCALE MODEL")
print("="*70)
evaluator_multi = ModelEvaluator(MULTI_MODEL_PATH, DATA_DIR, model_arch='multi')
evaluator_multi.load_model(strict=False, attention_mode="uniform")
evaluator_multi.load_test_data(batch_size=32)
metrics_multi = evaluator_multi.evaluate()
evaluator_multi.print_metrics(metrics_multi)

print("\n" + "="*70)
print("CREATING COMPARISON PLOTS")
print("="*70)

# Create comparison plot with both AUC and AUPRC
evaluators = {
    'Multi Trained': evaluator_single,
    'Multi Uniform': evaluator_multi
}

ModelEvaluator.plot_comparison(
    evaluators,
    save_path=SAVE_DIR / 'auc_auprc_comparison.png'
)

print("\n✓ All plots created successfully!")
print(f"\nComparison Summary:")
print(f"  Single-Scale - AUC: {metrics_single['auc_roc']:.4f}, AUPRC: {metrics_single['auprc']:.4f}")
print(f"  Multi-Scale  - AUC: {metrics_multi['auc_roc']:.4f}, AUPRC: {metrics_multi['auprc']:.4f}")
