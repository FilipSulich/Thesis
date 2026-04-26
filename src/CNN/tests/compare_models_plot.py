"""
Compare and plot AUC/AUPRC curves for Single, Multi Uniform, and Multi Trained models
"""

from pathlib import Path
from model_evaluator import ModelEvaluator

def plot_model_comparison():
    """
    Load three models (single, multi-uniform, multi-trained) and plot their ROC/AUPRC curves
    """
    
    # Paths
    data_dir = Path('/data/data_final/utrp_dataset_db8')
    single_model_path = Path('/data/results_final/single/final_checkpoint.pt')
    multi_model_path = Path('/data/results_new/multi/final_checkpoint.pt')
    hybrid_model_path = Path('/data/results_final/hybrid/best_checkpoint.pt')
    
    output_dir = Path('/data/results_new/comparison_plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed for reproducibility
    ModelEvaluator.set_seed(42)
    
    evaluators = {}
    
    # 1. Single model
    # print("\n" + "="*70)
    # print("LOADING SINGLE MODEL")
    # print("="*70)
    # evaluator_single = ModelEvaluator(single_model_path, data_dir, model_arch='single')
    # evaluator_single.load_model(strict=True, attention_mode='trained')
    # evaluator_single.load_test_data(batch_size=32)
    # metrics_single = evaluator_single.evaluate()
    # evaluator_single.print_metrics(metrics_single)
    # evaluators['Single-scale'] = evaluator_single
    #
    # # 2. Multi model with uniform attention
    print("\n" + "="*70)
    print("LOADING MULTI MODEL - UNIFORM ATTENTION")
    print("="*70)
    evaluator_multi_uniform = ModelEvaluator(multi_model_path, data_dir, model_arch='multi')
    evaluator_multi_uniform.load_model(strict=True, attention_mode='uniform')
    evaluator_multi_uniform.load_test_data(batch_size=32)
    metrics_multi_uniform = evaluator_multi_uniform.evaluate()
    evaluator_multi_uniform.print_metrics(metrics_multi_uniform)
    evaluators['Multi-scale Uniform'] = evaluator_multi_uniform
    
    # 3. Multi model with trained attention
    print("\n" + "="*70)
    print("LOADING MULTI MODEL - TRAINED ATTENTION")
    print("="*70)
    evaluator_multi_trained = ModelEvaluator(multi_model_path, data_dir, model_arch='multi')
    evaluator_multi_trained.load_model(strict=True, attention_mode='trained')
    evaluator_multi_trained.load_test_data(batch_size=32)
    metrics_multi_trained = evaluator_multi_trained.evaluate()
    evaluator_multi_trained.print_metrics(metrics_multi_trained)
    evaluators['Multi-scale Trained'] = evaluator_multi_trained

    # # 3. Multi model with trained attention
    # print("\n" + "="*70)
    # print("LOADING Hybrid MODEL - uniform ATTENTION")
    # print("="*70)
    # evaluator_hybrid_uniform = ModelEvaluator(hybrid_model_path, data_dir, model_arch='hybrid')
    # evaluator_hybrid_uniform.load_model(strict=True, attention_mode='uniform')
    # evaluator_hybrid_uniform.load_test_data(batch_size=32)
    # metrics_hybrid_uniform = evaluator_hybrid_uniform.evaluate()
    # evaluator_hybrid_uniform.print_metrics(metrics_hybrid_uniform)
    # evaluators['Hybrid'] = evaluator_hybrid_uniform
    
    # Plot comparison
    print("\n" + "="*70)
    print("PLOTTING COMPARISON")
    print("="*70)
    save_path = output_dir / 'comparison_auc_auprc.png'
    ModelEvaluator.plot_comparison(evaluators, save_path=save_path)
    evaluators = {
        # 'Single-scale': evaluator_single,
        'Multi-scale Uniform': evaluator_multi_uniform,
        'Multi-scale Trained': evaluator_multi_trained,
        # 'Hybrid': evaluator_hybrid_uniform
    }

    ModelEvaluator.plot_confusion_matrices_comparison(
        evaluators,
        normalize=True,
        save_path= output_dir / 'confusion_matrix.png',
        title='Model Comparison - Confusion Matrices'
    )

    # print(f"\n✓ Comparison complete!")
    # print(f"  Single:        AUC={metrics_single['auc_roc']:.4f}, AUPRC={metrics_single['auprc']:.4f}")
    # print(f"  Multi Uniform: AUC={metrics_multi_uniform['auc_roc']:.4f}, AUPRC={metrics_multi_uniform['auprc']:.4f}")
    # print(f"  Multi Trained: AUC={metrics_multi_trained['auc_roc']:.4f}, AUPRC={metrics_multi_trained['auprc']:.4f}")

if __name__ == '__main__':
    plot_model_comparison()

