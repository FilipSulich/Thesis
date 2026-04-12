from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import SingleScaleCNN, MultiScaleCNN, ResNet18Transfer
from dataset import UTRPDataset, get_all_patient_ids
from cross_validation import create_holdout_test_set, get_remaining_labels
from training import CrossValidationRunner, set_seed


HDF5_PATH = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp2/utrp_dataset_simple.h5')
RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results/cnn_small_2')
SEED = 42

HYPERPARAM_GRID = {
    'lr': [1e-3],  # Start with just one LR
    'dropout': [0.5, 0.7],  # Higher dropout for small dataset
    'batch_size': [32]  # Larger batches for stability
}

RESNET_HYPERPARAM_GRID = {
    'lr': [1e-4, 1e-5],  # Very low LR to prevent overfitting
    'dropout': [0.7, 0.8],  # Very high dropout
    'batch_size': [16, 32],  # Larger batches
    'freeze_backbone': [False, True]  # Fine-tune all layers
}


def generate_gradcam_examples(
    model,
    test_dataset: UTRPDataset,
    save_path: Path,
    n_examples: int = 8
):
    """
    Generate GradCAM heatmaps for example images.

    Args:
        model: Trained model
        test_dataset: Test dataset
        save_path: Path to save figure
        n_examples: Number of examples (4 pre-AF, 4 NSR)
    """
    print(f"\nGenerating GradCAM examples...")

    # Find examples from each class
    preaf_indices = [i for i, label in enumerate(test_dataset.labels) if label == 1]
    nsr_indices = [i for i, label in enumerate(test_dataset.labels) if label == 0]

    # Sample random examples
    np.random.seed(SEED)
    preaf_samples = np.random.choice(preaf_indices, min(n_examples // 2, len(preaf_indices)), replace=False)
    nsr_samples = np.random.choice(nsr_indices, min(n_examples // 2, len(nsr_indices)), replace=False)

    # Determine target layer for GradCAM
    # For custom CNNs: use last conv layer (conv4)
    # For ResNet: use layer4
    if isinstance(model, ResNet18Transfer):
        target_layer = model.resnet.layer4[-1].conv2
    elif isinstance(model, (SingleScaleCNN, MultiScaleCNN)):
        target_layer = model.conv4
    else:
        print("Unknown model type, skipping GradCAM")
        return

    # Create figure
    fig, axes = plt.subplots(2, n_examples, figsize=(3 * n_examples, 6))

    for col_idx, (sample_idx, label_name) in enumerate(
        list(zip(preaf_samples, ['Pre-AF'] * len(preaf_samples))) +
        list(zip(nsr_samples, ['NSR'] * len(nsr_samples)))
    ):
        # Get image
        utrp, label = test_dataset[sample_idx]
        utrp_batch = utrp.unsqueeze(0)  # Add batch dimension

        # Generate GradCAM
        try:
            cam = model.grad_cam(utrp_batch, target_layer)

            # Plot original UTRP (average channels if multi-scale)
            if test_dataset.mode == 'multi':
                img = utrp.mean(dim=0).numpy()
            else:
                img = utrp[0].numpy()

            axes[0, col_idx].imshow(img, cmap='gray')
            axes[0, col_idx].set_title(f"{label_name}\n(True: {label})")
            axes[0, col_idx].axis('off')

            # Plot GradCAM overlay
            axes[1, col_idx].imshow(img, cmap='gray')
            axes[1, col_idx].imshow(cam, cmap='jet', alpha=0.5)
            axes[1, col_idx].set_title('GradCAM')
            axes[1, col_idx].axis('off')

        except Exception as e:
            print(f"Failed to generate GradCAM for sample {sample_idx}: {e}")
            axes[0, col_idx].text(0.5, 0.5, 'Error', ha='center', va='center')
            axes[1, col_idx].text(0.5, 0.5, 'Error', ha='center', va='center')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"GradCAM examples saved to {save_path}")
    plt.close()


def main(model, arch, epochs):
    """Main training workflow."""

    # Set seed
    set_seed(SEED)

    print("\n" + "="*80)
    print("UTRP PRE-AF CLASSIFICATION")
    print("="*80)
    print(f"Model: {model} scale")
    print(f"Architecture: {arch}")
    print(f"Dataset: {HDF5_PATH}")
    print(f"Results dir: {RESULTS_DIR}")
    # Inside run_classification.py / main()
    # 1. Get ALL unique subjects and their labels from the HDF5 metadata
    all_subjects, all_labels = get_all_patient_ids(HDF5_PATH)

    test_patients, train_val_patients = create_holdout_test_set(
        all_subjects,
        all_labels,
        n_test_per_class=5
    )

    # Note: train_val_patients is now a list of RECORD IDs, not subject IDs
    # We'll handle label lookup inside the CrossValidationRunner

    # Select model class and hyperparameter grid
    if arch == 'resnet':
        if model == 'single':
            model_class = lambda **kwargs: ResNet18Transfer(mode='single', **kwargs)
        else:
            model_class = lambda **kwargs: ResNet18Transfer(mode='multi', **kwargs)
        hyperparam_grid = RESNET_HYPERPARAM_GRID
    else:  # custom
        if model == 'single':
            model_class = SingleScaleCNN
        else:
            model_class = MultiScaleCNN
        hyperparam_grid = HYPERPARAM_GRID

    # Create results directory
    experiment_name = f"{model}_{arch}"
    experiment_dir = RESULTS_DIR / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Run cross-validation
    cv_runner = CrossValidationRunner(
        model_class=model_class,
        hdf5_path=HDF5_PATH,
        mode=model,
        results_dir=experiment_dir,
        n_folds=5,
        seed=SEED
    )

    results = cv_runner.run(
        hyperparam_grid=hyperparam_grid,
        train_patients=train_val_patients,  # These are record IDs now
        test_patients=test_patients,
        epochs=epochs,
    )

    # Generate GradCAM examples
    test_dataset = UTRPDataset(HDF5_PATH, mode=model, augment=False, patient_ids=test_patients)

    generate_gradcam_examples(
        results['final_model'],
        test_dataset,
        save_path=experiment_dir / 'grad_cam_examples.png',
        n_examples=8
    )

    # Print final summary
    print(f"\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nBest configuration:")
    for key, value in results['best_config'].items():
        if key not in ['fold_results']:
            print(f"  {key}: {value}")

    print(f"\nTest set performance:")
    for metric, value in results['test_metrics'].items():
        if metric != 'confusion_matrix':
            print(f"  {metric}: {value:.4f}")

    print(f"\nAll results saved to: {experiment_dir}")


if __name__ == '__main__':
    model = "single"  # Try single first
    arch = "resnet"   # Try simpler custom CNN
    epochs = 100

    main(model, arch, epochs)