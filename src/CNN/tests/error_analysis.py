"""
Patient-level ensemble evaluation using segment-wise predictions.
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix
from src.CNN.utils.datasets import CombinedH5Dataset
from sklearn.model_selection import GroupKFold

from src.CNN.models.models import FullBandCNN, MultiScaleCNN
from src.CNN.utils.datasets import H5Dataset
from src.CNN.utils.evaluation import get_predictions, find_optimal_threshold_youden
from src.CNN.utils.utils import get_device
from src.wavelet_selection_tests.utils import get_subject_group

DATA_ROOT = Path(__file__).parent.parent.parent.parent

H5_DIR = DATA_ROOT / 'data' / 'processed_data' / 'urp_no_denoise'
H5_DIR_DENOISED = DATA_ROOT / 'data' / 'processed_data' / 'urp_denoise'

RESULTS_BASE = DATA_ROOT / 'results_final' / 'models_final_weights'

RESULTS_SINGLE = RESULTS_BASE / 'single'
RESULTS_MULTI = RESULTS_BASE / 'multi'
RESULTS_MULTI_DENOISED = RESULTS_BASE / 'multi_denoise'

OUTPUT_DIR = DATA_ROOT / 'results_final' / 'patient_level_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 32
DEVICE = get_device()
np.random.seed(42)

SUBBAND_CONFIGS = {
    'single': None,
    'multi': ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8'],
}

def compute_mean_youden_threshold_from_folds(model_class, model_params, checkpoint_dir, dataset):
    """
    Compute mean Youden index threshold across all CV folds.

    Args:
        model_class: Model class
        model_params: Dict of model parameters
        checkpoint_dir: Directory containing fold checkpoints
        dataset: Combined dataset (train+val)
        device: Torch device
        batch_size: Batch size for evaluation

    Returns:
        Mean threshold across all folds
    """

    groups = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf = GroupKFold(n_splits=10)
    dummy_X = np.zeros(len(dataset))
    fold_splits = list(gkf.split(dummy_X, dataset.labels, groups))

    thresholds = []

    for fold_idx, (_, va_idx) in enumerate(fold_splits):
        checkpoint_path = checkpoint_dir / f'fold{fold_idx}_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / f'fold_{fold_idx + 1}_best.pt'

        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model = model_class(**model_params).to(DEVICE)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            continue

        model.load_state_dict(state_dict)
        model.eval()

        va_loader = DataLoader(
            torch.utils.data.Subset(dataset, va_idx),
            batch_size=BATCH_SIZE,
            shuffle=False
        )

        probs, labels = get_predictions(model, va_loader, DEVICE)
        threshold = find_optimal_threshold_youden(labels, probs)
        thresholds.append(threshold)

    mean_threshold = np.mean(thresholds)
    std_threshold = np.std(thresholds)

    return mean_threshold, thresholds, std_threshold


def aggregate_patient_predictions(probs, record_ids):
    """
    Aggregate segment-wise predictions to patient-level.

    Args:
        probs: Array of predicted probabilities for each segment
        record_ids: Array of record IDs for each segment (each ID is a patient ID)

    Returns:
        patient_ids: List of unique patient IDs
        patient_probs: Mean probability for each patient
    """
    # record_ids are already patient IDs
    unique_patients = np.unique(record_ids)

    patient_probs = []

    for patient in unique_patients:
        patient_mask = record_ids == patient
        patient_segment_probs = probs[patient_mask]

        # Average probabilities across segments
        mean_prob = np.mean(patient_segment_probs)
        patient_probs.append(mean_prob)

    return list(unique_patients), np.array(patient_probs)


def evaluate_patient_level(model_class, model_params, checkpoint_dir, test_dataset, threshold):
    """
    Evaluate patient-level performance using ensemble of all fold models.

    Args:
        model_class: Model class
        model_params: Dict of model parameters
        checkpoint_dir: Directory containing fold checkpoints
        test_dataset: Test dataset
        threshold: Classification threshold
        device: Torch device
        batch_size: Batch size

    Returns:
        Dict with patient-level metrics
    """
    # Load all fold models and collect predictions
    all_fold_probs = []

    for fold_idx in range(10):
        checkpoint_path = checkpoint_dir / f'fold{fold_idx}_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / f'fold_{fold_idx + 1}_best.pt'

        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model = model_class(**model_params).to(DEVICE)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            continue

        model.load_state_dict(state_dict)
        model.eval()

        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        probs, labels = get_predictions(model, test_loader, DEVICE)
        all_fold_probs.append(probs)

    # Average predictions across all folds (ensemble)
    mean_probs = np.mean(all_fold_probs, axis=0)

    # Aggregate to patient level
    patient_ids, patient_probs = aggregate_patient_predictions(mean_probs, test_dataset.record_ids)

    # Get patient labels
    patient_labels = []
    for patient in patient_ids:
        patient_mask = test_dataset.record_ids == patient
        patient_label = test_dataset.labels[patient_mask][0]
        patient_labels.append(patient_label)
    patient_labels = np.array(patient_labels)

    # Make predictions
    patient_preds = (patient_probs >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(patient_labels, patient_preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # ROC and AUPRC
    fpr, tpr, _ = roc_curve(patient_labels, patient_probs)
    auc_roc = auc(fpr, tpr)
    auprc = average_precision_score(patient_labels, patient_probs)

    return {
        'n_patients': len(patient_ids),
        'patient_ids': patient_ids,
        'patient_probs': patient_probs.tolist(),
        'patient_labels': patient_labels.tolist(),
        'patient_preds': patient_preds.tolist(),
        'threshold': threshold,
        'auc': float(auc_roc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def main():
    print("\nLoading datasets...")

    dataset_single = CombinedH5Dataset(H5_DIR, mode='single')
    dataset_multi = CombinedH5Dataset(H5_DIR, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    dataset_multi_denoised = CombinedH5Dataset(H5_DIR_DENOISED, mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])

    test_dataset_single = H5Dataset(H5_DIR / 'test.h5', mode='single')
    test_dataset_multi = H5Dataset(H5_DIR / 'test.h5', mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])
    test_dataset_multi_denoised = H5Dataset(H5_DIR_DENOISED / 'test.h5', mode='multi', subband_keys=SUBBAND_CONFIGS['multi'])

    # Model configurations
    models = {
        'Full-band': {
            'model_class': FullBandCNN,
            'model_params': {'dropout': 0.5},
            'checkpoint_dir': RESULTS_SINGLE,
            'train_dataset': dataset_single,
            'test_dataset': test_dataset_single
        },
        'Multi-scale': {
            'model_class': MultiScaleCNN,
            'model_params': {'n_subbands': 7, 'dropout': 0.5},
            'checkpoint_dir': RESULTS_MULTI,
            'train_dataset': dataset_multi,
            'test_dataset': test_dataset_multi
        },
        'Multi-scale denoised': {
            'model_class': MultiScaleCNN,
            'model_params': {'n_subbands': 7, 'dropout': 0.5},
            'checkpoint_dir': RESULTS_MULTI_DENOISED,
            'train_dataset': dataset_multi_denoised,
            'test_dataset': test_dataset_multi_denoised
        }
    }

    results = {}

    print("Evaluating models: ")
    #model evaluation
    for model_name, config in models.items():
        mean_threshold, fold_thresholds, std_threshold = compute_mean_youden_threshold_from_folds(
            model_class=config['model_class'],
            model_params=config['model_params'],
            checkpoint_dir=config['checkpoint_dir'],
            dataset=config['train_dataset'],
        )

        test_results = evaluate_patient_level(
            model_class=config['model_class'],
            model_params=config['model_params'],
            checkpoint_dir=config['checkpoint_dir'],
            test_dataset=config['test_dataset'],
            threshold=mean_threshold,
        )

        test_results['mean_youden_threshold'] = mean_threshold
        test_results['fold_youden_thresholds'] = fold_thresholds
        test_results['std_youden_thresholds'] = std_threshold

        results[model_name] = test_results


    output_path = OUTPUT_DIR / 'patient_level_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Analysis done and results saved to: {output_path}")


if __name__ == '__main__':
    main()