"""
model evaluation utilities.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_curve, auc, average_precision_score,
    accuracy_score, f1_score, confusion_matrix
)
from sklearn.model_selection import GroupKFold

from src.wavelet_selection_tests.utils import get_subject_group


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    collect predictions from model.

    args:
        model: trained model
        loader: dataloader
        device: torch device

    returns:
        probs: array of predicted probabilities
        labels: array of true labels
    """
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(prob.cpu().numpy())
        labels.extend(y.numpy())

    return np.array(probs), np.array(labels)


@torch.no_grad()
def get_attention_weights(model, loader, device):
    """
    extract attention weights from multi-scale model.

    args:
        model: trained multi-scale model with get_attention_weights method
        loader: dataloader
        device: torch device

    returns:
        mean attention weights across all samples
    """
    if not hasattr(model, 'get_attention_weights'):
        return None

    model.eval()
    all_weights = []

    for x, _ in loader:
        x = x.to(device)
        weights = model.get_attention_weights(x).cpu().numpy()
        all_weights.append(weights)

    # concatenate all batches then average
    all_weights = np.concatenate(all_weights, axis=0)
    return np.mean(all_weights, axis=0)


# ============================================================================
# metrics computation
# ============================================================================

def compute_metrics(labels, probs, threshold=0.5):
    """
    compute classification metrics.

    args:
        labels: true labels
        probs: predicted probabilities
        threshold: classification threshold

    returns:
        dict with all metrics
    """
    preds = (probs >= threshold).astype(int)

    # roc and auprc
    fpr, tpr, _ = roc_curve(labels, probs)
    auc_roc = auc(fpr, tpr)
    auprc = average_precision_score(labels, probs)

    # classification metrics
    accuracy = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, zero_division=0)

    # confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    return {
        'auc': float(auc_roc),
        'auprc': float(auprc),
        'accuracy': float(accuracy),
        'f1': float(f1),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold_youden(labels, probs):
    """
    find Youden index classification threshold.

    args:
        labels: true labels
        probs: predicted probabilities
    returns:
        optimal threshold at Youden Index
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])


def evaluate_model_folds(model_class, model_params, checkpoint_dir, dataset, device, batch_size=32):
    """
    evaluate model across all folds

    args:
        model_class: model class (FullbandCNN or MultiScaleCNN)
        model_params: dict of model parameters
        checkpoint_dir: directory containing fold checkpoints
        dataset: combined dataset
        device: torch device
        batch_size: batch size for evaluation

    returns:
        list of fold results
    """
    np.random.seed(42)

    groups = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf = GroupKFold(n_splits=10)
    dummy_X = np.zeros(len(dataset))
    fold_splits = list(gkf.split(dummy_X, dataset.labels, groups))

    fold_results = []

    for fold_idx, (_, va_idx) in enumerate(fold_splits):
        checkpoint_path = checkpoint_dir / f'fold{fold_idx}_best.pt'
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / f'fold_{fold_idx + 1}_best.pt'

        if not checkpoint_path.exists():
            continue

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = model_class(**model_params).to(device)

        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            continue

        model.load_state_dict(state_dict)
        model.eval()

        va_loader = DataLoader(Subset(dataset, va_idx), batch_size=batch_size, shuffle=False)

        probs, labels = get_predictions(model, va_loader, device)

        metrics = compute_metrics(labels, probs)
        metrics['fold'] = fold_idx
        metrics['probs'] = probs
        metrics['labels'] = labels

        fold_results.append(metrics)

    return fold_results
#
# def aggregate_fold_results(fold_results, metrics=['auc', 'auprc', 'sensitivity', 'specificity']):
#     """
#     aggregate metrics across folds.
#
#     args:
#         fold_results: list of fold result dicts
#         metrics: list of metric names to aggregate
#
#     returns:
#         dict with mean, std, and per-fold values for each metric
#     """
#     aggregated = {}
#
#     for metric in metrics:
#         values = [fold[metric] for fold in fold_results]
#         aggregated[metric] = {
#             'mean': float(np.mean(values)),
#             'std': float(np.std(values)),
#             'values': values
#         }
#
#     return aggregated

#
# def print_aggregated_results(aggregated, model_name):
#     """
#     print aggregated results in a formatted table.
#
#     args:
#         aggregated: dict from aggregate_fold_results
#         model_name: name of model
#     """
#     print(f"\n{model_name} - aggregated results:")
#     print("=" * 60)
#     print(f"{'metric':<15s} {'mean':>10s} {'std':>10s}")
#     print("-" * 60)
#
#     for metric, stats in aggregated.items():
#         print(f"{metric:<15s} {stats['mean']:>10.4f} {stats['std']:>10.4f}")
