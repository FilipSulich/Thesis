import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import (
    roc_curve, auc, average_precision_score,
    confusion_matrix
)
from sklearn.model_selection import GroupKFold

from src.wavelet_selection_tests.utils import get_subject_group


@torch.no_grad()
def get_predictions(model, loader, device):
    """
    collects predicted probabilities and true labels from a trained model on a dataset

    Args:
        model (nn.Module): trained PyTorch model for evaluation
        loader (DataLoader): dataloader providing batches of data
        device (torch.device): device to run inference on (cpu or cuda)

    Returns:
        tuple: (np.ndarray, np.ndarray) containing predicted probabilities for positive class and true labels
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
    extracts and averages attention weights from a multi-scale model across all samples in a dataset

    Args:
        model (nn.Module): trained multi-scale model
        loader (DataLoader): dataloader providing batches of data
        device (torch.device): device to run inference on (cpu or cuda)

    Returns:
        np.ndarray: mean attention weights across all samples, or None if model lacks attention mechanism
    """
    if not hasattr(model, 'get_attention_weights'):
        return None

    model.eval()
    all_weights = []

    for x, _ in loader:
        x = x.to(device)
        weights = model.get_attention_weights(x).cpu().numpy()
        all_weights.append(weights)

    all_weights = np.concatenate(all_weights, axis=0)
    return np.mean(all_weights, axis=0)


def compute_metrics(labels, probs, threshold=0.5):
    """
    computes classification metrics: AUC, AUPRC, sensitivity, and specificity

    Args:
        labels (np.ndarray): true binary labels
        probs (np.ndarray): predicted probabilities for positive class
        threshold (float): classification threshold for converting probabilities to predictions, defaults to 0.5

    Returns:
        dict: dictionary containing all computed metrics including auc, auprc, sensitivity, specificity, and confusion matrix values
    """
    preds = (probs >= threshold).astype(int)

    fpr, tpr, _ = roc_curve(labels, probs)
    auc_roc = auc(fpr, tpr)
    auprc = average_precision_score(labels, probs)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'auc': float(auc_roc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def find_optimal_threshold_youden(labels, probs):
    """
    finds the optimal classification threshold by maximizing Youden's Index (sensitivity + specificity - 1)

    Args:
        labels (np.ndarray): true binary labels
        probs (np.ndarray): predicted probabilities for positive class

    Returns:
        float: optimal threshold value that maximizes the Youden Index
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return float(thresholds[optimal_idx])


def evaluate_model_folds(model_class, model_params, checkpoint_dir, dataset, device, batch_size=32):
    """
    evaluates a trained model across all cross-validation folds by loading checkpoints and computing metrics

    Args:
        model_class (type): model class to instantiate (e.g., FullBandCNN or MultiScaleCNN)
        model_params (dict): dictionary of model initialization parameters
        checkpoint_dir (Path): directory containing fold checkpoint files (fold*_best.pt)
        dataset (Dataset): combined dataset containing all samples
        device (torch.device): device to run evaluation on (cpu or cuda)
        batch_size (int): batch size for evaluation dataloaders, defaults to 32

    Returns:
        list: list of dictionaries containing metrics, predictions, and metadata for each fold
    """
    from pathlib import Path
    checkpoint_dir = Path(checkpoint_dir)

    fold_results = []

    checkpoint_files = sorted(checkpoint_dir.glob('fold*_best.pt'))

    for checkpoint_path in checkpoint_files:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        fold_idx = checkpoint.get('fold', None)
        if fold_idx is None:
            import re
            match = re.search(r'fold(\d+)', checkpoint_path.name)
            if match:
                fold_idx = int(match.group(1))
            else:
                continue

        print(f"evaluating fold {fold_idx}...")

        if 'val_indices' in checkpoint:
            va_idx = checkpoint['val_indices']
        else:
            groups = np.array([get_subject_group(r) for r in dataset.record_ids])
            gkf = GroupKFold(n_splits=10)
            dummy_X = np.zeros(len(dataset))
            fold_splits = list(gkf.split(dummy_X, dataset.labels, groups))

            if fold_idx < len(fold_splits):
                _, va_idx = fold_splits[fold_idx]
            else:
                continue

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

        if 'val_record_ids' in checkpoint:
            metrics['val_record_ids'] = checkpoint['val_record_ids']

        fold_results.append(metrics)

    return fold_results