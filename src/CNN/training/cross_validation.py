import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ( average_precision_score,confusion_matrix, roc_curve, auc as sklearn_auc)
from src.CNN.models.models import FullBandCNN, MultiScaleCNN
from src.CNN.training.data_loaders import FullDataset, AugmentedSubset
from src.CNN.utils.utils import DEVICE, get_subject_group, set_seed

DATA_ROOT = Path(__file__).parent.parent.parent.parent

H5_DIR = DATA_ROOT / 'data' / 'processed_data' / 'urp_no_denoise'
SAVE_DIR_BASE = DATA_ROOT / 'saved_models'

# Model type: 'single' or 'multi_denoise'
MODEL_TYPE = 'multi'

if MODEL_TYPE == 'multi_denoise':
    H5_DIR = DATA_ROOT / 'data' / 'processed_data' / 'urp_denoise'

# Hyperparameters
N_FOLDS = 10
N_SUBBANDS = 7
SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
DROPOUT = 0.5
PATIENCE = 20
SCHED_PAT = 5
GRAD_CLIP = 1.0
SEED = 42

SAVE_DIR = SAVE_DIR_BASE / MODEL_TYPE
SAVE_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels = [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        probs = torch.softmax(logits, dim=1)[:, 1]

        total_loss += loss.item() * len(y)
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Compute metrics
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    auc_roc = sklearn_auc(fpr, tpr)
    auprc = average_precision_score(all_labels, all_probs)

    # Find best threshold using Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    preds = (all_probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(all_labels, preds).ravel()

    metrics = {
        'auc_roc': float(auc_roc),
        'auprc': float(auprc),
        'accuracy': float((tp + tn) / (tp + tn + fp + fn)),
        'f1': float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0,
        'sensitivity': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'specificity': float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0,
        'precision': float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        'recall': float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }

    return total_loss / n, correct / n, metrics, all_probs, all_labels

@torch.no_grad()
def collect_attention_weights(model, loader, device):
    """Collect mean attention weights for multi-scale models."""
    if not hasattr(model, 'get_attention_weights'):
        return None

    model.eval()
    all_weights = []

    for x, _ in loader:
        x = x.to(device)
        weights = model.get_attention_weights(x).detach().cpu().numpy()
        all_weights.append(weights)

    # Concatenate all batches first (they may have different batch sizes)
    all_weights = np.concatenate(all_weights, axis=0)
    return np.mean(all_weights, axis=0)

def run_cross_validation():
    """Run 10-fold cross-validation with patient-level splitting."""

    # Load dataset
    mode = 'single' if MODEL_TYPE == 'single' else 'multi_denoise'
    dataset = FullDataset(H5_DIR, mode=mode, cache=True)

    # Get patient groups
    groups = np.array([get_subject_group(r) for r in dataset.record_ids])

    # Initialize fold splitting
    gkf = GroupKFold(n_splits=N_FOLDS)
    dummy_X = np.zeros(len(dataset))

    # Storage for results
    fold_results = []
    fold_probs = []
    fold_labels = []
    fold_attn_weights = []

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(dummy_X, dataset.labels, groups), start=1):

        n_tr_nsr = int((dataset.labels[tr_idx] == 0).sum())
        n_tr_paf = int((dataset.labels[tr_idx] == 1).sum())
        w_preaf = n_tr_nsr / n_tr_paf if n_tr_paf > 0 else 1.0
        class_weights = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(DEVICE)

        tr_subset = AugmentedSubset(dataset, tr_idx, augment=True) #only augment training data
        va_subset = Subset(dataset, va_idx)

        tr_loader = DataLoader(tr_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        va_loader = DataLoader(va_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        # Initialize model
        set_seed(SEED + fold)

        if MODEL_TYPE == 'single':
            model = FullBandCNN(DROPOUT).to(DEVICE)
        else:
            model = MultiScaleCNN(N_SUBBANDS, DROPOUT).to(DEVICE)

        if fold == 1:
            print(f"Model parameters: {model.count_params():,}\n")

        # Optimizer and scheduler
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=SCHED_PAT
        )

        # Training loop
        best_auc = 0.0
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, DEVICE)
            va_loss, va_acc, va_metrics, va_probs, va_labels = eval_epoch(model, va_loader, criterion, DEVICE)

            scheduler.step(va_metrics['auc_roc'])

            # Early stopping based on validation AUC
            if va_metrics['auc_roc'] > best_auc:
                best_auc = va_metrics['auc_roc']
                patience_counter = 0

                # Save checkpoint
                checkpoint = {
                    'fold': fold,
                    'epoch': epoch,
                    'model_type': MODEL_TYPE,
                    'state_dict': model.state_dict(),
                    'val_auc': va_metrics['auc_roc'],
                    'val_metrics': va_metrics,
                    'config': {
                        'n_subbands': N_SUBBANDS if MODEL_TYPE != 'single' else 1,
                        'dropout': DROPOUT,
                        'lr': LR,
                        'seed': SEED,
                        'w_preaf': float(w_preaf),
                    }
                }
                torch.save(checkpoint, SAVE_DIR / f'fold_{fold}_best.pt')
            else:
                patience_counter += 1

            if (epoch % 10 == 0) or (epoch == 1):
                print(f"Epoch {epoch:3d}: train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} val_loss={va_loss:.4f} val_auc={va_metrics['auc_roc']:.4f} (best={best_auc:.4f})")

            if patience_counter >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        # Load best model and collect final results
        checkpoint = torch.load(SAVE_DIR / f'fold_{fold}_best.pt', map_location=DEVICE)
        model.load_state_dict(checkpoint['state_dict'])

        _, _, final_metrics, final_probs, final_labels = eval_epoch(model, va_loader, criterion, DEVICE)

        # Collect attention weights for multi_denoise-scale models
        attn_weights = collect_attention_weights(model, va_loader, DEVICE)

        fold_results.append(final_metrics)
        fold_probs.append(final_probs)
        fold_labels.append(final_labels)
        if attn_weights is not None:
            fold_attn_weights.append(attn_weights)

        print(f"\nFold {fold} final results:")
        print(f"AUC: {final_metrics['auc_roc']:.4f}")
        print(f"AUPRC:   {final_metrics['auprc']:.4f}")

    # Aggregate results
    def aggregate(fold_results):
        keys = fold_results[0].keys()
        return {
            k: {
                'values': [f[k] for f in fold_results],
                'mean': float(np.mean([f[k] for f in fold_results])),
                'std': float(np.std([f[k] for f in fold_results])),
                'median': float(np.median([f[k] for f in fold_results])),
                'min': float(np.min([f[k] for f in fold_results])),
                'max': float(np.max([f[k] for f in fold_results])),
            }
            for k in keys if k not in ['tp', 'tn', 'fp', 'fn']
        }

    aggregated = aggregate(fold_results)

    # Save results
    results = {
        'model_type': MODEL_TYPE,
        'config': {
            'n_folds': N_FOLDS,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'dropout': DROPOUT,
            'patience': PATIENCE,
            'seed': SEED,
        },
        'aggregated': aggregated,
        'per_fold': fold_results,
    }

    with open(SAVE_DIR / 'kfold_results.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    run_cross_validation()
