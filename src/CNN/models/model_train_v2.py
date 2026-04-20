import json
import os
import random
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, roc_curve,
                              confusion_matrix)
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.CNN.dataset.dataset import UTRPDataset
from src.CNN.models.models import ZhuResNetSingleScale
from src.CNN.models.complexity_test import LateFusionWithAttention, HybridFusionWithAttention, SingleScaleCNN


# ── Helpers ────────────────────────────────────────────────────────────────────

def _dataset_mode(model_arch: str) -> str:
    """Map model arch string to dataset mode string."""
    return 'both' if model_arch == 'hybrid' else model_arch


def _forward(model, batch, model_arch: str, device):
    """Unpack batch and run forward pass for any arch."""
    if model_arch == 'hybrid':
        x_single, x_multi, labels = batch
        logits = model(x_single.to(device), x_multi.to(device))
    else:
        x, labels = batch
        logits = model(x.to(device))
    return logits, labels.to(device)


# ── Training / evaluation ──────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer,
                device, model_arch: str) -> tuple:
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    for batch in tqdm(loader, desc='Train', leave=False):
        logits, labels = _forward(model, batch, model_arch, device)
        optimizer.zero_grad()
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)

    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, model_arch: str) -> tuple:
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_probs, all_labels  = [], []

    for batch in tqdm(loader, desc='Eval', leave=False):
        logits, labels = _forward(model, batch, model_arch, device)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
        all_probs.extend(torch.softmax(logits, 1)[:, 1].cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.5
    return (total_loss / n, correct / n, auc,
            np.array(all_probs), np.array(all_labels))


def threshold_analysis(va_probs, va_true, te_probs, te_true) -> dict:
    """Find F1-optimal and Youden threshold on val, apply to test."""
    threshs = np.arange(0.05, 0.95, 0.01)
    f1s        = [f1_score(va_true, (va_probs >= t).astype(int), zero_division=0)
                  for t in threshs]
    f1_thresh  = float(threshs[np.argmax(f1s)])

    fpr, tpr, roc_t = roc_curve(va_true, va_probs)
    youden_thresh   = float(roc_t[np.argmax(tpr - fpr)])

    results = {}
    for name, thresh in [('default_0.50',            0.5),
                          (f'f1_opt_{f1_thresh:.2f}', f1_thresh),
                          (f'youden_{youden_thresh:.2f}', youden_thresh)]:
        preds = (te_probs >= thresh).astype(int)
        cm    = confusion_matrix(te_true, preds)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results[name] = {
            'threshold':   thresh,
            'accuracy':    float(accuracy_score(te_true, preds)),
            'precision':   float(precision_score(te_true, preds, zero_division=0)),
            'recall':      float(recall_score(te_true, preds, zero_division=0)),
            'f1':          float(f1_score(te_true, preds, zero_division=0)),
            'sensitivity': float(tp / (tp + fn)) if tp + fn > 0 else 0.0,
            'specificity': float(tn / (tn + fp)) if tn + fp > 0 else 0.0,
            'cm':          cm.tolist(),
        }
    return results

def save_plots(history: dict, te_probs, te_true, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    ep = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(ep, history['train_loss'], label='Train')
    axes[0].plot(ep, history['val_loss'],   label='Val')
    axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(0.3)

    axes[1].plot(ep, history['train_acc'], label='Train')
    axes[1].plot(ep, history['val_acc'],   label='Val')
    axes[1].set_title('Accuracy'); axes[1].set_ylim(0, 1)
    axes[1].legend(); axes[1].grid(0.3)

    axes[2].plot(ep, history['val_auc'], color='green')
    axes[2].set_title('Val AUC'); axes[2].set_ylim(0, 1); axes[2].grid(0.3)

    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(te_true, te_probs)
    auc = roc_auc_score(te_true, te_probs)
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC={auc:.4f}')
    plt.plot([0, 1], [0, 1], '--', color='grey')
    plt.xlabel('FPR'); plt.ylabel('TPR')
    plt.title('ROC Curve'); plt.legend(); plt.grid(0.3)
    plt.tight_layout()
    plt.savefig(save_dir / 'roc_curve.png', dpi=150)
    plt.close()
    print(f"  Plots saved → {save_dir}")


# ── Main training function ─────────────────────────────────────────────────────
def train_model(
    data_dir:    Path,
    results_dir: Path,
    model_arch:  str  = 'single',
    epochs:      int  = 150,
    patience:    int  = 20,
    batch_size:  int  = 32,
    lr:          float = 1e-4,
    dropout:     float = 0.5,
    seed:        int  = 42,
    augment:     bool = True,
) -> Dict:

    # Set all random seeds for reproducibility
    import random
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Ensure deterministic behavior on GPU/MPS
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # MPS determinism (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        os.environ['PYTHONHASHSEED'] = str(seed)
    
    results_dir.mkdir(parents=True, exist_ok=True)

    device = (torch.device('mps')  if torch.backends.mps.is_available()  else
              torch.device('cuda') if torch.cuda.is_available() else
              torch.device('cpu'))
    print(f"\n{'='*60}")
    print(f"Model:  {model_arch}  |  Device: {device}  |  Seed: {seed}")
    print(f"LR: {lr}  WD: 1e-4  Dropout: {dropout}  Augment: {augment}")
    print(f"{'='*60}")

    ds_mode = _dataset_mode(model_arch)
    train_ds = UTRPDataset(data_dir, split='train', mode=ds_mode, augment=augment)
    val_ds   = UTRPDataset(data_dir, split='val',   mode=ds_mode, augment=False)
    test_ds  = UTRPDataset(data_dir, split='test',  mode=ds_mode, augment=False)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")

    tr_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    va_loader = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    te_loader = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    if model_arch == 'single':
        model = SingleScaleCNN(dropout=dropout).to(device)
    elif model_arch == 'multi':
        model = LateFusionWithAttention(n_subbands=7, dropout=dropout).to(device)
    else:
        model = HybridFusionWithAttention(n_subbands=7, dropout=dropout).to(device)
    print(f"Parameters: {model.count_params():,}")

    # Class-weighted loss
    dist    = train_ds.get_class_distribution()
    n_nsr   = dist[0]; n_preaf = dist[1]
    w_preaf = min(n_nsr / max(1, n_preaf), 2.0)
    cw      = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(device)
    print(f"Class weights: NSR=1.00  Pre-AF={w_preaf:.3f}")

    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7
    )

    # Training loop
    history    = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'val_auc': []}
    best_auc   = 0.0
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_epoch(model, tr_loader, criterion, optimizer, device, model_arch)
        va_loss, va_acc, va_auc, _, _ = eval_epoch(model, va_loader, criterion, device, model_arch)

        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_auc'].append(va_auc)

        lr_now = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch:3d}  tr={tr_loss:.4f}/{tr_acc:.3f}  "
              f"va={va_loss:.4f}/{va_acc:.3f}  auc={va_auc:.3f}  lr={lr_now:.2e}")

        scheduler.step(va_auc)

        if va_auc > best_auc:
            best_auc   = va_auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
            # ── Save best model checkpoint ─────────────────────────────────
            torch.save({
                'epoch':       epoch,
                'model_arch':  model_arch,
                'state_dict':  best_state,
                'val_auc':     best_auc,
                'optimizer':   optimizer.state_dict(),
                'config': {
                    'lr': lr, 'dropout': dropout,
                    'batch_size': batch_size, 'seed': seed,
                },
            }, results_dir / 'final_checkpoint.pt')
            print(f"  → Best val AUC: {best_auc:.4f}  (saved)")
        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # Restore best weights
    model.load_state_dict(best_state)

    # Collect val probs for threshold tuning
    _, _, _, va_probs, va_true = eval_epoch(model, va_loader, criterion, device, model_arch)
    _, _, _, te_probs, te_true = eval_epoch(model, te_loader, criterion, device, model_arch)

    te_auc    = roc_auc_score(te_true, te_probs)
    thresh_r  = threshold_analysis(va_probs, va_true, te_probs, te_true)

    # Print threshold analysis
    print(f"\nTest AUC: {te_auc:.4f}")
    for name, m in thresh_r.items():
        print(f"  {name}: Acc={m['accuracy']:.3f} Prec={m['precision']:.3f} "
              f"Rec={m['recall']:.3f} F1={m['f1']:.3f} "
              f"Sens={m['sensitivity']:.3f} Spec={m['specificity']:.3f}")

    # ── Save final model (last epoch weights, for reference) ───────────────
    torch.save({
        'epoch':      epoch,
        'model_arch': model_arch,
        'state_dict': model.state_dict(),
        'val_auc':    best_auc,
        'config': {
            'lr': lr, 'dropout': dropout,
            'batch_size': batch_size, 'seed': seed,
        },
    }, results_dir / 'final_checkpoint.pt')

    metrics = {
        'model_arch':  model_arch,
        'seed':        seed,
        'best_val_auc': float(best_auc),
        'test_auc':    float(te_auc),
        'thresholds':  thresh_r,
        'history':     history,
        'class_weights': {'nsr': 1.0, 'preaf': float(w_preaf)},
    }

    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    save_plots(history, te_probs, te_true, results_dir)
    print(f"Results saved → {results_dir}")
    return metrics


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    DATA_DIR    = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data_final/utrp_dataset_db8_no_denoise')
    RESULTS_ROOT = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_new')

    CONFIGS = [
        # dict(model_arch='single', lr=1e-4, dropout=0.5, epochs=150, patience=25),
        # dict(model_arch='multi',  lr=1e-4, dropout=0.5, epochs=150, patience=25),
        dict(model_arch='hybrid', lr=1e-4, dropout=0.5, epochs=100, patience=20),
    ]

    all_results = {}
    for cfg in CONFIGS:
        arch        = cfg['model_arch']
        results_dir = RESULTS_ROOT / arch   # each arch gets its own subdirectory
        result      = train_model(
            data_dir=DATA_DIR,
            results_dir=results_dir,
            **cfg,
            batch_size=32,
            seed=42,
            augment=True,
        )
        all_results[arch] = result

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"{'Arch':<10} {'Val AUC':>10} {'Test AUC':>10}")
    print('-' * 32)
    for arch, r in all_results.items():
        print(f"{arch:<10} {r['best_val_auc']:>10.4f} {r['test_auc']:>10.4f}")