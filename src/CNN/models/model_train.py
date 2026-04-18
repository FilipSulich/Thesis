import json
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import sys
import os

from src.CNN.models.complexity_test import LateFusionWithAttention, HybridFusionWithAttention

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.CNN.dataset.dataset import ZhuDataset
from models import ZhuResNetSingleScale, TinyCNN


def train_zhu_exact(
    hdf5_path: Path,
    results_dir: Path,
    epochs: int = 100,
    patience: int = 20,
    model_arch: str = 'single',
    lr: float = 1e-5,
    dropout: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
    use_bce_pos_weight: bool = False,
    scheduler_type: str = 'reduce',  # 'reduce' or 'cosine'
    weight_decay: float = 0.0,
) -> Dict:
    torch.manual_seed(seed)
    # np.random.seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)


    device = (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

    print(f"Device: {device}")
    print(f"HDF5: {hdf5_path}")
    print(f"Results: {results_dir}")

    # json_path = "/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/zhu_afpdb/zhu_afpdb_final.split.json"
    json_path = None
    train_ds = ZhuDataset(hdf5_path, split='train', mode=model_arch, augment=False, json_path=json_path)
    val_ds = ZhuDataset(hdf5_path, split='val', mode=model_arch, augment=False, json_path=json_path)
    test_ds = ZhuDataset(hdf5_path, split='test', mode=model_arch, augment=False, json_path=json_path)

    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")
    print(f"Train dist: {train_ds.get_class_distribution()}")
    print(f"Val dist:   {val_ds.get_class_distribution()}")
    print(f"Test dist: {test_ds.get_class_distribution()}")

    x, y = train_ds[0]
    print(f"Shape: {x.shape}")
    print(f"Label: {y}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,shuffle=True, num_workers=0)

    val_loader = DataLoader(val_ds, batch_size=batch_size,shuffle=False, num_workers=0)

    test_loader = DataLoader(test_ds, batch_size=batch_size,shuffle=False, num_workers=0)

    sample_x, sample_y = next(iter(train_loader))
    expected_shapes = {
        'single': '[B, 3, 256, 256]',
        'multi': '[B, 9, 256, 256]',
        'hybrid': '[B, 12, 256, 256]'
    }
    #
    print(f"\nInput shape: {sample_x.shape}  (expected: {expected_shapes[model_arch]})")
    print(f"Label dtype: {sample_y.dtype}  (expected: torch.int64 for CrossEntropyLoss)")

    if model_arch == 'single':
        model = ZhuResNetSingleScale(dropout=dropout).to(device)
        # model = TinyCNN().to(device)
        # model = Tier4MediumCNN(dropout=dropout).to(device)
        # model = Tier5ResNet(dropout=dropout).to(device)
    elif model_arch == 'multi':
        # model = EarlyFusionCNN(dropout=dropout).to(device)
        # model = LateFusionCNN(dropout=dropout).to(device)
        model = LateFusionWithAttention(dropout=dropout).to(device)
    else:  # hybrid
        # model = HybridFusionCNN(dropout=dropout).to(device)
        model = HybridFusionWithAttention(dropout=dropout).to(device)

    print(f"\nModel: ZhuResNet{model_arch.capitalize()}")
    print(f"Parameters: {model.count_params():,}")

    # ── Loss / Pos-weight options ───────────────────────────────────────────
    train_dist = train_ds.get_class_distribution()
    n_nsr = train_dist.get(0, 1)
    n_preaf = train_dist.get(1, 1)
    ratio = float(n_nsr) / float(max(1, n_preaf))
    w_preaf = min(ratio, 2.0)

    print(f"\nClass distribution (train): NSR={n_nsr} Pre-AF={n_preaf} (ratio={ratio:.2f})")

    if use_bce_pos_weight:
        # BCEWithLogitsLoss expects a single logit and float labels
        pos_weight = torch.tensor(max(1.0, ratio), dtype=torch.float32).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using BCEWithLogitsLoss with pos_weight={pos_weight.item():.3f}")
    else:
        class_weights = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0)
        print(f"Using CrossEntropyLoss with class_weights: NSR=1.0 Pre-AF={w_preaf:.3f}")

    # --- Optimizer ---
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)

    # --- Scheduler selection ---
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=1e-7)
        print(f"Scheduler: CosineAnnealingLR(T_max={epochs})")
    else:
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=0.01,
            steps_per_epoch=len(train_loader),
            epochs=100, pct_start=0.3
        )
        print(scheduler)
        print(f"Scheduler: ReduceLROnPlateau(patience=10, factor=0.5)")

    print(f"Optimizer: {optimizer}(lr={lr}, weight_decay={weight_decay})")
    print(f"Early stopping: patience={patience} on val loss")

    # ── Training Loop ──────────────────────────────────────────────────────
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'val_auc': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_val_auc = 0

    best_state = None
    no_improve = 0

    print(f"\n{'=' * 70}")
    print(f"Training")
    print(f"{'=' * 70}")

    for epoch in range(epochs):
        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        tr_loss, tr_correct, tr_total = 0.0, 0, 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False):
            if model_arch == 'hybrid':
                x_ss, x_ms, labels = batch
                x_ss = x_ss.to(device)
                x_ms = x_ms.to(device)
                labels = labels.to(device)
                logits = model(x_ss, x_ms)
                batch_size = len(labels)
            else:
                utrps, labels = batch
                utrps = utrps.to(device)
                labels = labels.to(device)
                logits = model(utrps)
                batch_size = len(labels)

            optimizer.zero_grad()
            if use_bce_pos_weight:
                # convert 2-logit output -> single logit: logit1 - logit0
                single_logit = (logits[:, 1] - logits[:, 0]).squeeze()
                labels_float = labels.float()
                loss = criterion(single_logit, labels_float)
            else:
                loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping (standard practice, not mentioned in paper)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            tr_loss += loss.item() * batch_size
            if use_bce_pos_weight:
                preds_batch = (torch.sigmoid(single_logit) > 0.5).long()
                tr_correct += (preds_batch == labels).sum().item()
            else:
                tr_correct += (logits.argmax(1) == labels).sum().item()
            tr_total += batch_size

        tr_loss /= tr_total
        tr_acc = tr_correct / tr_total

        # ── Validation ─────────────────────────────────────────────────────


        model.eval()
        va_loss, va_correct, va_total = 0.0, 0, 0
        va_probs, va_true = [], []

        with torch.no_grad():
            for batch in val_loader:
                if model_arch == 'hybrid':
                    x_ss, x_ms, labels = batch
                    x_ss = x_ss.to(device)
                    x_ms = x_ms.to(device)
                    labels = labels.to(device)
                    logits = model(x_ss, x_ms)
                else:
                    utrps, labels = batch
                    utrps = utrps.to(device)
                    labels = labels.to(device)
                    logits = model(utrps)

                if use_bce_pos_weight:
                    single_logit = (logits[:, 1] - logits[:, 0]).squeeze()
                    labels_float = labels.float()
                    loss = criterion(single_logit, labels_float)
                else:
                    loss = criterion(logits, labels)

                va_loss += loss.item() * len(labels)
                if use_bce_pos_weight:
                    probs = torch.sigmoid(single_logit).cpu().numpy()
                    va_correct += ((probs > 0.5).astype(int) == labels.cpu().numpy()).sum()
                else:
                    va_correct += (logits.argmax(1) == labels).sum().item()
                va_total += len(labels)

                if use_bce_pos_weight:
                    probs = torch.sigmoid(single_logit).cpu().numpy()
                else:
                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                va_probs.extend(probs)
                va_true.extend(labels.cpu().numpy())

        va_loss /= va_total
        va_acc = va_correct / va_total
        va_auc = roc_auc_score(va_true, va_probs) if len(set(va_true)) > 1 else 0.5

        # Step scheduler
        if scheduler_type == 'cosine':
            scheduler.step()
        else:
            scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)

        # Record history
        history['train_loss'].append(tr_loss)
        history['val_loss'].append(va_loss)
        history['train_acc'].append(tr_acc)
        history['val_acc'].append(va_acc)
        history['val_auc'].append(va_auc)

        print(f"Epoch {epoch + 1:3d}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f} va_auc={va_auc:.3f} lr={current_lr:.6f}")

        # ── Early Stopping (on val loss) ──────────────────────────────────
        if va_loss < best_val_loss:
            best_val_loss = va_loss
            no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            print(f"  → New best val loss: {best_val_loss:.4f}")
        else :
            no_improve += 1

        if va_auc > best_val_auc:
            best_val_auc = va_auc
        #     no_improve = 0
        #     best_state = {k: v.clone() for k, v in model.state_dict().items()}
        #     print(f"  → New best val auc: {best_val_auc:.4f}")
        # else:
        #     no_improve += 1

        if no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1} "
                  f"(best val loss: {best_val_loss:.4f})")
            break

    # ── Restore Best Model ─────────────────────────────────────────────────
    model.load_state_dict(best_state)
    print(f"\nRestored best model (val loss: {best_val_loss:.4f})")

    # Verify restoration BEFORE calibration
    model.eval()
    chk_loss, chk_total = 0.0, 0
    with torch.no_grad():
        for batch in val_loader:
            if model_arch == 'hybrid':
                x_ss, x_ms, labels = batch
                x_ss = x_ss.to(device)
                x_ms = x_ms.to(device)
                labels = labels.to(device)
                logits = model(x_ss, x_ms)
            else:
                utrps, labels = batch
                utrps = utrps.to(device)
                labels = labels.to(device)
                logits = model(utrps)

            if use_bce_pos_weight:
                single_logit = (logits[:, 1] - logits[:, 0]).squeeze()
                labels_float = labels.float()
                loss = criterion(single_logit, labels_float)
            else:
                loss = criterion(logits, labels)
            chk_loss += loss.item() * len(labels)
            chk_total += len(labels)
    chk_loss /= chk_total
    print(f"Restored val loss: {chk_loss:.4f}  (should match {best_val_loss:.4f})")

    eval_model = model

    # ── Test Evaluation ────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"Test Evaluation")
    print(f"{'=' * 70}")

    model.eval()
    te_probs, te_preds, te_true = [], [], []
    import numpy as np

    with torch.no_grad():
        for batch in test_loader:
            # Use calibrated evaluation model to get temperature-scaled logits
            if model_arch == 'hybrid':
                x_ss, x_ms, labels = batch
                x_ss = x_ss.to(device)
                x_ms = x_ms.to(device)
                logits = eval_model(x_ss, x_ms)
            else:
                utrps, labels = batch
                utrps = utrps.to(device)
                logits = eval_model(utrps)
            if use_bce_pos_weight:
                single_logit = (logits[:, 1] - logits[:, 0]).squeeze()
                probs = torch.sigmoid(single_logit).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            if use_bce_pos_weight:
                preds = (probs >= 0.5).astype(int)
            else:
                preds = logits.argmax(1).cpu().numpy()

            te_probs.extend(probs)
            te_preds.extend(preds)
            te_true.extend(labels.numpy())

    te_probs = np.array(te_probs)
    te_preds = np.array(te_preds)
    te_true = np.array(te_true)

    from sklearn.metrics import f1_score, roc_curve
    import numpy as np

    # After training, on val set:
    # Evaluate validation with calibrated model as well
    eval_model.eval()
    va_probs, va_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            if model_arch == 'hybrid':
                x_ss, x_ms, labels = batch
                x_ss = x_ss.to(device)
                x_ms = x_ms.to(device)
                logits = eval_model(x_ss, x_ms)
            else:
                utrps, labels = batch
                utrps = utrps.to(device)
                logits = eval_model(utrps)

            if use_bce_pos_weight:
                single_logit = (logits[:, 1] - logits[:, 0]).squeeze()
                probs = torch.sigmoid(single_logit).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            va_probs.extend(probs)
            va_true.extend(labels.numpy())

    va_probs = np.array(va_probs)
    va_true = np.array(va_true)

    # Find threshold that maximises F1 on val set
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = [f1_score(va_true, (va_probs >= t).astype(int), zero_division=0)
                 for t in thresholds]
    best_thresh = thresholds[np.argmax(f1_scores)]
    best_f1_val = max(f1_scores)
    print(f"Optimal threshold: {best_thresh:.2f}  Val F1: {best_f1_val:.3f}")

    # Also find threshold for sensitivity=specificity (Youden's J)
    fpr, tpr, roc_thresh = roc_curve(va_true, va_probs)
    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_thresh = roc_thresh[youden_idx]
    print(f"Youden threshold:  {youden_thresh:.2f}  "
          f"Sens={tpr[youden_idx]:.3f}  Spec={1 - fpr[youden_idx]:.3f}")

    # Apply both thresholds to test set
    for name, thresh in [('Default (0.5)', 0.5),
                         (f'F1-optimal ({best_thresh:.2f})', best_thresh),
                         (f'Youden ({youden_thresh:.2f})', youden_thresh)]:
        preds = (te_probs >= thresh).astype(int)
        from sklearn.metrics import classification_report
        print(f"\n--- {name} ---")
        print(classification_report(te_true, preds,target_names=['NSR', 'Pre-AF']))
    print("Probabilities:")
    print(f"  Pre-AF (label=1) mean prob: {te_probs[te_true == 1].mean():.3f}")
    print(f"  NSR (label=0) mean prob:    {te_probs[te_true == 0].mean():.3f}")

    # Use optimal threshold for final metrics
    te_preds_optimal = (te_probs >= best_thresh).astype(int)

    # Confusion matrix with optimal threshold
    cm = confusion_matrix(te_true, te_preds_optimal)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Metrics using optimal threshold
    metrics = {
        'accuracy': accuracy_score(te_true, te_preds_optimal),
        'precision': precision_score(te_true, te_preds_optimal, zero_division=0),
        'recall': recall_score(te_true, te_preds_optimal, zero_division=0),
        'f1': f1_score(te_true, te_preds_optimal, zero_division=0),
        'auc': roc_auc_score(te_true, te_probs),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'best_val_loss': best_val_loss,
        'optimal_threshold': best_thresh,
    }

    # Save raw prediction arrays (useful for offline diagnostics)
    try:
        np.save(results_dir / 'te_true.npy', te_true)
        np.save(results_dir / 'te_prob.npy', te_probs)
        np.save(results_dir / 'va_true.npy', va_true)
        np.save(results_dir / 'va_prob.npy', va_probs)
    except Exception as e:
        print(f"Warning: failed to save prediction arrays: {e}")

    print(f"\n{'=' * 70}")
    print(f"TEST RESULTS")
    print(f"{'=' * 70}")
    for k, v in metrics.items():
        print(f"  {k:15s}: {v:.4f}  ({v * 100:.2f}%)")
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"            NSR  Pre-AF")
    print(f"  Actual NSR   {tn:4d}  {fp:4d}")
    print(f"        Pre-AF {fn:4d}  {tp:4d}")

    # ── Analyze Attention Weights (if model supports it) ──────────────────
    if hasattr(model, 'get_attention_weights'):
        print(f"\n{'=' * 70}")
        print(f"ATTENTION WEIGHTS ANALYSIS")
        print(f"{'=' * 70}")

        # Collect attention weights across test set
        model.eval()
        all_attention_weights = []

        with torch.no_grad():
            for batch in test_loader:
                if model_arch == 'hybrid':
                    x_ss, x_ms, labels = batch
                    x_ss = x_ss.to(device)
                    x_ms = x_ms.to(device)
                    # Forward pass to update attention weights
                    _ = model(x_ss, x_ms)
                    # Get attention weights (method signature varies by model)
                    try:
                        # Try as a method with input
                        attn_weights = model.get_attention_weights(torch.cat([x_ss, x_ms], dim=1))
                    except TypeError:
                        # Try as a method without input (stored in model)
                        attn_weights = model.get_attention_weights()
                else:
                    utrps, labels = batch
                    utrps = utrps.to(device)
                    # Forward pass to update attention weights
                    _ = model(utrps)
                    # Get attention weights (method signature varies by model)
                    try:
                        # Try as a method with input
                        attn_weights = model.get_attention_weights(utrps)
                    except TypeError:
                        # Try as a method without input (stored in model)
                        attn_weights = model.get_attention_weights()

                all_attention_weights.append(attn_weights.cpu())

        # Stack and compute mean across all test samples
        all_attention_weights = torch.cat(all_attention_weights, dim=0)
        mean_attention = all_attention_weights.mean(dim=0)

        # Print mean attention weight for each subband
        subband_names = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
        print(f"\nMean attention weight per subband (across test set):")
        print(f"{'Subband':<10} {'Weight':>10}")
        print("-" * 21)

        for i, name in enumerate(subband_names[:len(mean_attention)]):
            weight = mean_attention[i].item()
            print(f"{name:<10} {weight:>10.4f}")

        # Identify most and least attended subbands
        max_idx = mean_attention.argmax().item()
        min_idx = mean_attention.argmin().item()

        print(f"\nMost attended subband:  {subband_names[max_idx]} ({mean_attention[max_idx].item():.4f})")
        print(f"Least attended subband: {subband_names[min_idx]} ({mean_attention[min_idx].item():.4f})")

        # Save attention weights to metrics
        metrics['attention_weights'] = {
            name: mean_attention[i].item()
            for i, name in enumerate(subband_names[:len(mean_attention)])
        }

    torch.save({
        'state_dict': model.state_dict(),
        'metrics': metrics,
        'history': history,
        'config': {
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'patience': patience,
            'seed': seed,
        }
    }, results_dir / 'zhu_exact_best.pt')

    # Save metrics JSON
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Plot training curves
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ep = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(ep, history['train_loss'], label='Train')
    axes[0].plot(ep, history['val_loss'], label='Val')
    axes[0].axhline(best_val_loss, color='red', ls='--',
                    label=f'Best={best_val_loss:.3f}')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(ep, history['train_acc'], label='Train')
    axes[1].plot(ep, history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Val AUC
    axes[2].plot(ep, history['val_auc'], color='green', label='Val AUC')
    axes[2].set_title('Validation AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    plt.close()

    fpr, tpr, _ = roc_curve(te_true, te_probs)
    roc_auc = metrics['auc']

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: ({model_arch})')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'roc_curve.png', dpi=150)
    plt.close()

    # Plot training curves
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))  # Changed from 1, 3 to 1, 4
    ep = range(1, len(history['train_loss']) + 1)

    # 1. Loss
    axes[0].plot(ep, history['train_loss'], label='Train')
    axes[0].plot(ep, history['val_loss'], label='Val')
    axes[0].axhline(best_val_loss, color='red', ls='--', label=f'Best={best_val_loss:.3f}')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # 2. Accuracy
    axes[1].plot(ep, history['train_acc'], label='Train')
    axes[1].plot(ep, history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # 3. Val AUC
    axes[2].plot(ep, history['val_auc'], color='green', label='Val AUC')
    axes[2].set_title('Validation AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(0, 1)
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    # 4. Learning Rate (NEW)
    axes[3].plot(ep, history['lr'], color='purple', label='LR')
    axes[3].set_title('Learning Rate')
    axes[3].set_xlabel('Epoch')
    axes[3].set_yscale('log')  # Use log scale because LR changes by orders of magnitude
    axes[3].legend()
    axes[3].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    plt.close()

    print(f"\nSaved results → {results_dir}")
    print(f"  - zhu_exact_best.pt")
    print(f"  - metrics.json")
    print(f"  - training_curves.png")

    return {'metrics': metrics, 'history': history, 'model': model}


def train_with_kfold(
    hdf5_path: Path,
    results_dir: Path,
    n_folds: int = 5,
    epochs: int = 100,
    patience: int = 20,
    model_arch: str = 'single',
    lr: float = 1e-5,
    dropout: float = 0.5,
    batch_size: int = 32,
    seed: int = 42,
) -> Dict:
    """
    Train with k-fold cross-validation on train+val, evaluate on test.

    Args:
        n_folds: Number of folds for cross-validation
        ... (other args same as train_zhu_exact)

    Returns:
        Dict with aggregated metrics across folds and test performance
    """
    import numpy as np

    torch.manual_seed(seed)
    np.random.seed(seed)

    results_dir.mkdir(parents=True, exist_ok=True)
    device = (torch.device('mps') if torch.backends.mps.is_available()
              else torch.device('cpu'))

    print(f"{'=' * 70}")
    print(f"K-FOLD CROSS-VALIDATION (k={n_folds})")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"HDF5: {hdf5_path}")
    print(f"Results: {results_dir}")

    # Load datasets
    # We'll combine train+val for k-fold CV, keep test separate
    train_ds = ZhuDataset(hdf5_path, split='train', mode=model_arch,
                          augment=False, json_path=None)  # Will add augmentation per fold
    val_ds = ZhuDataset(hdf5_path, split='val', mode=model_arch,
                        augment=False, json_path=None)
    test_ds = ZhuDataset(hdf5_path, split='test', mode=model_arch,
                         augment=False, json_path=None)

    # Combine train+val for cross-validation
    # We need to get all indices and labels
    from torch.utils.data import ConcatDataset
    trainval_ds = ConcatDataset([train_ds, val_ds])

    # Extract labels for stratified splitting
    trainval_labels = []
    for i in range(len(train_ds)):
        _, label = train_ds[i]
        trainval_labels.append(label)
    for i in range(len(val_ds)):
        _, label = val_ds[i]
        trainval_labels.append(label)
    trainval_labels = np.array(trainval_labels)

    print(f"\nDataset sizes:")
    print(f"  Train+Val: {len(trainval_ds)} samples")
    print(f"  Test: {len(test_ds)} samples")
    print(f"  Class distribution (train+val): "
          f"NSR={np.sum(trainval_labels==0)}, Pre-AF={np.sum(trainval_labels==1)}")

    # Stratified K-Fold
    skf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    fold_results = []
    fold_models = []

    # Perform k-fold cross-validation
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(trainval_ds)),
                                                            trainval_labels)):
        print(f"\n{'=' * 70}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'=' * 70}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")

        # Create subset datasets for this fold
        train_subset = Subset(trainval_ds, train_idx)
        val_subset = Subset(trainval_ds, val_idx)

        # Create data loaders
        train_loader = DataLoader(train_subset, batch_size=batch_size,
                                   shuffle=True, num_workers=0)
        val_loader = DataLoader(val_subset, batch_size=batch_size,
                                shuffle=False, num_workers=0)

        # Initialize model for this fold
        if model_arch == 'single':
            model = ZhuResNetSingleScale(dropout=dropout).to(device)
        elif model_arch == 'multi':
            model = LateFusionWithAttention(n_subbands=9, dropout=dropout).to(device)
        else:  # hybrid
            model = HybridFusionWithAttention(n_subbands=9, dropout=dropout).to(device)

        print(f"  Model: {model.__class__.__name__}")
        print(f"  Parameters: {model.count_params():,}")

        # Loss and optimizer
        train_labels_fold = trainval_labels[train_idx]
        n_nsr = np.sum(train_labels_fold == 0)
        n_preaf = np.sum(train_labels_fold == 1)
        w_preaf = min(n_nsr / n_preaf, 2.0)
        class_weights = torch.tensor([1.0, w_preaf], dtype=torch.float32).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        print(f"  Class weights: NSR=1.00, Pre-AF={w_preaf:.2f}")

        # Training loop for this fold
        history = {
            'train_loss': [], 'val_loss': [], 'train_acc': [],
            'val_acc': [], 'val_auc': [], 'lr': []
        }

        best_val_loss = float('inf')
        best_val_auc = 0
        best_state = None
        no_improve = 0

        for epoch in range(epochs):
            # Train
            model.train()
            tr_loss, tr_correct, tr_total = 0.0, 0, 0

            for batch in tqdm(train_loader,
                             desc=f'Fold {fold+1} Epoch {epoch+1}/{epochs} [Train]',
                             leave=False):
                if model_arch == 'hybrid':
                    x_ss, x_ms, labels = batch
                    x_ss, x_ms, labels = x_ss.to(device), x_ms.to(device), labels.to(device)
                    logits = model(x_ss, x_ms)
                    batch_size_actual = len(labels)
                else:
                    utrps, labels = batch
                    utrps, labels = utrps.to(device), labels.to(device)
                    logits = model(utrps)
                    batch_size_actual = len(labels)

                optimizer.zero_grad()
                loss = criterion(logits, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                tr_loss += loss.item() * batch_size_actual
                tr_correct += (logits.argmax(1) == labels).sum().item()
                tr_total += batch_size_actual

            tr_loss /= tr_total
            tr_acc = tr_correct / tr_total

            # Validation
            model.eval()
            va_loss, va_correct, va_total = 0.0, 0, 0
            va_probs, va_true = [], []

            with torch.no_grad():
                for batch in val_loader:
                    if model_arch == 'hybrid':
                        x_ss, x_ms, labels = batch
                        x_ss, x_ms, labels = x_ss.to(device), x_ms.to(device), labels.to(device)
                        logits = model(x_ss, x_ms)
                    else:
                        utrps, labels = batch
                        utrps, labels = utrps.to(device), labels.to(device)
                        logits = model(utrps)

                    loss = criterion(logits, labels)
                    va_loss += loss.item() * len(labels)
                    va_correct += (logits.argmax(1) == labels).sum().item()
                    va_total += len(labels)

                    probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                    va_probs.extend(probs)
                    va_true.extend(labels.cpu().numpy())

            va_loss /= va_total
            va_acc = va_correct / va_total
            va_auc = roc_auc_score(va_true, va_probs) if len(set(va_true)) > 1 else 0.5

            scheduler.step(va_loss)
            current_lr = optimizer.param_groups[0]['lr']

            history['train_loss'].append(tr_loss)
            history['val_loss'].append(va_loss)
            history['train_acc'].append(tr_acc)
            history['val_acc'].append(va_acc)
            history['val_auc'].append(va_auc)
            history['lr'].append(current_lr)

            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f"  Epoch {epoch+1:3d}: tr_acc = {tr_acc} tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} va_acc={va_acc:.3f} va_auc={va_auc:.3f}")

            # Early stopping
            if va_loss < best_val_loss:
                best_val_loss = va_loss
                best_val_auc = va_auc
                no_improve = 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                no_improve += 1

            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

        # Restore best model for this fold
        model.load_state_dict(best_state)

        # Save fold results
        fold_result = {
            'fold': fold + 1,
            'best_val_loss': best_val_loss,
            'best_val_auc': best_val_auc,
            'history': history
        }
        fold_results.append(fold_result)
        fold_models.append(model.state_dict())

        print(f"  Fold {fold+1} best val loss: {best_val_loss:.4f}, AUC: {best_val_auc:.4f}")

    # Aggregate fold results
    print(f"\n{'=' * 70}")
    print(f"CROSS-VALIDATION RESULTS")
    print(f"{'=' * 70}")

    val_losses = [f['best_val_loss'] for f in fold_results]
    val_aucs = [f['best_val_auc'] for f in fold_results]

    print(f"Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
    print(f"Val AUC:  {np.mean(val_aucs):.4f} ± {np.std(val_aucs):.4f}")

    # ── Evaluate on Test Set ───────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"TEST SET EVALUATION (averaged across {n_folds} folds)")
    print(f"{'=' * 70}")

    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # Evaluate each fold model on test set
    fold_test_metrics = []

    for fold_idx, model_state in enumerate(fold_models):
        # Load fold model
        if model_arch == 'single':
            model = ZhuResNetSingleScale(dropout=dropout).to(device)
        elif model_arch == 'multi':
            model = LateFusionWithAttention(n_subbands=9, dropout=dropout).to(device)
        else:  # hybrid
            model = HybridFusionWithAttention(n_subbands=9, dropout=dropout).to(device)

        model.load_state_dict(model_state)
        model.eval()

        te_probs, te_true = [], []

        with torch.no_grad():
            for batch in test_loader:
                if model_arch == 'hybrid':
                    x_ss, x_ms, labels = batch
                    x_ss, x_ms = x_ss.to(device), x_ms.to(device)
                    logits = model(x_ss, x_ms)
                else:
                    utrps, labels = batch
                    utrps = utrps.to(device)
                    logits = model(utrps)

                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                te_probs.extend(probs)
                te_true.extend(labels.numpy())

        te_probs = np.array(te_probs)
        te_true = np.array(te_true)

        # Find optimal threshold using validation predictions from this fold
        # (We stored the best model for each fold, so we use threshold 0.5 for test)
        te_preds = (te_probs >= 0.5).astype(int)

        # Compute metrics
        test_auc = roc_auc_score(te_true, te_probs)
        test_f1 = f1_score(te_true, te_preds, zero_division=0)
        test_acc = accuracy_score(te_true, te_preds)
        test_recall = recall_score(te_true, te_preds, zero_division=0)
        test_precision = precision_score(te_true, te_preds, zero_division=0)

        fold_test_metrics.append({
            'fold': fold_idx + 1,
            'auc': test_auc,
            'f1': test_f1,
            'accuracy': test_acc,
            'recall': test_recall,
            'precision': test_precision,
        })

    # Aggregate test metrics
    test_aucs = [m['auc'] for m in fold_test_metrics]
    test_f1s = [m['f1'] for m in fold_test_metrics]
    test_accs = [m['accuracy'] for m in fold_test_metrics]
    test_recalls = [m['recall'] for m in fold_test_metrics]
    test_precisions = [m['precision'] for m in fold_test_metrics]

    print(f"Test AUC:       {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
    print(f"Test F1:        {np.mean(test_f1s):.4f} ± {np.std(test_f1s):.4f}")
    print(f"Test Accuracy:  {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")
    print(f"Test Recall:    {np.mean(test_recalls):.4f} ± {np.std(test_recalls):.4f}")
    print(f"Test Precision: {np.mean(test_precisions):.4f} ± {np.std(test_precisions):.4f}")

    # Save all results
    cv_results = {
        'n_folds': n_folds,
        'fold_results': fold_results,
        'mean_val_loss': float(np.mean(val_losses)),
        'std_val_loss': float(np.std(val_losses)),
        'mean_val_auc': float(np.mean(val_aucs)),
        'std_val_auc': float(np.std(val_aucs)),
        'test_metrics': {
            'mean_auc': float(np.mean(test_aucs)),
            'std_auc': float(np.std(test_aucs)),
            'mean_f1': float(np.mean(test_f1s)),
            'std_f1': float(np.std(test_f1s)),
            'mean_accuracy': float(np.mean(test_accs)),
            'std_accuracy': float(np.std(test_accs)),
            'mean_recall': float(np.mean(test_recalls)),
            'std_recall': float(np.std(test_recalls)),
            'mean_precision': float(np.mean(test_precisions)),
            'std_precision': float(np.std(test_precisions)),
            'fold_metrics': fold_test_metrics,
        },
        'config': {
            'lr': lr,
            'dropout': dropout,
            'batch_size': batch_size,
            'patience': patience,
            'seed': seed,
        }
    }

    # Save CV results
    with open(results_dir / 'cv_results.json', 'w') as f:
        json.dump(cv_results, f, indent=2)

    # Save all fold models
    torch.save({
        'fold_models': fold_models,
        'cv_results': cv_results,
    }, results_dir / 'cv_models.pt')

    print(f"\nSaved cross-validation results → {results_dir}")
    print(f"  - cv_results.json")
    print(f"  - cv_models.pt")

    return cv_results


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--hdf5', type=Path, default=Path('/data/data_new/attention/utrp_new_denoise2.h5'))
    parser.add_argument('--results_dir', type=Path, default=Path('/results_new/attention_run'))
    parser.add_argument('--model_arch', choices=['single', 'multi', 'hybrid'], default='single')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use_bce_pos_weight', action='store_true')
    parser.add_argument('--scheduler_type', choices=['reduce', 'cosine'], default='reduce')
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--use_kfold', action='store_true')
    parser.add_argument('--n_folds', type=int, default=5)

    args = parser.parse_args()

    if args.use_kfold:
        train_with_kfold(
            hdf5_path=args.hdf5,
            results_dir=args.results_dir,
            n_folds=args.n_folds,
            epochs=args.epochs,
            patience=args.patience,
            model_arch=args.model_arch,
            lr=args.lr,
            dropout=args.dropout,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    else:
        train_zhu_exact(
            hdf5_path=args.hdf5,
            results_dir=args.results_dir,
            epochs=args.epochs,
            patience=args.patience,
            model_arch=args.model_arch,
            lr=args.lr,
            dropout=args.dropout,
            batch_size=args.batch_size,
            seed=args.seed,
            use_bce_pos_weight=args.use_bce_pos_weight,
            scheduler_type=args.scheduler_type,
            weight_decay=args.weight_decay,
        )

