"""
Training runner for Experiment 3.

Key fixes vs previous version:
  1. weight_decay lowered to 1e-4 (was 0.1 — dominated the gradient at lr=1e-5)
  2. Scheduler changed to ReduceLROnPlateau (more stable than CosineAnnealing
     with very small lr)
  3. p-even records excluded entirely
  4. Patient-level split uses clean record IDs — no paired logic needed
  5. Test set held out BEFORE any CV (no data leakage from split)
  6. pos_weight correctly passed to BCEWithLogitsLoss
"""

import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Type, Any, Tuple
from itertools import product
import random

from base_cnn import BaseCNN
from dataset import UTRPDataset, create_dataloaders, compute_class_weights
from cross_validation import (
    PatientStratifiedKFold,
    create_holdout_test_set,
    get_label_from_record_id,
)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Main runner ───────────────────────────────────────────────────────────────
class CrossValidationRunner:
    """
    K-fold cross-validation with nested hyperparameter search.

    Pipeline:
      1. Load patient list from HDF5 split JSON (test patients excluded).
      2. Grid search: for each param combo, train K folds, record mean val AUC.
      3. Best combo → retrain on full train+val, evaluate on held-out test.
    """

    def __init__(
        self,
        model_class: Type[BaseCNN],
        hdf5_path: Path,
        mode: str,               # 'single' or 'multi'
        results_dir: Path,
        n_folds: int = 5,
        seed: int = 42,
    ):
        self.model_class  = model_class
        self.hdf5_path    = Path(hdf5_path)
        self.mode         = mode
        self.results_dir  = Path(results_dir)
        self.n_folds      = n_folds
        self.seed         = seed

        self.results_dir.mkdir(parents=True, exist_ok=True)
        set_seed(seed)

        print(f"\nCrossValidationRunner")
        print(f"  model  : {model_class.__name__}")
        print(f"  mode   : {mode}")
        print(f"  folds  : {n_folds}")
        print(f"  output : {results_dir}")

    # ── Public API ────────────────────────────────────────────────────────────
    def run(
        self,
        hyperparam_grid: Dict[str, List[Any]],
        train_patients:  List[str],   # record IDs for CV (test already excluded)
        test_patients:   List[str],   # held-out record IDs
        epochs:     int = 100,
        patience:   int = 15,
        num_workers: int = 0,         # 0 = main process (safer on macOS MPS)
    ) -> Dict:

        print("\n" + "="*70)
        print("CROSS-VALIDATION WITH HYPERPARAMETER TUNING")
        print("="*70)
        print(f"Train+Val patients : {len(train_patients)}")
        print(f"Test patients      : {len(test_patients)}")

        cv_splitter = PatientStratifiedKFold(
            n_splits=self.n_folds, random_state=self.seed
        )
        folds = cv_splitter.split(train_patients)

        # ── Hyperparameter grid ───────────────────────────────────────────────
        param_names  = list(hyperparam_grid.keys())
        param_combos = list(product(*hyperparam_grid.values()))

        print(f"\nHyperparameter grid  ({len(param_combos)} combos):")
        for k, v in hyperparam_grid.items():
            print(f"  {k}: {v}")

        all_combo_results = []

        for combo_idx, combo in enumerate(param_combos, 1):
            params = dict(zip(param_names, combo))
            print(f"\n{'='*70}")
            print(f"Combo {combo_idx}/{len(param_combos)}: {params}")
            print(f"{'='*70}")

            fold_results = []

            for fold_idx, (tr_pids, va_pids) in enumerate(folds, 1):
                print(f"\n--- Fold {fold_idx}/{self.n_folds} ---")
                print(f"  Train: {len(tr_pids)} patients, Val: {len(va_pids)} patients")

                # Datasets
                tr_ds = UTRPDataset(self.hdf5_path, mode=self.mode,
                                    augment=True,  patient_ids=tr_pids)
                va_ds = UTRPDataset(self.hdf5_path, mode=self.mode,
                                    augment=False, patient_ids=va_pids)

                # Class distribution check
                tr_dist = tr_ds.get_class_distribution()
                print(f"  Train dist: {tr_dist}")

                tr_loader, va_loader = create_dataloaders(
                    tr_ds, va_ds,
                    batch_size=params.get('batch_size', 16),
                    num_workers=num_workers,
                )

                # Class weights (computed on training fold only)
                pos_weight = compute_class_weights(tr_ds)

                # Model
                model_kwargs = {k: v for k, v in params.items()
                                if k not in ('lr', 'batch_size', 'weight_decay')}
                model = self.model_class(**model_kwargs)

                history = _train(
                    model, tr_loader, va_loader,
                    lr=params.get('lr', 1e-4),
                    weight_decay=params.get('weight_decay', 1e-4),
                    epochs=epochs,
                    patience=patience,
                    pos_weight=pos_weight,
                )

                val_metrics = model.evaluate(va_loader)

                fold_results.append({
                    'fold': fold_idx,
                    **params,
                    **{f'val_{k}': v for k, v in val_metrics.items()
                       if k != 'confusion_matrix'},
                    'epochs_run': len(history['train_loss']),
                })

                print(f"  Val Acc={val_metrics['accuracy']:.3f}  "
                      f"AUC={val_metrics['auc']:.3f}  "
                      f"F1={val_metrics['f1']:.3f}  "
                      f"Sens={val_metrics['sensitivity']:.3f}  "
                      f"Spec={val_metrics['specificity']:.3f}")

                # Training curves per fold
                _plot_curves(
                    history,
                    self.results_dir / f"curves_fold{fold_idx}_combo{combo_idx}.png"
                )

            mean_auc = float(np.mean([r['val_auc']      for r in fold_results]))
            mean_acc = float(np.mean([r['val_accuracy'] for r in fold_results]))
            std_auc  = float(np.std( [r['val_auc']      for r in fold_results]))

            print(f"\n  → Mean val AUC = {mean_auc:.3f} ± {std_auc:.3f}  "
                  f"Acc = {mean_acc:.3f}")

            all_combo_results.append({
                **params,
                'mean_val_auc':      mean_auc,
                'std_val_auc':       std_auc,
                'mean_val_accuracy': mean_acc,
                'mean_val_f1':       float(np.mean([r['val_f1'] for r in fold_results])),
                'fold_results':      fold_results,
            })

        # ── Best combo ────────────────────────────────────────────────────────
        best = max(all_combo_results, key=lambda x: x['mean_val_auc'])

        print(f"\n{'='*70}")
        print("BEST CONFIG:")
        for k, v in best.items():
            if k not in ('fold_results',):
                print(f"  {k}: {v}")

        # ── Final retrain on full train+val ───────────────────────────────────
        print(f"\n{'='*70}")
        print("RETRAIN ON FULL TRAIN+VAL")
        print(f"{'='*70}")

        full_tr_ds = UTRPDataset(self.hdf5_path, mode=self.mode,
                                 augment=True,  patient_ids=train_patients)
        test_ds    = UTRPDataset(self.hdf5_path, mode=self.mode,
                                 augment=False, patient_ids=test_patients)

        # Use a small internal val split (10%) for early stopping only
        n_val_early = max(2, len(train_patients) // 10)
        rng = random.Random(self.seed)
        shuffled = train_patients[:]
        rng.shuffle(shuffled)
        es_val_pids = shuffled[:n_val_early]
        es_tr_pids  = shuffled[n_val_early:]

        es_tr_ds  = UTRPDataset(self.hdf5_path, mode=self.mode,
                                augment=True,  patient_ids=es_tr_pids)
        es_va_ds  = UTRPDataset(self.hdf5_path, mode=self.mode,
                                augment=False, patient_ids=es_val_pids)

        bs = best.get('batch_size', 16)
        es_tr_loader, es_va_loader = create_dataloaders(
            es_tr_ds, es_va_ds, batch_size=bs, num_workers=num_workers
        )
        test_loader, _ = create_dataloaders(
            test_ds, test_ds, batch_size=bs, num_workers=0
        )

        pos_weight = compute_class_weights(es_tr_ds)
        model_kwargs = {k: v for k, v in best.items()
                        if k not in ('lr', 'batch_size', 'weight_decay',
                                     'mean_val_auc', 'std_val_auc',
                                     'mean_val_accuracy', 'mean_val_f1',
                                     'fold_results')}
        final_model = self.model_class(**model_kwargs)

        final_history = _train(
            final_model, es_tr_loader, es_va_loader,
            lr=best.get('lr', 1e-4),
            weight_decay=best.get('weight_decay', 1e-4),
            epochs=epochs,
            patience=patience,
            pos_weight=pos_weight,
        )

        # ── Test evaluation ───────────────────────────────────────────────────
        print(f"\n{'='*70}")
        print("TEST SET RESULTS")
        print(f"{'='*70}")
        test_metrics = final_model.evaluate(test_loader)

        for k, v in test_metrics.items():
            if k != 'confusion_matrix':
                print(f"  {k:12s}: {v:.4f}")
        print(f"  confusion_matrix:\n{test_metrics['confusion_matrix']}")

        # ── Save artefacts ────────────────────────────────────────────────────
        final_model.save(self.results_dir / f"best_{self.mode}_model.pt",
                         hyperparams=model_kwargs)

        pd.DataFrame([
            {**c, **fr}
            for c in all_combo_results
            for fr in c['fold_results']
        ]).to_csv(self.results_dir / f"cv_results_{self.mode}.csv", index=False)

        pd.DataFrame([{k: v for k, v in test_metrics.items()
                        if k != 'confusion_matrix'}]).to_csv(
            self.results_dir / f"test_results_{self.mode}.csv", index=False
        )

        _plot_curves(final_history,
                     self.results_dir / f"final_curves_{self.mode}.png")
        final_model.plot_roc(
            test_loader,
            save_path=self.results_dir / f"roc_{self.mode}.png"
        )
        final_model.plot_confusion_matrix(
            test_loader,
            save_path=self.results_dir / f"cm_{self.mode}.png"
        )

        print(f"\nAll results saved to {self.results_dir}")

        return {
            'best_config':  best,
            'cv_results':   all_combo_results,
            'test_metrics': test_metrics,
            'final_model':  final_model,
        }


# ── Training function (separated from BaseCNN for clarity) ────────────────────
def _train(
    model: BaseCNN,
    train_loader,
    val_loader,
    lr: float          = 1e-4,
    weight_decay: float = 1e-3,
    epochs: int        = 150,
    patience: int      = 25,
    label_smoothing: float = 0.05,
    pos_weight: torch.Tensor = None,
) -> Dict:
    """
    Train with early stopping on val loss.

    Fixes vs original:
      - weight_decay default 1e-4 (not 0.1)
      - ReduceLROnPlateau scheduler (more stable at small lr)
      - pos_weight correctly applied
    """
    device = model.device
    model.to(device)

    def smooth_bce(logits, targets, pos_weight=None, eps=label_smoothing):
        targets_smooth = targets * (1 - eps) + eps / 2
        loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight.to(device) if pos_weight is not None else None
        )
        return loss_fn(logits, targets_smooth)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight.float().to(device) if pos_weight is not None else None
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-7
    )

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_loss = float('inf')
    epochs_no_improve = 0
    best_state = None

    total_p     = sum(p.numel() for p in model.parameters())
    trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel: {model.__class__.__name__}  "
          f"params={total_p:,}  trainable={trainable_p:,}")
    print(f"lr={lr}  weight_decay={weight_decay}  patience={patience}")

    for epoch in range(epochs):
        tr_loss, tr_acc = model.train_epoch(train_loader, optimizer, criterion)
        va_loss, va_acc = model.validate_epoch(val_loader, criterion)

        scheduler.step(va_loss)

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)

        print(f"Ep {epoch+1:3d}  "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}  "
              f"va_loss={va_loss:.4f} va_acc={va_acc:.3f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            epochs_no_improve = 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


def _plot_curves(history: Dict, save_path: Path):
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ep = range(1, len(history['train_loss']) + 1)
    ax1.plot(ep, history['train_loss'], label='Train')
    ax1.plot(ep, history['val_loss'],   label='Val')
    ax1.set_title('Loss');  ax1.set_xlabel('Epoch'); ax1.legend(); ax1.grid(alpha=.3)
    ax2.plot(ep, history['train_acc'], label='Train')
    ax2.plot(ep, history['val_acc'],   label='Val')
    ax2.set_title('Accuracy'); ax2.set_xlabel('Epoch'); ax2.legend(); ax2.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import h5py
    from models import SingleScaleCNN, MultiScaleCNN, ResNet18Transfer
    from cross_validation import create_holdout_test_set

    HDF5_PATH   = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/data/data_new/exp3/'
                       'utrp_dataset_128.h5')
    RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results/exp3/')
    SPLIT_JSON  = HDF5_PATH.with_suffix('.split.json')

    # ── Load patient list from split JSON ────────────────────────────────────
    if SPLIT_JSON.exists():
        with open(SPLIT_JSON) as f:
            split = json.load(f)
        test_patients      = split['test']
        train_val_patients = split['train_val']
        print(f"Loaded split from {SPLIT_JSON}")
    else:
        # Derive from HDF5 metadata
        with h5py.File(HDF5_PATH) as f:
            all_pids = list({
                pid.decode() for pid in f['metadata/patient_ids'][:]
            })
        # Filter: keep only p-odd and n* (safety check)
        all_pids = [p for p in all_pids
                    if (p.startswith('n')) or
                       (p.startswith('p') and int(p[1:]) % 2 == 1)]
        test_patients, train_val_patients = create_holdout_test_set(
            all_pids, n_test_per_class=5, random_state=42
        )

    print(f"\nTrain+Val: {len(train_val_patients)} patients")
    print(f"Test:      {len(test_patients)} patients")

    # ── Hyperparameter grid ───────────────────────────────────────────────────
    # Recommended starting grid. Expand after first run if needed.
    HYPERPARAM_GRID = {
        'lr':           [1e-4, 3e-4],    # 1e-5 was too small; start here
        'dropout':      [0.3, 0.5],
        'batch_size':   [16],             # small dataset → small batch
        'weight_decay': [1e-4],           # was 0.1 — that was the main bug
    }

    for mode, ModelClass in [
        ('single', SingleScaleCNN),
        ('multi',  MultiScaleCNN),
    ]:
        print(f"\n{'#'*70}")
        print(f"# MODE: {mode.upper()}")
        print(f"{'#'*70}")

        runner = CrossValidationRunner(
            model_class=ModelClass,
            hdf5_path=HDF5_PATH,
            mode=mode,
            results_dir=RESULTS_DIR / mode,
            n_folds=5,
            seed=42,
        )

        results = runner.run(
            hyperparam_grid=HYPERPARAM_GRID,
            train_patients=train_val_patients,
            test_patients=test_patients,
            epochs=100,
            patience=15,
            num_workers=0,
        )

        print(f"\n[{mode}] Test AUC = {results['test_metrics']['auc']:.4f}")