"""
Post-training analysis: attention weights + ROC curves from saved CV checkpoints.
Run after k-fold CV training is complete — no retraining needed.
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import auc as sklearn_auc, roc_curve


SUBBAND_KEYS = ['D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']


# ══════════════════════════════════════════════════════════════════════════════
# Attention weights
# ══════════════════════════════════════════════════════════════════════════════

def extract_attention_weights(save_dir: Path,
                               n_folds: int = 10,
                               subband_keys: list = SUBBAND_KEYS) -> np.ndarray:
    """
    Extract learned attention weights from saved fold checkpoints.
    Uses sigmoid(final_linear_bias) — the learned baseline weight per subband.

    Args:
        save_dir:     directory containing fold_N_best.pt checkpoints
        n_folds:      number of folds
        subband_keys: subband names in order

    Returns:
        fold_weights: (n_folds, n_subbands) float array
    """
    fold_weights = []

    for fold in range(1, n_folds + 1):
        ckpt_path = save_dir / f'fold_{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        ckpt  = torch.load(ckpt_path, map_location='cpu')
        state = ckpt['state_dict']

        # attention.2.bias: (n_subbands,) — bias of final Linear before Sigmoid
        bias         = state['attention.2.bias'].numpy()
        attn_weights = torch.sigmoid(torch.tensor(bias)).numpy()
        fold_weights.append(attn_weights)

        print(f"  Fold {fold}  (epoch {ckpt['epoch']}, "
              f"val AUC {ckpt['val_auc']:.4f}):")
        for name, w in zip(subband_keys, attn_weights):
            bar = '█' * int(w * 20)
            print(f"    {name}: {w:.4f}  {bar}")

    return np.array(fold_weights)   # (n_folds, n_subbands)


def plot_attention_weights(fold_weights: np.ndarray,
                            subband_keys: list,
                            save_path: Path) -> None:
    """
    Bar chart of mean ± std attention weights across CV folds.

    Args:
        fold_weights: (n_folds, n_subbands) array
        subband_keys: subband names in order
        save_path:    where to save the figure
    """
    means = fold_weights.mean(axis=0)
    stds  = fold_weights.std(axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(subband_keys))

    ax.bar(x, means, yerr=stds, capsize=5,
           color='steelblue', alpha=0.8, edgecolor='navy', width=0.6)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.2,
               label='Uniform baseline (0.5)')
    ax.set_xticks(x)
    ax.set_xticklabels(subband_keys, fontsize=11)
    ax.set_xlabel('Subband', fontsize=12)
    ax.set_ylabel('Attention weight (sigmoid)', fontsize=12)
    ax.set_title(f'Learned attention weights across {len(fold_weights)}-fold CV\n'
                 '(mean ± std)', fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Attention weight plot saved → {save_path}")

    # Print summary
    print(f"\n  {'Subband':<8} {'Mean':>8} {'Std':>8}  Rank")
    print(f"  {'-'*32}")
    ranked   = np.argsort(means)[::-1]
    rank_map = {i: r+1 for r, i in enumerate(ranked)}
    for i, (name, m, s) in enumerate(zip(subband_keys, means, stds)):
        print(f"  {name:<8} {m:>8.4f} {s:>8.4f}   #{rank_map[i]}")

    top3 = [subband_keys[i] for i in ranked[:3]]
    print(f"\n  Top 3 subbands (for ablation): {top3}")


# ══════════════════════════════════════════════════════════════════════════════
# ROC curves
# ══════════════════════════════════════════════════════════════════════════════

def load_fold_probs(save_dir: Path,
                    n_folds: int = 10,
                    key: str = 'trained') -> tuple:
    """
    Load per-fold predicted probabilities and labels from saved JSON.

    Expects save_dir/kfold_results.json produced by run_kfold_cv.
    key: 'trained_attention', 'uniform_attention', or 'results' (single-scale)

    Returns:
        fold_probs_list:  list of (N,) arrays
        fold_labels_list: list of (N,) arrays
    """
    results_path = save_dir / 'kfold_results.json'
    if not results_path.exists():
        raise FileNotFoundError(f"kfold_results.json not found in {save_dir}")

    with open(results_path) as f:
        results = json.load(f)

    # Resolve key
    if key not in results:
        available = list(results.keys())
        raise KeyError(f"Key '{key}' not in results. Available: {available}")

    per_fold = results[key]['per_fold']

    # kfold_results.json stores metrics but NOT raw probs/labels
    # Raw probs must be re-collected from checkpoints + dataset
    # If you want to avoid rerunning, save them during training
    # For now return None to signal they need to be collected separately
    print(f"  Note: raw probabilities not stored in JSON.")
    print(f"  Use collect_fold_probs() to get them from checkpoints.")
    return None, None


def collect_fold_probs(save_dir: Path,
                        dataset,         # CombinedPNGDataset instance
                        model_class,     # LateFusionWithAttention or SingleScaleCNN
                        model_kwargs: dict,
                        n_folds: int,
                        device,
                        uniform: bool = False) -> tuple:
    """
    Re-run inference on each fold's val set using saved checkpoints.
    This is the correct way to get fold-level probs for ROC curves.

    Args:
        save_dir:     directory with fold_N_best.pt files
        dataset:      CombinedPNGDataset (train+val pool, same as training)
        model_class:  model class to instantiate
        model_kwargs: kwargs for model_class (e.g. n_subbands=7, dropout=0.5)
        n_folds:      number of folds
        device:       torch device
        uniform:      if True, zero attention before inference

    Returns:
        fold_probs_list:  list of (N_val,) arrays
        fold_labels_list: list of (N_val,) arrays
    """
    from sklearn.model_selection import GroupKFold
    from torch.utils.data import DataLoader, Subset

    def get_subject_group(record_id):
        m = re.match(r'n(\d+)', record_id)
        if m: return int(m.group(1))
        m = re.match(r'p(\d+)', record_id)
        if m:
            num = int(m.group(1))
            return 100 + num // 2
        return -1

    groups  = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf     = GroupKFold(n_splits=n_folds)
    dummy_X = np.zeros(len(dataset))

    fold_probs_list  = []
    fold_labels_list = []

    for fold, (_, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups), start=1):

        ckpt_path = save_dir / f'fold_{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  Fold {fold}: checkpoint not found, skipping")
            continue

        ckpt  = torch.load(ckpt_path, map_location=device)
        model = model_class(**model_kwargs).to(device)
        model.load_state_dict(ckpt['state_dict'])

        if uniform and hasattr(model, 'set_uniform_attention'):
            model.set_uniform_attention()

        model.eval()

        va_loader = DataLoader(
            Subset(dataset, va_idx),
            batch_size=32, shuffle=False, num_workers=0,
        )

        probs_fold, labels_fold = [], []
        with torch.no_grad():
            for x, y in va_loader:
                logits = model(x.to(device))
                p      = torch.softmax(logits, 1)[:, 1].cpu().numpy()
                probs_fold.extend(p)
                labels_fold.extend(y.numpy())

        fold_probs_list.append(np.array(probs_fold))
        fold_labels_list.append(np.array(labels_fold))
        print(f"  Fold {fold}: {len(probs_fold)} samples collected")

    return fold_probs_list, fold_labels_list


def plot_roc_cv(fold_probs_list: list,
                fold_labels_list: list,
                title: str,
                save_path: Path,
                colour: str = '#1f77b4') -> float:
    """
    Plot per-fold ROC curves with ±1 std shading and bold mean curve.

    Returns mean AUC across folds.
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    fig, ax = plt.subplots(figsize=(6, 6))

    for probs, labels in zip(fold_probs_list, fold_labels_list):
        fpr, tpr, _ = roc_curve(labels, probs)
        fold_auc    = sklearn_auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr    = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax.plot(fpr, tpr, color=colour, alpha=0.15, lw=0.8)

    mean_tpr     = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr      = np.std(tprs, axis=0)
    mean_auc     = float(np.mean(aucs))
    std_auc      = float(np.std(aucs))

    ax.plot(mean_fpr, mean_tpr, color=colour, lw=2.5,
            label=f'Mean ROC (AUC = {mean_auc:.3f} ± {std_auc:.3f})')
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color=colour, label='±1 std')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')

    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ROC saved → {save_path}  (mean AUC={mean_auc:.4f})")
    return mean_auc


def plot_roc_comparison(models_data: dict,
                         save_path: Path) -> None:
    """
    Plot mean ROC curves for multiple models on one axes for comparison.

    Args:
        models_data: {
            'Model name': {
                'probs':  list of (N,) arrays per fold,
                'labels': list of (N,) arrays per fold,
                'colour': hex colour string,
            }, ...
        }
        save_path: where to save
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    mean_fpr = np.linspace(0, 1, 100)

    for model_name, data in models_data.items():
        tprs, aucs = [], []
        for probs, labels in zip(data['probs'], data['labels']):
            fpr, tpr, _ = roc_curve(labels, probs)
            aucs.append(sklearn_auc(fpr, tpr))
            interp        = np.interp(mean_fpr, fpr, tpr)
            interp[0]     = 0.0
            tprs.append(interp)

        mean_tpr     = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        std_tpr      = np.std(tprs, axis=0)
        mean_auc     = float(np.mean(aucs))
        std_auc      = float(np.std(aucs))
        col          = data.get('colour', '#1f77b4')

        ax.plot(mean_fpr, mean_tpr, color=col, lw=2.5,
                label=f'{model_name} (AUC={mean_auc:.3f} ± {std_auc:.3f})')
        ax.fill_between(mean_fpr,
                        np.clip(mean_tpr - std_tpr, 0, 1),
                        np.clip(mean_tpr + std_tpr, 0, 1),
                        alpha=0.12, color=col)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Chance')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC comparison — 10-fold CV\n(mean ± 1 std)', fontsize=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison ROC saved → {save_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Top-level runner
# ══════════════════════════════════════════════════════════════════════════════

def run_post_training_analysis(
    multi_save_dir:  Path,
    single_save_dir: Path,
    dataset_multi,        # CombinedPNGDataset for multi
    dataset_single,       # CombinedPNGDataset for single
    model_multi_class,    # LateFusionWithAttention
    model_single_class,   # SingleScaleCNN
    multi_kwargs:  dict,  # e.g. {'n_subbands': 7, 'dropout': 0.5}
    single_kwargs: dict,  # e.g. {'dropout': 0.5}
    device,
    n_folds:   int  = 10,
    output_dir: Path = None,
) -> dict:
    """
    Full post-training analysis:
      1. Extract + plot attention weights from multi-scale checkpoints
      2. Collect fold probs for multi (trained + uniform) and single
      3. Plot individual ROC curves per model
      4. Plot comparison ROC on one axes

    Returns dict with all fold_probs and fold_weights.
    """
    if output_dir is None:
        output_dir = multi_save_dir / 'analysis'
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── 1. Attention weights ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("1. ATTENTION WEIGHTS")
    print(f"{'='*60}")

    fold_weights = extract_attention_weights(
        multi_save_dir, n_folds=n_folds, subband_keys=SUBBAND_KEYS
    )
    plot_attention_weights(
        fold_weights, SUBBAND_KEYS,
        save_path=output_dir / 'attention_weights.png',
    )
    np.save(output_dir / 'fold_attention_weights.npy', fold_weights)
    results['fold_attention_weights'] = fold_weights

    # ── 2. Collect fold probs ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("2. COLLECTING FOLD PROBABILITIES")
    print(f"{'='*60}")

    print("\nMulti-scale — trained attention:")
    probs_multi_trained, labels_multi = collect_fold_probs(
        multi_save_dir, dataset_multi, model_multi_class,
        multi_kwargs, n_folds, device, uniform=False,
    )

    print("\nMulti-scale — uniform attention:")
    probs_multi_uniform, _ = collect_fold_probs(
        multi_save_dir, dataset_multi, model_multi_class,
        multi_kwargs, n_folds, device, uniform=True,
    )

    print("\nSingle-scale:")
    probs_single, labels_single = collect_fold_probs(
        single_save_dir, dataset_single, model_single_class,
        single_kwargs, n_folds, device, uniform=False,
    )

    results.update({
        'probs_multi_trained': probs_multi_trained,
        'probs_multi_uniform': probs_multi_uniform,
        'probs_single':        probs_single,
        'labels_multi':        labels_multi,
        'labels_single':       labels_single,
    })

    # ── 3. Individual ROC plots ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("3. INDIVIDUAL ROC CURVES")
    print(f"{'='*60}")

    plot_roc_cv(
        probs_multi_trained, labels_multi,
        title='Multi-scale CNN — Trained Attention\n10-fold CV ROC',
        save_path=output_dir / 'roc_multi_trained.png',
        colour='#d62728',
    )
    plot_roc_cv(
        probs_multi_uniform, labels_multi,
        title='Multi-scale CNN — Uniform Attention\n10-fold CV ROC',
        save_path=output_dir / 'roc_multi_uniform.png',
        colour='#ff7f0e',
    )
    plot_roc_cv(
        probs_single, labels_single,
        title='Single-scale CNN\n10-fold CV ROC',
        save_path=output_dir / 'roc_single.png',
        colour='#1f77b4',
    )

    # ── 4. Comparison ROC ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("4. COMPARISON ROC CURVE")
    print(f"{'='*60}")

    plot_roc_comparison(
        models_data={
            'Single-scale':           {'probs': probs_single,
                                       'labels': labels_single,
                                       'colour': '#1f77b4'},
            'Multi-scale (trained)':  {'probs': probs_multi_trained,
                                       'labels': labels_multi,
                                       'colour': '#d62728'},
            'Multi-scale (uniform)':  {'probs': probs_multi_uniform,
                                       'labels': labels_multi,
                                       'colour': '#ff7f0e'},
        },
        save_path=output_dir / 'roc_comparison.png',
    )

    print(f"\n✓ Analysis complete → {output_dir}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, '/Users/filipsulich/Developer/uni/Thesis/Thesis')

    from src.CNN.models.models import SingleScaleCNN, MultiScaleCNN
    # Import your dataset and CV script classes
    from src.CNN.training.cross_validation import CombinedPNGDataset

    DEVICE = (torch.device('mps')  if torch.cuda.is_available() is False
                                      and hasattr(torch.backends, 'mps')
                                      and torch.backends.mps.is_available()
              else torch.device('cuda') if torch.cuda.is_available()
              else torch.device('cpu'))

    PNG_DIR        = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2')
    MULTI_SAVE_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/multi')
    SINGLE_SAVE_DIR= Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/single')
    OUTPUT_DIR     = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/analysis')

    dataset_multi  = CombinedPNGDataset(PNG_DIR, mode='multi',  cache=True)
    dataset_single = CombinedPNGDataset(PNG_DIR, mode='single', cache=True)

    run_post_training_analysis(
        multi_save_dir   = MULTI_SAVE_DIR,
        single_save_dir  = SINGLE_SAVE_DIR,
        dataset_multi    = dataset_multi,
        dataset_single   = dataset_single,
        model_multi_class  = MultiScaleCNN,
        model_single_class = SingleScaleCNN,
        multi_kwargs   = {'n_subbands': 7, 'dropout': 0.5},
        single_kwargs  = {'dropout': 0.5},
        device         = DEVICE,
        n_folds        = 10,
        output_dir     = OUTPUT_DIR,
    )