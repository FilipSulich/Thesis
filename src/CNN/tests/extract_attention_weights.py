import json
import re
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from tqdm import tqdm

from src.CNN.models.models import MultiScaleCNN
from src.CNN.training.cross_validation import (
    CombinedPNGDataset, get_subject_group,
    SUBBAND_KEYS, N_FOLDS, SEED, DEVICE, N_SUBBANDS
)

PNG_DIR     = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2_denoise')
# RESULTS_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/multi')
RESULTS_DIR = Path('/results_final/results_kfold2_denoise')
BATCH_SIZE  = 32
#

@torch.no_grad()
def collect_attention_weights_per_class(
        model, loader, device) -> dict:
    """
    Collect per-sample attention weights split by class label.

    Returns dict with:
        'all':   (N, n_subbands) — all samples
        'nsr':   (N_nsr, n_subbands) — NSR segments only
        'preaf': (N_preaf, n_subbands) — Pre-AF segments only
        'labels': (N,) — class labels
    """
    model.eval()
    all_weights = []
    all_labels  = []

    for x, y in tqdm(loader, desc="Collecting weights", leave=False):
        x = x.to(device)
        feats   = [model.branches[i](x[:, i*3:(i+1)*3])
                   for i in range(model.n_subbands)]
        stacked = torch.stack(feats, dim=1)          # (B, 7, 64)
        flat    = stacked.reshape(stacked.size(0), -1)
        weights = model.attention(flat)               # (B, 7) sigmoid outputs
        all_weights.append(weights.cpu().numpy())
        all_labels.extend(y.numpy())

    all_weights = np.concatenate(all_weights, axis=0)  # (N, 7)
    all_labels  = np.array(all_labels)                  # (N,)

    nsr_mask   = all_labels == 0
    preaf_mask = all_labels == 1

    return {
        'all':    all_weights,
        'nsr':    all_weights[nsr_mask],
        'preaf':  all_weights[preaf_mask],
        'labels': all_labels,
    }


def plot_per_class_attention(fold_weights_nsr:   np.ndarray,
                              fold_weights_preaf: np.ndarray,
                              subband_names:      list,
                              save_path:          Path) -> None:
    """
    Bar chart comparing mean attention weights for NSR vs Pre-AF
    across all folds, with ±1 std error bars.

    Args:
        fold_weights_nsr:   (n_folds, n_subbands) NSR mean per fold
        fold_weights_preaf: (n_folds, n_subbands) Pre-AF mean per fold
        subband_names:      ['D2',...,'D8']
        save_path:          output file path
    """
    nsr_mean   = fold_weights_nsr.mean(axis=0)
    nsr_std    = fold_weights_nsr.std(axis=0)
    preaf_mean = fold_weights_preaf.mean(axis=0)
    preaf_std  = fold_weights_preaf.std(axis=0)
    diff       = preaf_mean - nsr_mean

    n      = len(subband_names)
    x      = np.arange(n)
    width  = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: side-by-side bars ───────────────────────────────────────
    ax = axes[0]
    ax.bar(x - width/2, nsr_mean,   width, yerr=nsr_std,
           capsize=4, label='NSR',    color='#1f77b4', alpha=0.8)
    ax.bar(x + width/2, preaf_mean, width, yerr=preaf_std,
           capsize=4, label='Pre-AF', color='#d62728', alpha=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.2,
               label='Uniform baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(subband_names, fontsize=11)
    ax.set_xlabel('Subband', fontsize=12)
    ax.set_ylabel('Mean attention weight (sigmoid)', fontsize=12)
    ax.set_title('Attention weights by class\n(mean ± std across folds)',
                 fontsize=12)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # ── Right: difference (Pre-AF − NSR) ─────────────────────────────
    ax2   = axes[1]
    diff_std = np.sqrt(nsr_std**2 + preaf_std**2)   # propagated std
    colours  = ['#d62728' if d > 0 else '#1f77b4' for d in diff]
    ax2.bar(x, diff, yerr=diff_std, capsize=4,
            color=colours, alpha=0.8, edgecolor='black', lw=0.5)
    ax2.axhline(0, color='black', lw=1)
    ax2.set_xticks(x)
    ax2.set_xticklabels(subband_names, fontsize=11)
    ax2.set_xlabel('Subband', fontsize=12)
    ax2.set_ylabel('Δ attention (Pre-AF − NSR)', fontsize=12)
    ax2.set_title('Differential attention weight\n(Pre-AF minus NSR)',
                  fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')

    # Annotate positive/negative
    for i, (d, c) in enumerate(zip(diff, colours)):
        ax2.text(i, d + np.sign(d) * 0.01,
                 f'{d:+.3f}', ha='center', va='bottom' if d > 0 else 'top',
                 fontsize=9)

    plt.suptitle('Per-class attention weight analysis — 10-fold CV',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Plot saved → {save_path}")


def extract_weights_from_checkpoints():
    print("Loading combined dataset...")
    dataset = CombinedPNGDataset(PNG_DIR, mode='multi', cache=True)

    groups  = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf     = GroupKFold(n_splits=N_FOLDS)
    dummy_X = np.zeros(len(dataset))

    fold_weights_all   = []   # (n_folds, n_subbands)
    fold_weights_nsr   = []
    fold_weights_preaf = []

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups), start=1):

        print(f"\n{'='*60}")
        print(f"Fold {fold}/{N_FOLDS}")
        print(f"{'='*60}")

        ckpt_path = RESULTS_DIR / f'fold_{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  ❌ Checkpoint not found: {ckpt_path.name} — skipping")
            continue

        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        model      = MultiScaleCNN(N_SUBBANDS, dropout=0.5).to(DEVICE)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        print(f"  Val AUC: {checkpoint['val_auc']:.4f}  "
              f"Val set: {len(va_idx)} samples")

        va_loader = DataLoader(
            Subset(dataset, va_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
        )

        result = collect_attention_weights_per_class(model, va_loader, DEVICE)

        # Per-fold mean weights per class
        w_nsr   = result['nsr'].mean(axis=0)    # (7,)
        w_preaf = result['preaf'].mean(axis=0)  # (7,)
        w_all   = result['all'].mean(axis=0)    # (7,)

        fold_weights_all.append(w_all)
        fold_weights_nsr.append(w_nsr)
        fold_weights_preaf.append(w_preaf)

        # Per-fold printout
        n_nsr   = result['nsr'].shape[0]
        n_preaf = result['preaf'].shape[0]
        print(f"\n  {'Subband':<8} {'All':>8} {'NSR':>8} {'PreAF':>8} {'Diff':>8}")
        print(f"  {'-'*44}")
        for i, name in enumerate(SUBBAND_KEYS[:N_SUBBANDS]):
            diff = w_preaf[i] - w_nsr[i]
            print(f"  {name:<8} {w_all[i]:>8.4f} {w_nsr[i]:>8.4f} "
                  f"{w_preaf[i]:>8.4f} {diff:>+8.4f}")
        print(f"  (NSR n={n_nsr}, Pre-AF n={n_preaf})")

    if not fold_weights_all:
        print("\n❌ No checkpoints found.")
        return None

    fold_weights_all   = np.array(fold_weights_all)    # (n_folds, 7)
    fold_weights_nsr   = np.array(fold_weights_nsr)
    fold_weights_preaf = np.array(fold_weights_preaf)

    # ── Save arrays ───────────────────────────────────────────────────
    np.save(RESULTS_DIR / 'fold_attention_weights.npy',       fold_weights_all)
    np.save(RESULTS_DIR / 'fold_attention_weights_nsr.npy',   fold_weights_nsr)
    np.save(RESULTS_DIR / 'fold_attention_weights_preaf.npy', fold_weights_preaf)

    # ── Aggregate summary ─────────────────────────────────────────────
    names = SUBBAND_KEYS[:N_SUBBANDS]
    means_all   = fold_weights_all.mean(axis=0)
    means_nsr   = fold_weights_nsr.mean(axis=0)
    means_preaf = fold_weights_preaf.mean(axis=0)
    stds_all    = fold_weights_all.std(axis=0)

    print(f"\n{'='*65}")
    print("SUMMARY: Mean Attention Weights Across Folds")
    print(f"{'='*65}")
    print(f"{'Subband':<8} {'Mean (all)':>12} {'Std':>8} "
          f"{'NSR':>8} {'PreAF':>8} {'Diff':>8}  Rank")
    print('-' * 65)
    ranked   = np.argsort(means_all)[::-1]
    rank_map = {i: r+1 for r, i in enumerate(ranked)}
    for i, name in enumerate(names):
        diff = means_preaf[i] - means_nsr[i]
        print(f"{name:<8} {means_all[i]:>12.4f} {stds_all[i]:>8.4f} "
              f"{means_nsr[i]:>8.4f} {means_preaf[i]:>8.4f} "
              f"{diff:>+8.4f}  #{rank_map[i]}")

    top3 = [names[i] for i in ranked[:3]]
    print(f"\nTop 3 by mean attention: {top3}")
    print(f"{'='*65}\n")

    # ── Save JSON summary ─────────────────────────────────────────────
    summary = {
        'subband_keys': names,
        'n_folds':      len(fold_weights_all),
        'overall': {
            name: {
                'mean': float(means_all[i]),
                'std':  float(stds_all[i]),
                'nsr_mean':   float(means_nsr[i]),
                'preaf_mean': float(means_preaf[i]),
                'diff_preaf_minus_nsr': float(means_preaf[i] - means_nsr[i]),
            }
            for i, name in enumerate(names)
        },
        'fold_weights_all':   fold_weights_all.tolist(),
        'fold_weights_nsr':   fold_weights_nsr.tolist(),
        'fold_weights_preaf': fold_weights_preaf.tolist(),
    }
    json_path = RESULTS_DIR / 'attention_analysis.json'
    with open(json_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  JSON saved → {json_path}")

    # ── Plot ──────────────────────────────────────────────────────────
    plot_per_class_attention(
        fold_weights_nsr, fold_weights_preaf, names,
        save_path=RESULTS_DIR / 'attention_weights_per_class.png',
    )

    return {
        'all':   fold_weights_all,
        'nsr':   fold_weights_nsr,
        'preaf': fold_weights_preaf,
    }


def plot_simple_attention_bars(results_dir: Path = RESULTS_DIR, save_path: Path = None):
    """
    Create a simple bar chart showing mean attention weights across all samples.

    Args:
        results_dir: Directory containing attention_analysis.json
        save_path: Optional path to save the plot (default: results_dir/attention_weights_simple.png)
    """
    import json

    # Load attention analysis
    json_path = results_dir / 'attention_analysis.json'
    if not json_path.exists():
        raise FileNotFoundError(f"Attention analysis not found: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract subband names and mean weights
    subband_names = data['subband_keys']
    means = [data['overall'][name]['mean'] for name in subband_names]
    stds = [data['overall'][name]['std'] for name in subband_names]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(subband_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.8,
                   color='steelblue', edgecolor='black', linewidth=1.2)

    # Baseline at uniform attention
    uniform_weight = 1.0 / len(subband_names)
    ax.axhline(uniform_weight, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label=f'Uniform baseline ({uniform_weight:.3f})')

    # Labels and formatting
    ax.set_xlabel('Subband', fontsize=13, fontweight='bold')
    ax.set_ylabel('Mean Attention Weight', fontsize=13, fontweight='bold')
    ax.set_title('Mean Attention Weights Across All Samples\n(10-Fold Cross-Validation)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(subband_names, fontsize=12)
    ax.set_ylim(0, max(means) + max(stds) + 0.1)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.01,
                f'{mean:.3f}',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()

    # Save
    if save_path is None:
        save_path = results_dir / 'attention_weights_simple.png'

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Simple attention bar chart saved → {save_path}")

    # Also save as PDF
    save_path_pdf = save_path.with_suffix('.pdf')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"✅ PDF version saved → {save_path_pdf}")

    plt.close()

    # Print summary
    print(f"\n{'='*60}")
    print("MEAN ATTENTION WEIGHTS (All samples)")
    print(f"{'='*60}")
    print(f"{'Subband':<10} {'Mean':<10} {'Std':<10} {'Rank':<6}")
    print('-' * 60)

    # Sort by mean (descending)
    ranked = sorted(enumerate(means), key=lambda x: x[1], reverse=True)
    for rank, (idx, mean) in enumerate(ranked, 1):
        name = subband_names[idx]
        std = stds[idx]
        print(f"{name:<10} {mean:<10.4f} {std:<10.4f} #{rank}")

    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Extract weights from all folds and create detailed plots
    # extract_weights_from_checkpoints()

    # Create simple bar chart
    plot_simple_attention_bars()
