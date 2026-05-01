#!/usr/bin/env python3
"""
Compare ROC curves across single-scale, multi-scale trained, and multi-scale uniform attention.
Creates a unified plot showing all three model types with per-fold curves and mean curves.
"""

import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc
from scipy.stats import wilcoxon
from tqdm import tqdm

from src.CNN.models.models import SingleScaleCNN, MultiScaleCNN
from src.CNN.training.cross_validation import (
    CombinedPNGDataset, get_subject_group,
    N_FOLDS, SEED, DEVICE, N_SUBBANDS
)

# Paths
PNG_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2')
RESULTS_SINGLE = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/single')
RESULTS_MULTI = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2/multi')
OUTPUT_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/Thesis/results_kfold2')
BATCH_SIZE = 32


@torch.no_grad()
def get_predictions(model, loader, device) -> tuple:
    """
    Get predictions from model.
    Returns: (probabilities, labels) as numpy arrays
    """
    model.eval()
    all_probs, all_labels = [], []

    for x, y in tqdm(loader, desc="Getting predictions", leave=False):
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, 1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

    return np.array(all_probs), np.array(all_labels)


def compute_cohen_d(group1, group2):
    """
    Compute Cohen's d effect size.

    Args:
        group1, group2: Arrays of values to compare

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def collect_fold_predictions(model_type='single'):
    """
    Collect predictions from all folds for a given model type.

    Args:
        model_type: 'single' or 'multi'

    Returns:
        fold_data: list of dicts with 'probs_trained', 'probs_uniform', 'labels', 'auc_trained', 'auc_uniform'
    """
    print(f"\n{'='*70}")
    print(f"Collecting predictions for {model_type.upper()} model")
    print(f"{'='*70}")

    # Load dataset
    dataset = CombinedPNGDataset(PNG_DIR, mode=model_type, cache=True)

    # Recreate CV splits
    groups = np.array([get_subject_group(r) for r in dataset.record_ids])
    gkf = GroupKFold(n_splits=N_FOLDS)
    dummy_X = np.zeros(len(dataset))

    fold_data = []
    results_dir = RESULTS_SINGLE if model_type == 'single' else RESULTS_MULTI

    for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups), start=1):

        print(f"\nFold {fold}/{N_FOLDS}")

        # Check if checkpoint exists
        ckpt_path = results_dir / f'fold_{fold}_best.pt'
        if not ckpt_path.exists():
            print(f"  ❌ Checkpoint not found: {ckpt_path.name}")
            continue

        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        print(f"  ✓ Loaded checkpoint (Val AUC: {checkpoint['val_auc']:.4f})")

        # Create model
        if model_type == 'single':
            model = SingleScaleCNN(dropout=0.5).to(DEVICE)
        else:
            model = MultiScaleCNN(N_SUBBANDS, dropout=0.5).to(DEVICE)

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Create validation loader
        va_subset = Subset(dataset, va_idx)
        va_loader = DataLoader(
            va_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=DEVICE.type == 'cuda'
        )

        # Get predictions with trained weights
        probs_trained, labels = get_predictions(model, va_loader, DEVICE)

        # Compute AUC for trained model
        fpr_t, tpr_t, _ = roc_curve(labels, probs_trained)
        auc_trained = auc(fpr_t, tpr_t)

        fold_info = {
            'fold': fold,
            'probs_trained': probs_trained,
            'labels': labels,
            'probs_uniform': None,
            'auc_trained': float(auc_trained),
            'auc_uniform': None
        }

        # For multi-scale, also get predictions with uniform attention
        if model_type == 'multi':
            # Zero out attention weights to get uniform weighting (sigmoid(0) = 0.5)
            with torch.no_grad():
                for layer in model.attention:
                    if hasattr(layer, 'weight'):
                        layer.weight.zero_()
                    if hasattr(layer, 'bias'):
                        layer.bias.zero_()
            probs_uniform, _ = get_predictions(model, va_loader, DEVICE)
            fold_info['probs_uniform'] = probs_uniform

            # Compute uniform AUC
            fpr_u, tpr_u, _ = roc_curve(labels, probs_uniform)
            auc_uniform = auc(fpr_u, tpr_u)
            fold_info['auc_uniform'] = float(auc_uniform)

            print(f"  AUC trained: {auc_trained:.4f} | AUC uniform: {auc_uniform:.4f}")
        else:
            print(f"  AUC: {auc_trained:.4f}")

        fold_data.append(fold_info)

    return fold_data


def plot_mean_roc_curve(fold_probs_list, fold_labels_list, ax, color, label_prefix,
                        alpha_individual=0.1, linewidth_mean=2.5):
    """
    Plot per-fold ROC curves (faint) and mean ROC curve (bold) on given axes.

    Returns:
        mean_auc: float
    """
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs = [], []

    # Plot individual folds
    for probs, labels in zip(fold_probs_list, fold_labels_list):
        fpr, tpr, _ = roc_curve(labels, probs)
        fold_auc = auc(fpr, tpr)
        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        # Individual fold curves (very faint)
        ax.plot(fpr, tpr, color=color, alpha=alpha_individual, lw=0.8)

    # Compute mean and std
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = float(np.mean(aucs))
    std_auc = float(np.std(aucs))

    # Plot mean curve (bold)
    ax.plot(mean_fpr, mean_tpr, color=color, lw=linewidth_mean,
            label=f'{label_prefix} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

    # Shaded std region
    ax.fill_between(mean_fpr,
                    np.clip(mean_tpr - std_tpr, 0, 1),
                    np.clip(mean_tpr + std_tpr, 0, 1),
                    alpha=0.15, color=color)

    return mean_auc


def perform_statistical_tests(single_data, multi_data):
    """
    Perform Wilcoxon signed-rank test and compute Cohen's d
    between single-scale and multi-scale trained models.

    Args:
        single_data: List of fold data for single-scale
        multi_data: List of fold data for multi-scale

    Returns:
        dict with test results
    """
    print(f"\n{'='*70}")
    print("STATISTICAL TESTS: Single-scale vs Multi-scale (trained)")
    print(f"{'='*70}\n")

    # Extract AUCs
    single_aucs = np.array([f['auc_trained'] for f in single_data])
    multi_aucs = np.array([f['auc_trained'] for f in multi_data])

    # Ensure paired samples (same number of folds)
    n_folds = min(len(single_aucs), len(multi_aucs))
    if len(single_aucs) != len(multi_aucs):
        print(f"⚠ Warning: Different number of folds ({len(single_aucs)} vs {len(multi_aucs)})")
        print(f"  Using first {n_folds} folds for comparison\n")
        single_aucs = single_aucs[:n_folds]
        multi_aucs = multi_aucs[:n_folds]

    # Wilcoxon signed-rank test (paired, non-parametric)
    statistic, p_value = wilcoxon(multi_aucs, single_aucs, alternative='greater')

    # Cohen's d effect size
    cohens_d = compute_cohen_d(multi_aucs, single_aucs)

    # Summary statistics
    single_mean = np.mean(single_aucs)
    single_std = np.std(single_aucs, ddof=1)
    multi_mean = np.mean(multi_aucs)
    multi_std = np.std(multi_aucs, ddof=1)
    improvement = multi_mean - single_mean

    print("Per-fold AUCs:")
    print(f"{'Fold':<6} {'Single':<10} {'Multi':<10} {'Diff':>10}")
    print("-" * 70)
    for i, (s_auc, m_auc) in enumerate(zip(single_aucs, multi_aucs), 1):
        diff = m_auc - s_auc
        print(f"{i:<6} {s_auc:<10.4f} {m_auc:<10.4f} {diff:>+10.4f}")

    print(f"\n{'─'*70}")
    print("Summary Statistics:")
    print(f"  Single-scale:  {single_mean:.4f} ± {single_std:.4f}")
    print(f"  Multi-scale:   {multi_mean:.4f} ± {multi_std:.4f}")
    print(f"  Improvement:   {improvement:+.4f}")

    print(f"\n{'─'*70}")
    print("Wilcoxon Signed-Rank Test (paired, one-sided):")
    print(f"  H0: Multi-scale AUC ≤ Single-scale AUC")
    print(f"  H1: Multi-scale AUC > Single-scale AUC")
    print(f"  Test statistic: {statistic:.2f}")
    print(f"  p-value:        {p_value:.4f}")

    if p_value < 0.001:
        sig_str = "*** (p < 0.001)"
    elif p_value < 0.01:
        sig_str = "** (p < 0.01)"
    elif p_value < 0.05:
        sig_str = "* (p < 0.05)"
    else:
        sig_str = "(not significant)"

    print(f"  Significance:   {sig_str}")

    print(f"\n{'─'*70}")
    print("Effect Size (Cohen's d):")
    print(f"  Cohen's d: {cohens_d:.4f}")

    if abs(cohens_d) < 0.2:
        effect_str = "negligible"
    elif abs(cohens_d) < 0.5:
        effect_str = "small"
    elif abs(cohens_d) < 0.8:
        effect_str = "medium"
    else:
        effect_str = "large"

    print(f"  Interpretation: {effect_str} effect")
    print(f"{'='*70}\n")

    return {
        'test': 'wilcoxon_signed_rank',
        'alternative': 'greater',
        'n_folds': int(n_folds),
        'single_scale': {
            'mean_auc': float(single_mean),
            'std_auc': float(single_std),
            'aucs': single_aucs.tolist()
        },
        'multi_scale_trained': {
            'mean_auc': float(multi_mean),
            'std_auc': float(multi_std),
            'aucs': multi_aucs.tolist()
        },
        'improvement': float(improvement),
        'wilcoxon': {
            'statistic': float(statistic),
            'p_value': float(p_value),
            'significant_at_0.05': bool(p_value < 0.05)
        },
        'cohens_d': float(cohens_d),
        'effect_size_interpretation': effect_str
    }


def create_comparison_plot(single_data, multi_data):
    """
    Create a single plot comparing all three model configurations.
    """
    print(f"\n{'='*70}")
    print("Creating comparison plot")
    print(f"{'='*70}")

    fig, ax = plt.subplots(figsize=(8, 8))

    # Define colors for each model type
    colors = {
        'single': '#1f77b4',      # Blue
        'multi_trained': '#d62728',  # Red
        'multi_uniform': '#ff7f0e',  # Orange
    }

    mean_aucs = {}

    # 1. Single-scale model
    if single_data:
        print("\nPlotting single-scale model...")
        probs_list = [f['probs_trained'] for f in single_data]
        labels_list = [f['labels'] for f in single_data]
        mean_aucs['single'] = plot_mean_roc_curve(
            probs_list, labels_list, ax,
            color=colors['single'],
            label_prefix='Single-scale'
        )
        print(f"  Single-scale mean AUC: {mean_aucs['single']:.4f}")

    # 2. Multi-scale trained attention
    if multi_data:
        print("\nPlotting multi-scale trained attention...")
        probs_list = [f['probs_trained'] for f in multi_data]
        labels_list = [f['labels'] for f in multi_data]
        mean_aucs['multi_trained'] = plot_mean_roc_curve(
            probs_list, labels_list, ax,
            color=colors['multi_trained'],
            label_prefix='Multi-scale (trained)'
        )
        print(f"  Multi-scale trained mean AUC: {mean_aucs['multi_trained']:.4f}")

        # 3. Multi-scale uniform attention
        print("\nPlotting multi-scale uniform attention...")
        probs_list = [f['probs_uniform'] for f in multi_data]
        labels_list = [f['labels'] for f in multi_data]
        mean_aucs['multi_uniform'] = plot_mean_roc_curve(
            probs_list, labels_list, ax,
            color=colors['multi_uniform'],
            label_prefix='Multi-scale (uniform)'
        )
        print(f"  Multi-scale uniform mean AUC: {mean_aucs['multi_uniform']:.4f}")

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Chance', alpha=0.6)

    # Formatting
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('ROC Curve Comparison: 10-Fold Cross-Validation',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(fontsize=11, loc='lower right', framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / 'roc_comparison_all_models.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved comparison plot → {output_path}")

    # Also save as PDF for publications
    output_path_pdf = OUTPUT_DIR / 'roc_comparison_all_models.pdf'
    plt.savefig(output_path_pdf, bbox_inches='tight')
    print(f"✅ Saved PDF version → {output_path_pdf}")

    plt.close()

    return mean_aucs


def save_auc_results(single_data, multi_data, stats_results, mean_aucs):
    """
    Save all AUC results and statistical tests to JSON.
    """
    results = {
        'mean_aucs': mean_aucs,
        'n_folds_single': len(single_data) if single_data else 0,
        'n_folds_multi': len(multi_data) if multi_data else 0,
        'per_fold_aucs': {
            'single_scale': [
                {'fold': f['fold'], 'auc': f['auc_trained']}
                for f in single_data
            ] if single_data else [],
            'multi_scale_trained': [
                {'fold': f['fold'], 'auc': f['auc_trained']}
                for f in multi_data
            ] if multi_data else [],
            'multi_scale_uniform': [
                {'fold': f['fold'], 'auc': f['auc_uniform']}
                for f in multi_data if f['auc_uniform'] is not None
            ] if multi_data else [],
        },
        'statistical_tests': stats_results if stats_results else {}
    }

    results_path = OUTPUT_DIR / 'roc_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Saved comprehensive results → {results_path}\n")


def main():
    print(f"Device: {DEVICE}")
    print(f"Output directory: {OUTPUT_DIR}")

    # Collect predictions from single-scale models
    single_data = collect_fold_predictions(model_type='single')

    # Collect predictions from multi-scale models
    multi_data = collect_fold_predictions(model_type='multi')

    # Check if we have data
    if not single_data and not multi_data:
        print("\n❌ No data collected! Make sure you have trained models in:")
        print(f"   {RESULTS_SINGLE}")
        print(f"   {RESULTS_MULTI}")
        return

    # Perform statistical tests if we have both models
    stats_results = None
    if single_data and multi_data:
        stats_results = perform_statistical_tests(single_data, multi_data)

    # Create comparison plot
    mean_aucs = create_comparison_plot(single_data, multi_data)

    # Save comprehensive results
    save_auc_results(single_data, multi_data, stats_results, mean_aucs)

    # Print summary
    print(f"{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    if 'single' in mean_aucs:
        print(f"Single-scale mean AUC:        {mean_aucs['single']:.4f}")
    if 'multi_trained' in mean_aucs:
        print(f"Multi-scale (trained) AUC:    {mean_aucs['multi_trained']:.4f}")
    if 'multi_uniform' in mean_aucs:
        print(f"Multi-scale (uniform) AUC:    {mean_aucs['multi_uniform']:.4f}")

    if 'multi_trained' in mean_aucs and 'multi_uniform' in mean_aucs:
        improvement = mean_aucs['multi_trained'] - mean_aucs['multi_uniform']
        print(f"\nTrained vs Uniform improvement: {improvement:+.4f}")

    if 'single' in mean_aucs and 'multi_trained' in mean_aucs:
        improvement = mean_aucs['multi_trained'] - mean_aucs['single']
        print(f"Multi vs Single improvement:   {improvement:+.4f}")

    if stats_results and 'wilcoxon' in stats_results:
        p_val = stats_results['wilcoxon']['p_value']
        print(f"\nStatistical significance (Wilcoxon): p = {p_val:.4f}")
        print(f"Effect size (Cohen's d):             {stats_results['cohens_d']:.4f}")

    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
