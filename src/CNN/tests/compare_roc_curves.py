import json
import numpy as np
import torch
import matplotlib.pyplot as plt

from pathlib import Path
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    average_precision_score,
    precision_recall_curve
)
from scipy.stats import wilcoxon, friedmanchisquare
from tqdm import tqdm

from src.CNN.models.models import SingleScaleCNN, MultiScaleCNN
from src.CNN.training.cross_validation import (
    CombinedPNGDataset,
    get_subject_group,
    N_FOLDS,
    DEVICE,
    N_SUBBANDS
)

# =============================================================================
# CONFIG
# =============================================================================

PNG_DIR = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2')
PNG_DIR_DENOISE = Path('/Users/filipsulich/Developer/uni/Thesis/dataset/images2_denoise')

RESULTS_SINGLE = Path('/results_final/results_kfold2/single')
RESULTS_MULTI = Path('/results_final/results_kfold2/multi')
RESULTS_MULTI_DENOISE = Path('/results_final/results_kfold2_denoise/multi')

OUTPUT_DIR = Path('/results_final/results_kfold2')

BATCH_SIZE = 32

# =============================================================================
# HELPERS
# =============================================================================

@torch.no_grad()
def get_predictions(model, loader, device):

    model.eval()

    probs_all = []
    labels_all = []

    for x, y in tqdm(loader, leave=False):

        x = x.to(device)

        logits = model(x)

        probs = torch.softmax(logits, dim=1)[:, 1]

        probs_all.extend(probs.cpu().numpy())
        labels_all.extend(y.numpy())

    return np.array(probs_all), np.array(labels_all)
def create_auc_boxplot(single_data, multi_data, denoise_data):

    colors = {
        'single': '#1f77b4',
        'multi': '#d62728',
        'denoise': '#2ca02c'
    }

    single_aucs = []
    multi_aucs = []
    denoise_aucs = []

    # -------------------------------------------------------------------------
    # Compute fold AUCs
    # -------------------------------------------------------------------------

    for fold in single_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        single_aucs.append(auc_score)

    for fold in multi_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        multi_aucs.append(auc_score)

    for fold in denoise_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        denoise_aucs.append(auc_score)

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    data = [
        single_aucs,
        multi_aucs,
        denoise_aucs
    ]

    labels = [
        'Single-scale',
        'Multi-scale',
        'Multi-scale Denoised'
    ]

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False
    )

    # -------------------------------------------------------------------------
    # Box colors
    # -------------------------------------------------------------------------

    box_colors = [
        colors['single'],
        colors['multi'],
        colors['denoise']
    ]

    for patch, color in zip(box['boxes'], box_colors):

        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # -------------------------------------------------------------------------
    # Individual fold dots
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(42)

    for i, (vals, color) in enumerate(zip(data, box_colors), start=1):

        x_jitter = rng.normal(
            loc=i,
            scale=0.04,
            size=len(vals)
        )

        ax.scatter(
            x_jitter,
            vals,
            color=color,
            s=55,
            alpha=0.9,
            zorder=3
        )

    # -------------------------------------------------------------------------
    # Mean markers
    # -------------------------------------------------------------------------

    means = [
        np.mean(single_aucs),
        np.mean(multi_aucs),
        np.mean(denoise_aucs)
    ]

    ax.plot(
        [1, 2, 3],
        means,
        linestyle='None',
        marker='D',
        markersize=6,
        color='black',
        label='Mean AUC'
    )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    ax.set_xticks([1, 2, 3])

    ax.set_xticklabels(
        labels,
        fontsize=11
    )

    ax.set_ylabel(
        'AUC',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_title(
        'Comparison of 10-fold CV AUC Scores',
        fontsize=13,
        fontweight='bold'
    )

    ax.grid(
        alpha=0.3,
        linestyle='--'
    )

    ax.legend()

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    plt.savefig(
        OUTPUT_DIR / 'auc_boxplot.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        OUTPUT_DIR / 'auc_boxplot.pdf',
        bbox_inches='tight'
    )

    plt.close()

    print("✅ Saved AUC boxplot")


def create_auprc_boxplot(single_data, multi_data, denoise_data):

    colors = {
        'single': '#1f77b4',
        'multi': '#d62728',
        'denoise': '#2ca02c'
    }

    single_aucs = []
    multi_aucs = []
    denoise_aucs = []

    # -------------------------------------------------------------------------
    # Compute fold AUCs
    # -------------------------------------------------------------------------

    for fold in single_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        single_aucs.append(auc_score)

    for fold in multi_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        multi_aucs.append(auc_score)

    for fold in denoise_data:

        auc_score = auc(
            *roc_curve(
                fold['labels'],
                fold['probs']
            )[:2]
        )

        denoise_aucs.append(auc_score)

    # -------------------------------------------------------------------------
    # Create figure
    # -------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(8, 6))

    data = [
        single_aucs,
        multi_aucs,
        denoise_aucs
    ]

    labels = [
        'Single-scale',
        'Multi-scale',
        'Multi-scale Denoised'
    ]

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.55,
        showfliers=False
    )

    # -------------------------------------------------------------------------
    # Box colors
    # -------------------------------------------------------------------------

    box_colors = [
        colors['single'],
        colors['multi'],
        colors['denoise']
    ]

    for patch, color in zip(box['boxes'], box_colors):

        patch.set_facecolor(color)
        patch.set_alpha(0.35)

    # -------------------------------------------------------------------------
    # Individual fold dots
    # -------------------------------------------------------------------------

    rng = np.random.default_rng(42)

    for i, (vals, color) in enumerate(zip(data, box_colors), start=1):

        x_jitter = rng.normal(
            loc=i,
            scale=0.04,
            size=len(vals)
        )

        ax.scatter(
            x_jitter,
            vals,
            color=color,
            s=55,
            alpha=0.9,
            zorder=3
        )

    # -------------------------------------------------------------------------
    # Mean markers
    # -------------------------------------------------------------------------

    means = [
        np.mean(single_aucs),
        np.mean(multi_aucs),
        np.mean(denoise_aucs)
    ]

    ax.plot(
        [1, 2, 3],
        means,
        linestyle='None',
        marker='D',
        markersize=6,
        color='black',
        label='Mean AUC'
    )

    # -------------------------------------------------------------------------
    # Formatting
    # -------------------------------------------------------------------------

    ax.set_xticks([1, 2, 3])

    ax.set_xticklabels(
        labels,
        fontsize=11
    )

    ax.set_ylabel(
        'AUC',
        fontsize=12,
        fontweight='bold'
    )

    ax.set_title(
        'Comparison of 10-fold CV AUC Scores',
        fontsize=13,
        fontweight='bold'
    )

    ax.grid(
        alpha=0.3,
        linestyle='--'
    )

    ax.legend()

    plt.tight_layout()

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    plt.savefig(
        OUTPUT_DIR / 'auc_boxplot.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        OUTPUT_DIR / 'auc_boxplot.pdf',
        bbox_inches='tight'
    )

    plt.close()

    print("✅ Saved AUC boxplot")

def compute_metrics_from_probs(probs, labels):

    fpr, tpr, thresholds = roc_curve(labels, probs)

    auc_roc = auc(fpr, tpr)

    auprc = average_precision_score(labels, probs)

    j_scores = tpr - fpr

    best_idx = np.argmax(j_scores)

    best_threshold = thresholds[best_idx]

    preds = (probs >= best_threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return {
        'auc_roc': float(auc_roc),
        'auprc': float(auprc),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'threshold': float(best_threshold)
    }


def compute_cohens_d(x, y):

    pooled_std = np.sqrt(
        ((np.std(x, ddof=1) ** 2) + (np.std(y, ddof=1) ** 2)) / 2
    )

    return (np.mean(x) - np.mean(y)) / pooled_std


# =============================================================================
# LOAD FOLD PREDICTIONS
# =============================================================================

def collect_fold_predictions(
        model_type,
        dataset_path,
        results_dir,
        dataset_mode
):

    print(f"\n{'='*70}")
    print(f'Collecting predictions: {model_type}')
    print(f"{'='*70}")

    dataset = CombinedPNGDataset(
        dataset_path,
        mode=dataset_mode,
        cache=True
    )

    groups = np.array([
        get_subject_group(r)
        for r in dataset.record_ids
    ])

    gkf = GroupKFold(n_splits=N_FOLDS)

    dummy_X = np.zeros(len(dataset))

    fold_data = []

    for fold, (_, va_idx) in enumerate(
            gkf.split(dummy_X, dataset.labels, groups),
            start=1
    ):

        print(f'Fold {fold}/{N_FOLDS}')

        ckpt_path = results_dir / f'fold_{fold}_best.pt'

        checkpoint = torch.load(
            ckpt_path,
            map_location=DEVICE
        )

        if model_type == 'single':

            model = SingleScaleCNN(
                dropout=0.5
            ).to(DEVICE)

        else:

            model = MultiScaleCNN(
                N_SUBBANDS,
                dropout=0.5
            ).to(DEVICE)

        model.load_state_dict(checkpoint['state_dict'])

        va_subset = Subset(dataset, va_idx)

        loader = DataLoader(
            va_subset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        probs, labels = get_predictions(
            model,
            loader,
            DEVICE
        )

        fold_data.append({
            'fold': fold,
            'probs': probs,
            'labels': labels
        })

    return fold_data

def plot_mean_roc_curve(ax, fold_data, color, label):

    mean_fpr = np.linspace(0, 1, 200)

    tprs = []
    aucs = []

    for fold in fold_data:

        probs = fold['probs']
        labels = fold['labels']

        fpr, tpr, _ = roc_curve(labels, probs)

        fold_auc = auc(fpr, tpr)

        aucs.append(fold_auc)

        interp_tpr = np.interp(mean_fpr, fpr, tpr)

        interp_tpr[0] = 0

        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)

    mean_tpr[-1] = 1

    std_tpr = np.std(tprs, axis=0)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color=color,
        lw=2.5,
        label=f'{label} (AUC = {mean_auc:.3f} ± {std_auc:.3f})'
    )

    ax.fill_between(
        mean_fpr,
        np.clip(mean_tpr - std_tpr, 0, 1),
        np.clip(mean_tpr + std_tpr, 0, 1),
        color=color,
        alpha=0.15
    )

def plot_mean_pr_curve(ax, fold_data, color, label):

    mean_recall = np.linspace(0, 1, 200)

    precisions = []
    auprcs = []

    positive_ratios = []

    for fold in fold_data:

        probs = fold['probs']
        labels = fold['labels']

        positive_ratios.append(np.mean(labels))

        precision, recall, _ = precision_recall_curve(
            labels,
            probs
        )

        fold_auprc = auc(recall, precision)

        auprcs.append(fold_auprc)

        interp_precision = np.interp(
            mean_recall,
            recall[::-1],
            precision[::-1]
        )

        precisions.append(interp_precision)

    mean_precision = np.mean(precisions, axis=0)

    std_precision = np.std(precisions, axis=0)

    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)

    ax.plot(
        mean_recall,
        mean_precision,
        color=color,
        lw=2.5,
        label=f'{label} (AUPRC = {mean_auprc:.3f} ± {std_auprc:.3f})'
    )

    ax.fill_between(
        mean_recall,
        np.clip(mean_precision - std_precision, 0, 1),
        np.clip(mean_precision + std_precision, 0, 1),
        color=color,
        alpha=0.15
    )

    return np.mean(positive_ratios)

def create_plots(single_data, multi_data, denoise_data):

    colors = {
        'single': '#1f77b4',
        'multi': '#d62728',
        'denoise': '#2ca02c'
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 7)
    )

    # ROC

    plot_mean_roc_curve(
        axes[0],
        single_data,
        colors['single'],
        'Single-scale'
    )

    plot_mean_roc_curve(
        axes[0],
        multi_data,
        colors['multi'],
        'Multi-scale'
    )

    plot_mean_roc_curve(
        axes[0],
        denoise_data,
        colors['denoise'],
        'Multi-scale Denoised'
    )

    axes[0].plot([0, 1], [0, 1], 'k--')

    axes[0].set_title('Mean ROC Curves across 10-fold cross-validation')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # PR

    # -------------------------------------------------------------------------
    # PR
    # -------------------------------------------------------------------------

    baseline = plot_mean_pr_curve(
        axes[1],
        single_data,
        colors['single'],
        'Single-scale'
    )

    plot_mean_pr_curve(
        axes[1],
        multi_data,
        colors['multi'],
        'Multi-scale'
    )

    plot_mean_pr_curve(
        axes[1],
        denoise_data,
        colors['denoise'],
        'Multi-scale Denoised'
    )

    # No-skill baseline
    axes[1].axhline(
        baseline,
        linestyle='--',
        color='black',
        lw=1.5,
        alpha=0.8,
        label=f'No-skill baseline ({baseline:.3f})'
    )

    axes[1].set_title('Mean Precision-Recall Curves across 10-fold cross-validation')

    axes[1].set_xlabel('Recall')

    axes[1].set_ylabel('Precision')

    axes[1].legend()

    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / 'roc_pr_comparison_final.png',
        dpi=300,
        bbox_inches='tight'
    )

    plt.savefig(
        OUTPUT_DIR / 'roc_pr_comparison_final.pdf',
        bbox_inches='tight'
    )

    plt.close()


# =============================================================================
# STATISTICS
# =============================================================================

def perform_statistics(single_data, multi_data, denoise_data):

    metrics = {
        'auc_roc': [],
        'auprc': [],
        'sensitivity': [],
        'specificity': []
    }

    single_metrics = {k: [] for k in metrics}
    multi_metrics = {k: [] for k in metrics}
    denoise_metrics = {k: [] for k in metrics}

    for s, m, d in zip(single_data, multi_data, denoise_data):

        s_metrics = compute_metrics_from_probs(
            s['probs'],
            s['labels']
        )

        m_metrics = compute_metrics_from_probs(
            m['probs'],
            m['labels']
        )

        d_metrics = compute_metrics_from_probs(
            d['probs'],
            d['labels']
        )

        for key in metrics:

            single_metrics[key].append(s_metrics[key])
            multi_metrics[key].append(m_metrics[key])
            denoise_metrics[key].append(d_metrics[key])

    results = {}

    bonferroni_alpha = 0.05 / 3

    for metric in metrics:

        s_vals = np.array(single_metrics[metric])
        m_vals = np.array(multi_metrics[metric])
        d_vals = np.array(denoise_metrics[metric])

        friedman_stat, friedman_p = friedmanchisquare(
            s_vals,
            m_vals,
            d_vals
        )

        sm_test = wilcoxon(
            s_vals,
            m_vals
        )

        sd_test = wilcoxon(
            s_vals,
            d_vals
        )

        md_test = wilcoxon(
            m_vals,
            d_vals
        )

        results[metric] = {

            'friedman': {
                'statistic': float(friedman_stat),
                'p_value': float(friedman_p)
            },

            'single_vs_multi': {
                'p_value': float(sm_test.pvalue),
                'cohens_d': float(compute_cohens_d(m_vals, s_vals)),
                'significant': bool(sm_test.pvalue < bonferroni_alpha)
            },

            'single_vs_denoise': {
                'p_value': float(sd_test.pvalue),
                'cohens_d': float(compute_cohens_d(d_vals, s_vals)),
                'significant': bool(sd_test.pvalue < bonferroni_alpha)
            },

            'multi_vs_denoise': {
                'p_value': float(md_test.pvalue),
                'cohens_d': float(compute_cohens_d(d_vals, m_vals)),
                'significant': bool(md_test.pvalue < bonferroni_alpha)
            },

            'means': {
                'single': float(np.mean(s_vals)),
                'multi': float(np.mean(m_vals)),
                'denoise': float(np.mean(d_vals))
            },

            'stds': {
                'single': float(np.std(s_vals, ddof=1)),
                'multi': float(np.std(m_vals, ddof=1)),
                'denoise': float(np.std(d_vals, ddof=1))
            }
        }

    with open(OUTPUT_DIR / 'statistics.json', 'w') as f:

        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print('STATISTICAL RESULTS')
    print(f"{'='*70}")

    for metric, vals in results.items():

        print(f'\n{metric.upper()}')

        print(
            f"Friedman p-value: "
            f"{vals['friedman']['p_value']:.4f}"
        )

        print(
            f"Single vs Multi: "
            f"{vals['single_vs_multi']['p_value']:.4f}"
        )

        print(
            f"Single vs Denoise: "
            f"{vals['single_vs_denoise']['p_value']:.4f}"
        )

        print(
            f"Multi vs Denoise: "
            f"{vals['multi_vs_denoise']['p_value']:.4f}"
        )

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():

    print(f'Device: {DEVICE}')

    # -------------------------------------------------------------------------
    # LOAD PREDICTIONS
    # -------------------------------------------------------------------------

    single_data = collect_fold_predictions(
        model_type='single',
        dataset_path=PNG_DIR,
        results_dir=RESULTS_SINGLE,
        dataset_mode='single'
    )

    multi_data = collect_fold_predictions(
        model_type='multi',
        dataset_path=PNG_DIR,
        results_dir=RESULTS_MULTI,
        dataset_mode='multi'
    )

    denoise_data = collect_fold_predictions(
        model_type='multi',
        dataset_path=PNG_DIR_DENOISE,
        results_dir=RESULTS_MULTI_DENOISE,
        dataset_mode='multi'
    )

    # -------------------------------------------------------------------------
    # PLOTS
    # -------------------------------------------------------------------------

    create_plots(
        single_data,
        multi_data,
        denoise_data
    )

    create_auc_boxplot(
        single_data,multi_data, denoise_data
    )

    # -------------------------------------------------------------------------
    # STATISTICS
    # -------------------------------------------------------------------------

    perform_statistics(
        single_data,
        multi_data,
        denoise_data
    )

    print('\nAnalysis complete.')


if __name__ == '__main__':
    main()