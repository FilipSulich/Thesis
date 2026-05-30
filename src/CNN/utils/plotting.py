"""
plotting utilities for model evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


DEFAULT_COLORS = {
    'single': '#1f77b4',
    'multi': '#d62728',
    'denoise': '#2ca02c',
    'full': '#d62728',
    'high_freq': '#2ca02c',
    'low_freq': '#9467bd',
    'full_band': '#1f77b4',
    'full_multiscale': '#d62728',
    'attention_guided': '#17becf',
    'Full-band': '#1f77b4',
    'Multi-scale': '#d62728',
    'Multi-scale denoised': '#2ca02c',
    'High-frequency': '#2ca02c',
    'Low-frequency': '#9467bd'
}

PLOT_STYLE = {
    'figure.figsize': (12, 8),
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11
}

# ============================================================================

def plot_roc_curves(data_dict, colors=None, output_path=None, title='ROC Curves'):
    """
    plot roc curves for multiple models.

    args:
        data_dict: dict mapping model names to list of fold results each fold result should have 'probs' and 'labels'
        colors: dict mapping model names to colors
        output_path: path to save figure (optional)
        title: plot title

    returns:
        fig, ax
    """
    colors = colors or DEFAULT_COLORS
    mean_fpr = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, fold_results in data_dict.items():
        tprs = []
        aucs = []

        # collect roc curves from all folds
        for fold in fold_results:
            fpr, tpr, _ = roc_curve(fold['labels'], fold['probs'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

        # compute mean and std
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        # plot mean curve
        color = colors.get(model_name, '#000000')
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})')

        # plot std
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

    # diagonal reference line (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')

    return fig, ax


# ============================================================================
# precision-recall curves
# ============================================================================

def plot_pr_curves(data_dict, colors=None, output_path=None, title='Precision-Recall Curves'):
    """
    plot precision-recall curves for multiple models.

    args:
        data_dict: dict mapping model names to list of fold results
        colors: dict mapping model names to colors
        output_path: path to save figure
        title: plot title

    returns:
        fig, ax
    """
    colors = colors or DEFAULT_COLORS
    mean_recall = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(10, 8))

    # calculate positive class ratio for random baseline
    all_labels = []
    for fold_results in data_dict.values():
        for fold in fold_results:
            all_labels.extend(fold['labels'])
    pos_ratio = np.mean(all_labels)

    for model_name, fold_results in data_dict.items():
        precisions_interp = []
        auprcs = []

        # collect pr curves from all folds
        for fold in fold_results:
            precision, recall, _ = precision_recall_curve(fold['labels'], fold['probs'])
            # reverse for interpolation (recall must be increasing)
            precision_rev = precision[::-1]
            recall_rev = recall[::-1]
            prec_interp = np.interp(mean_recall, recall_rev, precision_rev)
            precisions_interp.append(prec_interp)

            auprc = average_precision_score(fold['labels'], fold['probs'])
            auprcs.append(auprc)

        # compute mean and std
        mean_precision = np.mean(precisions_interp, axis=0)
        std_precision = np.std(precisions_interp, axis=0)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)

        # plot mean curve
        color = colors.get(model_name, '#000000')
        ax.plot(mean_recall, mean_precision, color=color, lw=2.5, label=f'{model_name} (AUPRC = {mean_auprc:.3f} ± {std_auprc:.3f})')

        # plot std
        prec_upper = np.minimum(mean_precision + std_precision, 1)
        prec_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, prec_lower, prec_upper, color=color, alpha=0.2)

    # random classifier baseline
    ax.axhline(pos_ratio, color='k', linestyle='--', lw=1.5, alpha=0.6, label=f'Random ({pos_ratio:.3f})')
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')

    return fig, ax


def plot_metric_boxplots(data_dict, metrics=['auc', 'auprc'], colors=None, output_path=None, title='Model Comparison'):
    """
    plot boxplots comparing metrics across models.

    args:
        data_dict: dict mapping model names to list of fold results
        metrics: list of metric names to plot
        colors: dict mapping model names to colors
        output_path: path to save figure
        title: plot title

    returns:
        fig, axes
    """
    colors = colors or DEFAULT_COLORS
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        # collect data for all models
        data = []
        labels = []
        box_colors = []

        for model_name, fold_results in data_dict.items():
            values = [fold[metric] for fold in fold_results]
            data.append(values)
            labels.append(model_name)
            box_colors.append(colors.get(model_name, '#000000'))

        # create boxplot
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)

        # color boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        # scatter individual fold points
        for i, (values, color) in enumerate(zip(data, box_colors)):
            x_jitter = np.random.normal(i + 1, 0.04, len(values))
            ax.scatter(x_jitter, values, color=color, s=50, alpha=0.9, zorder=3)

        ax.set_ylabel(metric.upper(), fontsize=12, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3, linestyle='--', axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')

    return fig, axes


def plot_roc_and_pr_combined(data_dict, colors=None, output_path=None, title='ROC and Precision-Recall Curves'):
    """
    plot roc and pr curves side-by-side.

    args:
        data_dict: dict mapping model names to list of fold results
        colors: dict mapping model names to colors
        output_path: path to save figure
        title: overall title

    returns:
        fig, (ax1, ax2)
    """
    colors = colors or DEFAULT_COLORS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    mean_fpr = np.linspace(0, 1, 200)
    mean_recall = np.linspace(0, 1, 200)

    for model_name, fold_results in data_dict.items():
        tprs = []
        aucs = []

        for fold in fold_results:
            fpr, tpr, _ = roc_curve(fold['labels'], fold['probs'])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            aucs.append(auc(fpr, tpr))

        mean_tpr = np.mean(tprs, axis=0)
        std_tpr = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        color = colors.get(model_name, '#000000')
        ax1.plot(mean_fpr, mean_tpr, color=color, lw=2.5,label=f'{model_name} ({mean_auc:.3f} ± {std_auc:.3f})')
        ax1.fill_between(mean_fpr, np.maximum(mean_tpr - std_tpr, 0),np.minimum(mean_tpr + std_tpr, 1), color=color, alpha=0.2)

    # random classifier baseline
    ax1.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random (0.500)')
    ax1.set_xlabel('False Positive Rate', fontweight='bold')
    ax1.set_ylabel('True Positive Rate', fontweight='bold')
    ax1.set_title('ROC Curves', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3, linestyle='--')

    # pr curves (right panel)
    # calculate positive class ratio for random baseline
    all_labels = []
    for fold_results in data_dict.values():
        for fold in fold_results:
            all_labels.extend(fold['labels'])
    pos_ratio = np.mean(all_labels)

    for model_name, fold_results in data_dict.items():
        precisions_interp = []
        auprcs = []

        for fold in fold_results:
            precision, recall, _ = precision_recall_curve(fold['labels'], fold['probs'])
            precision_rev = precision[::-1]
            recall_rev = recall[::-1]
            prec_interp = np.interp(mean_recall, recall_rev, precision_rev)
            precisions_interp.append(prec_interp)
            auprcs.append(average_precision_score(fold['labels'], fold['probs']))

        mean_precision = np.mean(precisions_interp, axis=0)
        std_precision = np.std(precisions_interp, axis=0)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)

        color = colors.get(model_name, '#000000')
        ax2.plot(mean_recall, mean_precision, color=color, lw=2.5, label=f'{model_name} ({mean_auprc:.3f} ± {std_auprc:.3f})')
        ax2.fill_between(mean_recall, np.maximum(mean_precision - std_precision, 0), np.minimum(mean_precision + std_precision, 1), color=color, alpha=0.2)

    # random classifierbaseline
    ax2.axhline(pos_ratio, color='k', linestyle='--', lw=1.5, alpha=0.6, label=f'Random ({pos_ratio:.3f})')
    ax2.set_xlabel('Recall', fontweight='bold')
    ax2.set_ylabel('Precision', fontweight='bold')
    ax2.set_title('Precision-Recall Curves', fontweight='bold')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.savefig(str(output_path).replace('.png', '.pdf'), bbox_inches='tight')

    return fig, (ax1, ax2)
