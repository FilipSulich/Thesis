import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


DEFAULT_COLORS = {
    'full_multiscale': '#d62728',
    'Full-band': '#1f77b4',
    'Multi-scale': '#d62728',
    'Multi-scale denoised': '#2ca02c',
    'High-frequency': '#2ca02c',
    'Low-frequency': '#9467bd'
}

def plot_roc_curves(data_dict, colors=None, output_path=None, title='ROC Curves'):
    """
    plots ROC curves for multiple models with mean and standard deviation across folds

    Args:
        data_dict (dict): mapping of model names to list of fold results, each with 'probs' and 'labels'
        colors (dict): mapping of model names to color codes
        output_path (Path): path to save figure
        title (str): plot title

    Returns:
        tuple: (matplotlib figure, matplotlib axes) for the ROC plot
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
        ax.plot(mean_fpr, mean_tpr, color=color, lw=2.5, label=f'{model_name} (AUC = {mean_auc:.3f} +- {std_auc:.3f})')

        # plot std
        tpr_upper = np.minimum(mean_tpr + std_tpr, 1)
        tpr_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tpr_lower, tpr_upper, color=color, alpha=0.2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.6, label='Random (AUC=0.500)')

    ax.set_xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc='lower right', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')

    return fig, ax

def plot_pr_curves(data_dict, colors=None, output_path=None, title='Precision-Recall Curves'):
    """
    plots precision-recall curves for multiple models with mean and standard deviation across folds

    Args:
        data_dict (dict): mapping of model names to list of fold results
        colors (dict): mapping of model names to color codes
        output_path (Path): path to save figure
        title (str): plot title

    Returns:
        tuple: (matplotlib figure, matplotlib axes) for the PR plot
    """
    colors = colors or DEFAULT_COLORS
    mean_recall = np.linspace(0, 1, 200)

    fig, ax = plt.subplots(figsize=(10, 8))

    all_labels = []
    for fold_results in data_dict.values():
        for fold in fold_results:
            all_labels.extend(fold['labels'])
    pos_ratio = float(np.mean(all_labels))

    for model_name, fold_results in data_dict.items():
        precisions_interp = []
        auprcs = []

        for fold in fold_results:
            precision, recall, _ = precision_recall_curve(fold['labels'], fold['probs'])
            precision_rev = precision[::-1]
            recall_rev = recall[::-1]
            prec_interp = np.interp(mean_recall, recall_rev, precision_rev)
            precisions_interp.append(prec_interp)

            auprc = average_precision_score(fold['labels'], fold['probs'])
            auprcs.append(auprc)

        mean_precision = np.mean(precisions_interp, axis=0)
        std_precision = np.std(precisions_interp, axis=0)
        mean_auprc = np.mean(auprcs)
        std_auprc = np.std(auprcs)

        color = colors.get(model_name, '#000000')
        ax.plot(mean_recall, mean_precision, color=color, lw=2.5, label=f'{model_name} (AUPRC = {mean_auprc:.3f} +- {std_auprc:.3f})')

        prec_upper = np.minimum(mean_precision + std_precision, 1)
        prec_lower = np.maximum(mean_precision - std_precision, 0)
        ax.fill_between(mean_recall, prec_lower, prec_upper, color=color, alpha=0.2)

    ax.axhline(pos_ratio, color='k', linestyle='--', lw=1.5, alpha=0.6, label=f'Random ({pos_ratio:.3f})')
    ax.set_xlabel('Recall', fontweight='bold', fontsize=14)
    ax.set_ylabel('Precision', fontweight='bold', fontsize=14)
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.legend(loc='lower left', fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.0])

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')

    return fig, ax

def plot_metric_box_plots(data_dict, metrics=None, colors=None, output_path=None, title='Model Comparison'):
    """
    plots box plots comparing specified metrics across multiple models

    Args:
        data_dict (dict): mapping of model names to list of fold results
        metrics (list): list of metric names to plot
        colors (dict): mapping of model names to color codes
        output_path (Path): path to save figure
        title (str): overall plot title

    Returns:
        tuple: (matplotlib figure, list of matplotlib axes) for each metric
    """
    if metrics is None:
        metrics = ['auc', 'auprc']
    colors = colors or DEFAULT_COLORS
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))

    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        data = []
        labels = []
        box_colors = []

        for model_name, fold_results in data_dict.items():
            values = [fold[metric] for fold in fold_results]
            data.append(values)
            labels.append(model_name)
            box_colors.append(colors.get(model_name, '#000000'))

        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showfliers=False)

        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        for i, (values, color) in enumerate(zip(data, box_colors)):
            x_jitter = np.random.normal(i + 1, 0.04, len(values))
            ax.scatter(x_jitter, values, color=color, s=50, alpha=0.9, zorder=3)

        ax.set_ylabel(metric.upper(), fontsize=14, fontweight='bold')
        ax.set_title(f'{metric.upper()} Comparison', fontsize=16, fontweight='bold')
        ax.tick_params(axis='both', which='major', labelsize=11)
        ax.grid(alpha=0.3, linestyle='--', axis='y')

    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')

    return fig, axes

def plot_roc_and_pr_combined(data_dict, colors=None, output_path=None, title='ROC and Precision-Recall Curves'):
    """
    plots ROC and precision-recall curves side-by-side for model comparison

    Args:
        data_dict (dict): mapping of model names to list of fold results
        colors (dict): mapping of model names to color codes
        output_path (Path): path to save figure
        title (str): overall plot title

    Returns:
        tuple: (matplotlib figure, (ax1, ax2)) where ax1 is ROC plot and ax2 is PR plot
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
    ax1.set_xlabel('False Positive Rate', fontweight='bold', fontsize=14)
    ax1.set_ylabel('True Positive Rate', fontweight='bold', fontsize=14)
    ax1.set_title('ROC Curves', fontweight='bold', fontsize=16)
    ax1.legend(loc='lower right', fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=11)
    ax1.grid(alpha=0.3, linestyle='--')

    # pr curves (right panel)
    all_labels = []
    for fold_results in data_dict.values():
        for fold in fold_results:
            all_labels.extend(fold['labels'])
    pos_ratio = float(np.mean(all_labels))

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

    ax2.axhline(pos_ratio, color='k', linestyle='--', lw=1.5, alpha=0.6, label=f'Random ({pos_ratio:.3f})')
    ax2.set_xlabel('Recall', fontweight='bold', fontsize=14)
    ax2.set_ylabel('Precision', fontweight='bold', fontsize=14)
    ax2.set_title('Precision-Recall Curves', fontweight='bold', fontsize=16)
    ax2.legend(loc='lower left', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=11)
    ax2.grid(alpha=0.3, linestyle='--')

    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')

    return fig, (ax1, ax2)
