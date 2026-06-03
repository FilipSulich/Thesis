from src.CNN.utils.utils import (
    get_device,
    get_subject_group,
    set_seed,
    DEVICE,
)

from src.CNN.utils.datasets import (
    CombinedH5Dataset
)

from src.CNN.models.full_band_cnn import FullBandCNN
from src.CNN.models.multi_scale_cnn import MultiScaleCNN

from src.CNN.utils.evaluation import (
    get_predictions,
    get_attention_weights,
    compute_metrics,
    find_optimal_threshold_youden,
    evaluate_model_folds,
)

from src.CNN.utils.statistics import (
    compute_rosenthal_r,
    friedman_test,
    wilcoxon_pairwise,
    holm_bonferroni_correction,
    pairwise_wilcoxon_with_correction,
    compute_summary_stats,
    compare_models
)

from src.CNN.utils.plotting import (
    plot_roc_curves,
    plot_pr_curves,
    plot_metric_boxplots,
    plot_roc_and_pr_combined,
    DEFAULT_COLORS
)

__all__ = [
    # utils
    'get_device',
    'get_subject_group',
    'set_seed',
    'DEVICE',
    # datasets
    'CombinedH5Dataset',
    # models
    'FullBandCNN',
    'MultiScaleCNN',
    # evaluation
    'get_predictions',
    'get_attention_weights',
    'compute_metrics',
    'find_optimal_threshold_youden',
    'evaluate_model_folds',
    # statistics
    'compute_rosenthal_r',
    'friedman_test',
    'wilcoxon_pairwise',
    'holm_bonferroni_correction',
    'pairwise_wilcoxon_with_correction',
    'compute_summary_stats',
    'compare_models',
    # plotting
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_metric_boxplots',
    'plot_roc_and_pr_combined',
    'DEFAULT_COLORS',
]