"""decomposition strategy selection package."""

# config
from src.decomposition_strategy_selection.config import (
    ExperimentConfig,
    PIPELINE_LABELS,
    PIPELINE_COLORS,
    DEFAULT_METRICS,
)

# signals
from src.decomposition_strategy_selection.signals import (
    generate_composite_sine,
    generate_single_sine,
)

# normalization
from src.decomposition_strategy_selection.normalization import min_max_normalize

# embedding
from src.decomposition_strategy_selection.embedding import (
    estimate_tau,
    estimate_embedding_dim,
    estimate_embedding_params,
)

# recurrence
from src.decomposition_strategy_selection.recurrence import (
    compute_distance_matrix,
    compute_rqa_metrics,
    extract_rp_array,
)

# metrics
from src.decomposition_strategy_selection.metrics import (
    pad_matrix,
    hamming_distance,
    pattern_correlation,
    compute_structural_metrics,
)

# ground truth
from src.decomposition_strategy_selection.ground_truth import (
    generate_ground_truth,
    save_ground_truth,
    load_ground_truth,
    get_or_generate_ground_truth,
)

# pipeline analyzer
from src.decomposition_strategy_selection.pipeline_analyzer import (
    decompose_signal,
    analyze_pipeline,
    analyze_all_pipelines,
)

# statistical tests
from src.decomposition_strategy_selection.statistical_tests import (
    get_significance_notation,
    run_kruskal_wallis,
    run_dunn_posthoc,
    run_statistical_analysis,
    save_statistical_results,
)

# plotting
from src.decomposition_strategy_selection.plotting import (
    plot_boxplot_comparisons,
    plot_violinplot_comparisons,
)

# experiment
from src.decomposition_strategy_selection.experiment import (
    run_trials,
    run_experiment,
)

__all__ = [
    # config
    'ExperimentConfig',
    'PIPELINE_LABELS',
    'PIPELINE_COLORS',
    'DEFAULT_METRICS',
    # signals
    'generate_composite_sine',
    'generate_single_sine',
    # normalization
    'min_max_normalize',
    # embedding
    'estimate_tau',
    'estimate_embedding_dim',
    'estimate_embedding_params',
    # recurrence
    'compute_distance_matrix',
    'compute_rqa_metrics',
    'extract_rp_array',
    # metrics
    'pad_matrix',
    'hamming_distance',
    'pattern_correlation',
    'compute_structural_metrics',
    # ground truth
    'generate_ground_truth',
    'save_ground_truth',
    'load_ground_truth',
    'get_or_generate_ground_truth',
    # pipeline analyzer
    'decompose_signal',
    'analyze_pipeline',
    'analyze_all_pipelines',
    # statistical tests
    'get_significance_notation',
    'run_kruskal_wallis',
    'run_dunn_posthoc',
    'run_statistical_analysis',
    'save_statistical_results',
    # plotting
    'plot_boxplot_comparisons',
    'plot_violinplot_comparisons',
    # experiment
    'run_trials',
    'run_experiment',
]
