"""Analysis module for mechanistic interpretability."""

from .attention_analysis import (
    analyze_attention_patterns,
    compute_diagonal_score,
    visualize_attention,
    find_lookup_heads,
)
from .activation_patching import (
    activation_patching,
    PatchingResult,
    create_patching_pairs,
)
from .visualization import (
    plot_grokking_dynamics,
    plot_weight_norm,
    plot_stratified_accuracy,
    plot_attention_heatmap,
    plot_patching_heatmap,
)

__all__ = [
    # Attention analysis
    "analyze_attention_patterns",
    "compute_diagonal_score",
    "visualize_attention",
    "find_lookup_heads",
    # Activation patching
    "activation_patching",
    "PatchingResult",
    "create_patching_pairs",
    # Visualization
    "plot_grokking_dynamics",
    "plot_weight_norm",
    "plot_stratified_accuracy",
    "plot_attention_heatmap",
    "plot_patching_heatmap",
]
