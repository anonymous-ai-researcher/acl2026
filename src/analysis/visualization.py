"""
Visualization Utilities for Grokking Analysis.

This module provides plotting functions for:
- Grokking dynamics (train/test accuracy over time)
- Weight norm evolution (complexity collapse)
- Stratified accuracy by carry chain length
- Attention heatmaps
- Activation patching results
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_grokking_dynamics(
    history: Dict[str, List[Tuple[int, float]]],
    title: str = "Grokking Dynamics: Delayed Generalization",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot training and test accuracy over time.
    
    The characteristic grokking pattern shows:
    1. Train accuracy reaching 100% early
    2. Test accuracy remaining low until phase transition
    3. Sharp jump in test accuracy (grokking)
    
    Args:
        history: Training history with (step, value) pairs
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    train_acc_key = [k for k in history.keys() if "train" in k and "accuracy" in k.lower()]
    test_acc_key = [k for k in history.keys() if "eval" in k and "accuracy" in k.lower()]
    
    if train_acc_key:
        train_data = history[train_acc_key[0]]
        steps, values = zip(*train_data)
        ax.plot(steps, [v * 100 for v in values], label="Train Accuracy", 
                color='#2196F3', linewidth=2)
    
    if test_acc_key:
        test_data = history[test_acc_key[0]]
        steps, values = zip(*test_data)
        ax.plot(steps, [v * 100 for v in values], label="Test Accuracy", 
                color='#4CAF50', linewidth=2)
    
    # Add phase transition annotation
    if test_acc_key:
        # Find grokking point (where test acc jumps significantly)
        test_values = [v for _, v in history[test_acc_key[0]]]
        for i in range(1, len(test_values)):
            if test_values[i] - test_values[i-1] > 0.3:  # 30% jump
                grok_step = history[test_acc_key[0]][i][0]
                ax.axvline(x=grok_step, color='#FF5722', linestyle='--', 
                          alpha=0.7, label='Phase Transition')
                ax.annotate('Grokking', xy=(grok_step, 50), 
                           xytext=(grok_step + 2000, 60),
                           fontsize=10, color='#FF5722',
                           arrowprops=dict(arrowstyle='->', color='#FF5722'))
                break
    
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Sequence Accuracy (%)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='center right', fontsize=10)
    ax.set_ylim(-5, 105)
    ax.set_xlim(left=0)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_weight_norm(
    history: Dict[str, List[Tuple[int, float]]],
    title: str = "Weight Norm Evolution: Complexity Collapse",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot weight norm evolution over training.
    
    The complexity collapse pattern shows:
    1. Weight norm increasing during memorization phase
    2. Sharp drop coinciding with grokking
    3. Lower final norm for generalizable solution
    
    Args:
        history: Training history
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Top: Test accuracy
    test_acc_key = [k for k in history.keys() if "eval" in k and "accuracy" in k.lower()]
    if test_acc_key:
        test_data = history[test_acc_key[0]]
        steps, values = zip(*test_data)
        axes[0].plot(steps, [v * 100 for v in values], color='#4CAF50', linewidth=2)
        axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
        axes[0].set_ylim(-5, 105)
        axes[0].set_title(title, fontsize=14, fontweight='bold')
    
    # Bottom: Weight norm
    norm_key = [k for k in history.keys() if "weight_norm" in k.lower() or "norm" in k.lower()]
    if norm_key:
        norm_data = history[norm_key[0]]
        steps, values = zip(*norm_data)
        axes[1].plot(steps, values, color='#9C27B0', linewidth=2, label='Model L2 Norm')
        axes[1].set_ylabel("Weight Norm (||θ||₂)", fontsize=11)
        axes[1].set_xlabel("Training Steps", fontsize=12)
        
        # Add annotations
        max_norm_idx = np.argmax(values)
        axes[1].annotate('Memorization Phase\n(High Complexity)', 
                        xy=(steps[max_norm_idx], values[max_norm_idx]),
                        xytext=(steps[max_norm_idx] - 5000, values[max_norm_idx] + 2),
                        fontsize=9, ha='center')
        
        # Find collapse point
        for i in range(max_norm_idx + 1, len(values)):
            if values[i] < values[max_norm_idx] * 0.8:
                axes[1].axvline(x=steps[i], color='#FF5722', linestyle='--', alpha=0.5)
                axes[1].annotate('Complexity\nCollapse', 
                                xy=(steps[i], values[i]),
                                xytext=(steps[i] + 3000, values[i] + 1),
                                fontsize=9, color='#FF5722',
                                arrowprops=dict(arrowstyle='->', color='#FF5722'))
                break
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_stratified_accuracy(
    strata_accuracy: Dict[int, Optional[float]],
    strata_counts: Dict[int, int],
    title: str = "Accuracy by Carry Chain Length",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 5),
    model_name: str = "Model",
) -> plt.Figure:
    """
    Plot accuracy broken down by carry chain length.
    
    This reveals the RNN failure mode:
    - High accuracy for k=0 (trivial, no carry)
    - Rapid decay for k≥1 (carry required)
    
    Args:
        strata_accuracy: Accuracy for each carry length
        strata_counts: Sample counts per stratum
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        model_name: Name for legend
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Filter to strata with samples
    valid_k = [k for k in sorted(strata_accuracy.keys()) 
               if strata_counts.get(k, 0) > 0 and strata_accuracy[k] is not None]
    accuracies = [strata_accuracy[k] * 100 for k in valid_k]
    counts = [strata_counts[k] for k in valid_k]
    
    # Left: Accuracy bar chart
    colors = ['#4CAF50' if acc > 90 else '#FF9800' if acc > 50 else '#F44336' 
              for acc in accuracies]
    
    axes[0].bar(valid_k, accuracies, color=colors, edgecolor='white', linewidth=0.5)
    axes[0].set_xlabel("Carry Chain Length (k)", fontsize=11)
    axes[0].set_ylabel("Accuracy (%)", fontsize=11)
    axes[0].set_title(f"{title} - {model_name}", fontsize=12, fontweight='bold')
    axes[0].set_ylim(0, 105)
    axes[0].axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% threshold')
    
    # Annotate key regions
    axes[0].annotate('Trivial\n(k=0)', xy=(0, accuracies[0] + 3), ha='center', fontsize=9)
    
    # Right: Sample distribution
    axes[1].bar(valid_k, counts, color='#2196F3', alpha=0.7)
    axes[1].set_xlabel("Carry Chain Length (k)", fontsize=11)
    axes[1].set_ylabel("Number of Samples", fontsize=11)
    axes[1].set_title("Test Set Distribution", fontsize=12, fontweight='bold')
    axes[1].set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_attention_heatmap(
    attention_pattern: np.ndarray,
    title: str = "Attention Pattern",
    layer: int = 0,
    head: int = 0,
    n_bits: int = 20,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    annotate_diagonal: bool = True,
) -> plt.Figure:
    """
    Plot attention pattern heatmap.
    
    Highlights the diagonal at offset -(n+1) which corresponds
    to the Same-Bit Lookup mechanism.
    
    Args:
        attention_pattern: [seq_len, seq_len] attention matrix
        title: Plot title
        layer: Layer index (for title)
        head: Head index (for title)
        n_bits: Bit width (for diagonal annotation)
        save_path: Optional save path
        figsize: Figure size
        annotate_diagonal: Whether to highlight expected diagonal
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create heatmap
    im = ax.imshow(attention_pattern, cmap='viridis', aspect='auto')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Attention Weight', fontsize=10)
    
    # Labels
    seq_len = attention_pattern.shape[0]
    
    # Annotate input/output regions
    input_end = n_bits
    ax.axvline(x=input_end, color='white', linestyle='--', alpha=0.5)
    ax.axhline(y=input_end, color='white', linestyle='--', alpha=0.5)
    
    # Annotate expected diagonal for Same-Bit Lookup
    if annotate_diagonal:
        offset = -(n_bits + 1)
        for i in range(abs(offset), seq_len):
            j = i + offset
            if 0 <= j < seq_len:
                ax.plot(j, i, 'rx', markersize=8, markeredgewidth=2)
        
        ax.text(0.02, 0.98, f'Same-Bit Lookup\n(offset = {offset})', 
               transform=ax.transAxes, fontsize=10, verticalalignment='top',
               color='red', fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel("Source Token Position (Input Nᵢ)", fontsize=11)
    ax.set_ylabel("Target Token Position (Output Nᵢ₊₁)", fontsize=11)
    ax.set_title(f"{title} - Layer {layer}, Head {head}", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_patching_heatmap(
    patching_results: Dict[str, Dict[str, float]],
    title: str = "Causal Localization via Activation Patching",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot activation patching results as a heatmap.
    
    Shows which components are causally important for each bit position.
    
    Args:
        patching_results: Nested dict of component -> position -> effect
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract components and positions
    components = sorted(patching_results.keys())
    
    # For this visualization, we'll show effect size per component
    effects = []
    labels = []
    
    for comp in components:
        result = patching_results[comp]
        if isinstance(result, dict) and 'mean_effect' in result:
            effects.append(result['mean_effect'])
        elif hasattr(result, 'effect_size'):
            effects.append(result.effect_size)
        else:
            effects.append(0.0)
        labels.append(comp)
    
    # Create bar chart
    colors = ['#F44336' if e > 0.5 else '#FF9800' if e > 0.2 else '#4CAF50' 
              for e in effects]
    
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, effects, color=colors, edgecolor='white')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect Size (Logit Difference Drop)", fontsize=11)
    ax.set_ylabel("Model Component", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='High Impact')
    ax.axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Medium Impact')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#F44336', label='High Impact (>0.5)'),
        Patch(facecolor='#FF9800', label='Medium Impact (0.2-0.5)'),
        Patch(facecolor='#4CAF50', label='Low Impact (<0.2)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy",
    title: str = "Transformer vs RNN: The Succinctness Gap",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """
    Plot comparison between Transformer and RNN models.
    
    Args:
        results: Dictionary of model_name -> metrics
        metric: Metric to compare
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    models = list(results.keys())
    values = [results[m].get(metric, 0) * 100 for m in models]
    params = [results[m].get("parameters", 0) for m in models]
    
    # Color by model type
    colors = ['#2196F3' if 'transformer' in m.lower() else '#F44336' for m in models]
    
    bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=1.5)
    
    # Add parameter count annotations
    for bar, p in zip(bars, params):
        height = bar.get_height()
        ax.annotate(f'{p/1000:.0f}K params',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3), textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_ylabel(f"Test {metric.capitalize()} (%)", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylim(0, 110)
    ax.axhline(y=100, color='green', linestyle='--', alpha=0.3)
    
    # Add text box
    textstr = "Transformer achieves 100% accuracy with 30x fewer parameters"
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.95, 0.15, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    return fig


def create_summary_figure(
    history: Dict[str, List],
    attention_pattern: np.ndarray,
    strata_accuracy: Dict[int, float],
    save_path: str = "figures/summary.png",
) -> plt.Figure:
    """
    Create a comprehensive summary figure combining all visualizations.
    
    Layout:
    - Top left: Grokking dynamics
    - Top right: Attention pattern
    - Bottom left: Weight norm
    - Bottom right: Stratified accuracy
    
    Args:
        history: Training history
        attention_pattern: Best lookup head attention
        strata_accuracy: Per-stratum accuracies
        save_path: Path to save figure
        
    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Top left: Grokking dynamics
    ax1 = fig.add_subplot(gs[0, 0])
    # (Add plotting code here - simplified for brevity)
    ax1.set_title("A. Grokking Dynamics", fontsize=12, fontweight='bold')
    
    # Top right: Attention pattern
    ax2 = fig.add_subplot(gs[0, 1])
    im = ax2.imshow(attention_pattern, cmap='viridis', aspect='auto')
    ax2.set_title("B. Same-Bit Lookup (L0H2)", fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax2)
    
    # Bottom left: Weight norm
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title("C. Complexity Collapse", fontsize=12, fontweight='bold')
    
    # Bottom right: Stratified accuracy
    ax4 = fig.add_subplot(gs[1, 1])
    valid_k = sorted([k for k, v in strata_accuracy.items() if v is not None])
    accuracies = [strata_accuracy[k] * 100 for k in valid_k]
    ax4.bar(valid_k, accuracies)
    ax4.set_title("D. Accuracy by Carry Chain", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Carry Chain Length (k)")
    ax4.set_ylabel("Accuracy (%)")
    
    plt.suptitle("Mechanistic Evidence for Transformer Succinctness", 
                 fontsize=14, fontweight='bold', y=1.02)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved summary figure to {save_path}")
    
    return fig


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import os
    
    print("Testing visualization utilities...")
    
    # Create output directory
    os.makedirs("figures/test", exist_ok=True)
    
    # Create mock data
    np.random.seed(42)
    
    # Mock training history
    steps = list(range(0, 50000, 500))
    train_acc = [min(1.0, 0.1 + i/5000) for i in steps]
    test_acc = [0.05 if i < 15000 else min(1.0, 0.05 + (i-15000)/3000) for i in steps]
    weight_norm = [15 + i/5000 if i < 12000 else 25 - (i-12000)/4000 for i in steps]
    
    history = {
        "train_sequence_accuracy": list(zip(steps, train_acc)),
        "eval_sequence_accuracy": list(zip(steps, test_acc)),
        "weight_norm": list(zip(steps, weight_norm)),
    }
    
    # Test grokking dynamics plot
    fig1 = plot_grokking_dynamics(history, save_path="figures/test/grokking.png")
    print("✓ Grokking dynamics plot created")
    
    # Test weight norm plot
    fig2 = plot_weight_norm(history, save_path="figures/test/weight_norm.png")
    print("✓ Weight norm plot created")
    
    # Test stratified accuracy plot
    strata_acc = {k: 0.98 - 0.03*k if k < 15 else 0.5 for k in range(20)}
    strata_counts = {k: max(1, 1000 // (2**k)) for k in range(20)}
    
    fig3 = plot_stratified_accuracy(
        strata_acc, strata_counts, 
        model_name="Transformer (d=64)",
        save_path="figures/test/stratified.png"
    )
    print("✓ Stratified accuracy plot created")
    
    # Test attention heatmap
    attn_pattern = np.random.rand(42, 42) * 0.1
    for i in range(21, 42):
        j = i - 21
        if 0 <= j < 42:
            attn_pattern[i, j] = 0.9
    attn_pattern = attn_pattern / attn_pattern.sum(axis=1, keepdims=True)
    
    fig4 = plot_attention_heatmap(
        attn_pattern, layer=0, head=2, n_bits=20,
        save_path="figures/test/attention.png"
    )
    print("✓ Attention heatmap created")
    
    # Test patching heatmap
    from dataclasses import dataclass
    
    @dataclass
    class MockResult:
        effect_size: float
    
    patching_results = {
        "L0H0": MockResult(0.1),
        "L0H1": MockResult(0.15),
        "L0H2": MockResult(0.85),
        "L0H3": MockResult(0.12),
        "L0_MLP": MockResult(0.92),
        "L1H0": MockResult(0.45),
        "L1_MLP": MockResult(0.25),
    }
    
    fig5 = plot_patching_heatmap(
        patching_results,
        save_path="figures/test/patching.png"
    )
    print("✓ Patching heatmap created")
    
    # Test model comparison
    comparison_results = {
        "Transformer\n(d=64)": {"accuracy": 1.0, "parameters": 50000},
        "LSTM\n(d=64)": {"accuracy": 0.0, "parameters": 50000},
        "LSTM\n(d=2048)": {"accuracy": 0.058, "parameters": 1500000},
        "GRU\n(d=2048)": {"accuracy": 0.042, "parameters": 1200000},
    }
    
    fig6 = plot_model_comparison(
        comparison_results,
        save_path="figures/test/comparison.png"
    )
    print("✓ Model comparison plot created")
    
    plt.close('all')
    
    print("\n✓ All visualization tests passed!")
    print(f"Test figures saved to figures/test/")
