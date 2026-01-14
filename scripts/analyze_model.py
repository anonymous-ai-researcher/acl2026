#!/usr/bin/env python3
"""
Analyze trained Transformer for mechanistic interpretability.

This script performs:
1. Attention pattern analysis (Same-Bit Lookup verification)
2. Activation patching (causal circuit identification)
3. Weight norm tracking (complexity collapse verification)
4. Stratified accuracy analysis

Usage:
    python scripts/analyze_model.py --checkpoint outputs/exp_name/model_final.pt
    python scripts/analyze_model.py --checkpoint outputs/exp_name/model_final.pt --full
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.counter_dataset import (
    LargeCounterDataset,
    Tokenizer,
    create_dataloader,
    get_carry_chain_length,
)
from src.models.transformer import TransformerConfig, TransformerLM
from src.analysis.attention_analysis import (
    analyze_attention_patterns,
    compute_diagonal_score,
    find_lookup_heads,
    analyze_head_specialization,
)
from src.analysis.activation_patching import (
    ActivationPatcher,
    create_patching_pairs,
    identify_circuit_components,
)
from src.analysis.visualization import (
    plot_attention_heatmap,
    plot_patching_heatmap,
    plot_stratified_accuracy,
    create_summary_figure,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mechanistic analysis of trained Transformer"
    )
    
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for analysis results (default: same as checkpoint)"
    )
    parser.add_argument(
        "--n_samples", type=int, default=1000,
        help="Number of samples for analysis (default: 1000)"
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Run full analysis (slower but more comprehensive)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cpu, cuda)"
    )
    parser.add_argument(
        "--save_figures", action="store_true", default=True,
        help="Save analysis figures"
    )
    
    return parser.parse_args()


def setup_device(device_str: str) -> torch.device:
    """Setup compute device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_str)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint."""
    print(f"Loading checkpoint: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config
    if "config" in checkpoint:
        config_dict = checkpoint["config"]
    else:
        # Try to load from accompanying config file
        config_path = Path(checkpoint_path).parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                config_dict = json.load(f)
        else:
            raise ValueError("Could not find model configuration")
    
    # Create tokenizer
    tokenizer = Tokenizer()
    
    # Reconstruct model config
    n_bits = config_dict.get("n_bits", 20)
    model_config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=2 * n_bits + 3,
        n_layers=config_dict.get("n_layers", 2),
        n_heads=config_dict.get("n_heads", 4),
        d_model=config_dict.get("d_model", 64),
        d_ff=config_dict.get("d_ff", 256),
        dropout=0.0,
        use_rope=True,
        tie_weights=True,
    )
    
    # Create and load model
    model = TransformerLM(model_config)
    
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, tokenizer, config_dict


def analyze_attention(model, tokenizer, n_bits: int, device: torch.device, n_samples: int = 100):
    """Analyze attention patterns for Same-Bit Lookup."""
    print("\n" + "=" * 60)
    print("ATTENTION PATTERN ANALYSIS")
    print("=" * 60)
    
    # Create test samples
    test_numbers = list(range(min(n_samples, 2**n_bits)))
    dataset = LargeCounterDataset(test_numbers, n_bits, tokenizer)
    
    # Analyze patterns
    print("Computing attention patterns...")
    patterns = analyze_attention_patterns(model, dataset, device, max_samples=n_samples)
    
    # Compute diagonal scores for each head
    print("\nDiagonal Attention Scores (Same-Bit Lookup):")
    print(f"Expected offset: -{n_bits + 1} (retrieves corresponding bit from previous number)")
    print("-" * 50)
    
    expected_offset = n_bits + 1
    head_scores = {}
    
    for layer_idx in range(model.config.n_layers):
        for head_idx in range(model.config.n_heads):
            key = f"layer_{layer_idx}_head_{head_idx}"
            if key in patterns:
                score = compute_diagonal_score(
                    patterns[key], 
                    offset=expected_offset,
                    n_bits=n_bits
                )
                head_scores[(layer_idx, head_idx)] = score
                
                marker = "★" if score > 0.5 else " "
                print(f"  L{layer_idx}H{head_idx}: {score:.3f} {marker}")
    
    # Find lookup heads
    lookup_heads = find_lookup_heads(patterns, n_bits, threshold=0.5)
    
    print(f"\nIdentified Same-Bit Lookup Heads: {lookup_heads}")
    
    if lookup_heads:
        print("✓ Found attention heads implementing the theoretical -(n+1) offset!")
    else:
        print("✗ No clear Same-Bit Lookup heads found.")
        print("  The model may use a different mechanism or hasn't fully grokked.")
    
    return patterns, head_scores, lookup_heads


def analyze_patching(model, tokenizer, n_bits: int, device: torch.device, n_samples: int = 50):
    """Perform activation patching to identify causal components."""
    print("\n" + "=" * 60)
    print("ACTIVATION PATCHING ANALYSIS")
    print("=" * 60)
    
    # Create patching pairs
    print("Creating clean/corrupted input pairs...")
    pairs = create_patching_pairs(n_bits, tokenizer, n_pairs=n_samples)
    
    # Initialize patcher
    patcher = ActivationPatcher(model)
    
    # Analyze each component
    print("Patching model components...")
    
    results = {}
    components_to_patch = []
    
    # Add attention heads
    for layer in range(model.config.n_layers):
        for head in range(model.config.n_heads):
            components_to_patch.append(f"attn_L{layer}_H{head}")
        components_to_patch.append(f"mlp_L{layer}")
    
    for component in components_to_patch:
        # Parse component name
        if component.startswith("attn_"):
            parts = component.split("_")
            layer = int(parts[1][1:])
            head = int(parts[2][1:])
            component_type = "attention"
            component_idx = (layer, head)
        else:
            layer = int(component.split("_")[1][1:])
            component_type = "mlp"
            component_idx = layer
        
        # Compute patching effect
        total_effect = 0.0
        
        for clean_ids, corrupt_ids, target_pos in pairs:
            clean_ids = clean_ids.to(device).unsqueeze(0)
            corrupt_ids = corrupt_ids.to(device).unsqueeze(0)
            
            effect = patcher.activation_patching(
                clean_ids, corrupt_ids, target_pos,
                component_type, component_idx
            )
            total_effect += effect
        
        avg_effect = total_effect / len(pairs)
        results[component] = avg_effect
    
    # Print results
    print("\nPatching Results (Logit Difference Drop):")
    print("-" * 50)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for component, effect in sorted_results:
        importance = "HIGH" if effect > 0.5 else "MED" if effect > 0.2 else "LOW"
        print(f"  {component:<15}: {effect:.3f} [{importance}]")
    
    # Identify circuit components
    circuit = identify_circuit_components(results)
    
    print("\n" + "-" * 50)
    print("CIRCUIT DECOMPOSITION:")
    print(f"  Retrieval components: {circuit['retrieval']}")
    print(f"  Computation components: {circuit['computation']}")
    print(f"  Other components: {circuit['other']}")
    
    return results, circuit


def analyze_stratified_accuracy(model, tokenizer, n_bits: int, device: torch.device):
    """Analyze accuracy by carry chain length."""
    print("\n" + "=" * 60)
    print("STRATIFIED ACCURACY ANALYSIS")
    print("=" * 60)
    
    model.eval()
    
    # Group test numbers by carry chain length
    strata = {k: [] for k in range(n_bits)}
    
    for num in range(2**n_bits):
        k = get_carry_chain_length(num, n_bits)
        strata[k].append(num)
    
    # Evaluate each stratum
    results = {}
    
    print(f"{'Stratum':<10} {'Count':<10} {'Accuracy':<10} {'Description'}")
    print("-" * 60)
    
    for k in range(n_bits):
        if not strata[k]:
            continue
        
        # Sample from stratum
        sample_size = min(len(strata[k]), 100)
        sample = strata[k][:sample_size]
        
        # Create dataset
        dataset = LargeCounterDataset(sample, n_bits, tokenizer)
        loader = create_dataloader(dataset, batch_size=32, shuffle=False)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                
                # Forward pass
                outputs = model(input_ids)
                predictions = outputs.argmax(dim=-1)
                
                # Check sequence-level accuracy
                mask = labels != -100
                for i in range(len(input_ids)):
                    pred_seq = predictions[i][mask[i]]
                    label_seq = labels[i][mask[i]]
                    if torch.all(pred_seq == label_seq):
                        correct += 1
                    total += 1
        
        acc = correct / total if total > 0 else 0.0
        results[k] = {"accuracy": acc, "count": len(strata[k]), "sample_size": total}
        
        # Description
        if k == 0:
            desc = "No carry (trivial)"
        elif k == n_bits - 1:
            desc = "Global carry (hardest)"
        else:
            desc = f"Carry length {k+1}"
        
        print(f"k={k:<7} {len(strata[k]):<10} {acc:<10.2%} {desc}")
    
    # Summary
    print("\n" + "-" * 60)
    overall_acc = sum(r["accuracy"] * r["sample_size"] for r in results.values()) / sum(r["sample_size"] for r in results.values())
    print(f"Overall weighted accuracy: {overall_acc:.2%}")
    
    # Check for uniform accuracy (true algorithm) vs decay (heuristic)
    low_k_acc = np.mean([results[k]["accuracy"] for k in range(3) if k in results])
    high_k_acc = np.mean([results[k]["accuracy"] for k in range(n_bits-3, n_bits) if k in results])
    
    if high_k_acc > 0.9 and abs(low_k_acc - high_k_acc) < 0.1:
        print("✓ Uniform accuracy across strata - model learned the true algorithm!")
    elif high_k_acc < low_k_acc * 0.5:
        print("✗ Accuracy drops for deep carries - model using heuristic shortcuts.")
    
    return results


def compute_weight_statistics(model):
    """Compute weight norm and statistics."""
    print("\n" + "=" * 60)
    print("WEIGHT STATISTICS")
    print("=" * 60)
    
    total_norm = 0.0
    layer_norms = {}
    
    for name, param in model.named_parameters():
        norm = param.norm().item()
        layer_norms[name] = norm
        total_norm += norm ** 2
    
    total_norm = np.sqrt(total_norm)
    
    print(f"Total L2 weight norm: {total_norm:.4f}")
    print("\nTop parameters by norm:")
    
    sorted_norms = sorted(layer_norms.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, norm in sorted_norms:
        print(f"  {name}: {norm:.4f}")
    
    return total_norm, layer_norms


def generate_figures(
    model, patterns, patching_results, stratified_results,
    n_bits: int, output_dir: Path
):
    """Generate analysis figures."""
    print("\n" + "=" * 60)
    print("GENERATING FIGURES")
    print("=" * 60)
    
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Attention heatmap for best lookup head
    if patterns:
        # Find best head
        best_head = None
        best_score = 0
        
        for key, pattern in patterns.items():
            score = compute_diagonal_score(pattern, n_bits + 1, n_bits)
            if score > best_score:
                best_score = score
                best_head = key
        
        if best_head:
            fig = plot_attention_heatmap(
                patterns[best_head],
                title=f"Attention Pattern: {best_head} (Same-Bit Lookup)",
                n_bits=n_bits
            )
            fig.savefig(figures_dir / "attention_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: attention_heatmap.png")
    
    # 2. Patching heatmap
    if patching_results:
        fig = plot_patching_heatmap(patching_results, model.config.n_layers, model.config.n_heads)
        fig.savefig(figures_dir / "patching_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: patching_heatmap.png")
    
    # 3. Stratified accuracy
    if stratified_results:
        fig = plot_stratified_accuracy(stratified_results, n_bits)
        fig.savefig(figures_dir / "stratified_accuracy.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: stratified_accuracy.png")
    
    print(f"\nAll figures saved to: {figures_dir}")


def main():
    """Main analysis function."""
    args = parse_args()
    
    device = setup_device(args.device)
    
    # Load model
    model, tokenizer, config = load_checkpoint(args.checkpoint, device)
    n_bits = config.get("n_bits", 20)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.checkpoint).parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("TRANSFORMER MECHANISTIC ANALYSIS")
    print("=" * 60)
    print(f"Model: {args.checkpoint}")
    print(f"Output: {output_dir}")
    print(f"N bits: {n_bits}")
    print(f"Device: {device}")
    
    # Run analyses
    results = {}
    
    # 1. Attention analysis
    patterns, head_scores, lookup_heads = analyze_attention(
        model, tokenizer, n_bits, device, args.n_samples
    )
    results["attention"] = {
        "head_scores": {f"{k[0]}_{k[1]}": v for k, v in head_scores.items()},
        "lookup_heads": lookup_heads,
    }
    
    # 2. Weight statistics
    total_norm, layer_norms = compute_weight_statistics(model)
    results["weights"] = {
        "total_norm": total_norm,
        "layer_norms": layer_norms,
    }
    
    # 3. Stratified accuracy
    stratified_results = analyze_stratified_accuracy(model, tokenizer, n_bits, device)
    results["stratified_accuracy"] = {
        k: {"accuracy": v["accuracy"], "count": v["count"]}
        for k, v in stratified_results.items()
    }
    
    # 4. Activation patching (if full analysis)
    patching_results = None
    circuit = None
    if args.full:
        patching_results, circuit = analyze_patching(
            model, tokenizer, n_bits, device, args.n_samples // 10
        )
        results["patching"] = patching_results
        results["circuit"] = circuit
    
    # Generate figures
    if args.save_figures:
        generate_figures(
            model, patterns, patching_results, stratified_results,
            n_bits, output_dir
        )
    
    # Save results
    results_path = output_dir / "analysis_results.json"
    with open(results_path, "w") as f:
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        
        json.dump(convert(results), f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"✓ Same-Bit Lookup heads found: {len(lookup_heads)}")
    print(f"✓ Total weight norm: {total_norm:.2f}")
    
    # Check for successful grokking
    avg_acc = np.mean([v["accuracy"] for v in stratified_results.values()])
    if avg_acc > 0.95:
        print(f"✓ Model has grokked! Average accuracy: {avg_acc:.1%}")
        print("  The learned circuit aligns with B-RASP theory.")
    else:
        print(f"✗ Model accuracy: {avg_acc:.1%} - may not have fully grokked.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
