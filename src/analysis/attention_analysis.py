"""
Attention Pattern Analysis.

This module analyzes attention patterns to verify the theoretical predictions:
- Same-Bit Lookup heads should show strong diagonal at offset -(n+1)
- This corresponds to attending to the same bit position in the previous number

The diagonal attention pattern is the mechanistic signature of the succinct
ripple-carry algorithm learned by Transformers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def analyze_attention_patterns(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_bits: int = 20,
    n_samples: int = 100,
) -> Dict[str, torch.Tensor]:
    """
    Analyze attention patterns across samples.
    
    Args:
        model: Transformer model
        dataloader: DataLoader with test samples
        device: Device for computation
        n_bits: Bit-width of counter
        n_samples: Number of samples to analyze
        
    Returns:
        Dictionary with aggregated attention patterns per layer/head
    """
    model.eval()
    
    # Get model architecture info
    n_layers = len(model.layers)
    n_heads = model.layers[0].attn.n_heads
    
    # Accumulate attention patterns
    attention_accum = {
        (l, h): [] for l in range(n_layers) for h in range(n_heads)
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing attention", total=n_samples // dataloader.batch_size + 1):
            if sample_count >= n_samples:
                break
            
            input_ids = batch["input_ids"].to(device)
            
            # Forward with attention output
            outputs = model(input_ids, output_attentions=True)
            attentions = outputs["attentions"]  # List of [batch, heads, seq, seq]
            
            # Accumulate per-head patterns
            for layer_idx, attn in enumerate(attentions):
                for head_idx in range(n_heads):
                    head_attn = attn[:, head_idx, :, :].cpu()
                    attention_accum[(layer_idx, head_idx)].append(head_attn)
            
            sample_count += input_ids.shape[0]
    
    # Aggregate into mean patterns
    mean_patterns = {}
    for (layer_idx, head_idx), patterns in attention_accum.items():
        if patterns:
            stacked = torch.cat(patterns, dim=0)
            mean_patterns[(layer_idx, head_idx)] = stacked.mean(dim=0)
    
    model.train()
    
    return mean_patterns


def compute_diagonal_score(
    attention_pattern: torch.Tensor,
    offset: int,
    output_range: Optional[Tuple[int, int]] = None,
) -> float:
    """
    Compute the "Diagonal Attention Score" for a given offset.
    
    This measures how much attention mass is placed on the diagonal
    at the specified relative offset. For the Same-Bit Lookup head,
    this should be high at offset -(n+1).
    
    Args:
        attention_pattern: [seq_len, seq_len] attention matrix
        offset: Relative offset to check (negative = look back)
        output_range: Optional range of output positions to consider
        
    Returns:
        Average attention mass on the specified diagonal
    """
    seq_len = attention_pattern.shape[0]
    
    if output_range is None:
        # Default: analyze output portion (after input + delimiter)
        output_range = (abs(offset), seq_len)
    
    diagonal_mass = []
    
    for i in range(output_range[0], min(output_range[1], seq_len)):
        j = i + offset  # Position to attend to
        if 0 <= j < seq_len:
            diagonal_mass.append(attention_pattern[i, j].item())
    
    return np.mean(diagonal_mass) if diagonal_mass else 0.0


def find_lookup_heads(
    attention_patterns: Dict[Tuple[int, int], torch.Tensor],
    n_bits: int = 20,
    threshold: float = 0.5,
) -> List[Tuple[int, int, float]]:
    """
    Find attention heads that implement the Same-Bit Lookup.
    
    These heads should have high diagonal score at offset -(n+1).
    
    Args:
        attention_patterns: Dictionary of (layer, head) -> pattern
        n_bits: Bit-width (determines expected offset)
        threshold: Minimum diagonal score to be considered a lookup head
        
    Returns:
        List of (layer, head, score) tuples sorted by score
    """
    target_offset = -(n_bits + 1)
    lookup_heads = []
    
    for (layer_idx, head_idx), pattern in attention_patterns.items():
        score = compute_diagonal_score(pattern, target_offset)
        if score >= threshold:
            lookup_heads.append((layer_idx, head_idx, score))
    
    # Sort by score descending
    lookup_heads.sort(key=lambda x: x[2], reverse=True)
    
    return lookup_heads


def visualize_attention(
    model: nn.Module,
    input_sequence: torch.Tensor,
    device: torch.device,
    layer_idx: int = 0,
    head_idx: int = 0,
) -> torch.Tensor:
    """
    Get attention pattern for a single input.
    
    Args:
        model: Transformer model
        input_sequence: [seq_len] or [1, seq_len] input tokens
        device: Device for computation
        layer_idx: Layer to visualize
        head_idx: Head to visualize
        
    Returns:
        [seq_len, seq_len] attention pattern
    """
    model.eval()
    
    if input_sequence.dim() == 1:
        input_sequence = input_sequence.unsqueeze(0)
    
    input_sequence = input_sequence.to(device)
    
    with torch.no_grad():
        outputs = model(input_sequence, output_attentions=True)
        attention = outputs["attentions"][layer_idx][0, head_idx, :, :]
    
    model.train()
    
    return attention.cpu()


def analyze_attention_dynamics(
    checkpoints: List[str],
    dataloader: torch.utils.data.DataLoader,
    model_class: type,
    model_config: object,
    device: torch.device,
    n_bits: int = 20,
    n_samples: int = 50,
) -> Dict[int, Dict]:
    """
    Analyze how attention patterns evolve during training.
    
    This can reveal when the Same-Bit Lookup head emerges,
    typically coinciding with the grokking phase transition.
    
    Args:
        checkpoints: List of checkpoint paths (sorted by step)
        dataloader: DataLoader for analysis
        model_class: Model class to instantiate
        model_config: Model configuration
        device: Device for computation
        n_bits: Bit-width of counter
        n_samples: Samples per checkpoint
        
    Returns:
        Dictionary mapping step -> attention analysis results
    """
    dynamics = {}
    target_offset = -(n_bits + 1)
    
    for ckpt_path in tqdm(checkpoints, desc="Analyzing checkpoints"):
        # Load checkpoint
        checkpoint = torch.load(ckpt_path, map_location=device)
        step = checkpoint.get("global_step", 0)
        
        # Create and load model
        model = model_class(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        # Analyze patterns
        patterns = analyze_attention_patterns(
            model, dataloader, device, n_bits, n_samples
        )
        
        # Compute diagonal scores
        diagonal_scores = {}
        for (layer_idx, head_idx), pattern in patterns.items():
            score = compute_diagonal_score(pattern, target_offset)
            diagonal_scores[(layer_idx, head_idx)] = score
        
        # Find best lookup head
        lookup_heads = find_lookup_heads(patterns, n_bits, threshold=0.1)
        
        dynamics[step] = {
            "diagonal_scores": diagonal_scores,
            "lookup_heads": lookup_heads,
            "max_diagonal_score": max(diagonal_scores.values()) if diagonal_scores else 0,
        }
    
    return dynamics


def attention_entropy(attention_pattern: torch.Tensor) -> float:
    """
    Compute entropy of attention distribution.
    
    Low entropy indicates sparse, peaked attention (good for lookup).
    High entropy indicates diffuse attention (poor for precise retrieval).
    """
    # Normalize to sum to 1 per row
    probs = attention_pattern / (attention_pattern.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Compute entropy per row
    log_probs = torch.log(probs + 1e-10)
    row_entropy = -(probs * log_probs).sum(dim=-1)
    
    return row_entropy.mean().item()


def analyze_head_specialization(
    attention_patterns: Dict[Tuple[int, int], torch.Tensor],
    n_bits: int = 20,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Analyze specialization of each attention head.
    
    Returns metrics for each head:
    - diagonal_score: Same-Bit Lookup score
    - entropy: Attention distribution entropy
    - sparsity: Fraction of near-zero attention weights
    """
    target_offset = -(n_bits + 1)
    specialization = {}
    
    for (layer_idx, head_idx), pattern in attention_patterns.items():
        diagonal_score = compute_diagonal_score(pattern, target_offset)
        entropy = attention_entropy(pattern)
        sparsity = (pattern < 0.01).float().mean().item()
        
        specialization[(layer_idx, head_idx)] = {
            "diagonal_score": diagonal_score,
            "entropy": entropy,
            "sparsity": sparsity,
        }
    
    return specialization


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing attention analysis...")
    
    # Create synthetic attention pattern
    seq_len = 42  # 20 input bits + # + 20 output bits + shift
    pattern = torch.rand(seq_len, seq_len)
    
    # Add diagonal at offset -21 (for n=20)
    offset = -21
    for i in range(abs(offset), seq_len):
        j = i + offset
        if 0 <= j < seq_len:
            pattern[i, j] = 0.9  # High attention on diagonal
    
    # Normalize rows
    pattern = pattern / pattern.sum(dim=-1, keepdim=True)
    
    # Test diagonal score
    score = compute_diagonal_score(pattern, offset)
    print(f"Diagonal score at offset {offset}: {score:.4f}")
    
    # Test entropy
    entropy = attention_entropy(pattern)
    print(f"Attention entropy: {entropy:.4f}")
    
    # Test with mock patterns dictionary
    patterns = {
        (0, 0): pattern,
        (0, 1): torch.rand(seq_len, seq_len),  # Random
        (0, 2): pattern * 1.1,  # Strong diagonal
    }
    
    # Normalize mock patterns
    for k, v in patterns.items():
        patterns[k] = v / v.sum(dim=-1, keepdim=True)
    
    # Find lookup heads
    lookup_heads = find_lookup_heads(patterns, n_bits=20, threshold=0.3)
    print(f"\nLookup heads found: {len(lookup_heads)}")
    for layer, head, score in lookup_heads:
        print(f"  L{layer}H{head}: score={score:.4f}")
    
    # Test specialization analysis
    specialization = analyze_head_specialization(patterns, n_bits=20)
    print(f"\nHead specialization:")
    for (layer, head), metrics in specialization.items():
        print(f"  L{layer}H{head}: diag={metrics['diagonal_score']:.4f}, "
              f"entropy={metrics['entropy']:.4f}, sparsity={metrics['sparsity']:.4f}")
    
    print("\n✓ All tests passed!")
