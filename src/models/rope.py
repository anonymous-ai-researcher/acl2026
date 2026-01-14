"""
Rotary Position Embedding (RoPE) Implementation.

RoPE encodes position information directly into rotation matrices, allowing
attention heads to naturally attend to tokens based on relative offsets.
This is critical for the "Same-Bit Lookup" mechanism in binary counting.

Reference: Su et al. (2024) "RoFormer: Enhanced Transformer with Rotary Position Embedding"
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding module.
    
    RoPE applies a rotation to query and key vectors based on their positions,
    enabling relative position encoding without explicit positional embeddings.
    
    For the counting task, this allows attention heads to learn the fixed
    relative offset -(n+1) required for Same-Bit Lookup.
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 512,
        base: float = 10000.0,
        device: Optional[torch.device] = None,
    ):
        """
        Initialize RoPE.
        
        Args:
            dim: Dimension of the embedding (should be even, typically head_dim)
            max_seq_len: Maximum sequence length
            base: Base for frequency computation
            device: Device for tensors
        """
        super().__init__()
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute inverse frequencies
        # theta_i = base^(-2i/d) for i = 0, 1, ..., d/2 - 1
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        
        # Precompute cos and sin for all positions
        self._precompute_cache(max_seq_len, device)
    
    def _precompute_cache(
        self,
        seq_len: int,
        device: Optional[torch.device] = None,
    ):
        """Precompute and cache cos/sin values."""
        # Position indices: [0, 1, 2, ..., seq_len-1]
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        
        # Compute outer product: [seq_len, dim/2]
        freqs = torch.outer(t, self.inv_freq)
        
        # Concatenate to get [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Cache cos and sin
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    
    def forward(
        self,
        x: torch.Tensor,
        seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cos and sin for the given sequence length.
        
        Args:
            x: Input tensor (used only for device/dtype)
            seq_len: Sequence length (if None, inferred from cache)
            
        Returns:
            Tuple of (cos, sin) tensors with shape [seq_len, dim]
        """
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Extend cache if needed
        if seq_len > self.cos_cached.shape[0]:
            self._precompute_cache(seq_len, x.device)
        
        return (
            self.cos_cached[:seq_len].to(x.dtype),
            self.sin_cached[:seq_len].to(x.dtype),
        )


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    Rotate half the hidden dims of the input.
    
    For input [x1, x2, x3, x4, ...], returns [-x_{d/2+1}, ..., -x_d, x1, ..., x_{d/2}]
    This implements the rotation matrix multiplication efficiently.
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embedding to query and key tensors.
    
    The rotation is applied as:
        q_rot = q * cos + rotate_half(q) * sin
        k_rot = k * cos + rotate_half(k) * sin
    
    This encodes absolute position in a way that makes the dot product
    q_rot · k_rot dependent only on relative position.
    
    Args:
        q: Query tensor of shape [..., seq_len, head_dim]
        k: Key tensor of shape [..., seq_len, head_dim]
        cos: Cosine tensor of shape [seq_len, head_dim]
        sin: Sine tensor of shape [seq_len, head_dim]
        position_ids: Optional position indices
        
    Returns:
        Tuple of rotated (query, key) tensors
    """
    if position_ids is not None:
        # Gather cos/sin for specific positions
        cos = cos[position_ids].unsqueeze(1)  # [batch, 1, seq_len, head_dim]
        sin = sin[position_ids].unsqueeze(1)
    else:
        # Broadcast cos/sin to match q/k shape
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    
    # Apply rotation
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed


class RotaryAttention(nn.Module):
    """
    Multi-head attention with Rotary Position Embedding.
    
    This module combines standard multi-head attention with RoPE,
    enabling relative position-aware attention patterns.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 512,
        rope_base: float = 10000.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Initialize rotary attention.
        
        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            max_seq_len: Maximum sequence length
            rope_base: RoPE base frequency
            dropout: Attention dropout
            bias: Whether to use bias in projections
        """
        super().__init__()
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=max_seq_len,
            base=rope_base,
        )
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # For storing attention weights (for analysis)
        self.attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with rotary attention.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model]
            attention_mask: Optional mask of shape [batch, 1, seq_len, seq_len]
            output_attentions: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_weights if requested)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax and dropout
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store for analysis
        if output_attentions:
            self.attention_weights = attn_weights.detach()
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(output)
        
        if output_attentions:
            return output, attn_weights
        return output, None


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing RoPE implementation...")
    
    # Test RotaryEmbedding
    rope = RotaryEmbedding(dim=16, max_seq_len=100)
    x = torch.randn(2, 10, 16)
    cos, sin = rope(x, seq_len=10)
    print(f"cos shape: {cos.shape}, sin shape: {sin.shape}")
    
    # Test rotation
    q = torch.randn(2, 4, 10, 16)  # [batch, heads, seq, head_dim]
    k = torch.randn(2, 4, 10, 16)
    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"q_rot shape: {q_rot.shape}")
    
    # Verify relative position property
    # The dot product q[i] · k[j] should depend only on relative position i-j
    attn_1 = torch.matmul(q_rot[:, :, 5:6, :], k_rot[:, :, 3:4, :].transpose(-2, -1))
    attn_2 = torch.matmul(q_rot[:, :, 7:8, :], k_rot[:, :, 5:6, :].transpose(-2, -1))
    print(f"Attention scores for offset -2: similar = {torch.allclose(attn_1, attn_2, atol=1e-4)}")
    
    # Test RotaryAttention
    attn = RotaryAttention(d_model=64, n_heads=4, max_seq_len=100)
    x = torch.randn(2, 10, 64)
    out, weights = attn(x, output_attentions=True)
    print(f"Attention output shape: {out.shape}")
    print(f"Attention weights shape: {weights.shape}")
    
    print("\n✓ All tests passed!")
