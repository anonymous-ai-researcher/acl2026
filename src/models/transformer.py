"""
Transformer Language Model Implementation.

This module implements a GPT-style decoder-only Transformer with:
- Pre-LayerNorm architecture
- Rotary Position Embeddings (RoPE)
- GELU activation

The architecture is designed to be succinct (few layers/heads) while still
capable of learning the ripple-carry algorithm for binary counting.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RotaryEmbedding, apply_rotary_pos_emb


@dataclass
class TransformerConfig:
    """Configuration for the Transformer model."""
    
    vocab_size: int = 6          # {pad, 0, 1, #, bos, eos}
    d_model: int = 64            # Hidden dimension
    n_layers: int = 2            # Number of layers
    n_heads: int = 4             # Number of attention heads
    d_ff: int = 256              # Feedforward dimension
    max_seq_len: int = 512       # Maximum sequence length
    dropout: float = 0.0         # Dropout rate
    activation: str = "gelu"     # Activation function
    norm_first: bool = True      # Pre-LayerNorm
    rope_base: float = 10000.0   # RoPE base frequency
    init_std: float = 0.02       # Weight initialization std
    bias: bool = False           # Use bias in linear layers


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with Rotary Position Embedding.
    
    Key feature: RoPE enables precise relative addressing for the
    Same-Bit Lookup mechanism (attending to position -(n+1)).
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.scale = self.head_dim ** -0.5
        
        assert config.d_model % config.n_heads == 0
        
        # Projections
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=config.bias)
        
        # RoPE
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_seq_len=config.max_seq_len,
            base=config.rope_base,
        )
        
        self.dropout = nn.Dropout(config.dropout)
        
        # Cache for attention patterns (for interpretability)
        self._attention_weights = None
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            output_attentions: Whether to return attention weights
            
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: Optional [batch, n_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape to [batch, n_heads, seq_len, head_dim]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(x, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1,
        )
        attn_scores = attn_scores.masked_fill(causal_mask, float('-inf'))
        
        # Additional mask
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store for analysis
        if output_attentions:
            self._attention_weights = attn_weights.detach().clone()
        
        # Apply to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        out = self.out_proj(out)
        
        return out, attn_weights if output_attentions else None
    
    @property
    def attention_weights(self) -> Optional[torch.Tensor]:
        """Return cached attention weights."""
        return self._attention_weights


class FeedForward(nn.Module):
    """
    Feed-Forward Network.
    
    This module approximates the XOR/AND logic gates for carry computation.
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=config.bias)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
        
        if config.activation == "gelu":
            self.activation = F.gelu
        elif config.activation == "relu":
            self.activation = F.relu
        elif config.activation == "silu":
            self.activation = F.silu
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer block with attention and FFN.
    
    Uses Pre-LayerNorm architecture for training stability.
    """
    
    def __init__(self, config: TransformerConfig, layer_idx: int):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiHeadAttention(config)
        
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)
        
        self.norm_first = config.norm_first
    
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with residual connections."""
        
        if self.norm_first:
            # Pre-LayerNorm
            attn_out, attn_weights = self.attn(
                self.ln1(x), attention_mask, output_attentions
            )
            x = x + attn_out
            x = x + self.ffn(self.ln2(x))
        else:
            # Post-LayerNorm
            attn_out, attn_weights = self.attn(x, attention_mask, output_attentions)
            x = self.ln1(x + attn_out)
            x = self.ln2(x + self.ffn(x))
        
        return x, attn_weights


class TransformerLM(nn.Module):
    """
    Transformer Language Model for the LARGECOUNTER task.
    
    Architecture:
    - Token embedding (vocab → d_model)
    - Stack of Transformer blocks
    - Final layer norm
    - Output projection (d_model → vocab)
    
    Key design choices for succinctness:
    - Small d_model (64) and few layers (2)
    - RoPE for precise relative addressing
    - Pre-LayerNorm for training stability
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config, idx) for idx in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model)
        
        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying (embedding and output)
        self.lm_head.weight = self.token_emb.weight
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights with truncated normal."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
            output_attentions: Return attention weights
            output_hidden_states: Return hidden states
            
        Returns:
            Dictionary with logits, loss, and optional intermediate outputs
        """
        # Embed tokens
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # Collect outputs
        all_hidden_states = [] if output_hidden_states else None
        all_attentions = [] if output_attentions else None
        
        if output_hidden_states:
            all_hidden_states.append(x)
        
        # Apply transformer blocks
        for layer in self.layers:
            x, attn_weights = layer(x, attention_mask, output_attentions)
            
            if output_hidden_states:
                all_hidden_states.append(x)
            if output_attentions:
                all_attentions.append(attn_weights)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Output logits
        logits = self.lm_head(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        return {
            "logits": logits,
            "loss": loss,
            "hidden_states": all_hidden_states,
            "attentions": all_attentions,
        }
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation.
        
        Args:
            input_ids: [batch, seq_len] prefix tokens
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            
        Returns:
            Generated tokens [batch, seq_len + max_new_tokens]
        """
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(input_ids)
            logits = outputs["logits"][:, -1, :] / temperature
            
            # Optional top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        return input_ids
    
    def get_weight_norm(self) -> float:
        """Compute total L2 norm of all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def get_attention_patterns(self) -> List[torch.Tensor]:
        """Collect attention patterns from all layers."""
        patterns = []
        for layer in self.layers:
            if layer.attn.attention_weights is not None:
                patterns.append(layer.attn.attention_weights)
        return patterns
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_transformer(
    vocab_size: int = 6,
    d_model: int = 64,
    n_layers: int = 2,
    n_heads: int = 4,
    **kwargs,
) -> TransformerLM:
    """
    Factory function to create a Transformer model.
    
    Args:
        vocab_size: Vocabulary size
        d_model: Hidden dimension
        n_layers: Number of layers
        n_heads: Number of attention heads
        **kwargs: Additional config options
        
    Returns:
        TransformerLM model
    """
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=kwargs.get("d_ff", 4 * d_model),
        max_seq_len=kwargs.get("max_seq_len", 512),
        dropout=kwargs.get("dropout", 0.0),
        activation=kwargs.get("activation", "gelu"),
        norm_first=kwargs.get("norm_first", True),
        rope_base=kwargs.get("rope_base", 10000.0),
        init_std=kwargs.get("init_std", 0.02),
        bias=kwargs.get("bias", False),
    )
    
    return TransformerLM(config)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing TransformerLM...")
    
    # Create model
    config = TransformerConfig(
        vocab_size=6,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
    )
    model = TransformerLM(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model config: {config}")
    
    # Test forward pass
    batch_size, seq_len = 4, 42  # n=20: input(20) + #(1) + output(20) + 1 for shift
    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))
    labels[:, :21] = -100  # Mask input portion
    
    outputs = model(input_ids, labels=labels, output_attentions=True)
    
    print(f"\nForward pass:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Attention shapes: {[a.shape for a in outputs['attentions']]}")
    
    # Test generation
    prefix = torch.randint(0, 3, (2, 21))  # Just input + delimiter
    generated = model.generate(prefix, max_new_tokens=20)
    print(f"\nGeneration:")
    print(f"  Input shape: {prefix.shape}")
    print(f"  Output shape: {generated.shape}")
    
    # Test weight norm
    norm = model.get_weight_norm()
    print(f"\nWeight L2 norm: {norm:.4f}")
    
    print("\n✓ All tests passed!")
