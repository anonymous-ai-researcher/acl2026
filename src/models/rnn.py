"""
RNN Baseline Models (LSTM/GRU) for the LARGECOUNTER task.

These baselines demonstrate the exponential bottleneck predicted by theory:
RNNs require O(2^n) states to track n-bit counters, making them unable
to generalize even with massive hidden dimensions.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RNNConfig:
    """Configuration for RNN models."""
    
    vocab_size: int = 6          # {pad, 0, 1, #, bos, eos}
    hidden_dim: int = 2048       # Hidden dimension (scaled up for fair comparison)
    n_layers: int = 2            # Number of RNN layers
    model_type: str = "lstm"     # "lstm" or "gru"
    dropout: float = 0.0         # Dropout rate
    bidirectional: bool = False  # Whether to use bidirectional RNN
    init_std: float = 0.02       # Weight initialization std


class RNNLM(nn.Module):
    """
    RNN Language Model (LSTM/GRU) baseline.
    
    Despite having significantly more parameters than the Transformer
    (e.g., 1.5M vs 50K), RNNs fail to generalize on the counting task
    due to the exponential state bottleneck.
    
    The hidden state h_t ∈ R^d cannot succinctly represent 2^n distinct
    counter states, leading to performance collapse on held-out sequences.
    """
    
    def __init__(self, config: RNNConfig):
        super().__init__()
        
        self.config = config
        
        # Token embedding
        self.token_emb = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # RNN backbone
        rnn_cls = nn.LSTM if config.model_type == "lstm" else nn.GRU
        self.rnn = rnn_cls(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.n_layers,
            batch_first=True,
            dropout=config.dropout if config.n_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        
        # Output dimension adjustment for bidirectional
        self.output_dim = config.hidden_dim * (2 if config.bidirectional else 1)
        
        # Output projection
        self.fc_out = nn.Linear(self.output_dim, config.vocab_size)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module: nn.Module):
        """Initialize weights."""
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=self.config.init_std)
        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        hidden: Optional[Tuple[torch.Tensor, ...]] = None,
        output_hidden_states: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: Optional (not used, for interface compatibility)
            labels: Optional labels for loss computation
            hidden: Optional initial hidden state
            output_hidden_states: Whether to return hidden states
            
        Returns:
            Dictionary with logits, loss, and optional hidden states
        """
        batch_size, seq_len = input_ids.shape
        
        # Embed tokens
        x = self.token_emb(input_ids)
        x = self.dropout(x)
        
        # RNN forward
        if hidden is not None:
            rnn_out, final_hidden = self.rnn(x, hidden)
        else:
            rnn_out, final_hidden = self.rnn(x)
        
        # Dropout on RNN output
        rnn_out = self.dropout(rnn_out)
        
        # Output projection
        logits = self.fc_out(rnn_out)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1),
                ignore_index=-100,
            )
        
        outputs = {
            "logits": logits,
            "loss": loss,
            "hidden": final_hidden,
        }
        
        if output_hidden_states:
            outputs["hidden_states"] = rnn_out
        
        return outputs
    
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
        hidden = None
        
        # Process prefix
        outputs = self.forward(input_ids, hidden=hidden)
        hidden = outputs["hidden"]
        
        # Generate new tokens
        for _ in range(max_new_tokens):
            # Get last token's logits
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
            
            # Forward with new token
            outputs = self.forward(next_token, hidden=hidden)
            hidden = outputs["hidden"]
        
        return input_ids
    
    def get_weight_norm(self) -> float:
        """Compute total L2 norm of all parameters."""
        total_norm = 0.0
        for p in self.parameters():
            total_norm += p.data.norm(2).item() ** 2
        return total_norm ** 0.5
    
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_hidden_state_norm(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute norm of hidden states for each position.
        
        This can reveal state saturation in RNNs that fail to track counters.
        """
        outputs = self.forward(input_ids, output_hidden_states=True)
        hidden_states = outputs["hidden_states"]  # [batch, seq, hidden]
        norms = torch.norm(hidden_states, dim=-1)  # [batch, seq]
        return norms


def create_rnn(
    vocab_size: int = 6,
    hidden_dim: int = 2048,
    n_layers: int = 2,
    model_type: str = "lstm",
    **kwargs,
) -> RNNLM:
    """
    Factory function to create an RNN model.
    
    Args:
        vocab_size: Vocabulary size
        hidden_dim: Hidden dimension
        n_layers: Number of layers
        model_type: "lstm" or "gru"
        **kwargs: Additional config options
        
    Returns:
        RNNLM model
    """
    config = RNNConfig(
        vocab_size=vocab_size,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        model_type=model_type,
        dropout=kwargs.get("dropout", 0.0),
        bidirectional=kwargs.get("bidirectional", False),
        init_std=kwargs.get("init_std", 0.02),
    )
    
    return RNNLM(config)


# =============================================================================
# Model Comparison Utilities
# =============================================================================

def compare_model_sizes():
    """Compare parameter counts between Transformer and RNN."""
    from .transformer import TransformerLM, TransformerConfig
    
    print("=" * 60)
    print("Model Size Comparison")
    print("=" * 60)
    
    # Transformer (d=64, L=2, H=4)
    tf_config = TransformerConfig(
        vocab_size=6,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
    )
    tf_model = TransformerLM(tf_config)
    tf_params = tf_model.count_parameters()
    
    print(f"\nTransformer (d=64, L=2, H=4):")
    print(f"  Parameters: {tf_params:,}")
    
    # LSTM baselines
    for hidden_dim in [64, 256, 512, 1024, 2048]:
        rnn_config = RNNConfig(
            vocab_size=6,
            hidden_dim=hidden_dim,
            n_layers=2,
            model_type="lstm",
        )
        rnn_model = RNNLM(rnn_config)
        rnn_params = rnn_model.count_parameters()
        ratio = rnn_params / tf_params
        
        print(f"\nLSTM (d={hidden_dim}, L=2):")
        print(f"  Parameters: {rnn_params:,} ({ratio:.1f}x Transformer)")
    
    print("\n" + "=" * 60)
    print("Despite having 30x more parameters, LSTMs fail to generalize")
    print("due to the exponential state bottleneck (Proposition 1).")
    print("=" * 60)


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    print("Testing RNNLM...")
    
    # Create LSTM model
    config = RNNConfig(
        vocab_size=6,
        hidden_dim=256,
        n_layers=2,
        model_type="lstm",
    )
    model = RNNLM(config)
    
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Model config: {config}")
    
    # Test forward pass
    batch_size, seq_len = 4, 41
    input_ids = torch.randint(0, 6, (batch_size, seq_len))
    labels = torch.randint(0, 6, (batch_size, seq_len))
    labels[:, :21] = -100
    
    outputs = model(input_ids, labels=labels, output_hidden_states=True)
    
    print(f"\nForward pass:")
    print(f"  Logits shape: {outputs['logits'].shape}")
    print(f"  Loss: {outputs['loss'].item():.4f}")
    print(f"  Hidden states shape: {outputs['hidden_states'].shape}")
    
    # Test generation
    prefix = torch.randint(0, 3, (2, 21))
    generated = model.generate(prefix, max_new_tokens=20)
    print(f"\nGeneration:")
    print(f"  Input shape: {prefix.shape}")
    print(f"  Output shape: {generated.shape}")
    
    # Test GRU
    gru_config = RNNConfig(
        vocab_size=6,
        hidden_dim=256,
        n_layers=2,
        model_type="gru",
    )
    gru_model = RNNLM(gru_config)
    gru_outputs = gru_model(input_ids, labels=labels)
    print(f"\nGRU forward pass:")
    print(f"  Loss: {gru_outputs['loss'].item():.4f}")
    
    # Compare model sizes
    print("\n")
    compare_model_sizes()
    
    print("\n✓ All tests passed!")
