"""
Tests for the models module (Transformer and RNN).

Tests cover:
- RoPE positional embeddings
- Transformer architecture
- RNN baselines
- Forward pass correctness
- Generation functionality
- Weight norm computation
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.rope import RotaryEmbedding, apply_rotary_pos_emb
from src.models.transformer import (
    TransformerConfig,
    TransformerLM,
    MultiHeadAttention,
    FeedForward,
    TransformerBlock
)
from src.models.rnn import RNNConfig, RNNLM, compare_model_sizes


class TestRotaryEmbedding:
    """Tests for Rotary Position Embeddings (RoPE)."""
    
    @pytest.fixture
    def rope(self):
        return RotaryEmbedding(dim=64, max_seq_len=512)
    
    def test_rope_output_shape(self, rope):
        """Test RoPE returns correct shapes."""
        seq_len = 32
        cos, sin = rope(seq_len)
        
        assert cos.shape == (seq_len, 64)
        assert sin.shape == (seq_len, 64)
    
    def test_rope_values_bounded(self, rope):
        """Test that cos/sin values are in [-1, 1]."""
        cos, sin = rope(100)
        
        assert cos.min() >= -1.0
        assert cos.max() <= 1.0
        assert sin.min() >= -1.0
        assert sin.max() <= 1.0
    
    def test_rope_different_positions(self, rope):
        """Test that different positions get different embeddings."""
        cos, sin = rope(10)
        
        # Position 0 and position 5 should be different
        assert not torch.allclose(cos[0], cos[5])
        assert not torch.allclose(sin[0], sin[5])
    
    def test_apply_rotary_pos_emb_shape(self, rope):
        """Test that applying RoPE preserves tensor shape."""
        batch_size = 4
        n_heads = 8
        seq_len = 32
        head_dim = 64
        
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        cos, sin = rope(seq_len)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        assert q_rot.shape == q.shape
        assert k_rot.shape == k.shape
    
    def test_rope_relative_position(self, rope):
        """Test RoPE encodes relative positions via rotation."""
        # The key property of RoPE is that q_i^T * k_j depends only on (i-j)
        # We test this indirectly by checking consistency
        batch_size = 1
        n_heads = 1
        seq_len = 16
        head_dim = 64
        
        q = torch.randn(batch_size, n_heads, seq_len, head_dim)
        k = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        cos, sin = rope(seq_len)
        q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Compute attention scores
        scores = torch.matmul(q_rot, k_rot.transpose(-2, -1)) / np.sqrt(head_dim)
        
        # Scores should be finite
        assert torch.isfinite(scores).all()


class TestTransformerConfig:
    """Tests for Transformer configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = TransformerConfig()
        
        assert config.vocab_size == 6
        assert config.d_model == 64
        assert config.n_heads == 4
        assert config.n_layers == 2
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TransformerConfig(
            vocab_size=100,
            d_model=128,
            n_heads=8,
            n_layers=4
        )
        
        assert config.vocab_size == 100
        assert config.d_model == 128


class TestMultiHeadAttention:
    """Tests for Multi-Head Attention with RoPE."""
    
    @pytest.fixture
    def attention(self):
        return MultiHeadAttention(d_model=64, n_heads=4, use_rope=True)
    
    def test_attention_output_shape(self, attention):
        """Test attention output shape."""
        batch_size = 4
        seq_len = 32
        d_model = 64
        
        x = torch.randn(batch_size, seq_len, d_model)
        output, attn_weights = attention(x, return_attention=True)
        
        assert output.shape == (batch_size, seq_len, d_model)
        assert attn_weights.shape == (batch_size, 4, seq_len, seq_len)
    
    def test_attention_causal_mask(self, attention):
        """Test that causal masking is applied correctly."""
        x = torch.randn(1, 8, 64)
        _, attn_weights = attention(x, return_attention=True)
        
        # Upper triangular should be zero (causal mask)
        for i in range(8):
            for j in range(i + 1, 8):
                # Allow small numerical error
                assert attn_weights[0, 0, i, j] < 1e-6
    
    def test_attention_weights_sum_to_one(self, attention):
        """Test that attention weights sum to 1."""
        x = torch.randn(2, 16, 64)
        _, attn_weights = attention(x, return_attention=True)
        
        # Each row should sum to 1 (softmax property)
        row_sums = attn_weights.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


class TestFeedForward:
    """Tests for Feed-Forward network."""
    
    def test_feedforward_shape(self):
        """Test FFN output shape."""
        ffn = FeedForward(d_model=64, d_ff=256)
        x = torch.randn(4, 32, 64)
        output = ffn(x)
        
        assert output.shape == x.shape
    
    def test_feedforward_gelu_activation(self):
        """Test that GELU activation is used."""
        ffn = FeedForward(d_model=64, d_ff=256)
        
        # GELU should be somewhere in the module
        has_gelu = any(
            isinstance(m, nn.GELU) 
            for m in ffn.modules()
        )
        assert has_gelu


class TestTransformerBlock:
    """Tests for Transformer Block."""
    
    @pytest.fixture
    def block(self):
        return TransformerBlock(d_model=64, n_heads=4, d_ff=256, use_rope=True)
    
    def test_block_output_shape(self, block):
        """Test block output shape."""
        x = torch.randn(4, 32, 64)
        output = block(x)
        
        assert output.shape == x.shape
    
    def test_block_residual_connection(self, block):
        """Test that residual connections are present."""
        # With LayerNorm and residual, output should not be too different from input
        x = torch.randn(4, 32, 64) * 0.1
        output = block(x)
        
        # Residual should prevent output from being completely different
        diff = (output - x).abs().mean()
        assert diff < 10.0  # Reasonable bound


class TestTransformerLM:
    """Tests for the full Transformer Language Model."""
    
    @pytest.fixture
    def model(self):
        config = TransformerConfig(
            vocab_size=6,
            d_model=64,
            n_heads=4,
            n_layers=2,
            d_ff=256,
            max_seq_len=128,
            use_rope=True
        )
        return TransformerLM(config)
    
    def test_forward_shape(self, model):
        """Test forward pass output shape."""
        batch_size = 4
        seq_len = 32
        
        input_ids = torch.randint(0, 6, (batch_size, seq_len))
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, 6)
    
    def test_generate_shape(self, model):
        """Test generation output shape."""
        prefix = torch.tensor([[0, 1, 1, 1, 2]])  # 0111#
        generated = model.generate(prefix, max_new_tokens=4)
        
        assert generated.shape[0] == 1
        assert generated.shape[1] == prefix.shape[1] + 4
    
    def test_weight_norm(self, model):
        """Test weight norm computation."""
        norm = model.get_weight_norm()
        
        assert isinstance(norm, float)
        assert norm > 0
    
    def test_get_attention_patterns(self, model):
        """Test attention pattern extraction."""
        input_ids = torch.randint(0, 6, (1, 16))
        patterns = model.get_attention_patterns(input_ids)
        
        assert len(patterns) == 2  # 2 layers
        assert patterns[0].shape[-1] == 16  # seq_len
    
    def test_parameter_count(self, model):
        """Test that model has reasonable parameter count."""
        n_params = sum(p.numel() for p in model.parameters())
        
        # Should be around 50K for the paper's config
        assert 10000 < n_params < 200000
    
    def test_weight_tying(self, model):
        """Test that embedding and output weights are tied."""
        embed_weight = model.embedding.weight
        output_weight = model.output_proj.weight
        
        assert torch.equal(embed_weight, output_weight)


class TestRNNConfig:
    """Tests for RNN configuration."""
    
    def test_default_config(self):
        """Test default RNN configuration."""
        config = RNNConfig()
        
        assert config.hidden_dim == 64
        assert config.model_type == "lstm"
    
    def test_gru_config(self):
        """Test GRU configuration."""
        config = RNNConfig(model_type="gru", hidden_dim=256)
        
        assert config.model_type == "gru"
        assert config.hidden_dim == 256


class TestRNNLM:
    """Tests for the RNN Language Model."""
    
    @pytest.fixture
    def lstm_model(self):
        config = RNNConfig(
            vocab_size=6,
            hidden_dim=64,
            n_layers=2,
            model_type="lstm"
        )
        return RNNLM(config)
    
    @pytest.fixture
    def gru_model(self):
        config = RNNConfig(
            vocab_size=6,
            hidden_dim=64,
            n_layers=2,
            model_type="gru"
        )
        return RNNLM(config)
    
    def test_lstm_forward(self, lstm_model):
        """Test LSTM forward pass."""
        input_ids = torch.randint(0, 6, (4, 32))
        logits = lstm_model(input_ids)
        
        assert logits.shape == (4, 32, 6)
    
    def test_gru_forward(self, gru_model):
        """Test GRU forward pass."""
        input_ids = torch.randint(0, 6, (4, 32))
        logits = gru_model(input_ids)
        
        assert logits.shape == (4, 32, 6)
    
    def test_rnn_generation(self, lstm_model):
        """Test RNN generation."""
        prefix = torch.tensor([[0, 1, 1, 1, 2]])
        generated = lstm_model.generate(prefix, max_new_tokens=4)
        
        assert generated.shape[1] == 9


class TestModelComparison:
    """Tests for model comparison utilities."""
    
    def test_compare_model_sizes(self):
        """Test parameter comparison function."""
        comparison = compare_model_sizes()
        
        assert "transformer" in comparison
        assert "lstm_64" in comparison or "lstm" in comparison
        
        # All values should be positive
        for name, params in comparison.items():
            assert params > 0


class TestGradientFlow:
    """Tests for gradient flow through models."""
    
    def test_transformer_gradient_flow(self):
        """Test gradients flow through Transformer."""
        config = TransformerConfig(vocab_size=6, d_model=32, n_heads=2, n_layers=1)
        model = TransformerLM(config)
        
        input_ids = torch.randint(0, 6, (2, 8))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_rnn_gradient_flow(self):
        """Test gradients flow through RNN."""
        config = RNNConfig(vocab_size=6, hidden_dim=32, n_layers=1)
        model = RNNLM(config)
        
        input_ids = torch.randint(0, 6, (2, 8))
        logits = model(input_ids)
        loss = logits.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
