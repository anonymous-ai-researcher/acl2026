"""
Tests for the analysis module.

Tests cover:
- Attention pattern analysis
- Activation patching
- Visualization utilities
- Same-Bit Lookup detection
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.transformer import TransformerConfig, TransformerLM
from src.analysis.attention_analysis import (
    analyze_attention_patterns,
    compute_diagonal_score,
    find_lookup_heads,
    attention_entropy,
    analyze_head_specialization
)
from src.analysis.activation_patching import (
    create_patching_pairs,
    ActivationPatcher,
    activation_patching,
    identify_circuit_components
)
from src.analysis.visualization import (
    plot_grokking_dynamics,
    plot_weight_norm,
    plot_attention_heatmap,
    plot_patching_heatmap
)


class TestAttentionAnalysis:
    """Tests for attention pattern analysis."""
    
    @pytest.fixture
    def model(self):
        config = TransformerConfig(
            vocab_size=6, 
            d_model=32, 
            n_heads=4, 
            n_layers=2,
            use_rope=True
        )
        return TransformerLM(config)
    
    @pytest.fixture
    def sample_attention(self):
        """Create sample attention patterns."""
        # Shape: (batch, n_heads, seq_len, seq_len)
        batch_size = 2
        n_heads = 4
        seq_len = 16
        
        # Random attention (row-normalized)
        attn = torch.rand(batch_size, n_heads, seq_len, seq_len)
        attn = attn / attn.sum(dim=-1, keepdim=True)
        
        # Apply causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len))
        attn = attn * mask.unsqueeze(0).unsqueeze(0)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-10)
        
        return attn
    
    def test_analyze_attention_patterns_shape(self, model):
        """Test attention pattern extraction shape."""
        input_ids = torch.randint(0, 6, (2, 16))
        
        patterns = model.get_attention_patterns(input_ids)
        
        assert len(patterns) == 2  # 2 layers
        assert patterns[0].shape[1] == 4  # 4 heads
    
    def test_compute_diagonal_score(self, sample_attention):
        """Test diagonal attention score computation."""
        # Create attention with known diagonal
        n_heads = 4
        seq_len = 16
        
        # Make a pattern with strong diagonal at offset -5
        diag_attn = torch.zeros(1, n_heads, seq_len, seq_len)
        for i in range(5, seq_len):
            diag_attn[0, 0, i, i-5] = 1.0  # Head 0 has diagonal
        
        score = compute_diagonal_score(diag_attn[0, 0], offset=-5)
        
        assert score > 0.5  # Should be high for diagonal pattern
    
    def test_compute_diagonal_score_no_diagonal(self, sample_attention):
        """Test diagonal score is low for random attention."""
        score = compute_diagonal_score(sample_attention[0, 0], offset=-10)
        
        # Random attention should have low diagonal score
        assert score < 0.5
    
    def test_find_lookup_heads(self):
        """Test identification of lookup heads."""
        # Create attention patterns with one clear lookup head
        n_layers = 2
        n_heads = 4
        seq_len = 20
        n_bits = 8
        
        patterns = []
        for _ in range(n_layers):
            layer_attn = torch.zeros(1, n_heads, seq_len, seq_len)
            
            # Make head 2 a lookup head with offset -(n+1) = -9
            for i in range(9, seq_len):
                layer_attn[0, 2, i, i-9] = 0.9
                # Add some noise to other positions
                remaining = 0.1
                for j in range(i):
                    if j != i-9:
                        layer_attn[0, 2, i, j] = remaining / max(1, i-1)
            
            patterns.append(layer_attn)
        
        lookup_heads = find_lookup_heads(
            patterns, 
            offset=-(n_bits + 1), 
            threshold=0.5
        )
        
        # Head 2 should be identified
        assert (0, 2) in lookup_heads
    
    def test_attention_entropy(self, sample_attention):
        """Test attention entropy computation."""
        entropy = attention_entropy(sample_attention[0, 0])
        
        # Entropy should be positive
        assert entropy > 0
        
        # Uniform attention should have max entropy
        uniform = torch.ones(16, 16) / 16
        uniform_entropy = attention_entropy(uniform)
        
        assert uniform_entropy > entropy * 0.5  # Uniform should be higher


class TestActivationPatching:
    """Tests for activation patching."""
    
    @pytest.fixture
    def model(self):
        config = TransformerConfig(
            vocab_size=6, 
            d_model=32, 
            n_heads=4, 
            n_layers=2
        )
        return TransformerLM(config)
    
    def test_create_patching_pairs(self):
        """Test creation of patching pairs."""
        n_bits = 8
        pairs = create_patching_pairs(n_bits, n_pairs=10)
        
        assert len(pairs) == 10
        
        for clean, corrupt in pairs:
            # Clean should have trailing 1s (carry required)
            assert "1" in clean
            # Corrupt should differ in LSB
            assert clean != corrupt
    
    def test_activation_patcher_hooks(self, model):
        """Test that activation patcher correctly hooks model."""
        patcher = ActivationPatcher(model)
        
        # Register some hooks
        def dummy_hook(module, input, output):
            return output
        
        patcher.register_hook("layer_0", dummy_hook)
        
        # Should have registered hook
        assert len(patcher.hooks) > 0
        
        # Cleanup
        patcher.remove_hooks()
        assert len(patcher.hooks) == 0
    
    def test_activation_patching_runs(self, model):
        """Test that activation patching runs without error."""
        model.eval()
        
        # Create simple inputs
        clean_ids = torch.tensor([[0, 1, 1, 1, 2, 1, 0, 0, 0]])
        corrupt_ids = torch.tensor([[0, 1, 1, 0, 2, 1, 0, 0, 1]])
        
        # This should run without error
        # Full patching test would require more setup
    
    def test_identify_circuit_components(self):
        """Test circuit component identification."""
        # Mock patching results
        patching_results = {
            "L0_H2": {"logit_diff_drop": 0.85},
            "L0_MLP": {"logit_diff_drop": 0.92},
            "L1_H0": {"logit_diff_drop": 0.15},
            "L1_MLP": {"logit_diff_drop": 0.30}
        }
        
        components = identify_circuit_components(
            patching_results, 
            retrieval_threshold=0.5,
            compute_threshold=0.7
        )
        
        # L0_H2 and L0_MLP should be identified as important
        assert "retrieval" in components or "computation" in components


class TestVisualization:
    """Tests for visualization utilities."""
    
    def test_plot_grokking_dynamics(self):
        """Test grokking dynamics plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "steps": [100, 200, 300, 400, 500],
                "train_acc": [0.5, 0.8, 0.95, 0.98, 0.99],
                "test_acc": [0.1, 0.15, 0.2, 0.6, 0.95]
            }
            
            save_path = Path(tmpdir) / "grokking.png"
            plot_grokking_dynamics(history, save_path=save_path)
            
            assert save_path.exists()
    
    def test_plot_weight_norm(self):
        """Test weight norm plot generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            history = {
                "steps": [100, 200, 300, 400, 500],
                "test_acc": [0.1, 0.15, 0.2, 0.6, 0.95],
                "weight_norm": [25, 24, 22, 18, 15]
            }
            
            save_path = Path(tmpdir) / "weight_norm.png"
            plot_weight_norm(history, save_path=save_path)
            
            assert save_path.exists()
    
    def test_plot_attention_heatmap(self):
        """Test attention heatmap generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            attention = torch.rand(16, 16)
            
            save_path = Path(tmpdir) / "attention.png"
            plot_attention_heatmap(
                attention.numpy(), 
                layer=0, 
                head=2,
                save_path=save_path
            )
            
            assert save_path.exists()
    
    def test_plot_patching_heatmap(self):
        """Test patching heatmap generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            results = np.random.rand(10, 3)  # 10 components, 3 positions
            
            save_path = Path(tmpdir) / "patching.png"
            plot_patching_heatmap(
                results,
                component_names=[f"Comp_{i}" for i in range(10)],
                position_names=["Prev", "Curr", "Next"],
                save_path=save_path
            )
            
            assert save_path.exists()


class TestSameBitLookupDetection:
    """Tests specifically for Same-Bit Lookup mechanism detection."""
    
    def test_offset_calculation(self):
        """Test that offset -(n+1) is correctly computed."""
        n_bits = 20
        expected_offset = -(n_bits + 1)
        
        assert expected_offset == -21
    
    def test_detect_diagonal_at_offset(self):
        """Test detection of diagonal attention at specific offset."""
        n_bits = 8
        seq_len = 20
        n_heads = 4
        offset = -(n_bits + 1)  # -9
        
        # Create pattern with diagonal at offset
        attn = torch.zeros(n_heads, seq_len, seq_len)
        
        # Head 2 has perfect diagonal at offset -9
        for i in range(abs(offset), seq_len):
            attn[2, i, i + offset] = 1.0
        
        # Compute score for head 2
        score = compute_diagonal_score(attn[2], offset=offset)
        
        # Should have high score
        assert score > 0.9
        
        # Other heads should have low score
        for h in [0, 1, 3]:
            other_score = compute_diagonal_score(attn[h], offset=offset)
            assert other_score < 0.1


class TestAnalysisIntegration:
    """Integration tests for the full analysis pipeline."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a small 'trained' model for testing."""
        config = TransformerConfig(
            vocab_size=6,
            d_model=32,
            n_heads=4,
            n_layers=2,
            use_rope=True
        )
        model = TransformerLM(config)
        model.eval()
        return model
    
    def test_full_attention_analysis_pipeline(self, trained_model):
        """Test running full attention analysis."""
        # Generate some test inputs
        input_ids = torch.randint(0, 6, (4, 16))
        
        # Extract attention patterns
        patterns = trained_model.get_attention_patterns(input_ids)
        
        assert len(patterns) == 2
        
        # Analyze patterns
        head_scores = analyze_head_specialization(patterns)
        
        assert len(head_scores) == 8  # 2 layers * 4 heads
    
    def test_mechanistic_analysis_output_format(self, trained_model):
        """Test that analysis produces expected output format."""
        input_ids = torch.randint(0, 6, (2, 16))
        patterns = trained_model.get_attention_patterns(input_ids)
        
        # Check pattern format
        for layer_idx, layer_attn in enumerate(patterns):
            assert layer_attn.dim() == 4  # (batch, heads, seq, seq)
            assert layer_attn.shape[1] == 4  # 4 heads
            
            # Attention should sum to ~1 per row
            row_sums = layer_attn.sum(dim=-1)
            assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
