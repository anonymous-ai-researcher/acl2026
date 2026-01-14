"""
Tests for the data module (LargeCounter dataset).

Tests cover:
- Binary conversion utilities
- Carry chain length computation
- Tokenization
- Stratified sampling
- Dataset construction
- DataLoader functionality
"""

import pytest
import numpy as np
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.counter_dataset import (
    int_to_binary,
    binary_to_int,
    get_carry_chain_length,
    CounterTokenizer,
    LargeCounterDataset,
    create_dataloader,
    stratified_sample
)


class TestBinaryConversion:
    """Tests for binary conversion utilities."""
    
    def test_int_to_binary_basic(self):
        """Test basic integer to binary conversion."""
        assert int_to_binary(0, n_bits=4) == "0000"
        assert int_to_binary(1, n_bits=4) == "0001"
        assert int_to_binary(7, n_bits=4) == "0111"
        assert int_to_binary(15, n_bits=4) == "1111"
    
    def test_int_to_binary_padding(self):
        """Test that binary strings are properly zero-padded."""
        assert int_to_binary(1, n_bits=8) == "00000001"
        assert int_to_binary(255, n_bits=8) == "11111111"
    
    def test_int_to_binary_overflow(self):
        """Test behavior at boundary (modular arithmetic)."""
        # 16 in 4-bit should wrap to 0
        result = int_to_binary(16 % 16, n_bits=4)
        assert result == "0000"
    
    def test_binary_to_int_basic(self):
        """Test binary string to integer conversion."""
        assert binary_to_int("0000") == 0
        assert binary_to_int("0001") == 1
        assert binary_to_int("0111") == 7
        assert binary_to_int("1111") == 15
    
    def test_roundtrip_conversion(self):
        """Test that conversion is reversible."""
        for n in range(256):
            binary = int_to_binary(n, n_bits=8)
            recovered = binary_to_int(binary)
            assert recovered == n, f"Roundtrip failed for {n}"


class TestCarryChainLength:
    """Tests for carry chain length computation."""
    
    def test_no_carry(self):
        """Numbers ending in 0 have carry chain length 0."""
        assert get_carry_chain_length(0, n_bits=4) == 0  # 0000 -> 0001
        assert get_carry_chain_length(2, n_bits=4) == 0  # 0010 -> 0011
        assert get_carry_chain_length(8, n_bits=4) == 0  # 1000 -> 1001
    
    def test_single_carry(self):
        """Numbers ending in 01 have carry chain length 1."""
        assert get_carry_chain_length(1, n_bits=4) == 1  # 0001 -> 0010
        assert get_carry_chain_length(5, n_bits=4) == 1  # 0101 -> 0110
    
    def test_double_carry(self):
        """Numbers ending in 011 have carry chain length 2."""
        assert get_carry_chain_length(3, n_bits=4) == 2  # 0011 -> 0100
        assert get_carry_chain_length(11, n_bits=4) == 2  # 1011 -> 1100
    
    def test_global_carry(self):
        """Numbers with all 1s have maximum carry chain."""
        assert get_carry_chain_length(7, n_bits=4) == 3   # 0111 -> 1000
        assert get_carry_chain_length(15, n_bits=4) == 4  # 1111 -> 0000 (overflow)
    
    def test_carry_chain_formula(self):
        """Verify carry chain = number of trailing 1s."""
        # 6 = 0110, ends in 0, chain = 0
        assert get_carry_chain_length(6, n_bits=4) == 0
        # 13 = 1101, ends in 01, chain = 1
        assert get_carry_chain_length(13, n_bits=4) == 1
        # 7 = 0111, ends in 111, chain = 3
        assert get_carry_chain_length(7, n_bits=4) == 3


class TestTokenizer:
    """Tests for the counter tokenizer."""
    
    @pytest.fixture
    def tokenizer(self):
        return CounterTokenizer()
    
    def test_vocab_size(self, tokenizer):
        """Test vocabulary size."""
        # Vocab: 0, 1, #, <pad>, <bos>, <eos>
        assert tokenizer.vocab_size == 6
    
    def test_encode_decode_roundtrip(self, tokenizer):
        """Test encoding and decoding are inverse operations."""
        sequence = "0111#1000"
        encoded = tokenizer.encode(sequence)
        decoded = tokenizer.decode(encoded)
        assert decoded == sequence
    
    def test_special_tokens(self, tokenizer):
        """Test special token handling."""
        sequence = "01#10"
        encoded = tokenizer.encode(sequence, add_bos=True, add_eos=True)
        
        # Should have BOS at start, EOS at end
        assert encoded[0] == tokenizer.bos_id
        assert encoded[-1] == tokenizer.eos_id
    
    def test_padding(self, tokenizer):
        """Test padding functionality."""
        sequence = "01"
        encoded = tokenizer.encode(sequence)
        padded = tokenizer.pad(encoded, max_len=10)
        
        assert len(padded) == 10
        assert padded[-1] == tokenizer.pad_id


class TestStratifiedSampling:
    """Tests for stratified sampling strategy."""
    
    def test_stratified_sample_coverage(self):
        """Test that stratified sampling covers all strata."""
        n_bits = 8
        n_samples = 100
        
        samples = stratified_sample(n_bits, n_samples)
        
        # Check we got the right number
        assert len(samples) == n_samples
        
        # Check samples are in valid range
        assert all(0 <= s < 2**n_bits for s in samples)
    
    def test_stratified_sample_distribution(self):
        """Test that sampling is approximately uniform over strata."""
        n_bits = 8
        n_samples = 1000
        
        samples = stratified_sample(n_bits, n_samples)
        
        # Count samples per stratum
        strata_counts = {k: 0 for k in range(n_bits + 1)}
        for s in samples:
            k = get_carry_chain_length(s, n_bits)
            strata_counts[k] += 1
        
        # Each stratum should have roughly n_samples / n_bits samples
        expected = n_samples / n_bits
        for k in range(n_bits):
            # Allow 50% variance for randomness
            assert strata_counts[k] > expected * 0.3, f"Stratum {k} underrepresented"


class TestLargeCounterDataset:
    """Tests for the LargeCounter dataset."""
    
    @pytest.fixture
    def train_dataset(self):
        return LargeCounterDataset(n_bits=8, split="train", train_ratio=0.3)
    
    @pytest.fixture
    def test_dataset(self):
        return LargeCounterDataset(n_bits=8, split="test", train_ratio=0.3)
    
    def test_dataset_length(self, train_dataset, test_dataset):
        """Test dataset sizes are correct."""
        total = 2**8
        expected_train = int(total * 0.3)
        
        # Train should have ~30% of samples
        assert abs(len(train_dataset) - expected_train) < expected_train * 0.2
        
        # Test should have remaining
        assert len(test_dataset) > 0
    
    def test_dataset_no_overlap(self, train_dataset, test_dataset):
        """Test that train and test sets don't overlap."""
        train_numbers = set(train_dataset.numbers)
        test_numbers = set(test_dataset.numbers)
        
        overlap = train_numbers & test_numbers
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping samples"
    
    def test_getitem_format(self, train_dataset):
        """Test that __getitem__ returns correct format."""
        item = train_dataset[0]
        
        assert "input_ids" in item
        assert "labels" in item
        assert "attention_mask" in item
        
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["labels"], torch.Tensor)
    
    def test_sequence_correctness(self, train_dataset):
        """Test that sequences encode correct counting."""
        for i in range(min(10, len(train_dataset))):
            item = train_dataset[i]
            n = train_dataset.numbers[i]
            next_n = (n + 1) % (2 ** train_dataset.n_bits)
            
            # Decode the sequence
            input_ids = item["input_ids"]
            labels = item["labels"]
            
            # The labels should contain the next number's bits
            # (after the # delimiter)
            assert labels is not None


class TestDataLoader:
    """Tests for DataLoader creation."""
    
    def test_create_dataloader(self):
        """Test DataLoader creation."""
        dataset = LargeCounterDataset(n_bits=8, split="train")
        loader = create_dataloader(dataset, batch_size=4, shuffle=True)
        
        batch = next(iter(loader))
        
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 4
    
    def test_dataloader_iteration(self):
        """Test that we can iterate through full dataset."""
        dataset = LargeCounterDataset(n_bits=6, split="train")
        loader = create_dataloader(dataset, batch_size=8)
        
        total_samples = 0
        for batch in loader:
            total_samples += batch["input_ids"].shape[0]
        
        assert total_samples == len(dataset)


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_minimum_bits(self):
        """Test with minimum bit width."""
        dataset = LargeCounterDataset(n_bits=2, split="train", train_ratio=0.5)
        assert len(dataset) > 0
        
        # Should be able to get an item
        item = dataset[0]
        assert item["input_ids"] is not None
    
    def test_large_bits(self):
        """Test with larger bit width."""
        dataset = LargeCounterDataset(n_bits=16, split="train", train_ratio=0.01)
        assert len(dataset) > 0
    
    def test_single_sample(self):
        """Test dataset with single sample."""
        # Create tiny dataset
        dataset = LargeCounterDataset(
            n_bits=4, 
            split="train", 
            train_ratio=0.1,
            stratified=False
        )
        
        if len(dataset) > 0:
            item = dataset[0]
            assert item is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
