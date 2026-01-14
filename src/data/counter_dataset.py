"""
LARGECOUNTER Dataset Implementation.

This module implements the n-bit binary counter dataset for studying the
succinctness hypothesis in Transformers vs RNNs.

The task: Given a binary number N_i, predict N_{i+1} = (N_i + 1) mod 2^n
Example: 0111 (7) → 1000 (8)
"""

import random
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Utility Functions
# =============================================================================

def int_to_binary(num: int, n_bits: int) -> str:
    """
    Convert integer to big-endian binary string with zero-padding.
    
    Args:
        num: Integer to convert
        n_bits: Number of bits in output
        
    Returns:
        Binary string representation (MSB first)
        
    Example:
        >>> int_to_binary(7, 4)
        '0111'
    """
    return format(num, f'0{n_bits}b')


def binary_to_int(binary_str: str) -> int:
    """
    Convert binary string to integer.
    
    Args:
        binary_str: Binary string (MSB first)
        
    Returns:
        Integer value
        
    Example:
        >>> binary_to_int('0111')
        7
    """
    return int(binary_str, 2)


def get_carry_chain_length(num: int, n_bits: int) -> int:
    """
    Compute the carry chain length for incrementing a number.
    
    The carry chain length is the number of consecutive 1s at the LSB end,
    which determines how many bits will flip during increment.
    
    Args:
        num: The number to analyze
        n_bits: Total number of bits
        
    Returns:
        Length of carry chain (0 to n_bits)
        
    Example:
        >>> get_carry_chain_length(7, 4)  # 0111 → needs full propagation
        3
        >>> get_carry_chain_length(6, 4)  # 0110 → only LSB flips
        0
    """
    if num == 0:
        return 0
    
    # Count trailing ones
    count = 0
    while num & 1:
        count += 1
        num >>= 1
        if count >= n_bits:
            break
    return count


def stratified_sample(
    n_bits: int,
    n_samples: int,
    seed: Optional[int] = None
) -> List[int]:
    """
    Sample numbers with uniform distribution over carry chain lengths.
    
    This ensures the model sees equal examples of all carry propagation depths,
    preventing collapse into low-bit heuristics.
    
    Args:
        n_bits: Bit-width of the counter
        n_samples: Total number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled integers
        
    Note:
        The stratification ensures:
        - Stratum k contains numbers ending in ...0(1)_k
        - Each stratum is sampled with probability 1/n
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    samples = []
    samples_per_stratum = n_samples // n_bits
    remainder = n_samples % n_bits
    
    for k in range(n_bits):
        # Stratum k: numbers with exactly k trailing ones
        # These end in pattern: ...X01...1 (k ones at the end for k > 0)
        # Or ...X0 for k = 0
        
        stratum_samples = samples_per_stratum + (1 if k < remainder else 0)
        
        if k == 0:
            # Numbers ending in 0: all even numbers
            candidates = [i for i in range(0, 2**n_bits, 2)]
        elif k == n_bits:
            # Special case: all 1s (only 2^n - 1)
            candidates = [2**n_bits - 1]
        else:
            # Numbers ending in exactly k ones: pattern ...01...1 (k ones)
            # The (k+1)th bit from LSB must be 0
            candidates = []
            suffix = (1 << k) - 1  # k ones
            for prefix in range(2**(n_bits - k - 1)):
                # Ensure the (k+1)th bit is 0
                num = (prefix << (k + 1)) | suffix
                if num < 2**n_bits:
                    candidates.append(num)
        
        if candidates:
            sampled = random.choices(candidates, k=min(stratum_samples, len(candidates) * 10))
            sampled = sampled[:stratum_samples]
            samples.extend(sampled)
    
    random.shuffle(samples)
    return samples[:n_samples]


# =============================================================================
# Tokenization
# =============================================================================

class CounterTokenizer:
    """
    Tokenizer for binary counter sequences.
    
    Vocabulary: {0, 1, #, <pad>, <bos>, <eos>}
    """
    
    def __init__(self, special_tokens: bool = True):
        """
        Initialize tokenizer.
        
        Args:
            special_tokens: Whether to include BOS/EOS tokens
        """
        self.special_tokens = special_tokens
        
        # Build vocabulary
        self.token_to_id = {
            '<pad>': 0,
            '0': 1,
            '1': 2,
            '#': 3,
        }
        
        if special_tokens:
            self.token_to_id['<bos>'] = 4
            self.token_to_id['<eos>'] = 5
        
        self.id_to_token = {v: k for k, v in self.token_to_id.items()}
        self.vocab_size = len(self.token_to_id)
        self.pad_id = self.token_to_id['<pad>']
    
    def encode(self, sequence: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode a string sequence to token IDs.
        
        Args:
            sequence: Input string (e.g., "0111#1000")
            add_special_tokens: Whether to add BOS/EOS
            
        Returns:
            List of token IDs
        """
        tokens = []
        if add_special_tokens and self.special_tokens:
            tokens.append(self.token_to_id['<bos>'])
        
        for char in sequence:
            if char in self.token_to_id:
                tokens.append(self.token_to_id[char])
            else:
                raise ValueError(f"Unknown token: {char}")
        
        if add_special_tokens and self.special_tokens:
            tokens.append(self.token_to_id['<eos>'])
        
        return tokens
    
    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        """
        Decode token IDs back to string.
        
        Args:
            token_ids: List of token IDs
            skip_special: Whether to skip special tokens
            
        Returns:
            Decoded string
        """
        special_ids = {0, 4, 5} if skip_special else {0}
        chars = []
        for tid in token_ids:
            if tid not in special_ids:
                chars.append(self.id_to_token.get(tid, '?'))
        return ''.join(chars)


# =============================================================================
# Dataset
# =============================================================================

class LargeCounterDataset(Dataset):
    """
    PyTorch Dataset for the LARGECOUNTER task.
    
    Each sample consists of:
    - Input: N_i in binary, followed by delimiter '#'
    - Target: N_{i+1} = (N_i + 1) mod 2^n in binary
    
    The dataset supports stratified sampling to ensure uniform
    exposure to all carry chain lengths.
    """
    
    def __init__(
        self,
        n_bits: int = 20,
        numbers: Optional[List[int]] = None,
        stratified: bool = True,
        n_samples: Optional[int] = None,
        seed: Optional[int] = None,
        train_ratio: float = 0.3,
        split: str = "train",
    ):
        """
        Initialize the dataset.
        
        Args:
            n_bits: Bit-width of the counter
            numbers: Explicit list of numbers (overrides sampling)
            stratified: Use stratified sampling
            n_samples: Number of samples (if not using explicit numbers)
            seed: Random seed
            train_ratio: Fraction for training split
            split: "train" or "test"
        """
        self.n_bits = n_bits
        self.max_num = 2 ** n_bits
        self.tokenizer = CounterTokenizer(special_tokens=False)
        
        if numbers is not None:
            self.numbers = numbers
        else:
            # Generate train/test split
            all_numbers = list(range(self.max_num))
            
            if seed is not None:
                random.seed(seed)
            random.shuffle(all_numbers)
            
            split_idx = int(len(all_numbers) * train_ratio)
            
            if split == "train":
                candidate_numbers = all_numbers[:split_idx]
            else:
                candidate_numbers = all_numbers[split_idx:]
            
            if stratified and split == "train":
                # Apply stratified sampling to training set
                n_target = n_samples if n_samples else len(candidate_numbers)
                self.numbers = stratified_sample(n_bits, n_target, seed)
            else:
                self.numbers = candidate_numbers
                if n_samples and n_samples < len(self.numbers):
                    self.numbers = random.sample(self.numbers, n_samples)
        
        # Precompute sequences
        self._precompute_sequences()
    
    def _precompute_sequences(self):
        """Precompute all input/target pairs."""
        self.samples = []
        
        for num in self.numbers:
            next_num = (num + 1) % self.max_num
            
            # Create input sequence: N_i followed by #
            input_binary = int_to_binary(num, self.n_bits)
            target_binary = int_to_binary(next_num, self.n_bits)
            
            # Full sequence for autoregressive training
            full_sequence = input_binary + '#' + target_binary
            
            # Store
            self.samples.append({
                'number': num,
                'next_number': next_num,
                'input_binary': input_binary,
                'target_binary': target_binary,
                'full_sequence': full_sequence,
                'carry_length': get_carry_chain_length(num, self.n_bits),
            })
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            Dictionary with:
            - input_ids: Full sequence tokens for autoregressive modeling
            - labels: Shifted targets (-100 for non-predicted positions)
            - attention_mask: Attention mask
            - carry_length: Carry chain length (for stratified evaluation)
        """
        sample = self.samples[idx]
        
        # Encode full sequence
        full_tokens = self.tokenizer.encode(sample['full_sequence'])
        
        # Create input_ids and labels for causal LM
        # Labels are shifted: we predict the next token
        input_ids = torch.tensor(full_tokens[:-1], dtype=torch.long)
        labels = torch.tensor(full_tokens[1:], dtype=torch.long)
        
        # Mask labels before the delimiter - we only want to predict the output
        # Input format: N_i # N_{i+1}
        # We want to predict only N_{i+1}
        delimiter_pos = self.n_bits  # Position of '#' in input_ids
        labels[:delimiter_pos] = -100  # Ignore loss before and at delimiter
        
        attention_mask = torch.ones_like(input_ids)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'carry_length': sample['carry_length'],
            'number': sample['number'],
        }
    
    def get_stratum_indices(self) -> Dict[int, List[int]]:
        """
        Get indices grouped by carry chain length.
        
        Returns:
            Dictionary mapping carry length to list of sample indices
        """
        strata = {}
        for idx, sample in enumerate(self.samples):
            k = sample['carry_length']
            if k not in strata:
                strata[k] = []
            strata[k].append(idx)
        return strata
    
    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for DataLoader.
        
        Handles padding for variable-length sequences (though in this task
        all sequences have the same length).
        """
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        carry_lengths = torch.tensor([item['carry_length'] for item in batch])
        numbers = torch.tensor([item['number'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'carry_lengths': carry_lengths,
            'numbers': numbers,
        }


def create_dataloaders(
    n_bits: int = 20,
    train_ratio: float = 0.3,
    batch_size: int = 512,
    stratified: bool = True,
    seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create training and test dataloaders.
    
    Args:
        n_bits: Bit-width of counter
        train_ratio: Fraction for training
        batch_size: Batch size
        stratified: Use stratified sampling
        seed: Random seed
        num_workers: DataLoader workers
        pin_memory: Pin memory for faster GPU transfer
        
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    train_dataset = LargeCounterDataset(
        n_bits=n_bits,
        stratified=stratified,
        seed=seed,
        train_ratio=train_ratio,
        split="train",
    )
    
    test_dataset = LargeCounterDataset(
        n_bits=n_bits,
        stratified=False,  # Test set uses all remaining numbers
        seed=seed,
        train_ratio=train_ratio,
        split="test",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=LargeCounterDataset.collate_fn,
        drop_last=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=LargeCounterDataset.collate_fn,
    )
    
    return train_loader, test_loader


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    # Test the dataset
    print("Testing LargeCounterDataset...")
    
    # Create dataset
    dataset = LargeCounterDataset(n_bits=4, seed=42, train_ratio=0.5, split="train")
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Input IDs: {sample['input_ids']}")
    print(f"  Labels: {sample['labels']}")
    print(f"  Carry length: {sample['carry_length']}")
    
    # Test tokenizer
    tokenizer = CounterTokenizer()
    decoded = tokenizer.decode(sample['input_ids'].tolist())
    print(f"  Decoded input: {decoded}")
    
    # Test stratification
    strata = dataset.get_stratum_indices()
    print(f"\nStrata distribution:")
    for k, indices in sorted(strata.items()):
        print(f"  Carry length {k}: {len(indices)} samples")
    
    # Test dataloader
    train_loader, test_loader = create_dataloaders(
        n_bits=4, batch_size=4, seed=42, num_workers=0
    )
    
    batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✓ All tests passed!")
