"""
Pytest configuration file.

This module configures pytest for the Transformer Grokking project.
"""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def small_n_bits():
    """Return a small n_bits value for fast testing."""
    return 6


@pytest.fixture(scope="session")
def default_config():
    """Return default configuration dictionary."""
    return {
        "n_bits": 6,
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 128,
        "batch_size": 4,
        "max_steps": 10
    }


def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    import torch
    
    skip_gpu = pytest.mark.skip(reason="GPU not available")
    
    for item in items:
        if "gpu" in item.keywords and not torch.cuda.is_available():
            item.add_marker(skip_gpu)
