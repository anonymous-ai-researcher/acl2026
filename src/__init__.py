"""
Transformer Grokking Counting - A mechanistic study of succinct algorithmic learning.

This package provides tools for:
- Training Transformers and RNNs on the LARGECOUNTER task
- Analyzing grokking dynamics and weight norm evolution
- Mechanistic interpretability via attention analysis and activation patching
- Extending analysis to pre-trained LLMs
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

from . import data
from . import models
from . import training
from . import analysis
