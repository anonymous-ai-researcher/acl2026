"""Training module for grokking experiments."""

from .trainer import (
    Trainer,
    TrainingConfig,
    create_optimizer,
    create_scheduler,
    compute_metrics,
    evaluate_stratified,
)

__all__ = [
    "Trainer",
    "TrainingConfig",
    "create_optimizer",
    "create_scheduler",
    "compute_metrics",
    "evaluate_stratified",
]
