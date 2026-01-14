"""
Training Module for Grokking Experiments.

This module implements the training loop with special attention to:
- High weight decay (λ=1.0) to induce grokking
- Weight norm tracking for complexity collapse detection
- Stratified evaluation to detect carry chain failures
- Extended training beyond convergence to allow phase transitions
"""

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    
    # Optimizer
    learning_rate: float = 1e-3
    weight_decay: float = 1.0     # High for grokking
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-8
    
    # Scheduler
    scheduler_type: str = "cosine"
    warmup_steps: int = 1000
    
    # Training
    max_steps: int = 50000        # Extended for grokking
    batch_size: int = 512
    gradient_clip: float = 1.0
    
    # Evaluation
    eval_interval: int = 500
    log_interval: int = 100
    save_interval: int = 5000
    
    # Checkpointing
    output_dir: str = "outputs"
    save_total_limit: int = 5
    
    # Hardware
    device: str = "cuda"
    fp16: bool = False
    bf16: bool = False
    
    # Reproducibility
    seed: int = 42
    
    # Logging
    use_wandb: bool = False
    wandb_project: str = "transformer-grokking"
    wandb_run_name: Optional[str] = None


def create_optimizer(
    model: nn.Module,
    config: TrainingConfig,
) -> Optimizer:
    """
    Create AdamW optimizer with weight decay.
    
    High weight decay is critical for inducing grokking:
    - Forces the model to find low-norm solutions
    - Penalizes high-complexity memorization circuits
    - Enables "slingshot" into algorithmic solutions
    """
    # Separate parameters that should/shouldn't have weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias", "ln"]
    
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": config.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = AdamW(
        optimizer_grouped_parameters,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )
    
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: TrainingConfig,
) -> _LRScheduler:
    """Create learning rate scheduler with warmup."""
    
    def lr_lambda(current_step: int) -> float:
        # Warmup
        if current_step < config.warmup_steps:
            return float(current_step) / float(max(1, config.warmup_steps))
        
        # Cosine decay
        if config.scheduler_type == "cosine":
            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.max_steps - config.warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        # Linear decay
        elif config.scheduler_type == "linear":
            progress = float(current_step - config.warmup_steps) / float(
                max(1, config.max_steps - config.warmup_steps)
            )
            return max(0.0, 1.0 - progress)
        
        # Constant
        else:
            return 1.0
    
    return LambdaLR(optimizer, lr_lambda)


def compute_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bits: int = 20,
) -> Dict[str, float]:
    """
    Compute sequence-level accuracy and other metrics.
    
    A prediction is correct only if ALL n bits match the ground truth.
    This is the strict evaluation used in the paper.
    """
    # Get predictions
    predictions = logits.argmax(dim=-1)
    
    # Mask where labels are valid (not -100)
    valid_mask = labels != -100
    
    # Token-level accuracy
    correct_tokens = (predictions == labels) | ~valid_mask
    token_accuracy = correct_tokens[valid_mask].float().mean().item()
    
    # Sequence-level accuracy (all tokens in output must be correct)
    # For each sequence, check if all valid positions are correct
    batch_size = labels.shape[0]
    seq_correct = []
    
    for i in range(batch_size):
        valid_positions = valid_mask[i]
        if valid_positions.sum() > 0:
            seq_correct.append(
                (predictions[i][valid_positions] == labels[i][valid_positions]).all().item()
            )
    
    sequence_accuracy = sum(seq_correct) / len(seq_correct) if seq_correct else 0.0
    
    return {
        "token_accuracy": token_accuracy,
        "sequence_accuracy": sequence_accuracy,
    }


def evaluate_stratified(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    n_bits: int = 20,
) -> Dict[str, Any]:
    """
    Stratified evaluation by carry chain length.
    
    This reveals the failure mode of RNNs:
    - High accuracy on k=0 (trivial, no carry)
    - Rapid decay for k≥1 (carry required)
    """
    model.eval()
    
    # Collect results by stratum
    strata_results = {k: {"correct": 0, "total": 0} for k in range(n_bits)}
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            carry_lengths = batch["carry_lengths"]
            
            outputs = model(input_ids)
            logits = outputs["logits"]
            predictions = logits.argmax(dim=-1)
            
            # Check sequence correctness
            valid_mask = labels != -100
            
            for i in range(input_ids.shape[0]):
                valid_pos = valid_mask[i]
                if valid_pos.sum() > 0:
                    correct = (predictions[i][valid_pos] == labels[i][valid_pos]).all().item()
                    k = carry_lengths[i].item()
                    
                    strata_results[k]["total"] += 1
                    strata_results[k]["correct"] += int(correct)
                    total_correct += int(correct)
                    total_samples += 1
    
    # Compute per-stratum accuracy
    strata_accuracy = {}
    for k, result in strata_results.items():
        if result["total"] > 0:
            strata_accuracy[k] = result["correct"] / result["total"]
        else:
            strata_accuracy[k] = None
    
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0.0
    
    model.train()
    
    return {
        "overall_accuracy": overall_accuracy,
        "strata_accuracy": strata_accuracy,
        "strata_counts": {k: v["total"] for k, v in strata_results.items()},
    }


class Trainer:
    """
    Trainer for grokking experiments.
    
    Key features:
    - Extended training beyond convergence
    - Weight norm tracking for complexity collapse
    - Stratified evaluation
    - Checkpoint management
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        config: TrainingConfig,
        n_bits: int = 20,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        self.n_bits = n_bits
        
        # Device setup
        self.device = torch.device(config.device)
        self.model.to(self.device)
        
        # Optimizer and scheduler
        self.optimizer = create_optimizer(model, config)
        self.scheduler = create_scheduler(self.optimizer, config)
        
        # Mixed precision
        self.scaler = GradScaler() if config.fp16 else None
        self.autocast_dtype = torch.float16 if config.fp16 else (
            torch.bfloat16 if config.bf16 else torch.float32
        )
        
        # Tracking
        self.global_step = 0
        self.best_accuracy = 0.0
        self.history = {
            "train_loss": [],
            "train_accuracy": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "weight_norm": [],
            "learning_rate": [],
        }
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Wandb
        self.wandb_run = None
        if config.use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=config.wandb_project,
                    name=config.wandb_run_name,
                    config=vars(config),
                )
            except ImportError:
                logger.warning("wandb not installed, skipping logging")
    
    def train(self) -> Dict[str, Any]:
        """
        Main training loop.
        
        Trains until max_steps, even after training loss converges,
        to allow the grokking phase transition to occur.
        """
        logger.info(f"Starting training for {self.config.max_steps} steps")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Weight decay: {self.config.weight_decay}")
        
        self.model.train()
        train_iter = iter(self.train_dataloader)
        
        pbar = tqdm(total=self.config.max_steps, desc="Training")
        
        while self.global_step < self.config.max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_dataloader)
                batch = next(train_iter)
            
            # Training step
            metrics = self._training_step(batch)
            
            # Logging
            if self.global_step % self.config.log_interval == 0:
                self._log_metrics(metrics, prefix="train")
            
            # Evaluation
            if self.global_step % self.config.eval_interval == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(eval_metrics, prefix="eval")
                
                # Save if best
                if eval_metrics["sequence_accuracy"] > self.best_accuracy:
                    self.best_accuracy = eval_metrics["sequence_accuracy"]
                    self._save_checkpoint("best_model.pt")
            
            # Periodic save
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f"checkpoint_{self.global_step}.pt")
            
            pbar.update(1)
            pbar.set_postfix({
                "loss": f"{metrics['loss']:.4f}",
                "acc": f"{metrics.get('sequence_accuracy', 0):.2%}",
                "norm": f"{metrics['weight_norm']:.2f}",
            })
            
            self.global_step += 1
        
        pbar.close()
        
        # Final evaluation
        final_metrics = self._evaluate()
        final_stratified = evaluate_stratified(
            self.model, self.eval_dataloader, self.device, self.n_bits
        )
        
        # Save final
        self._save_checkpoint("final_model.pt")
        self._save_history()
        
        logger.info(f"Training complete. Best accuracy: {self.best_accuracy:.2%}")
        logger.info(f"Final accuracy: {final_metrics['sequence_accuracy']:.2%}")
        
        return {
            "best_accuracy": self.best_accuracy,
            "final_metrics": final_metrics,
            "stratified_results": final_stratified,
            "history": self.history,
        }
    
    def _training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        self.optimizer.zero_grad()
        
        # Forward with optional mixed precision
        with autocast(device_type=self.device.type, dtype=self.autocast_dtype):
            outputs = self.model(input_ids, labels=labels)
            loss = outputs["loss"]
        
        # Backward
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip
            )
            self.optimizer.step()
        
        self.scheduler.step()
        
        # Compute metrics
        with torch.no_grad():
            metrics = compute_metrics(outputs["logits"], labels, self.n_bits)
        
        # Track weight norm (critical for grokking analysis)
        weight_norm = self.model.get_weight_norm()
        
        return {
            "loss": loss.item(),
            "weight_norm": weight_norm,
            "learning_rate": self.scheduler.get_last_lr()[0],
            **metrics,
        }
    
    @torch.no_grad()
    def _evaluate(self) -> Dict[str, float]:
        """Full evaluation on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch in self.eval_dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            outputs = self.model(input_ids, labels=labels)
            total_loss += outputs["loss"].item() * input_ids.shape[0]
            
            # Sequence accuracy
            predictions = outputs["logits"].argmax(dim=-1)
            valid_mask = labels != -100
            
            for i in range(input_ids.shape[0]):
                valid_pos = valid_mask[i]
                if valid_pos.sum() > 0:
                    correct = (predictions[i][valid_pos] == labels[i][valid_pos]).all()
                    total_correct += int(correct.item())
                    total_samples += 1
        
        self.model.train()
        
        return {
            "loss": total_loss / total_samples if total_samples > 0 else 0.0,
            "sequence_accuracy": total_correct / total_samples if total_samples > 0 else 0.0,
        }
    
    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log metrics to console and wandb."""
        # Update history
        for key, value in metrics.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if full_key not in self.history:
                self.history[full_key] = []
            self.history[full_key].append((self.global_step, value))
        
        # Log to wandb
        if self.wandb_run is not None:
            import wandb
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
            log_dict["step"] = self.global_step
            wandb.log(log_dict)
        
        # Log to console (summary)
        if prefix == "eval":
            logger.info(
                f"Step {self.global_step}: "
                f"eval_loss={metrics.get('loss', 0):.4f}, "
                f"eval_acc={metrics.get('sequence_accuracy', 0):.2%}"
            )
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "best_accuracy": self.best_accuracy,
            "config": vars(self.config),
        }
        
        path = self.output_dir / filename
        torch.save(checkpoint, path)
        logger.debug(f"Saved checkpoint to {path}")
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent."""
        checkpoints = sorted(
            self.output_dir.glob("checkpoint_*.pt"),
            key=lambda x: int(x.stem.split("_")[1]),
        )
        
        if len(checkpoints) > self.config.save_total_limit:
            for ckpt in checkpoints[:-self.config.save_total_limit]:
                ckpt.unlink()
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.best_accuracy = checkpoint.get("best_accuracy", 0.0)
        logger.info(f"Loaded checkpoint from {checkpoint_path} (step {self.global_step})")


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    from src.data import create_dataloaders
    from src.models import TransformerLM, TransformerConfig
    
    print("Testing Trainer...")
    
    # Create small model
    config = TransformerConfig(
        vocab_size=6,
        d_model=32,
        n_layers=1,
        n_heads=2,
    )
    model = TransformerLM(config)
    
    # Create small dataloaders
    train_loader, test_loader = create_dataloaders(
        n_bits=8,
        batch_size=32,
        num_workers=0,
    )
    
    # Create trainer
    train_config = TrainingConfig(
        max_steps=100,
        eval_interval=50,
        log_interval=20,
        output_dir="outputs/test",
        use_wandb=False,
    )
    
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        eval_dataloader=test_loader,
        config=train_config,
        n_bits=8,
    )
    
    # Short training run
    results = trainer.train()
    
    print(f"\nTraining complete:")
    print(f"  Best accuracy: {results['best_accuracy']:.2%}")
    print(f"  Final accuracy: {results['final_metrics']['sequence_accuracy']:.2%}")
    
    print("\n✓ All tests passed!")
