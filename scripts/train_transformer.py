#!/usr/bin/env python3
"""
Train Transformer on LARGECOUNTER task.

This script trains a GPT-style Transformer with RoPE positional embeddings
on the binary counting task, demonstrating grokking and succinctness.

Usage:
    python scripts/train_transformer.py --config configs/default.yaml
    python scripts/train_transformer.py --n_bits 20 --weight_decay 1.0
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.counter_dataset import (
    LargeCounterDataset,
    Tokenizer,
    create_dataloader,
    stratified_sample,
)
from src.models.transformer import TransformerConfig, TransformerLM
from src.training.trainer import Trainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train Transformer on LARGECOUNTER task"
    )
    
    # Config file
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file"
    )
    
    # Data arguments
    parser.add_argument(
        "--n_bits", type=int, default=20,
        help="Number of bits for binary counting (default: 20)"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.3,
        help="Fraction of state space for training (default: 0.3)"
    )
    parser.add_argument(
        "--stratified", action="store_true", default=True,
        help="Use stratified sampling (default: True)"
    )
    
    # Model arguments
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of transformer layers (default: 2)"
    )
    parser.add_argument(
        "--n_heads", type=int, default=4,
        help="Number of attention heads (default: 4)"
    )
    parser.add_argument(
        "--d_model", type=int, default=64,
        help="Model dimension (default: 64)"
    )
    parser.add_argument(
        "--d_ff", type=int, default=256,
        help="Feedforward dimension (default: 256)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate (default: 0.0)"
    )
    
    # Training arguments
    parser.add_argument(
        "--max_steps", type=int, default=50000,
        help="Maximum training steps (default: 50000)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512,
        help="Batch size (default: 512)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1.0,
        help="Weight decay for AdamW (default: 1.0, critical for grokking)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500,
        help="Warmup steps (default: 500)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Gradient clipping (default: 1.0)"
    )
    
    # Logging arguments
    parser.add_argument(
        "--output_dir", type=str, default="outputs",
        help="Output directory (default: outputs)"
    )
    parser.add_argument(
        "--exp_name", type=str, default=None,
        help="Experiment name (default: auto-generated)"
    )
    parser.add_argument(
        "--log_interval", type=int, default=100,
        help="Logging interval (default: 100)"
    )
    parser.add_argument(
        "--eval_interval", type=int, default=500,
        help="Evaluation interval (default: 500)"
    )
    parser.add_argument(
        "--save_interval", type=int, default=5000,
        help="Checkpoint save interval (default: 5000)"
    )
    parser.add_argument(
        "--use_wandb", action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="transformer-grokking",
        help="W&B project name"
    )
    
    # Hardware arguments
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=4,
        help="DataLoader workers (default: 4)"
    )
    parser.add_argument(
        "--mixed_precision", type=str, default="no",
        choices=["no", "fp16", "bf16"],
        help="Mixed precision training"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Merge config file with command line arguments (CLI takes precedence)."""
    # Flatten nested config
    flat_config = {}
    for section, values in config.items():
        if isinstance(values, dict):
            flat_config.update(values)
        else:
            flat_config[section] = values
    
    # Update args with config values (only if not set via CLI)
    for key, value in flat_config.items():
        if hasattr(args, key):
            # Only update if using default value
            if getattr(args, key) == parse_args.__wrapped__()[key] if hasattr(parse_args, '__wrapped__') else True:
                setattr(args, key, value)
    
    return args


def setup_device(device_str: str) -> torch.device:
    """Setup compute device."""
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using Apple MPS")
        else:
            device = torch.device("cpu")
            print("Using CPU")
    else:
        device = torch.device(device_str)
        print(f"Using device: {device}")
    
    return device


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_datasets(args) -> tuple:
    """Create train and test datasets with stratified sampling."""
    tokenizer = Tokenizer()
    
    # Calculate number of training samples
    total_states = 2 ** args.n_bits
    n_train = int(total_states * args.train_ratio)
    
    print(f"\nDataset Configuration:")
    print(f"  Bit width: {args.n_bits}")
    print(f"  Total state space: {total_states:,}")
    print(f"  Training samples: {n_train:,} ({args.train_ratio*100:.0f}%)")
    
    if args.stratified:
        print("  Sampling: Stratified by carry chain length")
        train_numbers = stratified_sample(args.n_bits, n_train)
    else:
        print("  Sampling: Uniform random")
        import random
        train_numbers = random.sample(range(total_states), n_train)
    
    # Create test set (remaining numbers)
    train_set = set(train_numbers)
    test_numbers = [i for i in range(total_states) if i not in train_set]
    
    print(f"  Test samples: {len(test_numbers):,}")
    
    train_dataset = LargeCounterDataset(train_numbers, args.n_bits, tokenizer)
    test_dataset = LargeCounterDataset(test_numbers, args.n_bits, tokenizer)
    
    return train_dataset, test_dataset, tokenizer


def create_model(args, tokenizer: Tokenizer) -> TransformerLM:
    """Create Transformer model."""
    config = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=2 * args.n_bits + 3,  # N_i # N_{i+1} with special tokens
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_rope=True,
        tie_weights=True,
    )
    
    model = TransformerLM(config)
    
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: GPT-style Transformer with RoPE")
    print(f"  Layers: {args.n_layers}")
    print(f"  Heads: {args.n_heads}")
    print(f"  Model dim: {args.d_model}")
    print(f"  FF dim: {args.d_ff}")
    print(f"  Total parameters: {n_params:,}")
    print(f"  Trainable parameters: {n_trainable:,}")
    
    return model


def main():
    """Main training function."""
    args = parse_args()
    
    # Load config file if provided
    if args.config:
        config = load_config(args.config)
        # Simple merge: config values as defaults
        for section in config.values():
            if isinstance(section, dict):
                for key, value in section.items():
                    if hasattr(args, key) and getattr(args, key) is None:
                        setattr(args, key, value)
    
    # Setup
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"transformer_n{args.n_bits}_d{args.d_model}_wd{args.weight_decay}_{timestamp}"
    
    # Create output directory
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LARGECOUNTER Transformer Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    
    # Save config
    config_save_path = output_dir / "config.json"
    with open(config_save_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Config saved to: {config_save_path}")
    
    # Create datasets
    train_dataset, test_dataset, tokenizer = create_datasets(args)
    
    # Create dataloaders
    train_loader = create_dataloader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    
    # Create model
    model = create_model(args, tokenizer)
    model = model.to(device)
    
    # Training config
    training_config = TrainingConfig(
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        mixed_precision=args.mixed_precision,
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Weight decay: {args.weight_decay}")
    print(f"  Warmup steps: {args.warmup_steps}")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        eval_loader=test_loader,
        config=training_config,
        device=device,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_name=args.exp_name,
    )
    
    # Train
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)
    
    try:
        history = trainer.train()
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        
        # Report final results
        if history["test_accuracy"]:
            final_test_acc = history["test_accuracy"][-1]
            print(f"Final test accuracy: {final_test_acc:.2%}")
            
            # Check for grokking
            early_accs = history["test_accuracy"][:len(history["test_accuracy"])//3]
            late_accs = history["test_accuracy"][-len(history["test_accuracy"])//3:]
            
            if early_accs and late_accs:
                early_mean = sum(early_accs) / len(early_accs)
                late_mean = sum(late_accs) / len(late_accs)
                
                if early_mean < 0.2 and late_mean > 0.9:
                    print("✓ Grokking detected! Model transitioned from memorization to generalization.")
                elif late_mean > 0.95:
                    print("✓ Model achieved near-perfect generalization.")
                else:
                    print("Model did not fully generalize. Consider adjusting hyperparameters.")
        
        # Save final model
        final_path = output_dir / "model_final.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "history": history,
        }, final_path)
        print(f"Final model saved to: {final_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        # Save checkpoint
        interrupt_path = output_dir / "model_interrupted.pt"
        torch.save(model.state_dict(), interrupt_path)
        print(f"Interrupted model saved to: {interrupt_path}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
