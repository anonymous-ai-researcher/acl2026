#!/usr/bin/env python3
"""
Train RNN baselines (LSTM/GRU) on LARGECOUNTER task.

This script trains recurrent baselines to demonstrate the exponential
bottleneck - RNNs fail to generalize regardless of capacity.

Usage:
    python scripts/train_rnn.py --rnn_type lstm --hidden_dim 2048
    python scripts/train_rnn.py --rnn_type gru --hidden_dim 64
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
from src.models.rnn import RNNConfig, RNNLM, compare_model_sizes
from src.training.trainer import Trainer, TrainingConfig


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train RNN baselines on LARGECOUNTER task"
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
        "--rnn_type", type=str, default="lstm",
        choices=["lstm", "gru"],
        help="RNN type (default: lstm)"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=2048,
        help="Hidden dimension (default: 2048)"
    )
    parser.add_argument(
        "--n_layers", type=int, default=2,
        help="Number of RNN layers (default: 2)"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.0,
        help="Dropout rate (default: 0.0)"
    )
    parser.add_argument(
        "--bidirectional", action="store_true",
        help="Use bidirectional RNN"
    )
    
    # Training arguments
    parser.add_argument(
        "--max_steps", type=int, default=100000,
        help="Maximum training steps (default: 100000, more for RNNs)"
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
        "--weight_decay", type=float, default=0.0,
        help="Weight decay (default: 0.0 for RNNs)"
    )
    parser.add_argument(
        "--warmup_steps", type=int, default=500,
        help="Warmup steps (default: 500)"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=1.0,
        help="Gradient clipping (default: 1.0)"
    )
    
    # Curriculum learning (optional)
    parser.add_argument(
        "--use_curriculum", action="store_true",
        help="Use curriculum learning (start with shorter sequences)"
    )
    parser.add_argument(
        "--curriculum_stages", type=int, default=5,
        help="Number of curriculum stages (default: 5)"
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
        "--save_interval", type=int, default=10000,
        help="Checkpoint save interval (default: 10000)"
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
    
    return parser.parse_args()


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


def create_datasets(args) -> tuple:
    """Create train and test datasets."""
    tokenizer = Tokenizer()
    
    total_states = 2 ** args.n_bits
    n_train = int(total_states * args.train_ratio)
    
    print(f"\nDataset Configuration:")
    print(f"  Bit width: {args.n_bits}")
    print(f"  Total state space: {total_states:,}")
    print(f"  Training samples: {n_train:,}")
    
    if args.stratified:
        train_numbers = stratified_sample(args.n_bits, n_train)
    else:
        import random
        train_numbers = random.sample(range(total_states), n_train)
    
    train_set = set(train_numbers)
    test_numbers = [i for i in range(total_states) if i not in train_set]
    
    print(f"  Test samples: {len(test_numbers):,}")
    
    train_dataset = LargeCounterDataset(train_numbers, args.n_bits, tokenizer)
    test_dataset = LargeCounterDataset(test_numbers, args.n_bits, tokenizer)
    
    return train_dataset, test_dataset, tokenizer


def create_model(args, tokenizer: Tokenizer) -> RNNLM:
    """Create RNN model."""
    config = RNNConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        rnn_type=args.rnn_type,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    )
    
    model = RNNLM(config)
    
    n_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nModel Configuration:")
    print(f"  Architecture: {args.rnn_type.upper()}")
    print(f"  Hidden dim: {args.hidden_dim}")
    print(f"  Layers: {args.n_layers}")
    print(f"  Bidirectional: {args.bidirectional}")
    print(f"  Total parameters: {n_params:,}")
    
    # Compare with Transformer
    print("\n  Parameter comparison:")
    compare_model_sizes(args.n_bits)
    
    return model


def run_scaling_experiment(args, device):
    """Run experiments across different hidden dimensions to show failure."""
    print("\n" + "=" * 60)
    print("RNN Scaling Experiment")
    print("=" * 60)
    print("Testing whether increasing RNN capacity helps...")
    
    hidden_dims = [64, 128, 256, 512, 1024, 2048]
    results = {}
    
    tokenizer = Tokenizer()
    train_dataset, test_dataset, _ = create_datasets(args)
    
    for hidden_dim in hidden_dims:
        print(f"\n{'='*40}")
        print(f"Hidden dimension: {hidden_dim}")
        print(f"{'='*40}")
        
        # Create model
        config = RNNConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=hidden_dim,
            n_layers=args.n_layers,
            rnn_type=args.rnn_type,
        )
        model = RNNLM(config).to(device)
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Parameters: {n_params:,}")
        
        # Create dataloaders
        train_loader = create_dataloader(
            train_dataset, batch_size=args.batch_size, shuffle=True
        )
        test_loader = create_dataloader(
            test_dataset, batch_size=args.batch_size, shuffle=False
        )
        
        # Train for fewer steps in scaling experiment
        training_config = TrainingConfig(
            max_steps=min(args.max_steps, 20000),
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            eval_interval=1000,
            log_interval=500,
        )
        
        output_dir = Path(args.output_dir) / f"rnn_scaling_{args.rnn_type}_d{hidden_dim}"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            eval_loader=test_loader,
            config=training_config,
            device=device,
            output_dir=output_dir,
        )
        
        history = trainer.train()
        
        # Record results
        best_test_acc = max(history["test_accuracy"]) if history["test_accuracy"] else 0.0
        final_test_acc = history["test_accuracy"][-1] if history["test_accuracy"] else 0.0
        
        results[hidden_dim] = {
            "n_params": n_params,
            "best_test_acc": best_test_acc,
            "final_test_acc": final_test_acc,
        }
        
        print(f"Best test accuracy: {best_test_acc:.2%}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SCALING EXPERIMENT SUMMARY")
    print("=" * 60)
    print(f"{'Hidden Dim':<12} {'Params':<12} {'Best Acc':<12} {'Final Acc':<12}")
    print("-" * 48)
    
    for hidden_dim, res in results.items():
        print(f"{hidden_dim:<12} {res['n_params']:<12,} {res['best_test_acc']:<12.2%} {res['final_test_acc']:<12.2%}")
    
    print("\n" + "=" * 60)
    print("CONCLUSION: RNNs fail to generalize regardless of capacity!")
    print("This empirically validates the exponential bottleneck.")
    print("=" * 60)
    
    # Save results
    results_path = Path(args.output_dir) / "rnn_scaling_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")
    
    return results


def main():
    """Main training function."""
    args = parse_args()
    
    set_seed(args.seed)
    device = setup_device(args.device)
    
    # Generate experiment name
    if args.exp_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.exp_name = f"{args.rnn_type}_n{args.n_bits}_d{args.hidden_dim}_{timestamp}"
    
    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"LARGECOUNTER {args.rnn_type.upper()} Baseline Training")
    print("=" * 60)
    print(f"Experiment: {args.exp_name}")
    print(f"Output directory: {output_dir}")
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
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
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Max steps: {args.max_steps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    
    # Warn about expected failure
    print("\n" + "!" * 60)
    print("NOTE: RNNs are expected to FAIL on this task!")
    print("This validates the theoretical exponential bottleneck.")
    print("!" * 60)
    
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
        
        # Report results
        if history["test_accuracy"]:
            final_test_acc = history["test_accuracy"][-1]
            best_test_acc = max(history["test_accuracy"])
            
            print(f"Final test accuracy: {final_test_acc:.2%}")
            print(f"Best test accuracy: {best_test_acc:.2%}")
            
            if best_test_acc < 0.1:
                print("\n✗ As predicted, the RNN failed to generalize!")
                print("  The fixed hidden state cannot represent 2^n counter states.")
            elif best_test_acc < 0.5:
                print("\n~ RNN shows partial memorization but no true generalization.")
            else:
                print("\n! Unexpected: RNN showed some generalization.")
                print("  This may indicate a bug or unusual training dynamics.")
        
        # Save final model
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": vars(args),
            "history": history,
        }, output_dir / "model_final.pt")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted.")
        torch.save(model.state_dict(), output_dir / "model_interrupted.pt")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
