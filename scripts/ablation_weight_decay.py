#!/usr/bin/env python3
"""
Weight Decay Ablation Study for Grokking Analysis.

This script systematically varies the weight decay coefficient (λ) to empirically
validate that weight decay is the causal driver of the grokking phase transition.

As shown in Figure 6 of the paper:
- λ ≤ 0.01: Memorization regime (overfitting, no generalization)
- λ = 1.0: Grokking regime (complexity collapse, perfect generalization)  
- λ = 2.0: Underfitting regime (too much regularization)

Usage:
    python scripts/ablation_weight_decay.py [--config configs/default.yaml]
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.counter_dataset import LargeCounterDataset, create_dataloader
from src.models.transformer import TransformerLM, TransformerConfig
from src.training.trainer import (
    TrainingConfig, 
    create_optimizer, 
    create_scheduler,
    compute_metrics,
    evaluate_stratified
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weight Decay Ablation Study for Grokking"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--weight_decays",
        type=float,
        nargs="+",
        default=[0.0, 0.01, 0.1, 1.0, 2.0],
        help="Weight decay values to sweep"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=50000,
        help="Maximum training steps per run"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=500,
        help="Steps between evaluations"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for each run"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/ablation_weight_decay",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training"
    )
    parser.add_argument(
        "--n_bits",
        type=int,
        default=20,
        help="Number of bits for counter task"
    )
    return parser.parse_args()


class AblationRun:
    """Single ablation run with specific weight decay."""
    
    def __init__(
        self,
        weight_decay: float,
        seed: int,
        config: Dict,
        device: str,
        output_dir: Path
    ):
        self.weight_decay = weight_decay
        self.seed = seed
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # History tracking
        self.history = {
            "steps": [],
            "train_loss": [],
            "train_acc": [],
            "test_acc": [],
            "weight_norm": [],
            "weight_decay": weight_decay,
            "seed": seed
        }
    
    def setup_data(self, n_bits: int) -> Tuple[DataLoader, DataLoader]:
        """Create train and test dataloaders."""
        train_dataset = LargeCounterDataset(
            n_bits=n_bits,
            split="train",
            train_ratio=0.3,
            stratified=True
        )
        test_dataset = LargeCounterDataset(
            n_bits=n_bits,
            split="test", 
            train_ratio=0.3,
            stratified=True
        )
        
        train_loader = create_dataloader(
            train_dataset,
            batch_size=self.config.get("batch_size", 512),
            shuffle=True
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=self.config.get("batch_size", 512),
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def setup_model(self, vocab_size: int) -> TransformerLM:
        """Create Transformer model."""
        model_config = TransformerConfig(
            vocab_size=vocab_size,
            d_model=self.config.get("d_model", 64),
            n_heads=self.config.get("n_heads", 4),
            n_layers=self.config.get("n_layers", 2),
            d_ff=self.config.get("d_ff", 256),
            max_seq_len=self.config.get("max_seq_len", 128),
            dropout=self.config.get("dropout", 0.0),
            use_rope=True
        )
        model = TransformerLM(model_config)
        return model.to(self.device)
    
    def get_weight_norm(self, model: nn.Module) -> float:
        """Compute L2 norm of all model parameters."""
        total_norm = 0.0
        for param in model.parameters():
            if param.requires_grad:
                total_norm += param.data.norm(2).item() ** 2
        return np.sqrt(total_norm)
    
    def train_step(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Tuple[float, float]:
        """Single training step."""
        model.train()
        
        input_ids = batch["input_ids"].to(self.device)
        labels = batch["labels"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        optimizer.zero_grad()
        
        logits = model(input_ids, attention_mask=attention_mask)
        
        # Reshape for loss computation
        B, T, V = logits.shape
        loss = criterion(
            logits.view(-1, V),
            labels.view(-1)
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            accuracy = correct.sum().item() / mask.sum().item()
        
        return loss.item(), accuracy
    
    @torch.no_grad()
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate model on dataset."""
        model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        seq_correct = 0
        total_seqs = 0
        
        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.device)
            labels = batch["labels"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            logits = model(input_ids, attention_mask=attention_mask)
            
            B, T, V = logits.shape
            loss = criterion(logits.view(-1, V), labels.view(-1))
            total_loss += loss.item() * B
            
            # Token-level accuracy
            preds = logits.argmax(dim=-1)
            mask = labels != -100
            correct = (preds == labels) & mask
            total_correct += correct.sum().item()
            total_tokens += mask.sum().item()
            
            # Sequence-level accuracy (all bits must match)
            for i in range(B):
                seq_mask = mask[i]
                if seq_mask.sum() > 0:
                    seq_pred = preds[i][seq_mask]
                    seq_label = labels[i][seq_mask]
                    if (seq_pred == seq_label).all():
                        seq_correct += 1
                    total_seqs += 1
        
        return {
            "loss": total_loss / total_seqs,
            "token_acc": total_correct / total_tokens if total_tokens > 0 else 0,
            "seq_acc": seq_correct / total_seqs if total_seqs > 0 else 0
        }
    
    def run(
        self,
        n_bits: int,
        max_steps: int,
        eval_interval: int
    ) -> Dict:
        """Execute full training run."""
        print(f"\n{'='*60}")
        print(f"Starting run: weight_decay={self.weight_decay}, seed={self.seed}")
        print(f"{'='*60}")
        
        # Setup
        train_loader, test_loader = self.setup_data(n_bits)
        vocab_size = train_loader.dataset.tokenizer.vocab_size
        model = self.setup_model(vocab_size)
        
        # Optimizer with variable weight decay
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.get("learning_rate", 1e-3),
            weight_decay=self.weight_decay,
            betas=(0.9, 0.98)
        )
        
        scheduler = create_scheduler(
            optimizer,
            TrainingConfig(
                max_steps=max_steps,
                warmup_steps=self.config.get("warmup_steps", 1000)
            )
        )
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training loop
        step = 0
        train_iter = iter(train_loader)
        
        while step < max_steps:
            # Get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            # Train step
            loss, train_acc = self.train_step(model, batch, optimizer, criterion)
            scheduler.step()
            
            step += 1
            
            # Evaluate
            if step % eval_interval == 0 or step == max_steps:
                test_metrics = self.evaluate(model, test_loader, criterion)
                weight_norm = self.get_weight_norm(model)
                
                self.history["steps"].append(step)
                self.history["train_loss"].append(loss)
                self.history["train_acc"].append(train_acc)
                self.history["test_acc"].append(test_metrics["seq_acc"])
                self.history["weight_norm"].append(weight_norm)
                
                print(f"Step {step:5d} | "
                      f"Train Loss: {loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | "
                      f"Test Acc: {test_metrics['seq_acc']:.4f} | "
                      f"||W||: {weight_norm:.2f}")
                
                # Early stopping if perfect accuracy achieved
                if test_metrics["seq_acc"] >= 0.99:
                    print(f"  → Perfect generalization achieved at step {step}!")
        
        return self.history


def run_ablation(args: argparse.Namespace) -> Dict[str, List[Dict]]:
    """Run full ablation study across weight decays and seeds."""
    
    # Load config
    config = {}
    if os.path.exists(args.config):
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Results storage
    all_results = {}
    
    for wd in args.weight_decays:
        all_results[f"wd_{wd}"] = []
        
        for seed in args.seeds:
            run = AblationRun(
                weight_decay=wd,
                seed=seed,
                config=config,
                device=args.device,
                output_dir=output_dir
            )
            
            history = run.run(
                n_bits=args.n_bits,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval
            )
            
            all_results[f"wd_{wd}"].append(history)
            
            # Save individual run
            run_file = output_dir / f"run_wd{wd}_seed{seed}.json"
            with open(run_file, "w") as f:
                json.dump(history, f, indent=2)
    
    return all_results


def plot_ablation_results(
    results: Dict[str, List[Dict]],
    output_dir: Path
):
    """Generate ablation study visualization (Figure 6 from paper)."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Color map for weight decay values
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))
    
    # Plot 1: Test Accuracy Dynamics
    ax1 = axes[0]
    for (wd_key, runs), color in zip(results.items(), colors):
        wd = float(wd_key.split("_")[1])
        
        # Average across seeds
        all_steps = runs[0]["steps"]
        all_accs = np.array([r["test_acc"] for r in runs])
        mean_acc = np.mean(all_accs, axis=0)
        std_acc = np.std(all_accs, axis=0)
        
        ax1.plot(all_steps, mean_acc, color=color, label=f"λ={wd}", linewidth=2)
        ax1.fill_between(all_steps, mean_acc - std_acc, mean_acc + std_acc, 
                        color=color, alpha=0.2)
    
    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Test Accuracy", fontsize=12)
    ax1.set_title("(a) Test Accuracy Dynamics", fontsize=14)
    ax1.set_xscale("log")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)
    
    # Plot 2: Weight Norm Dynamics
    ax2 = axes[1]
    for (wd_key, runs), color in zip(results.items(), colors):
        wd = float(wd_key.split("_")[1])
        
        all_steps = runs[0]["steps"]
        all_norms = np.array([r["weight_norm"] for r in runs])
        mean_norm = np.mean(all_norms, axis=0)
        std_norm = np.std(all_norms, axis=0)
        
        ax2.plot(all_steps, mean_norm, color=color, label=f"λ={wd}", linewidth=2)
        ax2.fill_between(all_steps, mean_norm - std_norm, mean_norm + std_norm,
                        color=color, alpha=0.2)
    
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Weight Norm (||W||₂)", fontsize=12)
    ax2.set_title("(b) Weight Norm (L₂) Dynamics", fontsize=14)
    ax2.set_xscale("log")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "ablation_weight_decay.pdf", bbox_inches="tight", dpi=300)
    plt.savefig(output_dir / "ablation_weight_decay.png", bbox_inches="tight", dpi=150)
    plt.close()
    
    print(f"\nSaved ablation plots to {output_dir}")


def generate_summary_table(
    results: Dict[str, List[Dict]],
    output_dir: Path
):
    """Generate summary statistics table."""
    
    summary = []
    
    for wd_key, runs in results.items():
        wd = float(wd_key.split("_")[1])
        
        # Final test accuracy
        final_accs = [r["test_acc"][-1] for r in runs]
        mean_acc = np.mean(final_accs)
        std_acc = np.std(final_accs)
        
        # Final weight norm
        final_norms = [r["weight_norm"][-1] for r in runs]
        mean_norm = np.mean(final_norms)
        std_norm = np.std(final_norms)
        
        # Grokking step (first step where test_acc > 0.9)
        grok_steps = []
        for r in runs:
            for step, acc in zip(r["steps"], r["test_acc"]):
                if acc > 0.9:
                    grok_steps.append(step)
                    break
            else:
                grok_steps.append(float("inf"))
        
        mean_grok = np.mean([s for s in grok_steps if s < float("inf")])
        
        summary.append({
            "weight_decay": wd,
            "final_acc_mean": mean_acc,
            "final_acc_std": std_acc,
            "final_norm_mean": mean_norm,
            "final_norm_std": std_norm,
            "grokking_step": mean_grok if not np.isnan(mean_grok) else "N/A"
        })
    
    # Save summary
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print table
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    print(f"{'λ':>8} | {'Test Acc':>15} | {'Weight Norm':>15} | {'Grok Step':>10}")
    print("-"*80)
    
    for s in summary:
        grok_str = f"{s['grokking_step']:.0f}" if isinstance(s['grokking_step'], float) else s['grokking_step']
        print(f"{s['weight_decay']:>8.2f} | "
              f"{s['final_acc_mean']:.3f} ± {s['final_acc_std']:.3f} | "
              f"{s['final_norm_mean']:.2f} ± {s['final_norm_std']:.2f} | "
              f"{grok_str:>10}")
    
    print("="*80)


def main():
    """Main entry point for ablation study."""
    args = parse_args()
    
    print("="*60)
    print("Weight Decay Ablation Study for Grokking")
    print("="*60)
    print(f"Weight decay values: {args.weight_decays}")
    print(f"Seeds: {args.seeds}")
    print(f"Max steps: {args.max_steps}")
    print(f"Device: {args.device}")
    print(f"Output directory: {args.output_dir}")
    
    # Run ablation
    results = run_ablation(args)
    
    # Generate plots
    output_dir = Path(args.output_dir)
    plot_ablation_results(results, output_dir)
    
    # Generate summary
    generate_summary_table(results, output_dir)
    
    # Save all results
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nAblation study complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
