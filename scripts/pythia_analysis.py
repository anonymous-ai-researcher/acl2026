#!/usr/bin/env python3
"""
Pythia Model Analysis for Arithmetic Head Detection.

This script extends the mechanistic analysis to pre-trained LLMs (Pythia suite)
to detect "Arithmetic Heads" that exhibit sparse, relative routing patterns
similar to the Same-Bit Lookup mechanism discovered in synthetic experiments.

Key findings from Section 4.4 of the paper:
- Pythia-160M exhibits arithmetic heads in middle layers (e.g., Layer 8, Head 5)
- These heads attend to corresponding digits of operands when generating results
- Space-delimited prompting is crucial to overcome the "Tokenizer Barrier"

Usage:
    python scripts/pythia_analysis.py [--model pythia-160m] [--task addition]
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers library not installed. Install with: pip install transformers")

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Pythia models for arithmetic circuit signatures"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="EleutherAI/pythia-160m",
        help="Pythia model to analyze"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="addition",
        choices=["addition", "counting"],
        help="Task type for analysis"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to analyze"
    )
    parser.add_argument(
        "--n_digits",
        type=int,
        default=3,
        help="Number of digits for arithmetic task"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/pythia_analysis",
        help="Directory for saving results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for computation"
    )
    parser.add_argument(
        "--use_space_delimited",
        action="store_true",
        default=True,
        help="Use space-delimited prompting to overcome tokenizer barrier"
    )
    return parser.parse_args()


class ArithmeticPromptGenerator:
    """Generate arithmetic prompts for LLM analysis."""
    
    def __init__(self, n_digits: int = 3, use_space_delimited: bool = True):
        self.n_digits = n_digits
        self.use_space_delimited = use_space_delimited
    
    def generate_addition_prompt(self, ensure_carry: bool = True) -> Dict:
        """Generate a single addition problem.
        
        Args:
            ensure_carry: If True, ensure carry propagation occurs
            
        Returns:
            Dictionary with prompt, operands, and expected result
        """
        max_val = 10 ** self.n_digits - 1
        min_val = 10 ** (self.n_digits - 1)
        
        if ensure_carry:
            # Generate numbers likely to produce carries
            a = np.random.randint(min_val, max_val)
            b = np.random.randint(min_val, max_val)
            # Ensure at least one column produces a carry
            while not self._has_carry(a, b):
                a = np.random.randint(min_val, max_val)
                b = np.random.randint(min_val, max_val)
        else:
            a = np.random.randint(0, max_val)
            b = np.random.randint(0, max_val)
        
        result = a + b
        
        if self.use_space_delimited:
            # Space-delimited format for digit-level tokenization
            a_str = " ".join(str(a))
            b_str = " ".join(str(b))
            r_str = " ".join(str(result))
            prompt = f"{a_str} + {b_str} = "
            full = f"{a_str} + {b_str} = {r_str}"
        else:
            # Standard format
            prompt = f"{a} + {b} = "
            full = f"{a} + {b} = {result}"
        
        return {
            "prompt": prompt,
            "full": full,
            "a": a,
            "b": b,
            "result": result,
            "a_digits": [int(d) for d in str(a)],
            "b_digits": [int(d) for d in str(b)],
            "result_digits": [int(d) for d in str(result)]
        }
    
    def _has_carry(self, a: int, b: int) -> bool:
        """Check if addition produces at least one carry."""
        a_str = str(a).zfill(self.n_digits)
        b_str = str(b).zfill(self.n_digits)
        carry = 0
        for i in range(len(a_str) - 1, -1, -1):
            digit_sum = int(a_str[i]) + int(b_str[i]) + carry
            if digit_sum >= 10:
                return True
            carry = digit_sum // 10
        return False
    
    def generate_few_shot_prompt(self, n_shots: int = 5) -> str:
        """Generate few-shot examples for in-context learning."""
        examples = []
        for _ in range(n_shots):
            sample = self.generate_addition_prompt(ensure_carry=True)
            examples.append(sample["full"])
        
        return "\n".join(examples) + "\n"
    
    def generate_counting_prompt(self, start: int = 0) -> Dict:
        """Generate a counting prompt (binary increment)."""
        max_val = 2 ** self.n_digits - 1
        n = np.random.randint(start, max_val)
        next_n = (n + 1) % (2 ** self.n_digits)
        
        n_bin = format(n, f"0{self.n_digits}b")
        next_bin = format(next_n, f"0{self.n_digits}b")
        
        if self.use_space_delimited:
            n_str = " ".join(n_bin)
            next_str = " ".join(next_bin)
            prompt = f"{n_str} # "
            full = f"{n_str} # {next_str}"
        else:
            prompt = f"{n_bin} # "
            full = f"{n_bin} # {next_bin}"
        
        return {
            "prompt": prompt,
            "full": full,
            "n": n,
            "next_n": next_n,
            "n_bits": [int(b) for b in n_bin],
            "next_bits": [int(b) for b in next_bin]
        }


class PythiaAnalyzer:
    """Analyze Pythia model attention patterns for arithmetic circuits."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_space_delimited: bool = True
    ):
        if not HAS_TRANSFORMERS:
            raise RuntimeError("transformers library required for Pythia analysis")
        
        self.model_name = model_name
        self.device = device
        self.use_space_delimited = use_space_delimited
        
        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            output_attentions=True,
            trust_remote_code=True
        ).to(device)
        self.model.eval()
        
        # Get model config
        self.n_layers = self.model.config.num_hidden_layers
        self.n_heads = self.model.config.num_attention_heads
        
        print(f"Model loaded: {self.n_layers} layers, {self.n_heads} heads per layer")
    
    @torch.no_grad()
    def get_attention_patterns(
        self,
        prompt: str,
        max_new_tokens: int = 10
    ) -> Dict:
        """Extract attention patterns during generation.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Number of tokens to generate
            
        Returns:
            Dictionary with attention patterns and generation info
        """
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]
        prompt_len = input_ids.shape[1]
        
        # Storage for attention patterns
        all_attentions = []
        generated_tokens = []
        
        # Generate token by token to capture attention
        for _ in range(max_new_tokens):
            outputs = self.model(input_ids, output_attentions=True)
            
            # Get attention from all layers
            # Shape: (n_layers, batch, n_heads, seq_len, seq_len)
            attentions = torch.stack(outputs.attentions, dim=0)
            all_attentions.append(attentions[:, 0, :, -1, :].cpu())  # Last token's attention
            
            # Get next token
            next_token = outputs.logits[0, -1].argmax().unsqueeze(0).unsqueeze(0)
            generated_tokens.append(next_token.item())
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop at newline
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        full_output = self.tokenizer.decode(input_ids[0])
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # Stack attentions: (n_generated, n_layers, n_heads, seq_len)
        attention_stack = torch.stack(all_attentions, dim=0) if all_attentions else None
        
        return {
            "prompt": prompt,
            "generated": generated_text,
            "full_output": full_output,
            "prompt_len": prompt_len,
            "attention_patterns": attention_stack,
            "generated_tokens": generated_tokens,
            "input_tokens": self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        }
    
    def compute_diagonal_score(
        self,
        attention: torch.Tensor,
        target_offset: int,
        prompt_len: int
    ) -> float:
        """Compute diagonal attention score at specific offset.
        
        This measures how much attention mass is placed at the
        theoretically predicted relative position.
        
        Args:
            attention: Attention pattern (n_heads, seq_len)
            target_offset: Expected relative offset (e.g., -(n+1))
            prompt_len: Length of prompt (to identify output region)
            
        Returns:
            Average attention mass at target offset
        """
        seq_len = attention.shape[-1]
        scores = []
        
        for pos in range(prompt_len, seq_len):
            target_pos = pos + target_offset
            if 0 <= target_pos < seq_len:
                scores.append(attention[:, target_pos].mean().item())
        
        return np.mean(scores) if scores else 0.0
    
    def analyze_head_specialization(
        self,
        samples: List[Dict],
        task: str = "addition"
    ) -> Dict[Tuple[int, int], Dict]:
        """Analyze all heads for arithmetic specialization.
        
        Args:
            samples: List of analyzed samples with attention patterns
            task: Task type for expected offset calculation
            
        Returns:
            Dictionary mapping (layer, head) to specialization metrics
        """
        head_scores = {}
        
        for layer in range(self.n_layers):
            for head in range(self.n_heads):
                scores = []
                
                for sample in samples:
                    if sample["attention_patterns"] is None:
                        continue
                    
                    # Get attention pattern for this head
                    attn = sample["attention_patterns"][:, layer, head, :]
                    
                    # Compute metrics
                    # For addition: we expect attention to operand digits
                    # For counting: we expect attention at offset -(n+1)
                    
                    if task == "addition":
                        # Check if head attends to input digits
                        prompt_len = sample["prompt_len"]
                        input_attention = attn[:, :prompt_len].mean().item()
                        scores.append(input_attention)
                    else:
                        # Check diagonal offset
                        n_bits = len(sample.get("n_bits", [8]))
                        offset = -(n_bits + 1)
                        diag_score = self.compute_diagonal_score(
                            attn.mean(dim=0), offset, sample["prompt_len"]
                        )
                        scores.append(diag_score)
                
                head_scores[(layer, head)] = {
                    "mean_score": np.mean(scores) if scores else 0,
                    "std_score": np.std(scores) if scores else 0,
                    "n_samples": len(scores)
                }
        
        return head_scores
    
    def find_arithmetic_heads(
        self,
        head_scores: Dict,
        threshold: float = 0.3
    ) -> List[Tuple[int, int]]:
        """Identify heads with high arithmetic specialization.
        
        Args:
            head_scores: Output from analyze_head_specialization
            threshold: Minimum score for classification
            
        Returns:
            List of (layer, head) tuples for arithmetic heads
        """
        arithmetic_heads = []
        
        for (layer, head), metrics in head_scores.items():
            if metrics["mean_score"] > threshold:
                arithmetic_heads.append((layer, head))
        
        # Sort by score
        arithmetic_heads.sort(
            key=lambda x: head_scores[x]["mean_score"],
            reverse=True
        )
        
        return arithmetic_heads


def visualize_attention_pattern(
    attention: torch.Tensor,
    input_tokens: List[str],
    output_tokens: List[str],
    layer: int,
    head: int,
    save_path: Optional[Path] = None
):
    """Visualize attention pattern for a specific head.
    
    Recreates Figure 5 from the paper.
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get attention matrix
    attn_matrix = attention.numpy()
    
    # Create heatmap
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto")
    
    # Labels
    ax.set_xlabel("Input Prompt Tokens", fontsize=12)
    ax.set_ylabel("Generated Output Tokens", fontsize=12)
    ax.set_title(f"Pythia Attention: Layer {layer}, Head {head}", fontsize=14)
    
    # Token labels
    if len(input_tokens) <= 20:
        ax.set_xticks(range(len(input_tokens)))
        ax.set_xticklabels(input_tokens, rotation=45, ha="right")
    
    if len(output_tokens) <= 10:
        ax.set_yticks(range(len(output_tokens)))
        ax.set_yticklabels(output_tokens)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Attention Weight", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_head_specialization(
    head_scores: Dict,
    n_layers: int,
    n_heads: int,
    save_path: Optional[Path] = None
):
    """Visualize head specialization scores (Figure 7 from paper)."""
    
    # Create matrix
    score_matrix = np.zeros((n_heads, n_layers))
    
    for (layer, head), metrics in head_scores.items():
        score_matrix[head, layer] = metrics["mean_score"]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(score_matrix, cmap="viridis", aspect="auto")
    
    ax.set_xlabel("Layer Index", fontsize=12)
    ax.set_ylabel("Head Index", fontsize=12)
    ax.set_title("Pythia Attention Head Specificity\n(Diagonal Attention Score)", fontsize=14)
    
    ax.set_xticks(range(n_layers))
    ax.set_yticks(range(n_heads))
    
    # Annotate high-scoring heads
    for (layer, head), metrics in head_scores.items():
        if metrics["mean_score"] > 0.5:
            ax.annotate(
                f"L{layer}H{head}",
                (layer, head),
                fontsize=8,
                ha="center",
                va="center",
                color="white",
                fontweight="bold"
            )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Diagonal Attention Score", fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        plt.close()
    else:
        plt.show()


def run_analysis(args: argparse.Namespace):
    """Run full Pythia analysis."""
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = PythiaAnalyzer(
        model_name=args.model,
        device=args.device,
        use_space_delimited=args.use_space_delimited
    )
    
    # Initialize prompt generator
    prompt_gen = ArithmeticPromptGenerator(
        n_digits=args.n_digits,
        use_space_delimited=args.use_space_delimited
    )
    
    # Generate few-shot prefix
    few_shot_prefix = prompt_gen.generate_few_shot_prompt(n_shots=5)
    
    print(f"\nFew-shot prompt format:")
    print("-" * 40)
    print(few_shot_prefix[:200] + "...")
    print("-" * 40)
    
    # Collect samples
    samples = []
    print(f"\nAnalyzing {args.n_samples} samples...")
    
    for i in tqdm(range(args.n_samples)):
        # Generate problem
        if args.task == "addition":
            problem = prompt_gen.generate_addition_prompt(ensure_carry=True)
        else:
            problem = prompt_gen.generate_counting_prompt()
        
        # Full prompt with few-shot examples
        full_prompt = few_shot_prefix + problem["prompt"]
        
        # Get attention patterns
        result = analyzer.get_attention_patterns(
            full_prompt,
            max_new_tokens=args.n_digits + 2  # Result digits + buffer
        )
        
        # Merge problem info
        result.update(problem)
        samples.append(result)
    
    # Analyze head specialization
    print("\nAnalyzing head specialization...")
    head_scores = analyzer.analyze_head_specialization(samples, task=args.task)
    
    # Find arithmetic heads
    arithmetic_heads = analyzer.find_arithmetic_heads(head_scores, threshold=0.3)
    
    print(f"\nTop Arithmetic Heads (score > 0.3):")
    print("-" * 50)
    for layer, head in arithmetic_heads[:10]:
        score = head_scores[(layer, head)]["mean_score"]
        print(f"  Layer {layer:2d}, Head {head:2d}: score = {score:.3f}")
    
    # Visualize head specialization
    visualize_head_specialization(
        head_scores,
        analyzer.n_layers,
        analyzer.n_heads,
        save_path=output_dir / "head_specialization.png"
    )
    
    # Visualize top head's attention pattern
    if arithmetic_heads and samples:
        top_layer, top_head = arithmetic_heads[0]
        sample = samples[0]
        
        if sample["attention_patterns"] is not None:
            attn = sample["attention_patterns"][:, top_layer, top_head, :]
            
            visualize_attention_pattern(
                attn,
                sample["input_tokens"][-20:],  # Last 20 input tokens
                [f"out_{i}" for i in range(attn.shape[0])],
                top_layer,
                top_head,
                save_path=output_dir / f"attention_L{top_layer}H{top_head}.png"
            )
    
    # Save results
    results = {
        "model": args.model,
        "task": args.task,
        "n_samples": args.n_samples,
        "n_digits": args.n_digits,
        "use_space_delimited": args.use_space_delimited,
        "arithmetic_heads": [(l, h) for l, h in arithmetic_heads],
        "head_scores": {
            f"L{l}H{h}": metrics 
            for (l, h), metrics in head_scores.items()
        }
    }
    
    with open(output_dir / "analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}")
    
    # Print tokenizer comparison
    print("\n" + "="*60)
    print("TOKENIZER BARRIER ANALYSIS")
    print("="*60)
    
    test_input = "128 + 345"
    standard_tokens = analyzer.tokenizer.tokenize(test_input)
    
    test_spaced = "1 2 8 + 3 4 5"
    spaced_tokens = analyzer.tokenizer.tokenize(test_spaced)
    
    print(f"\nStandard BPE tokenization of '{test_input}':")
    print(f"  Tokens: {standard_tokens}")
    print(f"  Issue: Variable length tokens destroy positional offset -(n+1)")
    
    print(f"\nSpace-delimited tokenization of '{test_spaced}':")
    print(f"  Tokens: {spaced_tokens}")
    print(f"  Benefit: Fixed alignment. j-th digit at predictable offset.")


def main():
    """Main entry point."""
    args = parse_args()
    
    if not HAS_TRANSFORMERS:
        print("Error: transformers library required.")
        print("Install with: pip install transformers")
        sys.exit(1)
    
    print("="*60)
    print("Pythia Model Analysis for Arithmetic Circuits")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Task: {args.task}")
    print(f"N samples: {args.n_samples}")
    print(f"Space-delimited: {args.use_space_delimited}")
    print(f"Device: {args.device}")
    
    run_analysis(args)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
