#!/bin/bash
#===============================================================================
# Run All Experiments for "Do Transformers Grok Succinct Algorithms?"
#
# This script reproduces all experiments from the ACL 2026 paper.
# Results will be saved to the outputs/ directory.
#
# Usage:
#   chmod +x scripts/run_all_experiments.sh
#   ./scripts/run_all_experiments.sh [--quick]
#
# Options:
#   --quick     Run shortened experiments for testing (fewer steps/samples)
#   --gpu ID    Specify GPU device ID (default: 0)
#   --skip-rnn  Skip RNN baseline training (slow)
#   --skip-pythia  Skip Pythia analysis (requires HuggingFace)
#===============================================================================

set -e  # Exit on error

# Parse arguments
QUICK=false
GPU=0
SKIP_RNN=false
SKIP_PYTHIA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK=true
            shift
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --skip-rnn)
            SKIP_RNN=true
            shift
            ;;
        --skip-pythia)
            SKIP_PYTHIA=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set CUDA device
export CUDA_VISIBLE_DEVICES=$GPU

# Configuration based on mode
if [ "$QUICK" = true ]; then
    echo "Running in QUICK mode (reduced steps for testing)"
    MAX_STEPS=5000
    EVAL_INTERVAL=500
    N_SAMPLES=20
    SEEDS="42"
    WEIGHT_DECAYS="0.0 1.0"
else
    echo "Running in FULL mode (paper reproduction)"
    MAX_STEPS=50000
    EVAL_INTERVAL=500
    N_SAMPLES=100
    SEEDS="42 123 456 789 1337"
    WEIGHT_DECAYS="0.0 0.01 0.1 1.0 2.0"
fi

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/outputs"
CHECKPOINT_DIR="$OUTPUT_DIR/checkpoints"
FIGURES_DIR="$OUTPUT_DIR/figures"

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$FIGURES_DIR"

# Print configuration
echo ""
echo "==============================================================================="
echo "                     EXPERIMENT CONFIGURATION"
echo "==============================================================================="
echo "Project directory: $PROJECT_DIR"
echo "Output directory:  $OUTPUT_DIR"
echo "GPU device:        $GPU"
echo "Max steps:         $MAX_STEPS"
echo "Seeds:             $SEEDS"
echo "Weight decays:     $WEIGHT_DECAYS"
echo "==============================================================================="
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Check Python environment
echo "Checking Python environment..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if [ -f "src/__init__.py" ]; then
    echo "Project package found ✓"
else
    echo "Error: Project package not found. Run from project root."
    exit 1
fi

#===============================================================================
# Experiment 1: Main Transformer Training (Section 4.1, 4.2)
#===============================================================================
echo ""
echo "==============================================================================="
echo "EXPERIMENT 1: Transformer Training with Grokking"
echo "==============================================================================="
echo ""

for SEED in $SEEDS; do
    echo "Training Transformer (seed=$SEED)..."
    python scripts/train_transformer.py \
        --config configs/default.yaml \
        --max_steps $MAX_STEPS \
        --eval_interval $EVAL_INTERVAL \
        --seed $SEED \
        --output_dir "$OUTPUT_DIR/transformer_seed$SEED" \
        --save_checkpoints
    echo "  → Completed seed $SEED"
done

echo "Transformer training complete!"

#===============================================================================
# Experiment 2: RNN Baseline Training (Section 4.1)
#===============================================================================
if [ "$SKIP_RNN" = false ]; then
    echo ""
    echo "==============================================================================="
    echo "EXPERIMENT 2: RNN Baseline Training"
    echo "==============================================================================="
    echo ""
    
    # Train LSTM with various hidden dimensions
    for HIDDEN in 64 256 1024 2048; do
        echo "Training LSTM (hidden=$HIDDEN)..."
        python scripts/train_rnn.py \
            --model_type lstm \
            --hidden_dim $HIDDEN \
            --max_steps $MAX_STEPS \
            --eval_interval $EVAL_INTERVAL \
            --seed 42 \
            --output_dir "$OUTPUT_DIR/lstm_h$HIDDEN"
        echo "  → Completed LSTM h=$HIDDEN"
    done
    
    # Train GRU with largest hidden dimension
    echo "Training GRU (hidden=2048)..."
    python scripts/train_rnn.py \
        --model_type gru \
        --hidden_dim 2048 \
        --max_steps $MAX_STEPS \
        --eval_interval $EVAL_INTERVAL \
        --seed 42 \
        --output_dir "$OUTPUT_DIR/gru_h2048"
    
    echo "RNN training complete!"
else
    echo ""
    echo "Skipping RNN baseline training (--skip-rnn)"
fi

#===============================================================================
# Experiment 3: Weight Decay Ablation (Section 4.2, Appendix B.4)
#===============================================================================
echo ""
echo "==============================================================================="
echo "EXPERIMENT 3: Weight Decay Ablation Study"
echo "==============================================================================="
echo ""

python scripts/ablation_weight_decay.py \
    --weight_decays $WEIGHT_DECAYS \
    --max_steps $MAX_STEPS \
    --eval_interval $EVAL_INTERVAL \
    --seeds $SEEDS \
    --output_dir "$OUTPUT_DIR/ablation_weight_decay"

echo "Ablation study complete!"

#===============================================================================
# Experiment 4: Mechanistic Analysis (Section 4.3)
#===============================================================================
echo ""
echo "==============================================================================="
echo "EXPERIMENT 4: Mechanistic Analysis"
echo "==============================================================================="
echo ""

# Find best checkpoint
BEST_CKPT=$(find "$OUTPUT_DIR" -name "*.pt" -path "*transformer*" | head -1)

if [ -n "$BEST_CKPT" ]; then
    echo "Analyzing checkpoint: $BEST_CKPT"
    python scripts/analyze_model.py \
        --checkpoint "$BEST_CKPT" \
        --n_samples $N_SAMPLES \
        --output_dir "$OUTPUT_DIR/mechanistic_analysis"
else
    echo "Warning: No checkpoint found. Training a fresh model for analysis..."
    python scripts/train_transformer.py \
        --config configs/default.yaml \
        --max_steps $MAX_STEPS \
        --output_dir "$OUTPUT_DIR/analysis_model" \
        --save_checkpoints
    
    BEST_CKPT=$(find "$OUTPUT_DIR/analysis_model" -name "best_model.pt" | head -1)
    python scripts/analyze_model.py \
        --checkpoint "$BEST_CKPT" \
        --n_samples $N_SAMPLES \
        --output_dir "$OUTPUT_DIR/mechanistic_analysis"
fi

echo "Mechanistic analysis complete!"

#===============================================================================
# Experiment 5: Pythia Analysis (Section 4.4)
#===============================================================================
if [ "$SKIP_PYTHIA" = false ]; then
    echo ""
    echo "==============================================================================="
    echo "EXPERIMENT 5: Pythia LLM Analysis"
    echo "==============================================================================="
    echo ""
    
    # Check if transformers is installed
    if python -c "import transformers" 2>/dev/null; then
        python scripts/pythia_analysis.py \
            --model "EleutherAI/pythia-160m" \
            --task addition \
            --n_samples $N_SAMPLES \
            --n_digits 3 \
            --output_dir "$OUTPUT_DIR/pythia_analysis"
        echo "Pythia analysis complete!"
    else
        echo "Warning: transformers library not installed. Skipping Pythia analysis."
        echo "Install with: pip install transformers"
    fi
else
    echo ""
    echo "Skipping Pythia analysis (--skip-pythia)"
fi

#===============================================================================
# Generate Final Report
#===============================================================================
echo ""
echo "==============================================================================="
echo "Generating Final Report"
echo "==============================================================================="
echo ""

# Collect all results
REPORT_FILE="$OUTPUT_DIR/experiment_report.md"

cat > "$REPORT_FILE" << EOF
# Experiment Results: Do Transformers Grok Succinct Algorithms?

Generated: $(date)

## Configuration
- Max steps: $MAX_STEPS
- Seeds: $SEEDS
- Weight decays: $WEIGHT_DECAYS
- GPU: $GPU

## Results Summary

### 1. Transformer Training
EOF

# Add transformer results
for SEED in $SEEDS; do
    RESULT_FILE="$OUTPUT_DIR/transformer_seed$SEED/training_history.json"
    if [ -f "$RESULT_FILE" ]; then
        FINAL_ACC=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['test_acc'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
        echo "- Seed $SEED: Final test accuracy = $FINAL_ACC" >> "$REPORT_FILE"
    fi
done

cat >> "$REPORT_FILE" << EOF

### 2. RNN Baselines
EOF

if [ "$SKIP_RNN" = false ]; then
    for HIDDEN in 64 256 1024 2048; do
        RESULT_FILE="$OUTPUT_DIR/lstm_h$HIDDEN/training_history.json"
        if [ -f "$RESULT_FILE" ]; then
            FINAL_ACC=$(python -c "import json; d=json.load(open('$RESULT_FILE')); print(f\"{d['test_acc'][-1]:.4f}\")" 2>/dev/null || echo "N/A")
            echo "- LSTM (h=$HIDDEN): Final test accuracy = $FINAL_ACC" >> "$REPORT_FILE"
        fi
    done
else
    echo "Skipped (--skip-rnn)" >> "$REPORT_FILE"
fi

cat >> "$REPORT_FILE" << EOF

### 3. Weight Decay Ablation
See: outputs/ablation_weight_decay/ablation_summary.json

### 4. Mechanistic Analysis
See: outputs/mechanistic_analysis/

### 5. Pythia Analysis
EOF

if [ "$SKIP_PYTHIA" = false ]; then
    echo "See: outputs/pythia_analysis/" >> "$REPORT_FILE"
else
    echo "Skipped (--skip-pythia)" >> "$REPORT_FILE"
fi

echo "Report generated: $REPORT_FILE"

#===============================================================================
# Copy figures to central location
#===============================================================================
echo ""
echo "Collecting figures..."

# Copy all generated figures
find "$OUTPUT_DIR" -name "*.png" -exec cp {} "$FIGURES_DIR/" \; 2>/dev/null || true
find "$OUTPUT_DIR" -name "*.pdf" -exec cp {} "$FIGURES_DIR/" \; 2>/dev/null || true

echo "Figures collected in: $FIGURES_DIR"

#===============================================================================
# Final Summary
#===============================================================================
echo ""
echo "==============================================================================="
echo "                     ALL EXPERIMENTS COMPLETE!"
echo "==============================================================================="
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Figures saved to: $FIGURES_DIR"
echo "Report: $REPORT_FILE"
echo ""
echo "Key outputs:"
echo "  - Training curves: outputs/*/training_history.json"
echo "  - Ablation study: outputs/ablation_weight_decay/"
echo "  - Attention patterns: outputs/mechanistic_analysis/"
if [ "$SKIP_PYTHIA" = false ]; then
    echo "  - Pythia analysis: outputs/pythia_analysis/"
fi
echo ""
echo "To view the experiment report:"
echo "  cat $REPORT_FILE"
echo ""
echo "==============================================================================="
