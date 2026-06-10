#!/bin/bash
# =============================================================================
# Phase 2 – Coconut Curriculum Training
# =============================================================================
# Trains CoconutSwiModel from a Phase-1 SFT checkpoint using the multi-stage
# hidden-state-injection curriculum.
#
# Prerequisite:
#   Phase 1 SFT checkpoint already saved (see scripts/sft.sh).
#
# Usage:
#   bash scripts/train_phase2.sh <nproc_per_node> <phase1_checkpoint> <save_dir>
#
# Example:
#   bash scripts/train_phase2.sh 4 \
#       /data/jiayu/SwiLatent/outputs/phase1_sft/checkpoint \
#       /data/jiayu/SwiLatent/outputs/phase2_coconut
# =============================================================================
set -e

export PYTORCH_ALLOC_CONF=${PYTORCH_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DATA_FILE=${DATA_FILE:-$PROJECT_ROOT/data_hf/data/math_openr1_all.jsonl}

if [ "$#" -lt 3 ]; then
    echo "Usage: train_phase2.sh <nproc_per_node> <phase1_checkpoint> <save_dir>"
    echo "Tip: phase1_checkpoint can be a local path or HF repo id, e.g. LARK-Lab/SWITCH-Phase1-SFT-LoRA-Qwen3-8B"
    exit 1
fi

nproc_per_node=$1
phase1_checkpoint=$2
save_dir=$3
shift 3

# ── Paths ────────────────────────────────────────────────────────────────────
if [ ! -f "$DATA_FILE" ]; then
    echo "[ERROR] DATA_FILE not found: $DATA_FILE"
    echo "Set DATA_FILE=/absolute/path/to/your_annotated_data.jsonl and retry."
    exit 1
fi

if ! command -v torchrun >/dev/null 2>&1; then
    echo "[ERROR] torchrun not found in PATH. Activate the training environment first."
    exit 1
fi

# ── Curriculum hyperparameters ───────────────────────────────────────────────
MAX_LATENT_STAGE=8   # maximum curriculum stage index
C_THOUGHT=1          # phase2-1: latent tokens per replaced span; phase2-2: per-stage growth factor
EPOCHS_PER_STAGE=1   # how many epochs to train at each stage
PHASE2_STRATEGY=phase2-2  # phase2-1: sequential spans, phase2-2: parallel spans

# ── Optimiser ────────────────────────────────────────────────────────────────
LR=1e-4
BATCH_SIZE=2
MAX_LENGTH=2048
GRAD_ACCUM=8         # effective batch = BATCH_SIZE * GRAD_ACCUM = 16
FINETUNING_TYPE=lora
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
LORA_TARGET="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"
REPORT_TO=${REPORT_TO:-swanlab}
RUN_NAME=swilatent_phase2
LOGGING_STEPS=10

# ── Launch ───────────────────────────────────────────────────────────────────
mkdir -p "$save_dir"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$nproc_per_node" \
    "$PROJECT_ROOT/train_phase2.py" \
    --model_path        "$phase1_checkpoint" \
    --data_file         "$DATA_FILE" \
    --save_dir          "$save_dir" \
    --max_latent_stage  "$MAX_LATENT_STAGE" \
    --c_thought         "$C_THOUGHT" \
    --epochs_per_stage  "$EPOCHS_PER_STAGE" \
    --phase2_strategy   "$PHASE2_STRATEGY" \
    --lr                "$LR" \
    --batch_size        "$BATCH_SIZE" \
    --max_length        "$MAX_LENGTH" \
    --grad_accum        "$GRAD_ACCUM" \
    --finetuning_type   "$FINETUNING_TYPE" \
    --lora_r            "$LORA_R" \
    --lora_alpha        "$LORA_ALPHA" \
    --lora_dropout      "$LORA_DROPOUT" \
    --lora_target       "$LORA_TARGET" \
    --report_to         "$REPORT_TO" \
    --run_name          "$RUN_NAME" \
    --logging_steps     "$LOGGING_STEPS" \
    --gradient_checkpointing \
    --bf16 \
    "$@"
