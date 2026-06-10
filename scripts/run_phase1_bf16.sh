#!/bin/bash
# =============================================================================
# Phase 1 – SFT Training (bf16 LoRA, multi-GPU DDP)
# =============================================================================
# Trains Qwen3-8B with LoRA + modules_to_save on <swi>-annotated math data.
# Uses bf16 precision (no 4-bit quantization) for clean DDP multi-GPU support.
#
# Usage:
#   bash scripts/run_phase1_bf16.sh                    # defaults: 8 GPUs
#   GPU_IDS=0,1,2,3 NPROC=4 bash scripts/run_phase1_bf16.sh
#   GPU_IDS=0 NPROC=1 bash scripts/run_phase1_bf16.sh  # single GPU
# =============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ===== GPU =====
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ===== Paths =====
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
DATA_FILE=${DATA_FILE:-$PROJECT_ROOT/data_hf/data/math_openr1_all.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-$PROJECT_ROOT/outputs/phase1_bf16}

# ModelScope mirror (set USE_MODELSCOPE=1 if HuggingFace is unreachable)
if [ "${USE_MODELSCOPE:-0}" = "1" ]; then
    MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-8B}
    echo "[Phase 1] Downloading model via ModelScope: $MODELSCOPE_MODEL_ID"
    _TMPFILE=$(mktemp)
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
p = snapshot_download('${MODELSCOPE_MODEL_ID}')
with open('${_TMPFILE}', 'w') as f:
    f.write(p)
"
    MODEL_PATH=$(cat "$_TMPFILE")
    rm -f "$_TMPFILE"
    echo "[Phase 1] ModelScope local path: $MODEL_PATH"
fi

echo "============================================================"
echo "Phase 1 SFT (bf16 LoRA, ${NPROC} GPU DDP)"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"
echo "MODEL_PATH=$MODEL_PATH"
echo "DATA_FILE=$DATA_FILE"
echo "OUTPUT_DIR=$OUTPUT_DIR"
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "GPU info:"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
fi
echo "============================================================"

mkdir -p "$OUTPUT_DIR"

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/train_phase1.py" \
    --model_path          "$MODEL_PATH" \
    --data_file           "$DATA_FILE" \
    --output_dir          "$OUTPUT_DIR" \
    --num_train_epochs    3 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate       1e-4 \
    --warmup_ratio        0.05 \
    --max_length          4096 \
    --lora_r              32 \
    --lora_alpha          64 \
    --lora_dropout        0.05 \
    --gradient_checkpointing \
    --save_steps          100 \
    --save_total_limit    3 \
    --logging_steps       10 \
    --run_name            "swilatent_phase1_bf16_8gpu" \
    --report_to           swanlab
