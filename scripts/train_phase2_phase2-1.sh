#!/bin/bash
# =============================================================================
# Phase 2-1 (v4, bf16) – Sequential span replacement with <swi> in loss
# =============================================================================
# bf16 LoRA (no 4-bit quantization) for clean DDP multi-GPU support.
# <swi> label included in loss so the model keeps learning WHEN to open
# latent regions during Phase 2 training.
# =============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ===== GPU 选择 =====
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ===== 基础路径 =====
NPROC=${NPROC:-8}
PHASE1_CKPT=${PHASE1_CKPT:-$PROJECT_ROOT/outputs/phase1_bf16/final}
OUT1=${OUT1:-$PROJECT_ROOT/outputs/phase2_1_v4_bf16}
DATA_FILE=${DATA_FILE:-$PROJECT_ROOT/data_hf/data/math_openr1_all.jsonl}
USE_MODELSCOPE=${USE_MODELSCOPE:-1}
MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-8B}
ENABLE_GRADIENT_CHECKPOINTING=${ENABLE_GRADIENT_CHECKPOINTING:-0}
REPORT_TO=${REPORT_TO:-swanlab}

GC_FLAG=()
if [ "$ENABLE_GRADIENT_CHECKPOINTING" = "1" ]; then
  GC_FLAG+=(--gradient_checkpointing)
fi

echo "============================================================"
echo "Phase 2-1 (v4 bf16: <swi> in loss, DDP multi-GPU)"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"
echo "PHASE1_CKPT=$PHASE1_CKPT"
echo "OUT1=$OUT1"
echo "DATA_FILE=$DATA_FILE"
echo "USE_MODELSCOPE=$USE_MODELSCOPE"
echo "MODELSCOPE_MODEL_ID=$MODELSCOPE_MODEL_ID"
echo "GRADIENT_CHECKPOINTING=$ENABLE_GRADIENT_CHECKPOINTING"
echo "REPORT_TO=$REPORT_TO"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU memory:"
  nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader,nounits
fi
echo "============================================================"

export DATA_FILE

# ===== Phase2-1：逐个span替换 =====
# Key changes v4 (bf16):
#   - NO load_in_4bit: bf16 precision for clean DDP multi-GPU
#   - batch_size=2: H20 96GB can afford larger batches
#   - grad_accum=4: effective batch = 2 * 8 * 4 = 64
#   - epochs_per_stage=3: more training per stage
#   - stage0_epochs=3: SFT warm-up before latent injection
#   - warmup_ratio=0.05: cosine LR with warmup
USE_MODELSCOPE="$USE_MODELSCOPE" MODELSCOPE_MODEL_ID="$MODELSCOPE_MODEL_ID" BASE_MODEL_SOURCE=modelscope REPORT_TO="$REPORT_TO" \
bash "$PROJECT_ROOT/scripts/train_phase2.sh" \
  "$NPROC" \
  "$PHASE1_CKPT" \
  "$OUT1" \
  --phase2_strategy phase2-1 \
  --max_latent_stage 10 \
  --c_thought 1 \
  --run_name swilatent_phase2-1_v4_bf16 \
  --epochs_per_stage 3 \
  --stage0_epochs 3 \
  --warmup_ratio 0.05 \
  --uniform_prob 0.1 \
  --lr 1e-4 \
  --batch_size 1 \
  --grad_accum 8 \
  --finetuning_type lora \
  --max_length 2048 \
  "${GC_FLAG[@]}" \
  --bf16
