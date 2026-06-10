#!/bin/bash
# =============================================================================
# Phase 2-2 (v1, bf16) – Parallel span replacement with <swi> in loss
# =============================================================================
# All <swi>…</swi> spans are replaced simultaneously.
# At stage k, each span gets k × c_thought <latent> tokens.
#
# Difference from Phase 2-1:
#   phase2-1: stage k → leftmost k spans replaced, each with c_thought latents
#   phase2-2: stage k → ALL spans replaced, each with k*c_thought latents
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
PHASE1_CKPT=${PHASE1_CKPT:-$PROJECT_ROOT/outputs/phase2_1_v4_bf16/stage_010}
OUT=${OUT:-$PROJECT_ROOT/outputs/phase2_2_v1_bf16}
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
echo "Phase 2-2 (v1 bf16: parallel span replacement, DDP multi-GPU)"
echo "============================================================"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NPROC=$NPROC"
echo "PHASE1_CKPT=$PHASE1_CKPT"
echo "OUT=$OUT"
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

# ===== Phase2-2：并行span替换 =====
# Key design:
#   - phase2_strategy=phase2-2: all spans replaced in parallel
#   - c_thought=2: at stage k, each span gets k*2 latent tokens
#   - max_latent_stage=8: maximum stage index (up to 16 latent per span)
#   - max_total_latent=48: hard cap on total latent tokens per sample,
#     prevents OOM from samples with many spans (e.g. 12 spans × 16 = 192
#     would be capped to 48, i.e. first ~3 spans get latent, rest keep text)
#   - epochs_per_stage=3: thorough training at each stage
#   - stage0_epochs=3: SFT warm-up (no latent replacement)
#   - batch_size=1, grad_accum=8: effective batch = 1 * 8 * 8 = 64
USE_MODELSCOPE="$USE_MODELSCOPE" MODELSCOPE_MODEL_ID="$MODELSCOPE_MODEL_ID" BASE_MODEL_SOURCE=modelscope REPORT_TO="$REPORT_TO" \
bash "$PROJECT_ROOT/scripts/train_phase2.sh" \
  "$NPROC" \
  "$PHASE1_CKPT" \
  "$OUT" \
  --phase2_strategy phase2-2 \
  --max_latent_stage 8 \
  --c_thought 2 \
  --max_total_latent 64 \
  --run_name swilatent_phase2-2_v1_bf16 \
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
