#!/bin/bash
# =============================================================================
# Evaluate Phase 2-1 with LATENT INFERENCE on GSM8K and MATH-500
# =============================================================================
# Unlike run_eval_phase2.sh (standard generation), this uses CoconutSwiModel
# to perform hidden-state injection during <swi> regions.
# =============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

ADAPTER_PATH=${ADAPTER_PATH:-$PROJECT_ROOT/outputs/phase2_1_v4_bf16/stage_010}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

# Resolve base model via ModelScope if needed
USE_MODELSCOPE=${USE_MODELSCOPE:-1}
MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-8B}

BASE_MODEL_FLAG=""
if [ "$USE_MODELSCOPE" = "1" ]; then
    echo "[eval] Resolving base model via ModelScope: $MODELSCOPE_MODEL_ID"
    _TMPFILE=$(mktemp)
    python3 -c "
from modelscope.hub.snapshot_download import snapshot_download
p = snapshot_download('${MODELSCOPE_MODEL_ID}')
with open('${_TMPFILE}', 'w') as f:
    f.write(p)
"
    BASE_MODEL_PATH=$(cat "$_TMPFILE")
    rm -f "$_TMPFILE"
    echo "[eval] Base model path: $BASE_MODEL_PATH"
    BASE_MODEL_FLAG="--base_model $BASE_MODEL_PATH"
fi

echo "============================================================"
echo "Phase 2-1 LATENT INFERENCE Evaluation"
echo "============================================================"
echo "  Adapter       : $ADAPTER_PATH"
echo "  GPUs          : $NPROC"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "============================================================"

# ===== GSM8K =====
echo ""
echo ">>> [1/2] GSM8K — Latent Inference"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/scripts/eval_latent.py" \
    --adapter_path "$ADAPTER_PATH" \
    $BASE_MODEL_FLAG \
    --dataset gsm8k \
    --max_new_tokens "$MAX_NEW_TOKENS"

# ===== MATH-500 =====
echo ""
echo ">>> [2/2] MATH-500 — Latent Inference"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/scripts/eval_latent.py" \
    --adapter_path "$ADAPTER_PATH" \
    $BASE_MODEL_FLAG \
    --dataset math500 \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo ""
echo "============================================================"
echo "All latent inference evaluations finished."
echo "Results saved to $PROJECT_ROOT/logs/"
echo "============================================================"
