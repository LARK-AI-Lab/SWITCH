#!/bin/bash
# =============================================================================
# Evaluate Phase 2-1 (v4 bf16) checkpoint on GSM8K and MATH-500
# =============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ===== Configuration =====
ADAPTER_PATH=${ADAPTER_PATH:-$PROJECT_ROOT/outputs/phase2_1_v4_bf16/stage_010}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
NPROC=${NPROC:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-4096}

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export TOKENIZERS_PARALLELISM=false

# Resolve base model via ModelScope if needed
USE_MODELSCOPE=${USE_MODELSCOPE:-1}
MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-8B}

BASE_MODEL_PATH=""
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
fi

BASE_MODEL_FLAG=""
if [ -n "$BASE_MODEL_PATH" ] && [ -d "$BASE_MODEL_PATH" ]; then
    BASE_MODEL_FLAG="--base_model $BASE_MODEL_PATH"
fi

echo "============================================================"
echo "Phase 2-1 Evaluation"
echo "============================================================"
echo "  Adapter       : $ADAPTER_PATH"
echo "  Base model    : ${BASE_MODEL_PATH:-auto-detect from adapter_config}"
echo "  GPUs          : $NPROC x (CUDA_VISIBLE_DEVICES=$GPU_IDS)"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "============================================================"

# ===== GSM8K Evaluation =====
echo ""
echo ">>> [1/2] GSM8K Evaluation (full 1319 samples)"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/scripts/eval_gsm8k.py" \
    --adapter_path "$ADAPTER_PATH" \
    $BASE_MODEL_FLAG \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo ""
echo ">>> GSM8K evaluation complete."
echo ""

# ===== MATH-500 Evaluation =====
echo ""
echo ">>> [2/2] MATH-500 Evaluation (full 500 samples)"
echo ""

torchrun --standalone --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/scripts/eval_math500.py" \
    --adapter_path "$ADAPTER_PATH" \
    $BASE_MODEL_FLAG \
    --max_new_tokens "$MAX_NEW_TOKENS"

echo ""
echo ">>> MATH-500 evaluation complete."
echo ""
echo "============================================================"
echo "All evaluations finished. Results saved to $PROJECT_ROOT/logs/"
echo "============================================================"
