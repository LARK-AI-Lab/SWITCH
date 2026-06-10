#!/bin/bash
# =============================================================================
# Phase 3 – Custom GRPO RL with Latent Inference
# =============================================================================
# Uses CoconutSwiModel.generate_rl() for rollout with actual hidden-state
# injection, ensuring training-evaluation alignment.
#
# Usage:
#   conda activate qwen35_swift
#   cd .
#   USE_MODELSCOPE=1 bash scripts/train_phase3_grpo.sh
# =============================================================================
set -e

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ===== GPU =====
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ===== Paths =====
NPROC=${NPROC:-8}
MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/outputs/phase2_2_v1_bf16/stage_007}
DATA_FILE=${DATA_FILE:-$PROJECT_ROOT/data_hf/data/math_openr1_all.jsonl}
SAVE_DIR=${SAVE_DIR:-$PROJECT_ROOT/outputs/phase3_grpo_v1}
USE_MODELSCOPE=${USE_MODELSCOPE:-1}
MODELSCOPE_MODEL_ID=${MODELSCOPE_MODEL_ID:-Qwen/Qwen3-8B}

# ===== Hyperparameters =====
GROUP_SIZE=${GROUP_SIZE:-5}
MIN_LATENT_STEPS=${MIN_LATENT_STEPS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.95}
LATENT_NOISE_SCALE=${LATENT_NOISE_SCALE:-0.0}
RECORD_LATENT_ACTIONS=${RECORD_LATENT_ACTIONS:-0}
LATENT_REPLAY=${LATENT_REPLAY:-0}
LR=${LR:-1e-6}
EPOCHS=${EPOCHS:-3}
GRAD_ACCUM=${GRAD_ACCUM:-1}
CLIP_EPS=${CLIP_EPS:-0.2}
KL_COEF=${KL_COEF:-0.001}
SAVE_STEPS=${SAVE_STEPS:-200}
LOGGING_STEPS=${LOGGING_STEPS:-10}
REPORT_TO=${REPORT_TO:-swanlab}
RUN_NAME=${RUN_NAME:-swilatent_phase3_grpo_v1}

echo "============================================================"
echo "Phase 3: GRPO RL with Latent Inference"
echo "============================================================"
echo "  Model          : $MODEL_PATH"
echo "  Data           : $DATA_FILE"
echo "  Save dir       : $SAVE_DIR"
echo "  GPUs           : $NPROC"
echo "  Group size (G) : $GROUP_SIZE"
echo "  Min latent     : $MIN_LATENT_STEPS steps"
echo "  Temperature    : $TEMPERATURE"
echo "  Latent noise   : $LATENT_NOISE_SCALE"
echo "  Latent replay  : $LATENT_REPLAY"
echo "  LR             : $LR"
echo "  KL coef        : $KL_COEF"
echo "  Epochs         : $EPOCHS"
echo "  Grad accum     : $GRAD_ACCUM"
echo "============================================================"

USE_MODELSCOPE="$USE_MODELSCOPE" \
MODELSCOPE_MODEL_ID="$MODELSCOPE_MODEL_ID" \
torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="$NPROC" \
    "$PROJECT_ROOT/train_phase3_grpo.py" \
    --model_path "$MODEL_PATH" \
    --data_file "$DATA_FILE" \
    --save_dir "$SAVE_DIR" \
    --group_size "$GROUP_SIZE" \
    --min_latent_steps "$MIN_LATENT_STEPS" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --latent_noise_scale "$LATENT_NOISE_SCALE" \
    --lr "$LR" \
    --epochs "$EPOCHS" \
    --grad_accum "$GRAD_ACCUM" \
    --clip_eps "$CLIP_EPS" \
    --kl_coef "$KL_COEF" \
    --save_steps "$SAVE_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_NAME" \
    --bf16 \
    $(if [ "$RECORD_LATENT_ACTIONS" = "1" ]; then echo "--record_latent_actions"; fi) \
    $(if [ "$LATENT_REPLAY" = "1" ]; then echo "--latent_replay"; fi)
