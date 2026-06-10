#!/bin/bash
# =============================================================================
# Phase 1 – SFT Training via LLaMA-Factory
# =============================================================================
# Trains Qwen3-8B with LoRA on math_openr1_swi_sft (9030 math problems with
# <swi>...</swi> annotated reasoning).  Goal: model learns to emit <swi></swi>
# tags around compressible CoT segments.
#
# Prerequisites:
#   conda activate llamafac
#   python scripts/preprocess_sft_data.py   # already done
#
# Usage:
#   bash scripts/run_stage1_sft.sh [CUDA_DEVICES] [OUTPUT_DIR]
#   bash scripts/run_stage1_sft.sh 0,1 /data/jiayu/SwiLatent/outputs/phase1_sft
#
# Defaults: GPU=0, save to /data/jiayu/SwiLatent/outputs/phase1_sft
# =============================================================================
set -e

CUDA_DEVICES=${1:-"1"}
OUTPUT_DIR=${2:-"/data/jiayu/SwiLatent/outputs/phase1_sft"}
MODEL_PATH="${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218"
DATASET_DIR="/data/jiayu/LLaMA-Factory/data"

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_DISABLED=true
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

echo "=== Phase 1 SFT (QLoRA, single GPU) ==="
echo "Model      : $MODEL_PATH"
echo "Dataset    : math_openr1_swi_sft (9030 samples)"
echo "Output dir : $OUTPUT_DIR"
echo "GPU(s)     : $CUDA_DEVICES"
echo ""

llamafactory-cli train \
    --stage sft \
    --do_train True \
    --model_name_or_path "$MODEL_PATH" \
    --dataset math_openr1_swi_sft \
    --dataset_dir "$DATASET_DIR" \
    --template qwen3 \
    --add_special_tokens "<swi>,</swi>" \
    --finetuning_type lora \
    --lora_r 32 \
    --lora_alpha 64 \
    --lora_target all \
    --quantization_bit 4 \
    --quantization_type nf4 \
    --double_quantization True \
    --output_dir "$OUTPUT_DIR" \
    --resume_from_checkpoint "$OUTPUT_DIR/checkpoint-1400" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --optim paged_adamw_32bit \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.05 \
    --logging_steps 10 \
    --save_steps 200 \
    --save_total_limit 3 \
    --learning_rate 1e-4 \
    --num_train_epochs 3 \
    --max_length 4096 \
    --plot_loss \
    --bf16 \
    --report_to swanlab \
    --run_name "swilatent_phase1_sft_qwen3_8b"