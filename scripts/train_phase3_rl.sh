#!/bin/bash
# =============================================================================
# Phase 3 – verl GRPO RL Training for SwiLatent
# =============================================================================
# Prerequisites:
#   1. Merge Phase 2 LoRA → standalone model:
#      python scripts/merge_lora.py \
#          --base_model <base_model_path> \
#          --adapter_path outputs/phase2_2_v1_bf16/stage_008 \
#          --output_dir outputs/phase2_2_merged
#
#   2. Prepare data:
#      python scripts/prep_verl_data.py \
#          --input_file data_hf/data/math_openr1_all.jsonl \
#          --output_dir data_verl
#
#   3. Activate verl conda environment:
#      conda activate verl
# =============================================================================
set -ex

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}

# ===== Paths =====
MODEL_PATH=${MODEL_PATH:-$PROJECT_ROOT/outputs/phase2_2_merged}
TRAIN_DATA=${TRAIN_DATA:-$PROJECT_ROOT/data_verl/train.parquet}
VAL_DATA=${VAL_DATA:-$PROJECT_ROOT/data_verl/test.parquet}
REWARD_FN_PATH=${REWARD_FN_PATH:-$PROJECT_ROOT/src/rl/reward.py}

# ===== Hardware =====
N_GPUS=${N_GPUS:-8}
TP_SIZE=${TP_SIZE:-2}

# ===== Hyperparameters =====
TOTAL_EPOCHS=${TOTAL_EPOCHS:-15}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-128}
ROLLOUT_N=${ROLLOUT_N:-5}
LR=${LR:-1e-6}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}

echo "============================================================"
echo "Phase 3: verl GRPO RL Training"
echo "============================================================"
echo "  Model      : $MODEL_PATH"
echo "  Train data : $TRAIN_DATA"
echo "  Val data   : $VAL_DATA"
echo "  Reward fn  : $REWARD_FN_PATH"
echo "  GPUs       : $N_GPUS (TP=$TP_SIZE)"
echo "  Epochs     : $TOTAL_EPOCHS"
echo "  Batch size : $TRAIN_BATCH_SIZE"
echo "  Rollout N  : $ROLLOUT_N"
echo "  LR         : $LR"
echo "  LoRA       : rank=$LORA_RANK alpha=$LORA_ALPHA"
echo "============================================================"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.lora_rank=$LORA_RANK \
    actor_rollout_ref.model.lora_alpha=$LORA_ALPHA \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=$LR \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_SIZE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=$ROLLOUT_N \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.layered_summon=True \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    reward.custom_reward_function.path="$REWARD_FN_PATH" \
    reward.custom_reward_function.name=compute_score \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=swilatent_phase3_rl \
    trainer.experiment_name=grpo_v1 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=$TOTAL_EPOCHS \
    trainer.val_before_train=True \
    "$@"
