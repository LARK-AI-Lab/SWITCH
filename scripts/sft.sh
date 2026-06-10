set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen3_8b_sft_peft_sp2_npu.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=/data/jiayu/SwiReasoning/logs/sft_latents/gsm8k_teacher_128_sft.pt \
    data.val_files=/data/jiayu/SwiReasoning/data/datasets/gsm8k/test-00000-of-00001.parquet \
    data.max_length=2048 \
    data.truncation=right \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.train_batch_size=3 \
    data.micro_batch_size_per_gpu=1 \
    optim.lr=1e-4 \
    model.partial_pretrain=Qwen/Qwen3-8B \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear \
    model.fsdp_config.model_dtype=bf16 \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    trainer.default_local_dir=$save_path \
    trainer.project_name=gsm8k-sft \
    trainer.experiment_name=gsm8k-sft-qwen3-8b-instruct \
    trainer.logger=[console,swanlab] \
    trainer.total_epochs=2 \
    trainer.device=cuda \
    $@