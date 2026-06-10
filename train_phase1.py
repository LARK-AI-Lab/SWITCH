"""
Phase 1 – SFT Training for <swi> Tag Learning
===============================================
Trains Qwen3-8B with LoRA (bf16) on math problems whose solutions are
annotated with <swi>...</swi> tags marking compressible reasoning segments.

Uses HuggingFace Trainer with native DDP for multi-GPU training.
No 4-bit quantization — bf16 throughout for clean DDP compatibility.

Usage
-----
    # Single GPU
    python train_phase1.py --model_path Qwen/Qwen3-8B \
        --data_file data_hf/data/math_openr1_all.jsonl \
        --output_dir outputs/phase1_bf16

    # Multi-GPU (8 GPUs)
    torchrun --standalone --nproc_per_node=8 train_phase1.py \
        --model_path Qwen/Qwen3-8B \
        --data_file data_hf/data/math_openr1_all.jsonl \
        --output_dir outputs/phase1_bf16
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

SWI_START_TOKEN = "<swi>"
SWI_END_TOKEN = "</swi>"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SwiSFTDataset(Dataset):
    """Tokenised SFT dataset.

    Each sample is a (user-question, assistant-response) pair formatted with
    the tokenizer's chat template.  Instruction tokens are masked (-100) so
    the model only learns to predict the response.
    """

    def __init__(
        self,
        data_path: str,
        tokenizer: Any,
        max_length: int = 4096,
    ) -> None:
        with open(data_path) as f:
            raw = [json.loads(line) for line in f if line.strip()]
        self.data = [d for d in raw if d.get("is_correct", True)]
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Pre-compute template kwargs once — Qwen3 may support enable_thinking
        self._template_kwargs: dict[str, Any] = {}
        try:
            tokenizer.apply_chat_template(
                [{"role": "user", "content": "hi"}],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            self._template_kwargs["enable_thinking"] = False
        except (TypeError, ValueError, KeyError):
            pass

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        instruction = item["instruction"]
        output = item["output"]

        # Full conversation (prompt + response)
        full_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": output},
        ]
        # Get formatted text, then tokenize manually (robust across transformers versions)
        full_text = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
            **self._template_kwargs,
        )
        full_ids = self.tokenizer.encode(
            full_text, add_special_tokens=False, truncation=True,
            max_length=self.max_length,
        )

        prompt_messages = [{"role": "user", "content": instruction}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
            **self._template_kwargs,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Labels: mask the prompt portion with -100
        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(labels))
        for i in range(prompt_len):
            labels[i] = -100

        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Model setup helpers
# ---------------------------------------------------------------------------
class SFTCollator:
    """Simple padding collator that avoids tokenizer.pad() compatibility issues."""

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict]) -> dict:
        max_len = max(f["input_ids"].size(0) for f in features)
        input_ids, attention_mask, labels = [], [], []
        for f in features:
            pad_len = max_len - f["input_ids"].size(0)
            input_ids.append(torch.cat([
                f["input_ids"],
                torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
            ]))
            attention_mask.append(torch.cat([
                f["attention_mask"],
                torch.zeros(pad_len, dtype=torch.long),
            ]))
            labels.append(torch.cat([
                f["labels"],
                torch.full((pad_len,), -100, dtype=torch.long),
            ]))
        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def _init_new_token_embeddings(model, tokenizer, token_ids: list[int]) -> None:
    """Copy an anchor token's embedding into newly added tokens."""
    anchor_id = tokenizer.convert_tokens_to_ids("<<")
    if anchor_id == tokenizer.unk_token_id:
        anchor_id = tokenizer.convert_tokens_to_ids("<")
    if anchor_id == tokenizer.unk_token_id:
        anchor_id = tokenizer.pad_token_id

    embed = model.get_input_embeddings()
    anchor_vec = embed.weight.data[anchor_id].clone()
    for tid in token_ids:
        embed.weight.data[tid] = anchor_vec

    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        if embed.weight.data_ptr() != lm_head.weight.data_ptr():
            anchor_logit = lm_head.weight.data[anchor_id].clone()
            for tid in token_ids:
                lm_head.weight.data[tid] = anchor_logit


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 1 – SFT for <swi> tag learning")

    p.add_argument("--model_path", type=str, required=True,
                   help="HF model id or local path to Qwen3-8B.")
    p.add_argument("--data_file", type=str, required=True,
                   help="JSONL with instruction/input/output fields.")
    p.add_argument("--output_dir", type=str, required=True)

    # Training hyper-parameters
    p.add_argument("--num_train_epochs", type=int, default=3)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=4096)

    # LoRA
    p.add_argument("--lora_r", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Infrastructure
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_name", type=str, default="swilatent_phase1_sft_bf16")
    p.add_argument("--report_to", type=str, default="none",
                   choices=["none", "swanlab", "wandb", "tensorboard"])

    return p.parse_args()


def main() -> None:
    args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    # ── 1. Tokenizer ──────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_added = tokenizer.add_tokens(
        [SWI_START_TOKEN, SWI_END_TOKEN], special_tokens=True,
    )
    swi_start_id = tokenizer.convert_tokens_to_ids(SWI_START_TOKEN)
    swi_end_id = tokenizer.convert_tokens_to_ids(SWI_END_TOKEN)
    if local_rank == 0:
        print(f"[Phase 1] Added {n_added} tokens  "
              f"swi_start={swi_start_id}  swi_end={swi_end_id}")

    # ── 2. Model (bf16, NO quantization) ──────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    _init_new_token_embeddings(model, tokenizer, [swi_start_id, swi_end_id])

    # ── 3. LoRA with modules_to_save ──────────────────────────────────────
    target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        modules_to_save=["embed_tokens", "lm_head"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()

    if local_rank == 0:
        model.print_trainable_parameters()

    # ── 4. Dataset ────────────────────────────────────────────────────────
    dataset = SwiSFTDataset(args.data_file, tokenizer, args.max_length)
    if local_rank == 0:
        print(f"[Phase 1] Dataset: {len(dataset)} samples")

    # ── 5. Trainer ────────────────────────────────────────────────────────
    callbacks = []
    hf_report_to = args.report_to
    if args.report_to == "swanlab":
        hf_report_to = "none"
        try:
            from swanlab.integration.huggingface import SwanLabCallback
            callbacks.append(SwanLabCallback(
                project="SwiLatent",
                experiment_name=args.run_name,
                config=vars(args),
            ))
            if local_rank == 0:
                print("[Phase 1] SwanLab callback enabled")
        except ImportError:
            if local_rank == 0:
                print("[Phase 1] swanlab not installed, skipping. "
                      "Install with: pip install swanlab")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=4,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        seed=args.seed,
        run_name=args.run_name,
        report_to=hf_report_to,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=SFTCollator(pad_token_id=tokenizer.pad_token_id),
        callbacks=callbacks,
    )

    # ── 6. Train ──────────────────────────────────────────────────────────
    trainer.train()

    # ── 7. Save final checkpoint ──────────────────────────────────────────
    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    if local_rank == 0:
        print(f"[Phase 1] Training complete ✓  →  {final_dir}")


if __name__ == "__main__":
    main()
