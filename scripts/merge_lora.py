"""Merge Phase 2 LoRA adapter into a standalone model for vLLM.

vLLM cannot load LoRA adapters with ``modules_to_save`` (resized
embeddings). This script merges the adapter weights into the base model
and saves a full-weight checkpoint that vLLM can load directly.

Usage:
    python scripts/merge_lora.py \
        --base_model /root/.cache/modelscope/hub/models/Qwen/Qwen3-8B \
        --adapter_path outputs/phase2_2_v1_bf16/stage_008 \
        --output_dir outputs/phase2_2_merged
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--bf16", action="store_true", default=True)
    args = parser.parse_args()

    print(f"Loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path, trust_remote_code=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    if len(tokenizer) != model.config.vocab_size:
        print(f"Resizing embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    print(f"Loading adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(model, args.adapter_path)

    print("Merging LoRA weights...")
    model = model.merge_and_unload()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Saving merged model to {args.output_dir}")
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    print(f"Done. Model saved to {args.output_dir}")
    print(f"  Vocab size: {model.config.vocab_size}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")


if __name__ == "__main__":
    main()
