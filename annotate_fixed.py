"""
Fixed version of annotate_gsm8k.py that correctly handles GSM8K format.

Key fixes:
1. Treats entire solution as "thinking" content (no need for <think> tags)
2. Ensures final answer (#### X) is NOT wrapped in <swi> tags
3. Properly applies entropy-based switching logic
"""
import argparse
import json
import os
from glob import glob
from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_utils import set_seed, get_math_symbols_ids
import torch.nn.functional as F


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float('-inf'), logits)
    return logits


def annotate_teacher_forcing_fixed(
    model, 
    tokenizer, 
    input_ids, 
    attention_mask, 
    target_ids,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    min_p=0.0,
    alpha_0=1.0,
    beta_0=0.7,
    window_size=512,
    math_ids_tensor=None,
    switch_start_token="<swi>",
    switch_end_token="</swi>",
):
    """
    Fixed teacher forcing annotation for GSM8K-style data.
    
    Key changes:
    1. No assumption about <think> tags - treats entire target as reasoning
    2. Protects final answer (#### X) from being wrapped in <swi>
    3. Properly tracks entropy changes to decide when to use latent mode
    """
    if input_ids.dim() != 2 or input_ids.size(0) != 1:
        raise ValueError("input_ids must have shape (1, seq_len)")
    if target_ids.dim() != 2 or target_ids.size(0) != 1:
        raise ValueError("target_ids must have shape (1, seq_len)")

    device = input_ids.device
    if math_ids_tensor is not None:
        math_ids_tensor = math_ids_tensor.to(device)

    target_tokens = target_ids[0].tolist()
    total_tokens = len(target_tokens)
    if total_tokens == 0:
        return ""

    # Find where final answer starts (#### pattern)
    target_text = tokenizer.decode(target_tokens, skip_special_tokens=False)
    answer_start_idx = None
    if "####" in target_text:
        # Find token index where #### appears
        for i, token_id in enumerate(target_tokens):
            decoded_so_far = tokenizer.decode(target_tokens[:i+1], skip_special_tokens=False)
            if "####" in decoded_so_far and answer_start_idx is None:
                # Mark from this token onwards as "final answer region"
                answer_start_idx = i
                break

    E = model.get_input_embeddings().weight
    newline_token_id = tokenizer.convert_tokens_to_ids("\n")
    if newline_token_id is None:
        newline_token_id = tokenizer.eos_token_id

    # Mode: 0=soft (latent), 1=normal (direct output)
    mode = torch.ones(1, dtype=torch.long, device=device)  # Start in normal mode
    mode_stay_steps = torch.zeros(1, dtype=torch.long, device=device)

    tokens = []
    soft_flags = []

    attention_mask_local = attention_mask.clone()
    past_key_values = None
    last_emb = None
    cur_ref_entropy = None

    with torch.no_grad():
        for step, token_id in enumerate(target_tokens):
            # CRITICAL: Never mark final answer tokens as soft
            is_final_answer_region = (answer_start_idx is not None and step >= answer_start_idx)

            # Prepare model inputs
            if past_key_values is None:
                model_inputs = {
                    "input_ids": input_ids.clone(),
                    "attention_mask": attention_mask_local,
                }
            else:
                attention_mask_new = torch.ones((1, 1), dtype=attention_mask_local.dtype, device=device)
                attention_mask_local = torch.cat([attention_mask_local, attention_mask_new], dim=1)
                model_inputs = {
                    "inputs_embeds": last_emb.unsqueeze(1),
                    "attention_mask": attention_mask_local,
                    "past_key_values": past_key_values,
                }

            # Forward pass
            outputs = model(**model_inputs, use_cache=True)
            past_key_values = outputs.past_key_values
            logits_original = outputs.logits[:, -1, :]
            probs_original = F.softmax(logits_original, dim=-1)
            
            # Apply sampling filter
            logits = logits_original / temperature
            logits_filtered = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probs = F.softmax(logits_filtered, dim=-1)

            # Compute entropy (higher = more uncertain = should use latent)
            cur_entropy = -(probs_original * (probs_original.clamp(min=1e-12).log())).sum(dim=-1)

            # Mode switching logic based on entropy
            if step == 0:
                cur_ref_entropy = cur_entropy.clone()
                to_soft = torch.zeros_like(mode, dtype=torch.bool)
                to_normal = torch.zeros_like(mode, dtype=torch.bool)
            else:
                mode_stay_steps += 1
                allow_switch = (mode_stay_steps >= window_size)

                # High entropy → switch to soft (latent) mode
                # Low entropy → switch to normal mode
                to_soft = (mode == 1) & (cur_entropy > cur_ref_entropy) & allow_switch
                to_normal = (mode == 0) & (cur_entropy < cur_ref_entropy)

                # Apply mode transitions
                mode[to_soft] = 0  # Enter latent mode
                mode[to_normal] = 1  # Exit to normal mode
                mode_stay_steps[to_normal | to_soft] = 0
                cur_ref_entropy[to_normal | to_soft] = cur_entropy[to_normal | to_soft]

            # Determine if this token should use soft embedding
            is_soft = (mode == 0)
            
            # Override: math tokens and final answer always use normal mode
            if math_ids_tensor is not None:
                next_token_tensor = torch.tensor([token_id], device=device)
                is_math_token = (next_token_tensor.unsqueeze(-1) == math_ids_tensor).any(dim=-1)
                if is_math_token.item():
                    is_soft = torch.zeros_like(is_soft)
            
            if is_final_answer_region:
                is_soft = torch.zeros_like(is_soft)

            # Prepare embeddings
            normal_emb = E[token_id]
            soft_emb = torch.matmul(probs_original, E)

            # Mix embeddings for smooth transitions
            alpha = alpha_0 + (1 - alpha_0) * float(step) / float(total_tokens)
            beta = beta_0 + (1 - beta_0) * float(step) / float(total_tokens)

            if step == 0:
                soft_emb = 0.9 * soft_emb + 0.1 * E[newline_token_id]

            # Smooth transition when switching modes
            if to_soft.item():
                # Entering soft mode: mix with "thinking start" signal
                soft_emb = alpha * soft_emb + (1 - alpha) * E[newline_token_id]
            
            if to_normal.item():
                # Exiting soft mode: mix with "thinking end" signal
                normal_emb = beta * normal_emb + (1 - beta) * E[newline_token_id]

            # Choose embedding based on mode
            last_emb = torch.where(is_soft[:, None], soft_emb, normal_emb)

            # Record this token's status
            tokens.append(token_id)
            soft_flags.append(is_soft.item() and not is_final_answer_region)

    # Build annotated string
    annotated = []
    segment_tokens = []
    prev_flag = None

    def flush_segment(flag, segment):
        if not segment:
            return
        text = tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if flag:
            annotated.append(f"{switch_start_token}{text}{switch_end_token}")
        else:
            annotated.append(text)

    for token_id, flag in zip(tokens, soft_flags):
        if prev_flag is None:
            prev_flag = flag
        elif flag != prev_flag:
            flush_segment(prev_flag, segment_tokens)
            segment_tokens = []
            prev_flag = flag
        segment_tokens.append(token_id)

    flush_segment(prev_flag, segment_tokens)

    return "".join(annotated)


def build_prompt(tokenizer, question: str) -> str:
    """Build prompt without <think> tags for GSM8K."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant. Solve the math problem step by step."},
        {"role": "user", "content": question.strip()}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output_path", type=str, default="logs/gsm8k_swir_fixed.jsonl")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.8, help="Soft embedding mixing coefficient")
    parser.add_argument("--beta", type=float, default=0.7, help="Normal embedding mixing coefficient")
    parser.add_argument("--window_size", type=int, default=10, help="Min tokens before allowing mode switch")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--switch_start_token", type=str, default="<swi>")
    parser.add_argument("--switch_end_token", type=str, default="</swi>")
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device_map = {"": args.device}
    else:
        device_map = None

    print(f"Loading tokenizer and model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map=device_map,
    )
    device = model.device
    model.eval()

    math_ids_tensor = torch.tensor(list(get_math_symbols_ids(tokenizer)), device=device)

    # Load dataset
    if args.data_dir:
        pattern = os.path.join(args.data_dir, f"{args.split}-*.parquet")
        matching_files = sorted(glob(pattern))
        if not matching_files:
            raise FileNotFoundError(f"No parquet files found: {pattern}")
        dataset = load_dataset("parquet", data_files={"train": matching_files}, split="train")
    else:
        dataset = load_dataset("gsm8k", "main", split=args.split)
    
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = len(dataset)
    print(f"Processing {total} samples...")

    with open(args.output_path, "w", encoding="utf-8") as writer:
        for idx in range(total):
            sample = dataset[idx]
            question = sample["question"].strip()
            solution_text = str(sample["answer"]).strip()
            
            prompt = build_prompt(tokenizer, question)

            # Tokenize prompt
            model_inputs = tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
            ).to(device)

            # Tokenize target solution
            solution_tensor = tokenizer(
                solution_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"].to(device)

            if solution_tensor.numel() == 0:
                annotated_solution = solution_text
            else:
                annotated_solution = annotate_teacher_forcing_fixed(
                    model,
                    tokenizer,
                    input_ids=model_inputs["input_ids"],
                    attention_mask=model_inputs["attention_mask"],
                    target_ids=solution_tensor,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    alpha_0=args.alpha,
                    beta_0=args.beta,
                    window_size=args.window_size,
                    math_ids_tensor=math_ids_tensor,
                    switch_start_token=args.switch_start_token,
                    switch_end_token=args.switch_end_token,
                )

            record = {
                "instruction": question,
                "input": "",
                "output": annotated_solution,
            }
            writer.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (idx + 1) % 10 == 0 or (idx + 1) == total:
                print(f"Processed {idx + 1} / {total}")
                # Print example
                if (idx + 1) <= 5:
                    print(f"\n=== Example {idx + 1} ===")
                    print(f"Question: {question[:100]}...")
                    print(f"Annotated: {annotated_solution[:200]}...")
                    print()

    print(f"\n✓ Saved annotations to {args.output_path}")
    print(f"\nNext steps:")
    print(f"1. Inspect the output to verify <swi> placement")
    print(f"2. Convert to Stage 2 format: python convert_to_latent_format.py --input {args.output_path} --output gsm8k_stage2.json")


if __name__ == "__main__":
    main()
