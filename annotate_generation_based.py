"""
Generation-Based Annotation for SwiReasoning + Coconut.

Key insight: Instead of teacher forcing, let the model GENERATE its own solution,
track entropy during generation, and then annotate <swi> where entropy was high.

This avoids distribution mismatch and produces more natural <swi> placement.
"""
import argparse
import json
import os
from glob import glob
from datasets import load_dataset
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from generation_utils import set_seed, get_math_symbols_ids, apply_sampling_filter
from tqdm import tqdm
import re


def extract_answer(text):
    """Extract final numerical answer from various formats."""
    # Try #### format first (GSM8K style)
    match = re.search(r'####\s*(-?\d+(?:[.,]\d+)?)', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try \boxed{} format (LaTeX style)
    match = re.search(r'\\boxed\{([^}]*?(-?\d+(?:[.,]\d+)?)[^}]*?)\}', text)
    if match:
        # Extract just the number from the boxed content
        num_match = re.search(r'(-?\d+(?:[.,]\d+)?)', match.group(1))
        if num_match:
            return num_match.group(1).replace(',', '')

    # Try \fbox{} format (LaTeX style)
    match = re.search(r'\\fbox\{([^}]*?(-?\d+(?:[.,]\d+)?)[^}]*?)\}', text)
    if match:
        num_match = re.search(r'(-?\d+(?:[.,]\d+)?)', match.group(1))
        if num_match:
            return num_match.group(1).replace(',', '')
    
    # Try **$X** format
    match = re.search(r'\*\*\$?(-?\d+(?:[.,]\d+)?)\$?\*\*', text)
    if match:
        return match.group(1).replace(',', '')
    
    # Try to find last number in the text as fallback
    matches = re.findall(r'(-?\d+(?:[.,]\d+)?)', text)
    if matches:
        return matches[-1].replace(',', '')
    
    return None


def generate_with_entropy_tracking_batch(
    model,
    tokenizer,
    input_ids,
    attention_mask,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    max_new_tokens=2048,
):
    """
    Batched generation with entropy tracking.

    Returns:
        tokens_list: list of token-id lists for each sample
        entropies_list: list of entropy lists for each sample
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    tokens_list = [[] for _ in range(batch_size)]
    entropies_list = [[] for _ in range(batch_size)]

    past_key_values = None
    generated = input_ids.clone()
    attn_mask = attention_mask.clone()
    unfinished_idx = list(range(batch_size))

    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attn_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device)
            attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1)
            model_inputs = {
                "input_ids": next_tokens.unsqueeze(1),
                "past_key_values": past_key_values,
                "attention_mask": attn_mask,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        logits = outputs.logits[:, -1, :]

        # Compute entropy
        probs_for_entropy = F.softmax(logits / temperature, dim=-1)
        entropy = -(probs_for_entropy * (probs_for_entropy + 1e-12).log()).sum(dim=-1)

        # Apply sampling
        logits_filtered = apply_sampling_filter(
            logits / temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
        )
        probs_filtered = F.softmax(logits_filtered, dim=-1)

        if probs_filtered.shape[0] == 0:
            break

        next_tokens = torch.multinomial(probs_filtered, num_samples=1).squeeze(-1)

        for bi, orig in enumerate(unfinished_idx):
            token_id = next_tokens[bi].item()
            tokens_list[orig].append(token_id)
            entropies_list[orig].append(entropy[bi].item())

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break

        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attn_mask = attn_mask[keep_idx]

        if hasattr(past_key_values, "batch_select_indices"):
            keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=device)
            past_key_values.batch_select_indices(keep_idx_tensor)

    return tokens_list, entropies_list


def annotate_with_swir_logic(
    tokens,
    entropies,
    tokenizer,
    window_size=10,
    entropy_threshold_percentile=60,
    min_swi_length=3,
    protect_answer_region=True,
):
    """
    Annotate tokens with <swi> based on SWIR-style entropy analysis.
    
    Strategy:
    1. Identify high-entropy regions (uncertain/complex reasoning)
    2. Mark consecutive high-entropy tokens as candidates for <swi>
    3. Filter out short regions and protect final answer
    
    Args:
        tokens: list of token IDs
        entropies: list of entropy values
        window_size: smoothing window for entropy
        entropy_threshold_percentile: percentile threshold for "high entropy"
        min_swi_length: minimum tokens to wrap in <swi>
        protect_answer_region: don't mark final answer with <swi>
    """
    if len(tokens) == 0:
        return []
    
    # Decode to find answer region (protect both #### and \boxed{})
    full_text = tokenizer.decode(tokens, skip_special_tokens=False)
    answer_start_idx = None
    
    if protect_answer_region:
        # Find token index where answer markers appear
        answer_markers = ["####", "\\boxed", "\\fbox", "**Answer**", "Final Answer"]
        for marker in answer_markers:
            if marker in full_text:
                for i in range(len(tokens)):
                    decoded_so_far = tokenizer.decode(tokens[:i+1], skip_special_tokens=False)
                    if marker in decoded_so_far:
                        answer_start_idx = i
                        break
                if answer_start_idx is not None:
                    break
    
    # Compute smoothed entropy (moving average)
    smoothed_entropy = []
    for i in range(len(entropies)):
        start = max(0, i - window_size // 2)
        end = min(len(entropies), i + window_size // 2 + 1)
        avg = sum(entropies[start:end]) / (end - start)
        smoothed_entropy.append(avg)
    
    # Determine threshold (adaptive based on distribution)
    sorted_entropies = sorted(smoothed_entropy)
    threshold_idx = int(len(sorted_entropies) * entropy_threshold_percentile / 100)
    entropy_threshold = sorted_entropies[threshold_idx]
    
    # Mark high-entropy regions
    is_high_entropy = []
    for i, (token_id, entropy) in enumerate(zip(tokens, smoothed_entropy)):
        # Check if in answer region
        if answer_start_idx is not None and i >= answer_start_idx:
            is_high_entropy.append(False)
            continue
        
        # Check if high entropy
        if entropy > entropy_threshold:
            is_high_entropy.append(True)
        else:
            is_high_entropy.append(False)
    
    # Merge consecutive regions and filter by length
    regions = []  # [(start_idx, end_idx), ...]
    in_region = False
    region_start = None
    
    for i, is_high in enumerate(is_high_entropy):
        if is_high and not in_region:
            # Start new region
            region_start = i
            in_region = True
        elif not is_high and in_region:
            # End region
            region_length = i - region_start
            if region_length >= min_swi_length:
                regions.append((region_start, i))
            in_region = False
    
    # Handle last region
    if in_region and len(tokens) - region_start >= min_swi_length:
        regions.append((region_start, len(tokens)))
    
    # Create boolean mask
    should_mark = [False] * len(tokens)
    for start, end in regions:
        for i in range(start, end):
            should_mark[i] = True
    
    return should_mark


def build_annotated_output(tokens, should_mark, tokenizer, switch_start="<swi>", switch_end="</swi>"):
    """Build output string with <swi> annotations."""
    result = []
    current_segment = []
    prev_marked = None
    
    def flush_segment(is_marked, segment):
        if not segment:
            return
        text = tokenizer.decode(segment, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if is_marked:
            result.append(f"{switch_start}{text}{switch_end}")
        else:
            result.append(text)
    
    for token_id, marked in zip(tokens, should_mark):
        if prev_marked is None:
            prev_marked = marked
        elif marked != prev_marked:
            flush_segment(prev_marked, current_segment)
            current_segment = []
            prev_marked = marked
        current_segment.append(token_id)
    
    flush_segment(prev_marked, current_segment)
    return "".join(result)


def build_prompt(tokenizer, question: str, dataset: str) -> str:
    """Build prompt for GSM8K or MATH."""
    if dataset == "math":
        system_prompt = "You are a helpful assistant. Solve the math problem step by step and put the final answer in \\boxed{}." 
    else:
        system_prompt = "You are a helpful assistant. Solve the math problem step by step."
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question.strip()}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main():
    parser = argparse.ArgumentParser(description="Generation-based annotation for SwiReasoning")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--dataset", type=str, default="gsm8k", choices=["gsm8k", "math"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--split", type=str, default="train", choices=["train", "test"])
    parser.add_argument("--output_path", type=str, default="logs/gsm8k_swir_generation_based.jsonl")
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Existing output jsonl to skip already processed samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0)
    
    # Annotation parameters
    parser.add_argument("--window_size", type=int, default=10, 
                       help="Smoothing window for entropy")
    parser.add_argument("--entropy_percentile", type=int, default=60,
                       help="Percentile threshold for high entropy (0-100)")
    parser.add_argument("--min_swi_length", type=int, default=3,
                       help="Minimum tokens to mark with <swi>")
    parser.add_argument("--only_correct", action="store_true",
                       help="Only annotate samples where model gets correct answer")
    parser.add_argument("--num_shards", type=int, default=1,
                       help="Total number of dataset shards for parallel annotation")
    parser.add_argument("--shard_idx", type=int, default=0,
                       help="Shard index (0-based) for this worker")
    
    parser.add_argument("--switch_start_token", type=str, default="<swi>")
    parser.add_argument("--switch_end_token", type=str, default="</swi>")
    
    args = parser.parse_args()

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(args.device)
        device_map = {"": args.device}
    else:
        device_map = None

    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
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
        if args.dataset == "math":
            dataset = load_dataset("open-r1/OpenR1-Math-220k", "default", split=args.split)
        else:
            dataset = load_dataset("gsm8k", "main", split=args.split)
    
    if args.max_samples is not None:
        dataset = dataset.select(range(args.max_samples))
    if args.num_shards > 1:
        if not (0 <= args.shard_idx < args.num_shards):
            raise ValueError("shard_idx must be in [0, num_shards)")
        dataset = dataset.shard(num_shards=args.num_shards, index=args.shard_idx)

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total = len(dataset)
    print(f"Processing {total} samples...")
    print(f"Strategy: Generate solution → Track entropy → Annotate high-entropy regions with <swi>")
    print(f"Parameters: window={args.window_size}, percentile={args.entropy_percentile}, min_length={args.min_swi_length}")
    print()

    skip_indices = set()
    skip_questions = set()
    if args.resume_from:
        if os.path.exists(args.resume_from):
            with open(args.resume_from, "r", encoding="utf-8") as rf:
                for line in rf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if "dataset_index" in rec:
                        skip_indices.add(rec["dataset_index"])
                    if "instruction" in rec:
                        skip_questions.add(rec["instruction"])
            print(f"[Resume] Loaded {len(skip_indices)} indices and {len(skip_questions)} questions to skip.")
        else:
            print(f"[Resume] resume_from not found: {args.resume_from}")

    correct_count = 0
    annotated_count = 0
    
    with open(args.output_path, "w", encoding="utf-8") as writer:
        for start in tqdm(range(0, total, args.batch_size), desc="Annotating"):
            end = min(start + args.batch_size, total)
            indices = list(range(start, end))
            batch = dataset.select(indices)

            if args.dataset == "math":
                questions = [q.strip() for q in batch["problem"]]
                ground_truth_answers = [extract_answer(s) for s in batch["solution"]]
            else:
                questions = [q.strip() for q in batch["question"]]
                ground_truth_answers = [extract_answer(s) for s in batch["answer"]]

            # Apply resume skip
            keep = []
            for i, q in enumerate(questions):
                idx_val = indices[i]
                if idx_val in skip_indices:
                    continue
                if q in skip_questions:
                    continue
                keep.append(i)

            if not keep:
                continue

            questions = [questions[i] for i in keep]
            ground_truth_answers = [ground_truth_answers[i] for i in keep]
            indices = [indices[i] for i in keep]

            prompts = [build_prompt(tokenizer, q, args.dataset) for q in questions]

            model_inputs = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(device)

            tokens_list, entropies_list = generate_with_entropy_tracking_batch(
                model,
                tokenizer,
                model_inputs["input_ids"],
                model_inputs["attention_mask"],
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                max_new_tokens=args.max_new_tokens,
            )

            for idx_inner in range(len(questions)):
                tokens = tokens_list[idx_inner]
                entropies = entropies_list[idx_inner]

                if len(tokens) == 0:
                    continue

                generated_text = tokenizer.decode(tokens, skip_special_tokens=True)
                predicted_answer = extract_answer(generated_text)
                ground_truth_answer = ground_truth_answers[idx_inner]
                is_correct = (predicted_answer == ground_truth_answer)

                if is_correct:
                    correct_count += 1

                if args.only_correct and not is_correct:
                    continue

                should_mark = annotate_with_swir_logic(
                    tokens,
                    entropies,
                    tokenizer,
                    window_size=args.window_size,
                    entropy_threshold_percentile=args.entropy_percentile,
                    min_swi_length=args.min_swi_length,
                    protect_answer_region=True,
                )

                annotated_output = build_annotated_output(
                    tokens,
                    should_mark,
                    tokenizer,
                    switch_start=args.switch_start_token,
                    switch_end=args.switch_end_token,
                )

                if args.switch_start_token in annotated_output:
                    annotated_count += 1

                record = {
                    "instruction": questions[idx_inner],
                    "input": "",
                    "output": annotated_output,
                    "is_correct": is_correct,
                    "predicted_answer": predicted_answer,
                    "ground_truth_answer": ground_truth_answer,
                    "dataset_index": indices[idx_inner],
                }
                writer.write(json.dumps(record, ensure_ascii=False) + "\n")

                if start == 0 and idx_inner < 3:
                    print(f"\n{'='*80}")
                    print(f"Example {idx_inner + 1}:")
                    print(f"Question: {questions[idx_inner][:100]}...")
                    print(f"Correct: {is_correct} (Pred: {predicted_answer}, GT: {ground_truth_answer})")
                    print(f"Annotated output (first 300 chars):")
                    print(annotated_output[:300] + "...")
                    print(f"Has <swi>: {args.switch_start_token in annotated_output}")

    print(f"\n{'='*80}")
    print(f"✓ Saved annotations to {args.output_path}")
    print(f"\nStatistics:")
    print(f"  Total samples: {total}")
    print(f"  Correct answers: {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"  Samples with <swi>: {annotated_count} ({100*annotated_count/total:.1f}%)")
    print(f"\nNext steps:")
    print(f"1. Inspect output to verify <swi> placement")
    print(f"2. Convert to Stage 2 format:")
    print(f"   python convert_to_latent_format.py --input {args.output_path} --output gsm8k_stage2.json")
    print(f"3. Train Stage 1 (learn to output <swi>)")
    print(f"4. Train Stage 2 (learn Coconut latent reasoning)")


if __name__ == "__main__":
    main()
