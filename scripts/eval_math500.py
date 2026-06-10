"""
Evaluate a SwiLatent checkpoint on the MATH-500 benchmark.

Supports single-GPU and multi-GPU (via torchrun) evaluation.

Usage:
    # Single GPU
    python scripts/eval_math500.py --adapter_path outputs/phase2_1_v4_bf16/stage_010

    # Multi-GPU (8 cards)
    torchrun --standalone --nproc_per_node=8 scripts/eval_math500.py \
        --adapter_path outputs/phase2_1_v4_bf16/stage_010

    # Base model baseline
    python scripts/eval_math500.py --no_adapter --base_model /path/to/Qwen3-8B
"""

import argparse
import json
import os
import re
import sys
import time

import torch
import torch.distributed as dist
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from grader import math500_grader
    HAS_GRADER = True
except ImportError:
    HAS_GRADER = False

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def extract_boxed(text: str) -> str | None:
    """Extract the last \\boxed{...} content, handling nested braces."""
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth = 0
    start = idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return text[start:]


def simple_grade(pred_text: str, gold_answer: str) -> tuple[bool, str]:
    """Fallback grader when math_verify is not installed."""
    extracted = extract_boxed(pred_text)
    if extracted is None:
        numbers = re.findall(r"[-+]?\d*\.?\d+", pred_text.replace(",", ""))
        extracted = numbers[-1] if numbers else ""

    gold_clean = gold_answer.strip().replace(" ", "")
    pred_clean = extracted.strip().replace(" ", "")
    if gold_clean == pred_clean:
        return True, extracted
    try:
        return abs(float(pred_clean) - float(gold_clean)) < 1e-4, extracted
    except (ValueError, TypeError):
        return False, extracted


def grade(pred_text: str, gold_answer: str) -> tuple[bool, str]:
    if HAS_GRADER:
        try:
            return math500_grader(pred_text, gold_answer)
        except Exception:
            return simple_grade(pred_text, gold_answer)
    return simple_grade(pred_text, gold_answer)


def resolve_base_model(adapter_path, explicit_base):
    if explicit_base:
        return explicit_base
    cfg_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        raw = cfg.get("base_model_name_or_path", "")
        if raw and os.path.isdir(raw) and os.path.isfile(os.path.join(raw, "config.json")):
            return raw
        m = re.search(r"models--([^/]+)--([^/]+)", raw)
        if m:
            return f"{m.group(1)}/{m.group(2)}"
    return DEFAULT_BASE_MODEL


def build_prompt(tokenizer, problem):
    messages = [{"role": "user", "content": problem +
                 "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def count_swi(text):
    return text.count("<swi>"), text.count("</swi>")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SwiLatent on MATH-500")
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--no_adapter", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of MATH-500 samples (default: all 500)")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset", type=str, default="HuggingFaceH4/MATH-500",
                        help="HF dataset id for MATH-500")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)
    is_main = (rank == 0)

    if args.no_adapter:
        base_path = args.base_model or DEFAULT_BASE_MODEL
        adapter_path = None
        adapter_label = "none"
    else:
        adapter_path = args.adapter_path
        if adapter_path is None:
            if is_main:
                print("[error] Provide --adapter_path or --no_adapter")
            sys.exit(1)
        if not os.path.isabs(adapter_path) and not adapter_path.startswith((".", "/")):
            local = os.path.join(PROJECT_ROOT, adapter_path)
            if os.path.isdir(local):
                adapter_path = local
        base_path = resolve_base_model(adapter_path, args.base_model)
        adapter_label = os.path.basename(adapter_path.rstrip("/"))

    if is_main:
        if not HAS_GRADER:
            print("[warn] grader/math_verify not installed — using string/numeric matching.")
            print("       pip install math-verify latex2sympy2_extended\n")
        print("=" * 70)
        print("MATH-500 Evaluation")
        print(f"  Base model   : {base_path}")
        print(f"  Adapter      : {adapter_path or 'none'}")
        print(f"  Max tokens   : {args.max_new_tokens}")
        print(f"  N samples    : {args.n_samples or 'all (500)'}")
        print(f"  GPUs         : {world_size}")
        print("=" * 70)

    tok_source = adapter_path if adapter_path else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    if is_main:
        print("[eval] Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)
    if is_main:
        print(f"[eval] Base model loaded in {time.time()-t0:.1f}s")

    if adapter_path:
        if len(tokenizer) != model.config.vocab_size:
            model.resize_token_embeddings(len(tokenizer))
            if is_main:
                print(f"[eval] Resized embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
        if is_main:
            print("[eval] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # ── Load MATH-500 ──────────────────────────────────────────────────────
    if is_main:
        print(f"[eval] Loading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="test")
    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    total_len = len(dataset)

    chunk_size = (total_len + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, total_len)
    shard = dataset.select(range(start, end))

    if is_main:
        print(f"[eval] {total_len} total problems, rank {rank} evaluates [{start}:{end}] ({len(shard)} items)")

    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    rank_save_path = os.path.join(PROJECT_ROOT, "logs", f"_math500_rank{rank}.json")
    SAVE_EVERY = 10

    correct = 0
    total = 0
    details = []
    total_swi_open = 0
    total_swi_close = 0
    token_lens = []
    t_start = time.time()

    def _save_rank():
        with open(rank_save_path, "w") as _f:
            json.dump({
                "rank": rank, "correct": correct, "total": total,
                "swi_open": total_swi_open, "swi_close": total_swi_close,
                "token_lens": token_lens, "elapsed_s": time.time() - t_start,
                "details": details, "finished": False,
            }, _f, ensure_ascii=False)

    for i in tqdm(range(len(shard)), desc=f"[GPU{rank}] MATH-500", disable=(not is_main)):
        item = shard[i]
        problem = item["problem"]
        gold = item["answer"]

        prompt = build_prompt(tokenizer, problem)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )

        n_gen = output_ids.shape[1] - input_len
        token_lens.append(n_gen)
        generated = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=False)

        opens, closes = count_swi(generated)
        total_swi_open += opens
        total_swi_close += closes

        answer_text = generated
        think_end = generated.find("</think>")
        if think_end != -1:
            answer_text = generated[think_end + len("</think>"):]

        is_correct, extracted = grade(answer_text, gold)
        correct += int(is_correct)
        total += 1

        details.append({
            "idx": start + i,
            "problem": problem[:100],
            "gold": gold,
            "extracted": str(extracted),
            "correct": bool(is_correct),
            "swi_open": opens,
            "swi_close": closes,
            "n_tokens": n_gen,
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
        })

        if total % SAVE_EVERY == 0:
            _save_rank()

        if is_main and total % 10 == 0:
            acc = correct / total
            swi_rate = sum(1 for d in details if d["swi_open"] > 0) / total
            print(f"  [{total}/{len(shard)}] acc={acc:.4f} swi_rate={swi_rate:.2f}")

    _save_rank()
    elapsed = time.time() - t_start

    # ── Gather & merge ────────────────────────────────────────────────────
    if world_size > 1:
        with open(rank_save_path, "w") as f:
            json.dump({
                "rank": rank, "correct": correct, "total": total,
                "swi_open": total_swi_open, "swi_close": total_swi_close,
                "token_lens": token_lens, "elapsed_s": elapsed,
                "details": details, "finished": True,
            }, f, ensure_ascii=False)
        dist.barrier()

        if is_main:
            all_correct = all_total = all_swi_open = all_swi_close = 0
            all_token_lens, all_details = [], []
            all_elapsed = 0.0
            for r in range(world_size):
                rp = os.path.join(PROJECT_ROOT, "logs", f"_math500_rank{r}.json")
                with open(rp) as f:
                    rd = json.load(f)
                all_correct += rd["correct"]
                all_total += rd["total"]
                all_swi_open += rd["swi_open"]
                all_swi_close += rd["swi_close"]
                all_token_lens.extend(rd["token_lens"])
                all_details.extend(rd["details"])
                all_elapsed = max(all_elapsed, rd["elapsed_s"])
                os.remove(rp)
            all_details.sort(key=lambda d: d["idx"])
            correct, total = all_correct, all_total
            total_swi_open, total_swi_close = all_swi_open, all_swi_close
            token_lens, details, elapsed = all_token_lens, all_details, all_elapsed

    if is_main:
        n_with_swi = sum(1 for d in details if d["swi_open"] > 0)
        avg_tokens = sum(token_lens) / max(len(token_lens), 1)

        # Per-subject breakdown
        subjects = {}
        for d in details:
            s = d.get("subject", "unknown")
            if s not in subjects:
                subjects[s] = {"correct": 0, "total": 0}
            subjects[s]["total"] += 1
            if d["correct"]:
                subjects[s]["correct"] += 1

        summary = {
            "adapter": adapter_label,
            "base_model": base_path,
            "dataset": "math500",
            "n_gpus": world_size,
            "n_samples": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1), 4),
            "swi_stats": {
                "problems_with_swi": n_with_swi,
                "swi_rate": round(n_with_swi / max(total, 1), 4),
                "total_swi_open": total_swi_open,
                "total_swi_close": total_swi_close,
            },
            "token_stats": {
                "avg_tokens": round(avg_tokens, 1),
                "max_tokens": max(token_lens) if token_lens else 0,
                "min_tokens": min(token_lens) if token_lens else 0,
            },
            "per_subject": {s: {**v, "accuracy": round(v["correct"]/max(v["total"],1), 4)}
                           for s, v in sorted(subjects.items())},
            "elapsed_s": round(elapsed, 1),
        }

        print(f"\n{'='*70}")
        print(f"MATH-500 Results — {adapter_label} ({world_size} GPU{'s' if world_size>1 else ''})")
        print(f"{'='*70}")
        print(f"  Accuracy         : {correct}/{total} ({summary['accuracy']*100:.2f}%)")
        print(f"  <swi> rate       : {n_with_swi}/{total} ({summary['swi_stats']['swi_rate']*100:.1f}%)")
        print(f"  <swi> open/close : {total_swi_open}/{total_swi_close}")
        print(f"  Avg tokens       : {avg_tokens:.1f}")
        print(f"  Total time       : {elapsed:.0f}s")
        print(f"\n  Per-subject:")
        for s, v in sorted(subjects.items()):
            print(f"    {s:30s}: {v['correct']:>3}/{v['total']:>3} ({v['correct']/max(v['total'],1)*100:5.1f}%)")
        print(f"{'='*70}")

        save_path = args.save_path
        if save_path is None:
            n_tag = args.n_samples or "full"
            save_path = os.path.join(PROJECT_ROOT, "logs", f"math500_{adapter_label}_bf16_{n_tag}.json")

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({**summary, "details": details}, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] Results saved to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
