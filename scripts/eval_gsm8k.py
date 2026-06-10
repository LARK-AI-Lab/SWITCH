"""
Evaluate a SwiLatent checkpoint on the GSM8K benchmark.

Supports single-GPU and multi-GPU (via torchrun) evaluation. Each GPU
processes a disjoint shard of the test set; rank 0 merges and reports.

Usage:
    # Single GPU
    python scripts/eval_gsm8k.py --adapter_path outputs/phase2_1_v2/stage_008

    # Multi-GPU (4 cards)
    torchrun --standalone --nproc_per_node=4 scripts/eval_gsm8k.py \
        --adapter_path outputs/phase2_1_v2/stage_008

    # With 4-bit quantization
    torchrun --standalone --nproc_per_node=4 scripts/eval_gsm8k.py \
        --adapter_path outputs/phase2_1_v2/stage_008 --load_in_4bit

    # Quick test (50 samples)
    python scripts/eval_gsm8k.py --adapter_path outputs/phase2_1_v2/stage_008 --n_samples 50

    # Base model baseline
    python scripts/eval_gsm8k.py --no_adapter
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
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from grader import gsm8k_grader
    HAS_GRADER = True
except ImportError:
    HAS_GRADER = False

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def simple_grade(pred_text, gold):
    """Fallback grader: extract last number from prediction and compare."""
    numbers = re.findall(r"[-+]?\d*\.?\d+", pred_text.replace(",", ""))
    if not numbers:
        return False, "no_number"
    extracted = numbers[-1]
    gold_clean = gold.strip().replace(",", "")
    try:
        return abs(float(extracted) - float(gold_clean)) < 1e-3, extracted
    except ValueError:
        return extracted == gold_clean, extracted


def grade(pred_text, gold):
    if HAS_GRADER:
        try:
            return gsm8k_grader(pred_text, gold)
        except Exception:
            return simple_grade(pred_text, gold)
    return simple_grade(pred_text, gold)


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


def build_prompt(tokenizer, question):
    messages = [{"role": "user", "content": question +
                 "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True,
    )


def count_swi(text):
    return text.count("<swi>"), text.count("</swi>")


def main():
    parser = argparse.ArgumentParser(description="Evaluate SwiLatent on GSM8K")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to LoRA adapter (local dir or HF repo id)")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Override base model path")
    parser.add_argument("--no_adapter", action="store_true",
                        help="Evaluate base model without adapter")
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base model in 4-bit NF4 quantization")
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Number of GSM8K test samples (default: all 1319)")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Path to save merged results JSON")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)

    is_main = (rank == 0)

    # ── Resolve paths ─────────────────────────────────────────────────────────
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
            print("[warn] math_verify not installed — using simple numeric matching.")
            print("       pip install math-verify latex2sympy2_extended\n")
        print("=" * 70)
        print("GSM8K Evaluation")
        print(f"  Base model   : {base_path}")
        print(f"  Adapter      : {adapter_path or 'none'}")
        print(f"  4-bit quant  : {args.load_in_4bit}")
        print(f"  Max tokens   : {args.max_new_tokens}")
        print(f"  N samples    : {args.n_samples or 'all (1319)'}")
        print(f"  GPUs         : {world_size}")
        print("=" * 70)

    # ── Load tokenizer ────────────────────────────────────────────────────────
    tok_source = adapter_path if adapter_path else base_path
    tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load model (each GPU loads independently) ─────────────────────────────
    load_kwargs = dict(trust_remote_code=True)
    if args.load_in_4bit:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = {"": device}
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = {"": device}

    if is_main:
        print(f"[eval] Loading base model...")
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
            print(f"[eval] Loading LoRA adapter...")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    # ── Load & shard GSM8K ────────────────────────────────────────────────────
    dataset = load_dataset("gsm8k", "main", split="test")
    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    total_len = len(dataset)

    chunk_size = (total_len + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, total_len)
    shard = dataset.select(range(start, end))

    if is_main:
        print(f"[eval] {total_len} total problems, rank {rank} evaluates [{start}:{end}] ({len(shard)} items)")

    # ── Per-rank incremental save path ───────────────────────────────────────
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    rank_save_path = os.path.join(PROJECT_ROOT, "logs", f"_gsm8k_rank{rank}.json")
    SAVE_EVERY = 10

    def _save_rank_progress():
        """Write current rank results to disk (called every SAVE_EVERY items)."""
        rd = {
            "rank": rank, "correct": correct, "total": total,
            "swi_open": total_swi_open, "swi_close": total_swi_close,
            "token_lens": token_lens, "elapsed_s": time.time() - t_start,
            "details": details, "finished": False,
        }
        with open(rank_save_path, "w") as _f:
            json.dump(rd, _f, ensure_ascii=False)

    # ── Evaluate local shard ──────────────────────────────────────────────────
    correct = 0
    total = 0
    details = []
    total_swi_open = 0
    total_swi_close = 0
    token_lens = []
    t_start = time.time()

    for i in tqdm(range(len(shard)), desc=f"[GPU{rank}] GSM8K", disable=(not is_main)):
        item = shard[i]
        question = item["question"]
        gold = str(item["answer"]).split("####")[-1].strip()

        prompt = build_prompt(tokenizer, question)
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
            "question": question[:100],
            "gold": gold,
            "extracted": str(extracted),
            "correct": bool(is_correct),
            "swi_open": opens,
            "swi_close": closes,
            "n_tokens": n_gen,
        })

        if total % SAVE_EVERY == 0:
            _save_rank_progress()

        if is_main and total % 10 == 0:
            acc_so_far = correct / total
            swi_rate = sum(1 for d in details if d["swi_open"] > 0) / total
            print(f"  [{total}/{len(shard)}] acc={acc_so_far:.4f} swi_rate={swi_rate:.2f}")

    _save_rank_progress()

    elapsed = time.time() - t_start

    # ── Gather results from all ranks ─────────────────────────────────────────
    if world_size > 1:
        # Final save is already done above; mark as finished
        rank_final = {
            "rank": rank, "correct": correct, "total": total,
            "swi_open": total_swi_open, "swi_close": total_swi_close,
            "token_lens": token_lens, "elapsed_s": elapsed,
            "details": details, "finished": True,
        }
        with open(rank_save_path, "w") as f:
            json.dump(rank_final, f, ensure_ascii=False)

        dist.barrier()

        if is_main:
            all_correct = 0
            all_total = 0
            all_swi_open = 0
            all_swi_close = 0
            all_token_lens = []
            all_details = []
            all_elapsed = 0.0

            for r in range(world_size):
                rp = os.path.join(PROJECT_ROOT, "logs", f"_gsm8k_rank{r}.json")
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
            correct = all_correct
            total = all_total
            total_swi_open = all_swi_open
            total_swi_close = all_swi_close
            token_lens = all_token_lens
            details = all_details
            elapsed = all_elapsed
    # ── Print & save (rank 0 only) ────────────────────────────────────────────
    if is_main:
        n_with_swi = sum(1 for d in details if d["swi_open"] > 0)
        avg_tokens = sum(token_lens) / max(len(token_lens), 1)

        summary = {
            "adapter": adapter_label,
            "base_model": base_path,
            "load_in_4bit": args.load_in_4bit,
            "dataset": "gsm8k",
            "n_gpus": world_size,
            "n_samples": total,
            "correct": correct,
            "accuracy": round(correct / max(total, 1), 4),
            "swi_stats": {
                "problems_with_swi": n_with_swi,
                "swi_rate": round(n_with_swi / max(total, 1), 4),
                "total_swi_open": total_swi_open,
                "total_swi_close": total_swi_close,
                "tag_balance": "OK" if total_swi_open == total_swi_close else "MISMATCH",
            },
            "token_stats": {
                "avg_tokens": round(avg_tokens, 1),
                "max_tokens": max(token_lens) if token_lens else 0,
                "min_tokens": min(token_lens) if token_lens else 0,
            },
            "elapsed_s": round(elapsed, 1),
            "speed_samples_per_min": round(total / max(elapsed / 60, 0.01), 1),
        }

        print(f"\n{'='*70}")
        print(f"GSM8K Results — {adapter_label} ({world_size} GPU{'s' if world_size>1 else ''})")
        print(f"{'='*70}")
        print(f"  Accuracy         : {correct}/{total} ({summary['accuracy']*100:.2f}%)")
        print(f"  <swi> rate       : {n_with_swi}/{total} ({summary['swi_stats']['swi_rate']*100:.1f}%)")
        print(f"  <swi> open/close : {total_swi_open}/{total_swi_close} ({summary['swi_stats']['tag_balance']})")
        print(f"  Avg tokens       : {avg_tokens:.1f}")
        print(f"  Total time       : {elapsed:.0f}s ({summary['speed_samples_per_min']:.1f} samples/min)")
        print(f"{'='*70}")

        save_path = args.save_path
        if save_path is None:
            os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
            n_tag = args.n_samples or "full"
            q_tag = "4bit" if args.load_in_4bit else "bf16"
            save_path = os.path.join(
                PROJECT_ROOT, "logs",
                f"gsm8k_{adapter_label}_{q_tag}_{n_tag}.json"
            )

        result = {**summary, "details": details}
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n[eval] Results saved to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
