"""
Evaluate a SwiLatent checkpoint with **latent inference** on GSM8K / MATH-500.

Unlike eval_gsm8k.py / eval_math500.py which use standard autoregressive
generation, this script wraps the model in CoconutSwiModel and uses its
latent-mode generate():  when the model outputs <swi>, hidden states are
fed back as input embeddings for "silent thinking" until </swi> is predicted.

Supports single-GPU and multi-GPU (via torchrun).

Usage:
    # GSM8K with latent inference (8 GPU)
    torchrun --standalone --nproc_per_node=8 scripts/eval_latent.py \
        --adapter_path outputs/phase2_1_v4_bf16/stage_010 --dataset gsm8k

    # MATH-500 with latent inference
    torchrun --standalone --nproc_per_node=8 scripts/eval_latent.py \
        --adapter_path outputs/phase2_1_v4_bf16/stage_010 --dataset math500

    # Quick test (20 samples)
    python scripts/eval_latent.py \
        --adapter_path outputs/phase2_1_v4_bf16/stage_010 --dataset gsm8k --n_samples 20
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

from src.model.coconut_swi_model import CoconutSwiModel
from src.setup_tokens import (
    SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN, SwiTokenConfig,
)

try:
    from grader import gsm8k_grader, math500_grader
    HAS_GRADER = True
except ImportError:
    HAS_GRADER = False

DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


# ── Grading ───────────────────────────────────────────────────────────────────

def simple_grade_gsm8k(pred_text, gold):
    numbers = re.findall(r"[-+]?\d*\.?\d+", pred_text.replace(",", ""))
    if not numbers:
        return False, "no_number"
    extracted = numbers[-1]
    gold_clean = gold.strip().replace(",", "")
    try:
        return abs(float(extracted) - float(gold_clean)) < 1e-3, extracted
    except ValueError:
        return extracted == gold_clean, extracted


def extract_boxed(text):
    idx = text.rfind("\\boxed{")
    if idx == -1:
        return None
    depth, start = 0, idx + len("\\boxed{")
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            if depth == 0:
                return text[start:i]
            depth -= 1
    return text[start:]


def simple_grade_math(pred_text, gold):
    extracted = extract_boxed(pred_text)
    if extracted is None:
        numbers = re.findall(r"[-+]?\d*\.?\d+", pred_text.replace(",", ""))
        extracted = numbers[-1] if numbers else ""
    g = gold.strip().replace(" ", "")
    p = extracted.strip().replace(" ", "")
    if g == p:
        return True, extracted
    try:
        return abs(float(p) - float(g)) < 1e-4, extracted
    except (ValueError, TypeError):
        return False, extracted


def grade(pred_text, gold, dataset_name):
    if HAS_GRADER:
        try:
            if dataset_name == "gsm8k":
                return gsm8k_grader(pred_text, gold)
            else:
                return math500_grader(pred_text, gold)
        except Exception:
            pass
    if dataset_name == "gsm8k":
        return simple_grade_gsm8k(pred_text, gold)
    return simple_grade_math(pred_text, gold)


# ── Utilities ─────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate SwiLatent with latent inference on GSM8K / MATH-500")
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Override tokenizer path (default: same as adapter_path)")
    parser.add_argument("--dataset", type=str, default="gsm8k",
                        choices=["gsm8k", "math500"])
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--n_samples", type=int, default=None)
    parser.add_argument("--min_latent_steps", type=int, default=0,
                        help="Minimum latent steps per block before allowing "
                             "exit. Set to match training (e.g. max_latent_stage "
                             "* c_thought). 0 = no minimum.")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Distributed setup ─────────────────────────────────────────────────
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    torch.manual_seed(args.seed + rank)
    is_main = (rank == 0)

    adapter_path = args.adapter_path
    if not os.path.isabs(adapter_path) and not adapter_path.startswith((".", "/")):
        local = os.path.join(PROJECT_ROOT, adapter_path)
        if os.path.isdir(local):
            adapter_path = local
    base_path = resolve_base_model(adapter_path, args.base_model)
    adapter_label = os.path.basename(adapter_path.rstrip("/"))

    if is_main:
        if not HAS_GRADER:
            print("[warn] grader not available — using simple matching.")
            print("       pip install math-verify latex2sympy2_extended\n")
        print("=" * 70)
        print(f"SwiLatent LATENT INFERENCE — {args.dataset.upper()}")
        print("=" * 70)
        print(f"  Base model   : {base_path}")
        print(f"  Adapter      : {adapter_path}")
        print(f"  Mode         : LATENT INFERENCE (CoconutSwiModel)")
        print(f"  Min latent   : {args.min_latent_steps} steps/block")
        print(f"  Max tokens   : {args.max_new_tokens}")
        print(f"  N samples    : {args.n_samples or 'all'}")
        print(f"  GPUs         : {world_size}")
        print("=" * 70)

    # ── Load tokenizer ────────────────────────────────────────────────────
    tok_path = args.tokenizer_path or adapter_path
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ── Load base model + adapter ─────────────────────────────────────────
    if is_main:
        print("[eval] Loading base model...")
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    if is_main:
        print(f"[eval] Base model loaded in {time.time()-t0:.1f}s")

    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        if is_main:
            print(f"[eval] Resized embeddings -> {len(tokenizer)}")

    if is_main:
        print("[eval] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    # ── Wrap in CoconutSwiModel ───────────────────────────────────────────
    token_cfg = SwiTokenConfig(
        swi_start_id=tokenizer.convert_tokens_to_ids(SWI_START_TOKEN),
        swi_end_id=tokenizer.convert_tokens_to_ids(SWI_END_TOKEN),
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        eos_token_id=tokenizer.eos_token_id,
    )
    coconut_model = CoconutSwiModel(model, token_cfg)
    coconut_model.eval()

    if is_main:
        print(f"[eval] CoconutSwiModel ready — "
              f"swi_start={token_cfg.swi_start_id}, "
              f"swi_end={token_cfg.swi_end_id}, "
              f"latent={token_cfg.latent_id}")

    # ── Load & shard dataset ──────────────────────────────────────────────
    if args.dataset == "gsm8k":
        dataset = load_dataset("gsm8k", "main", split="test")
    else:
        dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")

    if args.n_samples:
        dataset = dataset.select(range(min(args.n_samples, len(dataset))))
    total_len = len(dataset)

    chunk_size = (total_len + world_size - 1) // world_size
    start_idx = rank * chunk_size
    end_idx = min(start_idx + chunk_size, total_len)
    shard = dataset.select(range(start_idx, end_idx))

    if is_main:
        print(f"[eval] {total_len} problems, rank {rank}: [{start_idx}:{end_idx}] ({len(shard)} items)\n")

    # ── Evaluate ──────────────────────────────────────────────────────────
    os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
    rank_file = os.path.join(
        PROJECT_ROOT, "logs", f"_latent_{args.dataset}_rank{rank}.json"
    )
    SAVE_EVERY = 5

    correct = 0
    total = 0
    details = []
    total_swi_open = 0
    total_swi_close = 0
    total_latent_steps = 0
    total_latent_blocks = 0
    token_lens = []
    t_start = time.time()

    def _save():
        with open(rank_file, "w") as _f:
            json.dump({
                "rank": rank, "correct": correct, "total": total,
                "swi_open": total_swi_open, "swi_close": total_swi_close,
                "total_latent_steps": total_latent_steps,
                "total_latent_blocks": total_latent_blocks,
                "token_lens": token_lens,
                "elapsed_s": time.time() - t_start,
                "details": details, "finished": False,
            }, _f, ensure_ascii=False)

    for i in tqdm(range(len(shard)),
                  desc=f"[GPU{rank}] {args.dataset} latent", disable=(not is_main)):
        item = shard[i]

        if args.dataset == "gsm8k":
            question = item["question"]
            gold = str(item["answer"]).split("####")[-1].strip()
        else:
            question = item["problem"]
            gold = item["answer"]

        prompt = build_prompt(tokenizer, question)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[1]

        output_ids, latent_info = coconut_model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=args.max_new_tokens,
            min_latent_steps=args.min_latent_steps,
        )

        n_gen = output_ids.shape[1] - input_len
        token_lens.append(n_gen)
        generated = tokenizer.decode(
            output_ids[0][input_len:], skip_special_tokens=False
        )

        opens, closes = count_swi(generated)
        total_swi_open += opens
        total_swi_close += closes

        n_blocks = len(latent_info)
        n_steps = sum(b["n_latent_steps"] for b in latent_info)
        total_latent_blocks += n_blocks
        total_latent_steps += n_steps

        answer_text = generated
        think_end = generated.find("</think>")
        if think_end != -1:
            answer_text = generated[think_end + len("</think>"):]

        is_correct, extracted = grade(answer_text, gold, args.dataset)
        correct += int(is_correct)
        total += 1

        detail = {
            "idx": start_idx + i,
            "question": question[:100],
            "gold": gold,
            "extracted": str(extracted),
            "correct": bool(is_correct),
            "swi_open": opens,
            "swi_close": closes,
            "n_tokens": n_gen,
            "latent_blocks": n_blocks,
            "latent_steps": n_steps,
            "latent_info": latent_info,
        }
        if args.dataset == "math500":
            detail["subject"] = item.get("subject", "")
            detail["level"] = item.get("level", "")
        details.append(detail)

        if total % SAVE_EVERY == 0:
            _save()

        if is_main and total % 5 == 0:
            acc = correct / total
            avg_lat = total_latent_steps / max(total_latent_blocks, 1)
            print(f"  [{total}/{len(shard)}] acc={acc:.4f} "
                  f"latent_blocks={total_latent_blocks} "
                  f"avg_steps/block={avg_lat:.1f}")

    _save()
    elapsed = time.time() - t_start

    # ── Gather & merge ────────────────────────────────────────────────────
    if world_size > 1:
        with open(rank_file, "w") as f:
            json.dump({
                "rank": rank, "correct": correct, "total": total,
                "swi_open": total_swi_open, "swi_close": total_swi_close,
                "total_latent_steps": total_latent_steps,
                "total_latent_blocks": total_latent_blocks,
                "token_lens": token_lens,
                "elapsed_s": elapsed,
                "details": details, "finished": True,
            }, f, ensure_ascii=False)
        dist.barrier()

        if is_main:
            all_c = all_t = all_so = all_sc = all_ls = all_lb = 0
            all_tl, all_d = [], []
            all_elapsed = 0.0
            for r in range(world_size):
                rp = os.path.join(
                    PROJECT_ROOT, "logs",
                    f"_latent_{args.dataset}_rank{r}.json",
                )
                with open(rp) as f:
                    rd = json.load(f)
                all_c += rd["correct"]
                all_t += rd["total"]
                all_so += rd["swi_open"]
                all_sc += rd["swi_close"]
                all_ls += rd["total_latent_steps"]
                all_lb += rd["total_latent_blocks"]
                all_tl.extend(rd["token_lens"])
                all_d.extend(rd["details"])
                all_elapsed = max(all_elapsed, rd["elapsed_s"])
                os.remove(rp)
            all_d.sort(key=lambda d: d["idx"])
            correct, total = all_c, all_t
            total_swi_open, total_swi_close = all_so, all_sc
            total_latent_steps, total_latent_blocks = all_ls, all_lb
            token_lens, details, elapsed = all_tl, all_d, all_elapsed

    # ── Print & save ──────────────────────────────────────────────────────
    if is_main:
        n_with_swi = sum(1 for d in details if d["swi_open"] > 0)
        avg_tokens = sum(token_lens) / max(len(token_lens), 1)
        avg_lat = total_latent_steps / max(total_latent_blocks, 1)

        summary = {
            "adapter": adapter_label,
            "base_model": base_path,
            "inference_mode": "latent",
            "min_latent_steps": args.min_latent_steps,
            "dataset": args.dataset,
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
            "latent_stats": {
                "total_latent_blocks": total_latent_blocks,
                "total_latent_steps": total_latent_steps,
                "avg_steps_per_block": round(avg_lat, 2),
            },
            "token_stats": {
                "avg_tokens": round(avg_tokens, 1),
                "max_tokens": max(token_lens) if token_lens else 0,
                "min_tokens": min(token_lens) if token_lens else 0,
            },
            "elapsed_s": round(elapsed, 1),
        }

        print(f"\n{'='*70}")
        print(f"{args.dataset.upper()} LATENT INFERENCE — "
              f"{adapter_label} ({world_size} GPUs)")
        print(f"{'='*70}")
        print(f"  Accuracy           : {correct}/{total} "
              f"({summary['accuracy']*100:.2f}%)")
        print(f"  <swi> rate         : {n_with_swi}/{total} "
              f"({summary['swi_stats']['swi_rate']*100:.1f}%)")
        print(f"  Latent blocks      : {total_latent_blocks}")
        print(f"  Total latent steps : {total_latent_steps}")
        print(f"  Avg steps/block    : {avg_lat:.1f}")
        print(f"  Avg output tokens  : {avg_tokens:.1f}")
        print(f"  Total time         : {elapsed:.0f}s")

        if args.dataset == "math500":
            subjects = {}
            for d in details:
                s = d.get("subject", "unknown")
                if s not in subjects:
                    subjects[s] = {"correct": 0, "total": 0}
                subjects[s]["total"] += 1
                if d["correct"]:
                    subjects[s]["correct"] += 1
            print(f"\n  Per-subject:")
            for s, v in sorted(subjects.items()):
                print(f"    {s:30s}: {v['correct']:>3}/{v['total']:>3} "
                      f"({v['correct']/max(v['total'],1)*100:5.1f}%)")
            summary["per_subject"] = {
                s: {**v, "accuracy": round(v["correct"]/max(v["total"],1), 4)}
                for s, v in sorted(subjects.items())
            }

        print(f"{'='*70}")

        save_path = args.save_path
        if save_path is None:
            n_tag = args.n_samples or "full"
            save_path = os.path.join(
                PROJECT_ROOT, "logs",
                f"{args.dataset}_{adapter_label}_latent_{n_tag}.json",
            )

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({**summary, "details": details}, f,
                      ensure_ascii=False, indent=2)
        print(f"\n[eval] Results saved to {save_path}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
