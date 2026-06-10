#!/usr/bin/env python3
"""Interpretability utilities for SwiLatent `<swi>` switch behavior.

This script is intentionally separate from training/eval code.  It supports:

1. teacher-forced entropy / p(<swi>) metrics on annotated SFT data
2. generation-time switch tracing
3. checkpoint-level metric comparison
4. switch-neighborhood curves around annotated `<swi>` positions
5. latent-state intervention / ablation during generation
6. logit-lens decoding of latent hidden states
7. lightweight hidden-state probing for switch-vs-control positions
8. markdown report generation

The teacher-forced and probe paths use incremental KV-cache forwarding so they
only materialize logits at selected positions instead of allocating full
sequence-by-vocabulary logits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from grader import gsm8k_grader, math500_grader  # noqa: E402
from src.model.coconut_swi_model import CoconutSwiModel  # noqa: E402
from src.setup_tokens import LATENT_TOKEN, SWI_END_TOKEN, SWI_START_TOKEN, SwiTokenConfig  # noqa: E402


DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"
JSON = Dict[str, object]


@dataclass
class LoadedModel:
    model: torch.nn.Module
    tokenizer: object
    token_cfg: SwiTokenConfig
    device: torch.device
    adapter_path: str
    label: str


def _resolve_path(path: str) -> str:
    if path is None:
        return path
    p = Path(path)
    if p.is_absolute():
        return str(p)
    candidate = PROJECT_ROOT / p
    return str(candidate if candidate.exists() else p)


def resolve_base_model(adapter_path: str, explicit_base: Optional[str] = None) -> str:
    if explicit_base:
        return explicit_base
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if cfg_path.is_file():
        with cfg_path.open() as f:
            cfg = json.load(f)
        raw = cfg.get("base_model_name_or_path", "")
        if raw and Path(raw).is_dir() and (Path(raw) / "config.json").is_file():
            return raw
        match = re.search(r"models--([^/]+)--([^/]+)", raw)
        if match:
            return f"{match.group(1)}/{match.group(2)}"
    return DEFAULT_BASE_MODEL


def load_model(
    adapter_path: str,
    base_model: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    device: str = "cuda",
    bf16: bool = True,
    use_modelscope: bool = False,
) -> LoadedModel:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = _resolve_path(adapter_path)
    tokenizer_source = _resolve_path(tokenizer_path) if tokenizer_path else adapter_path
    base_path = resolve_base_model(adapter_path, base_model)
    if use_modelscope:
        from modelscope.hub.snapshot_download import snapshot_download

        base_path = snapshot_download(os.environ.get("MODELSCOPE_MODEL_ID", "Qwen/Qwen3-8B"))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if len(tokenizer) != model.config.vocab_size:
        model.resize_token_embeddings(len(tokenizer))
    if (Path(adapter_path) / "adapter_config.json").is_file():
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    dev = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
    model.to(dev)
    token_cfg = SwiTokenConfig(
        swi_start_id=tokenizer.convert_tokens_to_ids(SWI_START_TOKEN),
        swi_end_id=tokenizer.convert_tokens_to_ids(SWI_END_TOKEN),
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        eos_token_id=tokenizer.eos_token_id,
    )
    return LoadedModel(
        model=model,
        tokenizer=tokenizer,
        token_cfg=token_cfg,
        device=dev,
        adapter_path=adapter_path,
        label=Path(adapter_path.rstrip("/")).name,
    )


def iter_jsonl(path: str, limit: Optional[int] = None) -> Iterable[JSON]:
    count = 0
    with open(_resolve_path(path)) as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit is not None and count >= limit:
                return


def phase2_prompt(sample: JSON) -> str:
    system = "You are a helpful assistant."
    question = str(sample.get("instruction", ""))
    inp = str(sample.get("input", ""))
    if inp:
        question = f"{question}\n{inp}"
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def chat_prompt(tokenizer, question: str, concise: bool = False) -> str:
    suffix = (
        "\nPlease reason step by step concisely, and put your final answer within \\boxed{}."
        if concise
        else "\nPlease reason step by step, and put your final answer within \\boxed{}."
    )
    messages = [{"role": "user", "content": question + suffix}]
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    except (TypeError, ValueError):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_annotated_text(sample: JSON, tokenizer, prompt_style: str, append_im_end: bool) -> str:
    prompt = phase2_prompt(sample) if prompt_style == "phase2" else chat_prompt(
        tokenizer, str(sample.get("instruction", ""))
    )
    output = str(sample.get("output", ""))
    if append_im_end and "<|im_end|>" not in output[-30:]:
        output += "<|im_end|>"
    return prompt + output


def tensorize(tokenizer, text: str, device: torch.device, max_length: int) -> torch.Tensor:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if max_length > 0:
        ids = ids[:max_length]
    return torch.tensor([ids], device=device, dtype=torch.long)


def logits_metrics(logits: torch.Tensor, swi_id: int, top_k: int = 5) -> JSON:
    logits = logits.float()
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = float(-(probs * log_probs).sum().item())
    p_swi = float(probs[swi_id].item())
    swi_logit = logits[swi_id]
    rank_swi = int((logits > swi_logit).sum().item() + 1)
    if logits.numel() > 1:
        tmp = logits.clone()
        tmp[swi_id] = -torch.inf
        margin = float((swi_logit - tmp.max()).item())
    else:
        margin = 0.0
    top_vals, top_ids = torch.topk(probs, k=min(top_k, probs.numel()))
    return {
        "entropy": entropy,
        "p_swi": p_swi,
        "rank_swi": rank_swi,
        "swi_logit_margin": margin,
        "top_token_ids": [int(x) for x in top_ids.tolist()],
        "top_probs": [float(x) for x in top_vals.tolist()],
    }


def selected_position_metrics(
    model,
    input_ids: torch.Tensor,
    target_positions: Sequence[int],
    swi_id: int,
    top_k: int,
    chunk_size: int,
    layer_indices: Optional[Sequence[int]] = None,
) -> Tuple[Dict[int, JSON], Dict[int, List[torch.Tensor]]]:
    """Return metrics for next-token logits predicting target_positions.

    For target position t, the model is advanced through token t-1 and the last
    logits of that prefix are used to score the target token at t.
    """
    if not target_positions:
        return {}, {}
    positions = sorted({int(p) for p in target_positions if p > 0 and p < input_ids.shape[1]})
    if not positions:
        return {}, {}

    metrics: Dict[int, JSON] = {}
    hidden_out: Dict[int, List[torch.Tensor]] = {}
    past = None
    cursor = 0
    total_len = input_ids.shape[1]
    device = input_ids.device
    need_hidden = layer_indices is not None

    with torch.no_grad():
        for target_pos in positions:
            last_logits = None
            last_hidden_states = None
            while cursor < target_pos:
                end = min(target_pos, cursor + chunk_size)
                chunk = input_ids[:, cursor:end]
                attn_len = end
                kwargs = {
                    "input_ids": chunk,
                    "attention_mask": torch.ones((1, attn_len), device=device, dtype=torch.long),
                    "position_ids": torch.arange(cursor, end, device=device).unsqueeze(0),
                    "use_cache": True,
                    "output_hidden_states": need_hidden,
                }
                if past is not None:
                    kwargs["past_key_values"] = past
                out = model(**kwargs)
                past = out.past_key_values
                last_logits = out.logits[0, -1, :]
                if need_hidden:
                    last_hidden_states = [h[0, -1, :].detach().cpu() for h in out.hidden_states]
                cursor = end
            if last_logits is None:
                continue
            row = logits_metrics(last_logits, swi_id=swi_id, top_k=top_k)
            row["target_token_id"] = int(input_ids[0, target_pos].item())
            row["target_pos"] = int(target_pos)
            row["sequence_len"] = int(total_len)
            metrics[target_pos] = row
            if need_hidden and last_hidden_states is not None:
                hidden_out[target_pos] = [
                    last_hidden_states[idx if idx >= 0 else len(last_hidden_states) + idx]
                    for idx in layer_indices
                ]
    return metrics, hidden_out


def find_teacher_forced_positions(
    ids: List[int],
    swi_start_id: int,
    swi_end_id: int,
    controls_per_event: int,
    rng: random.Random,
) -> List[Tuple[str, int, Optional[int]]]:
    special = {swi_start_id, swi_end_id}
    events: List[Tuple[str, int, Optional[int]]] = []
    switch_positions = [i for i, tok in enumerate(ids) if tok == swi_start_id and i > 0]
    end_positions = [i for i, tok in enumerate(ids) if tok == swi_end_id and i > 0]
    for pos in switch_positions:
        events.append(("swi_start", pos, pos))
        if pos - 1 > 0:
            events.append(("pre_swi_control", pos - 1, pos))
        if pos + 1 < len(ids):
            events.append(("post_swi_control", pos + 1, pos))
    for pos in end_positions:
        events.append(("swi_end", pos, pos))
    candidates = [
        i for i in range(1, len(ids))
        if ids[i] not in special and ids[i - 1] not in special
    ]
    rng.shuffle(candidates)
    n_random = min(len(candidates), max(controls_per_event * max(len(switch_positions), 1), controls_per_event))
    for pos in candidates[:n_random]:
        events.append(("random_control", pos, None))
    return events


def context_window(tokenizer, ids: List[int], pos: int, radius: int) -> str:
    left = max(0, pos - radius)
    right = min(len(ids), pos + radius + 1)
    return tokenizer.decode(ids[left:right], skip_special_tokens=False).replace("\n", "\\n")


def write_jsonl(path: str, rows: Iterable[JSON]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str, rows: List[JSON]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def cmd_teacher_forced(args: argparse.Namespace) -> None:
    loaded = load_model(
        args.adapter_path,
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        bf16=not args.fp16,
        use_modelscope=args.use_modelscope,
    )
    rng = random.Random(args.seed)
    rows: List[JSON] = []
    layer_indices = [int(x) for x in args.probe_layers.split(",")] if args.save_hidden_states else None
    hidden_rows: List[JSON] = []
    for sample_idx, sample in enumerate(tqdm(iter_jsonl(args.data_file, args.n_samples), desc="teacher-forced")):
        text = build_annotated_text(sample, loaded.tokenizer, args.prompt_style, args.append_im_end)
        input_ids = tensorize(loaded.tokenizer, text, loaded.device, args.max_length)
        ids = input_ids[0].tolist()
        events = find_teacher_forced_positions(
            ids,
            loaded.token_cfg.swi_start_id,
            loaded.token_cfg.swi_end_id,
            args.controls_per_event,
            rng,
        )
        positions = [pos for _, pos, _ in events]
        metrics, hidden = selected_position_metrics(
            loaded.model,
            input_ids,
            positions,
            loaded.token_cfg.swi_start_id,
            args.top_k,
            args.chunk_size,
            layer_indices=layer_indices,
        )
        entropies = [m["entropy"] for m in metrics.values()]
        sorted_entropy = sorted(entropies)
        for event_type, pos, anchor in events:
            if pos not in metrics:
                continue
            row = dict(metrics[pos])
            entropy = float(row["entropy"])
            percentile = (
                sum(1 for x in sorted_entropy if x <= entropy) / max(len(sorted_entropy), 1)
            )
            row.update({
                "checkpoint": loaded.label,
                "sample_idx": sample_idx,
                "dataset_index": sample.get("dataset_index", sample_idx),
                "event_type": event_type,
                "anchor_pos": anchor,
                "entropy_percentile_scored": percentile,
                "target_text": loaded.tokenizer.decode([row["target_token_id"]], skip_special_tokens=False),
                "context": context_window(loaded.tokenizer, ids, pos, args.context_radius),
                "instruction": str(sample.get("instruction", ""))[:300],
                "is_correct_annotation": sample.get("is_correct"),
            })
            rows.append(row)
            if args.save_hidden_states and pos in hidden:
                for layer_idx, vec in zip(layer_indices or [], hidden[pos]):
                    hidden_rows.append({
                        "sample_idx": sample_idx,
                        "event_type": event_type,
                        "target_pos": pos,
                        "layer": layer_idx,
                        "hidden": vec.tolist(),
                    })
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.hidden_output_jsonl and hidden_rows:
        write_jsonl(args.hidden_output_jsonl, hidden_rows)


def summarize_switch_window(rows: List[JSON]) -> List[JSON]:
    grouped: Dict[int, List[JSON]] = defaultdict(list)
    for row in rows:
        grouped[int(row["relative_pos"])].append(row)

    out = []
    for rel, group in sorted(grouped.items()):
        out.append({
            "relative_pos": rel,
            "n": len(group),
            "entropy_mean": mean([float(x["entropy"]) for x in group]),
            "p_swi_mean": mean([float(x["p_swi"]) for x in group]),
            "rank_swi_mean": mean([float(x["rank_swi"]) for x in group]),
            "margin_mean": mean([float(x["swi_logit_margin"]) for x in group]),
            "is_swi_token_rate": mean([1.0 if x.get("is_swi_token") else 0.0 for x in group]),
        })
    return out


def cmd_switch_window(args: argparse.Namespace) -> None:
    """Compute local metric curves around teacher-forced <swi> positions."""
    loaded = load_model(
        args.adapter_path,
        base_model=args.base_model,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        bf16=not args.fp16,
        use_modelscope=args.use_modelscope,
    )
    rows: List[JSON] = []
    for sample_idx, sample in enumerate(tqdm(iter_jsonl(args.data_file, args.n_samples), desc="switch-window")):
        text = build_annotated_text(sample, loaded.tokenizer, args.prompt_style, args.append_im_end)
        input_ids = tensorize(loaded.tokenizer, text, loaded.device, args.max_length)
        ids = input_ids[0].tolist()
        anchors = [i for i, tok in enumerate(ids) if tok == loaded.token_cfg.swi_start_id and i > 0]
        if not anchors:
            continue

        wanted = set()
        anchor_to_positions: Dict[int, List[Tuple[int, int]]] = {}
        for anchor_idx, anchor in enumerate(anchors):
            positions = []
            for rel in range(-args.window, args.window + 1):
                pos = anchor + rel
                if pos <= 0 or pos >= len(ids):
                    continue
                positions.append((rel, pos))
                wanted.add(pos)
            anchor_to_positions[anchor_idx] = positions

        metrics, _ = selected_position_metrics(
            loaded.model,
            input_ids,
            sorted(wanted),
            loaded.token_cfg.swi_start_id,
            args.top_k,
            args.chunk_size,
        )

        for anchor_idx, positions in anchor_to_positions.items():
            anchor = anchors[anchor_idx]
            for rel, pos in positions:
                if pos not in metrics:
                    continue
                row = dict(metrics[pos])
                token_id = int(ids[pos])
                row.update({
                    "checkpoint": loaded.label,
                    "sample_idx": sample_idx,
                    "dataset_index": sample.get("dataset_index", sample_idx),
                    "anchor_idx": anchor_idx,
                    "anchor_pos": anchor,
                    "relative_pos": rel,
                    "target_text": loaded.tokenizer.decode([token_id], skip_special_tokens=False),
                    "is_swi_token": token_id == loaded.token_cfg.swi_start_id,
                    "is_swi_end_token": token_id == loaded.token_cfg.swi_end_id,
                    "context": context_window(loaded.tokenizer, ids, pos, args.context_radius),
                    "instruction": str(sample.get("instruction", ""))[:300],
                    "is_correct_annotation": sample.get("is_correct"),
                })
                rows.append(row)

    summary = summarize_switch_window(rows)
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_csv:
        write_csv(args.summary_csv, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def load_eval_items(dataset: str, n_samples: Optional[int], seed: int) -> List[JSON]:
    from datasets import load_dataset

    if dataset == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        items = [{"question": x["question"], "gold": str(x["answer"]).split("####")[-1].strip()} for x in ds]
    elif dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        items = [{"question": x["problem"], "gold": x["answer"], "subject": x.get("subject", ""), "level": x.get("level", "")} for x in ds]
    else:
        items = []
        for i, x in enumerate(iter_jsonl(dataset, n_samples)):
            question = str(x.get("instruction", x.get("question", "")))
            if x.get("input"):
                question += "\n" + str(x["input"])
            gold = str(x.get("ground_truth_answer", x.get("answer", "")))
            items.append({"question": question, "gold": gold, "dataset_index": x.get("dataset_index", i)})
    if n_samples is not None and len(items) > n_samples:
        rng = random.Random(seed)
        indices = list(range(len(items)))
        rng.shuffle(indices)
        items = [items[i] for i in sorted(indices[:n_samples])]
    return items


def grade_answer(text: str, gold: str, dataset: str) -> Tuple[bool, str]:
    try:
        if dataset == "gsm8k":
            ok, extracted = gsm8k_grader(text, gold)
        else:
            ok, extracted = math500_grader(text, gold)
        return bool(ok), str(extracted)
    except Exception:
        numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
        extracted = numbers[-1] if numbers else ""
        return extracted == str(gold).strip(), extracted


def trace_generate_one(
    loaded: LoadedModel,
    question: str,
    max_new_tokens: int,
    min_latent_steps: int,
    top_k: int,
) -> Tuple[str, List[JSON], List[dict]]:
    model = loaded.model
    tokenizer = loaded.tokenizer
    cfg = loaded.token_cfg
    prompt = chat_prompt(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = input_ids.shape[1]

    events: List[JSON] = []
    latent_info: List[dict] = []
    generated: List[int] = []
    inputs_embeds = model.get_input_embeddings()(input_ids)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=torch.arange(prompt_len, device=loaded.device).unsqueeze(0),
        use_cache=True,
        output_hidden_states=False,
    )
    past = outputs.past_key_values
    cur_pos = prompt_len
    logits = outputs.logits[0, -1, :]

    for step in range(max_new_tokens):
        metrics = logits_metrics(logits, cfg.swi_start_id, top_k=top_k)
        token = int(torch.argmax(logits).item())
        metrics.update({
            "step": step,
            "position": cur_pos,
            "token_id": token,
            "token_text": tokenizer.decode([token], skip_special_tokens=False),
            "is_swi": token == cfg.swi_start_id,
            "is_eos": token == cfg.eos_token_id,
            "visible_prefix": tokenizer.decode(generated[-80:], skip_special_tokens=False).replace("\n", "\\n"),
        })
        events.append(metrics)
        generated.append(token)
        if token == cfg.eos_token_id:
            break
        if token == cfg.swi_start_id:
            emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            cur_pos += 1
            n_lat = 0
            natural_exit = False
            for _ in range(256):
                h = outputs.hidden_states[-1][:, -1:, :]
                outputs = model(
                    inputs_embeds=h,
                    attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                    position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values
                cur_pos += 1
                n_lat += 1
                exit_logits = outputs.logits[0, -1, :]
                exit_metrics = logits_metrics(exit_logits, cfg.swi_start_id, top_k=top_k)
                exit_metrics.update({
                    "step": step,
                    "position": cur_pos,
                    "event_type": "latent_exit_check",
                    "latent_step": n_lat,
                    "p_swi_end": float(F.softmax(exit_logits.float(), dim=-1)[cfg.swi_end_id].item()),
                })
                events.append(exit_metrics)
                if n_lat >= min_latent_steps and int(torch.argmax(exit_logits).item()) == cfg.swi_end_id:
                    natural_exit = True
                    break
            generated.append(cfg.swi_end_id)
            latent_info.append({"position": len(generated) - 1, "n_latent_steps": n_lat, "natural_exit": natural_exit})
            emb = model.get_input_embeddings()(torch.tensor([[cfg.swi_end_id]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
            )
            past = outputs.past_key_values
            cur_pos += 1
            logits = outputs.logits[0, -1, :]
            continue

        emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
        outputs = model(
            inputs_embeds=emb,
            attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
            position_ids=torch.tensor([[cur_pos]], device=loaded.device),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
        )
        past = outputs.past_key_values
        cur_pos += 1
        logits = outputs.logits[0, -1, :]
    return tokenizer.decode(generated, skip_special_tokens=False), events, latent_info


def generate_with_latent_intervention(
    loaded: LoadedModel,
    question: str,
    max_new_tokens: int,
    min_latent_steps: int,
    max_latent_steps_per_block: int,
    mode: str,
    seed: int,
) -> Tuple[str, List[dict]]:
    """Generate with optional intervention on latent hidden states.

    Modes:
      - normal: use latent hidden state unchanged
      - zero: replace each latent state with zeros
      - random_norm: replace with random direction at the original norm
      - skip: emit </swi> immediately with zero latent steps
    """
    model = loaded.model
    tokenizer = loaded.tokenizer
    cfg = loaded.token_cfg
    prompt = chat_prompt(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = input_ids.shape[1]

    generated: List[int] = []
    latent_info: List[dict] = []
    inputs_embeds = model.get_input_embeddings()(input_ids)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=torch.arange(prompt_len, device=loaded.device).unsqueeze(0),
        use_cache=True,
        output_hidden_states=False,
    )
    past = outputs.past_key_values
    cur_pos = prompt_len
    logits = outputs.logits[0, -1, :]

    for _ in range(max_new_tokens):
        token = int(torch.argmax(logits).item())
        generated.append(token)
        if token == cfg.eos_token_id:
            break

        if token == cfg.swi_start_id:
            emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            cur_pos += 1

            n_lat = 0
            natural_exit = False
            if mode != "skip":
                for _ in range(max_latent_steps_per_block):
                    h = outputs.hidden_states[-1][:, -1:, :]
                    if mode == "normal":
                        latent_input = h
                    elif mode == "zero":
                        latent_input = torch.zeros_like(h)
                    elif mode == "random_norm":
                        # Use a deterministic per-sample seed from caller while
                        # preserving the original hidden-state norm.
                        torch.manual_seed(seed + cur_pos)
                        rnd = torch.randn_like(h)
                        rnd = rnd / (rnd.norm(dim=-1, keepdim=True) + 1e-6)
                        latent_input = rnd * h.norm(dim=-1, keepdim=True)
                    else:
                        raise ValueError(f"Unknown intervention mode: {mode}")

                    outputs = model(
                        inputs_embeds=latent_input,
                        attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                        position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                        past_key_values=past,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    past = outputs.past_key_values
                    cur_pos += 1
                    n_lat += 1
                    exit_logits = outputs.logits[0, -1, :]
                    if n_lat >= min_latent_steps and int(torch.argmax(exit_logits).item()) == cfg.swi_end_id:
                        natural_exit = True
                        break

            generated.append(cfg.swi_end_id)
            latent_info.append({
                "position": len(generated) - 1,
                "n_latent_steps": n_lat,
                "natural_exit": natural_exit,
                "intervention": mode,
            })
            emb = model.get_input_embeddings()(torch.tensor([[cfg.swi_end_id]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
            )
            past = outputs.past_key_values
            cur_pos += 1
            logits = outputs.logits[0, -1, :]
            continue

        emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
        outputs = model(
            inputs_embeds=emb,
            attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
            position_ids=torch.tensor([[cur_pos]], device=loaded.device),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
        )
        past = outputs.past_key_values
        cur_pos += 1
        logits = outputs.logits[0, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=False), latent_info


def decode_hidden_topk(loaded: LoadedModel, h: torch.Tensor, top_k: int) -> JSON:
    """Apply the model's lm_head to a hidden state and decode top-k tokens."""
    base = loaded.model
    if hasattr(base, "base_model"):
        # PeftModel forwards lm_head access, but unwrap when needed.
        lm_head = getattr(base, "lm_head", None)
    else:
        lm_head = getattr(base, "lm_head", None)
    if lm_head is None:
        raise RuntimeError("Model does not expose lm_head for logit-lens decoding.")
    logits = lm_head(h).float().squeeze(0).squeeze(0)
    probs = F.softmax(logits, dim=-1)
    top_probs, top_ids = torch.topk(probs, k=min(top_k, probs.numel()))
    return {
        "top_token_ids": [int(x) for x in top_ids.tolist()],
        "top_tokens": [
            loaded.tokenizer.decode([int(x)], skip_special_tokens=False)
            for x in top_ids.tolist()
        ],
        "top_probs": [float(x) for x in top_probs.tolist()],
        "entropy": float(-(probs * torch.log(probs + 1e-12)).sum().item()),
    }


def generate_latent_logit_lens_one(
    loaded: LoadedModel,
    question: str,
    max_new_tokens: int,
    min_latent_steps: int,
    top_k: int,
    max_latent_records: int,
) -> Tuple[str, List[JSON], List[dict]]:
    """Generate normally and decode latent hidden states via lm_head."""
    model = loaded.model
    tokenizer = loaded.tokenizer
    cfg = loaded.token_cfg
    prompt = chat_prompt(tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt").to(loaded.device)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids))
    prompt_len = input_ids.shape[1]

    rows: List[JSON] = []
    latent_info: List[dict] = []
    generated: List[int] = []
    inputs_embeds = model.get_input_embeddings()(input_ids)
    outputs = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        position_ids=torch.arange(prompt_len, device=loaded.device).unsqueeze(0),
        use_cache=True,
        output_hidden_states=False,
    )
    past = outputs.past_key_values
    cur_pos = prompt_len
    logits = outputs.logits[0, -1, :]

    for step in range(max_new_tokens):
        token = int(torch.argmax(logits).item())
        generated.append(token)
        if token == cfg.eos_token_id:
            break
        if token == cfg.swi_start_id:
            visible_before = tokenizer.decode(generated[-120:], skip_special_tokens=False).replace("\n", "\\n")
            emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=True,
            )
            past = outputs.past_key_values
            cur_pos += 1
            n_lat = 0
            natural_exit = False
            block_idx = len(latent_info)
            for _ in range(256):
                h = outputs.hidden_states[-1][:, -1:, :]
                if len(rows) < max_latent_records:
                    lens = decode_hidden_topk(loaded, h, top_k)
                    lens.update({
                        "block_idx": block_idx,
                        "latent_step": n_lat + 1,
                        "position": cur_pos,
                        "visible_prefix": visible_before,
                    })
                    rows.append(lens)
                outputs = model(
                    inputs_embeds=h,
                    attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                    position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                    past_key_values=past,
                    use_cache=True,
                    output_hidden_states=True,
                )
                past = outputs.past_key_values
                cur_pos += 1
                n_lat += 1
                exit_logits = outputs.logits[0, -1, :]
                if n_lat >= min_latent_steps and int(torch.argmax(exit_logits).item()) == cfg.swi_end_id:
                    natural_exit = True
                    break
            generated.append(cfg.swi_end_id)
            latent_info.append({"position": len(generated) - 1, "n_latent_steps": n_lat, "natural_exit": natural_exit})
            emb = model.get_input_embeddings()(torch.tensor([[cfg.swi_end_id]], device=loaded.device))
            outputs = model(
                inputs_embeds=emb,
                attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
                position_ids=torch.tensor([[cur_pos]], device=loaded.device),
                past_key_values=past,
                use_cache=True,
                output_hidden_states=False,
            )
            past = outputs.past_key_values
            cur_pos += 1
            logits = outputs.logits[0, -1, :]
            continue

        emb = model.get_input_embeddings()(torch.tensor([[token]], device=loaded.device))
        outputs = model(
            inputs_embeds=emb,
            attention_mask=torch.ones((1, cur_pos + 1), device=loaded.device),
            position_ids=torch.tensor([[cur_pos]], device=loaded.device),
            past_key_values=past,
            use_cache=True,
            output_hidden_states=False,
        )
        past = outputs.past_key_values
        cur_pos += 1
        logits = outputs.logits[0, -1, :]

    return tokenizer.decode(generated, skip_special_tokens=False), rows, latent_info


def cmd_latent_logit_lens(args: argparse.Namespace) -> None:
    loaded = load_model(args.adapter_path, args.base_model, args.tokenizer_path, args.device, not args.fp16, args.use_modelscope)
    items = load_eval_items(args.dataset, args.n_samples, args.seed)
    rows: List[JSON] = []
    sample_rows: List[JSON] = []
    for sample_idx, item in enumerate(tqdm(items, desc="latent-logit-lens")):
        with torch.no_grad():
            generated, latent_rows, latent_info = generate_latent_logit_lens_one(
                loaded,
                item["question"],
                args.max_new_tokens,
                args.min_latent_steps,
                args.top_k,
                args.max_latent_records_per_sample,
            )
        answer_text = generated
        think_end = generated.find("</think>")
        if think_end != -1:
            answer_text = generated[think_end + len("</think>"):]
        ok, extracted = grade_answer(answer_text, item["gold"], "gsm8k" if args.dataset == "gsm8k" else "math500")
        for row in latent_rows:
            row.update({
                "checkpoint": loaded.label,
                "sample_idx": sample_idx,
                "question": item["question"][:300],
                "gold": item["gold"],
                "extracted": extracted,
                "correct": ok,
            })
            rows.append(row)
        sample_rows.append({
            "checkpoint": loaded.label,
            "sample_idx": sample_idx,
            "question": item["question"][:300],
            "gold": item["gold"],
            "extracted": extracted,
            "correct": ok,
            "n_swi": generated.count(SWI_START_TOKEN),
            "latent_blocks": len(latent_info),
            "latent_steps": sum(x["n_latent_steps"] for x in latent_info),
            "n_generated_tokens": len(loaded.tokenizer.encode(generated, add_special_tokens=False)),
            "generated": generated if args.save_full_text else generated[:1000],
        })
    write_jsonl(args.output_jsonl, rows)
    write_jsonl(args.sample_output_jsonl, sample_rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)


def cmd_generate_trace(args: argparse.Namespace) -> None:
    loaded = load_model(args.adapter_path, args.base_model, args.tokenizer_path, args.device, not args.fp16, args.use_modelscope)
    items = load_eval_items(args.dataset, args.n_samples, args.seed)
    rows: List[JSON] = []
    sample_rows: List[JSON] = []
    for sample_idx, item in enumerate(tqdm(items, desc="generate-trace")):
        with torch.no_grad():
            generated, events, latent_info = trace_generate_one(
                loaded,
                item["question"],
                args.max_new_tokens,
                args.min_latent_steps,
                args.top_k,
            )
        answer_text = generated
        think_end = generated.find("</think>")
        if think_end != -1:
            answer_text = generated[think_end + len("</think>"):]
        ok, extracted = grade_answer(answer_text, item["gold"], "gsm8k" if args.dataset == "gsm8k" else "math500")
        n_swi = generated.count(SWI_START_TOKEN)
        sample_rows.append({
            "checkpoint": loaded.label,
            "sample_idx": sample_idx,
            "question": item["question"][:300],
            "gold": item["gold"],
            "extracted": extracted,
            "correct": ok,
            "n_swi": n_swi,
            "latent_blocks": len(latent_info),
            "latent_steps": sum(x["n_latent_steps"] for x in latent_info),
            "n_generated_tokens": len(loaded.tokenizer.encode(generated, add_special_tokens=False)),
            "generated": generated if args.save_full_text else generated[:1000],
        })
        for ev in events:
            ev.update({
                "checkpoint": loaded.label,
                "sample_idx": sample_idx,
                "correct": ok,
                "gold": item["gold"],
            })
            rows.append(ev)
    write_jsonl(args.output_jsonl, rows)
    write_jsonl(args.sample_output_jsonl, sample_rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)


def summarize_intervention(rows: List[JSON]) -> List[JSON]:
    grouped: Dict[str, List[JSON]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mode"])].append(row)

    out = []
    for mode, group in sorted(grouped.items()):
        out.append({
            "mode": mode,
            "n": len(group),
            "accuracy": mean([1.0 if r["correct"] else 0.0 for r in group]),
            "answer_change_rate": mean([1.0 if r.get("answer_changed") else 0.0 for r in group]),
            "avg_tokens": mean([float(r["n_generated_tokens"]) for r in group]),
            "avg_swi": mean([float(r["n_swi"]) for r in group]),
            "avg_latent_blocks": mean([float(r["latent_blocks"]) for r in group]),
            "avg_latent_steps": mean([float(r["latent_steps"]) for r in group]),
        })
    return out


def cmd_latent_intervention(args: argparse.Namespace) -> None:
    loaded = load_model(args.adapter_path, args.base_model, args.tokenizer_path, args.device, not args.fp16, args.use_modelscope)
    items = load_eval_items(args.dataset, args.n_samples, args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if "normal" not in modes:
        modes = ["normal"] + modes

    rows: List[JSON] = []
    for sample_idx, item in enumerate(tqdm(items, desc="latent-intervention")):
        normal_ref: Optional[JSON] = None
        for mode in modes:
            with torch.no_grad():
                generated, latent_info = generate_with_latent_intervention(
                    loaded,
                    item["question"],
                    args.max_new_tokens,
                    args.min_latent_steps,
                    args.max_latent_steps_per_block,
                    mode,
                    args.seed + sample_idx * 1009,
                )
            answer_text = generated
            think_end = generated.find("</think>")
            if think_end != -1:
                answer_text = generated[think_end + len("</think>"):]
            ok, extracted = grade_answer(answer_text, item["gold"], "gsm8k" if args.dataset == "gsm8k" else "math500")
            row: JSON = {
                "checkpoint": loaded.label,
                "sample_idx": sample_idx,
                "mode": mode,
                "question": item["question"][:300],
                "gold": item["gold"],
                "extracted": extracted,
                "correct": ok,
                "n_swi": generated.count(SWI_START_TOKEN),
                "latent_blocks": len(latent_info),
                "latent_steps": sum(x["n_latent_steps"] for x in latent_info),
                "n_generated_tokens": len(loaded.tokenizer.encode(generated, add_special_tokens=False)),
                "generated": generated if args.save_full_text else generated[:1000],
            }
            if mode == "normal":
                normal_ref = {
                    "extracted": extracted,
                    "correct": ok,
                    "generated": generated,
                }
                row["answer_changed"] = False
                row["correct_changed"] = False
            elif normal_ref is not None:
                row["normal_extracted"] = normal_ref["extracted"]
                row["normal_correct"] = normal_ref["correct"]
                row["answer_changed"] = extracted != normal_ref["extracted"]
                row["correct_changed"] = bool(ok) != bool(normal_ref["correct"])
            rows.append(row)

    summary = summarize_intervention(rows)
    write_jsonl(args.output_jsonl, rows)
    if args.output_csv:
        write_csv(args.output_csv, rows)
    if args.summary_json:
        Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_json).write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    if args.summary_csv:
        write_csv(args.summary_csv, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def aggregate_rows(path: str) -> List[JSON]:
    return list(iter_jsonl(path))


def mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def group_name_from_sample(row: JSON) -> str:
    correct = bool(row.get("correct"))
    has_swi = int(row.get("n_swi", 0)) > 0
    return f"{'correct' if correct else 'wrong'}_{'with_swi' if has_swi else 'no_swi'}"


def cmd_generation_analysis(args: argparse.Namespace) -> None:
    """Analyze generation traces without rerunning the model."""
    sample_rows = aggregate_rows(args.samples_jsonl)
    event_rows = aggregate_rows(args.events_jsonl)

    sample_groups: Dict[str, List[JSON]] = defaultdict(list)
    for row in sample_rows:
        sample_groups[group_name_from_sample(row)].append(row)

    sample_summary = []
    for group_name, group in sorted(sample_groups.items()):
        sample_summary.append({
            "group": group_name,
            "n": len(group),
            "accuracy": mean([1.0 if r.get("correct") else 0.0 for r in group]),
            "avg_tokens": mean([float(r.get("n_generated_tokens", 0)) for r in group]),
            "avg_swi": mean([float(r.get("n_swi", 0)) for r in group]),
            "avg_latent_blocks": mean([float(r.get("latent_blocks", 0)) for r in group]),
            "avg_latent_steps": mean([float(r.get("latent_steps", 0)) for r in group]),
        })

    switch_groups: Dict[str, List[JSON]] = defaultdict(list)
    for row in event_rows:
        if row.get("is_swi") is True:
            switch_groups["correct" if row.get("correct") else "wrong"].append(row)

    switch_summary = []
    for group_name, group in sorted(switch_groups.items()):
        switch_summary.append({
            "group": group_name,
            "n": len(group),
            "entropy_mean": mean([float(r["entropy"]) for r in group if "entropy" in r]),
            "p_swi_mean": mean([float(r["p_swi"]) for r in group if "p_swi" in r]),
            "rank_swi_mean": mean([float(r["rank_swi"]) for r in group if "rank_swi" in r]),
            "margin_mean": mean([float(r["swi_logit_margin"]) for r in group if "swi_logit_margin" in r]),
        })

    exit_groups: Dict[Tuple[str, int], List[JSON]] = defaultdict(list)
    for row in event_rows:
        if row.get("event_type") != "latent_exit_check":
            continue
        key = ("correct" if row.get("correct") else "wrong", int(row.get("latent_step", -1)))
        exit_groups[key].append(row)

    exit_curve = []
    for (group_name, latent_step), group in sorted(exit_groups.items(), key=lambda x: (x[0][0], x[0][1])):
        exit_curve.append({
            "group": group_name,
            "latent_step": latent_step,
            "n": len(group),
            "p_swi_end_mean": mean([float(r["p_swi_end"]) for r in group if "p_swi_end" in r]),
        })

    result = {
        "sample_summary": sample_summary,
        "switch_summary": switch_summary,
        "exit_curve": exit_curve,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    if args.sample_group_csv:
        write_csv(args.sample_group_csv, sample_summary)
    if args.switch_group_csv:
        write_csv(args.switch_group_csv, switch_summary)
    if args.exit_curve_csv:
        write_csv(args.exit_curve_csv, exit_curve)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000])


def cmd_compare(args: argparse.Namespace) -> None:
    rows = []
    for file_path in args.inputs:
        for row in aggregate_rows(file_path):
            row = dict(row)
            row["_source"] = file_path
            rows.append(row)
    grouped: Dict[Tuple[str, str], List[JSON]] = defaultdict(list)
    for row in rows:
        grouped[(str(row.get("checkpoint", "")), str(row.get("event_type", row.get("is_swi", ""))))].append(row)
    out_rows = []
    for (ckpt, event), group in sorted(grouped.items()):
        out_rows.append({
            "checkpoint": ckpt,
            "event_type": event,
            "n": len(group),
            "entropy_mean": mean([float(x["entropy"]) for x in group if "entropy" in x]),
            "p_swi_mean": mean([float(x["p_swi"]) for x in group if "p_swi" in x]),
            "rank_swi_mean": mean([float(x["rank_swi"]) for x in group if "rank_swi" in x]),
            "margin_mean": mean([float(x["swi_logit_margin"]) for x in group if "swi_logit_margin" in x]),
        })
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output_json).write_text(json.dumps(out_rows, indent=2, ensure_ascii=False))
    if args.output_csv:
        write_csv(args.output_csv, out_rows)
    print(json.dumps(out_rows[:20], indent=2, ensure_ascii=False))


def cmd_probe(args: argparse.Namespace) -> None:
    loaded = load_model(args.adapter_path, args.base_model, args.tokenizer_path, args.device, not args.fp16, args.use_modelscope)
    rng = random.Random(args.seed)
    layer_indices = [int(x) for x in args.layers.split(",")]
    vectors: Dict[int, List[torch.Tensor]] = {l: [] for l in layer_indices}
    labels: List[int] = []
    meta: List[JSON] = []
    for sample_idx, sample in enumerate(tqdm(iter_jsonl(args.data_file, args.n_samples), desc="probe-collect")):
        text = build_annotated_text(sample, loaded.tokenizer, args.prompt_style, args.append_im_end)
        input_ids = tensorize(loaded.tokenizer, text, loaded.device, args.max_length)
        ids = input_ids[0].tolist()
        events = find_teacher_forced_positions(
            ids, loaded.token_cfg.swi_start_id, loaded.token_cfg.swi_end_id, 1, rng
        )
        events = [(e, p, a) for e, p, a in events if e in {"swi_start", "random_control"}]
        positions = [p for _, p, _ in events]
        _, hidden = selected_position_metrics(
            loaded.model, input_ids, positions, loaded.token_cfg.swi_start_id,
            args.top_k, args.chunk_size, layer_indices=layer_indices
        )
        for event_type, pos, _ in events:
            if pos not in hidden:
                continue
            labels.append(1 if event_type == "swi_start" else 0)
            meta.append({"sample_idx": sample_idx, "event_type": event_type, "target_pos": pos})
            for layer, vec in zip(layer_indices, hidden[pos]):
                vectors[layer].append(vec.float())

    results = []
    y = torch.tensor(labels, dtype=torch.float32)
    if len(set(labels)) < 2 or len(labels) < 10:
        raise RuntimeError("Need both switch and control examples for probing.")
    idx = torch.randperm(len(labels), generator=torch.Generator().manual_seed(args.seed))
    split = max(1, int(0.8 * len(labels)))
    train_idx, test_idx = idx[:split], idx[split:]
    for layer in layer_indices:
        X = torch.stack(vectors[layer])
        X = (X - X.mean(0, keepdim=True)) / (X.std(0, keepdim=True) + 1e-6)
        clf = torch.nn.Linear(X.shape[1], 1)
        opt = torch.optim.AdamW(clf.parameters(), lr=args.probe_lr, weight_decay=1e-3)
        for _ in range(args.probe_epochs):
            logits = clf(X[train_idx]).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, y[train_idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            test_logits = clf(X[test_idx]).squeeze(-1)
            pred = (torch.sigmoid(test_logits) > 0.5).float()
            acc = float((pred == y[test_idx]).float().mean().item())
        results.append({"layer": layer, "n": len(labels), "test_accuracy": acc})
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(json.dumps(results, indent=2, ensure_ascii=False))


def cmd_report(args: argparse.Namespace) -> None:
    lines = ["# Swi Switch Interpretability Report", ""]
    if args.teacher_forced:
        rows = aggregate_rows(args.teacher_forced)
        lines += ["## Teacher-Forced Switch Metrics", ""]
        for key in ["swi_start", "swi_end", "random_control", "pre_swi_control", "post_swi_control"]:
            group = [r for r in rows if r.get("event_type") == key]
            if not group:
                continue
            lines.append(
                f"- `{key}` n={len(group)} entropy={mean([float(r['entropy']) for r in group]):.3f} "
                f"p_swi={mean([float(r['p_swi']) for r in group]):.6f} "
                f"rank_swi={mean([float(r['rank_swi']) for r in group]):.1f}"
            )
        lines.append("")
    if args.generation_trace:
        rows = aggregate_rows(args.generation_trace)
        switch_rows = [r for r in rows if r.get("is_swi") is True]
        nonswitch = [r for r in rows if r.get("is_swi") is False and "entropy" in r]
        lines += ["## Generation Switch Trace", ""]
        lines.append(f"- switch events: {len(switch_rows)}")
        if switch_rows:
            lines.append(f"- switch entropy mean: {mean([float(r['entropy']) for r in switch_rows]):.3f}")
            lines.append(f"- switch p_swi mean: {mean([float(r['p_swi']) for r in switch_rows]):.6f}")
        if nonswitch:
            lines.append(f"- non-switch entropy mean: {mean([float(r['entropy']) for r in nonswitch]):.3f}")
            lines.append(f"- non-switch p_swi mean: {mean([float(r['p_swi']) for r in nonswitch]):.6f}")
        lines.append("")
    if args.probe_json and Path(args.probe_json).is_file():
        results = json.loads(Path(args.probe_json).read_text())
        lines += ["## Probe Results", ""]
        for r in results:
            lines.append(f"- layer {r['layer']}: test_accuracy={r['test_accuracy']:.3f} (n={r['n']})")
        lines.append("")
    lines += [
        "## Claim-Level Reading Guide",
        "",
        "- Retention is supported if annotated `<swi>` positions show higher `p(<swi>)`, lower rank, or larger margins than controls.",
        "- Entropy-driven switching is supported if switch positions are high entropy or high local entropy-percentile relative to controls.",
        "- A learned confident switch policy is supported if entropy is low but `p(<swi>)` and margin are high at switch positions.",
        "- A changed policy is suggested if generated switches do not align with annotated positions but correlate with different context features.",
        "",
    ]
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text("\n".join(lines))
    print(args.output_md)


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--base_model", default=None)
    parser.add_argument("--tokenizer_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_modelscope", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="SwiLatent switch interpretability tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_tf = sub.add_parser("teacher-forced")
    add_model_args(p_tf)
    p_tf.add_argument("--data_file", default=str(PROJECT_ROOT / "data_hf/data/math_openr1_all.jsonl"))
    p_tf.add_argument("--n_samples", type=int, default=50)
    p_tf.add_argument("--max_length", type=int, default=4096)
    p_tf.add_argument("--chunk_size", type=int, default=128)
    p_tf.add_argument("--top_k", type=int, default=5)
    p_tf.add_argument("--controls_per_event", type=int, default=2)
    p_tf.add_argument("--context_radius", type=int, default=24)
    p_tf.add_argument("--prompt_style", choices=["phase2", "chat"], default="phase2")
    p_tf.add_argument("--append_im_end", action="store_true")
    p_tf.add_argument("--seed", type=int, default=42)
    p_tf.add_argument("--output_jsonl", required=True)
    p_tf.add_argument("--output_csv", default=None)
    p_tf.add_argument("--save_hidden_states", action="store_true")
    p_tf.add_argument("--probe_layers", default="-1")
    p_tf.add_argument("--hidden_output_jsonl", default=None)
    p_tf.set_defaults(func=cmd_teacher_forced)

    p_win = sub.add_parser("switch-window")
    add_model_args(p_win)
    p_win.add_argument("--data_file", default=str(PROJECT_ROOT / "data_hf/data/math_openr1_all.jsonl"))
    p_win.add_argument("--n_samples", type=int, default=50)
    p_win.add_argument("--max_length", type=int, default=4096)
    p_win.add_argument("--chunk_size", type=int, default=128)
    p_win.add_argument("--top_k", type=int, default=5)
    p_win.add_argument("--window", type=int, default=8)
    p_win.add_argument("--context_radius", type=int, default=12)
    p_win.add_argument("--prompt_style", choices=["phase2", "chat"], default="phase2")
    p_win.add_argument("--append_im_end", action="store_true")
    p_win.add_argument("--output_jsonl", required=True)
    p_win.add_argument("--output_csv", default=None)
    p_win.add_argument("--summary_json", default=None)
    p_win.add_argument("--summary_csv", default=None)
    p_win.set_defaults(func=cmd_switch_window)

    p_gen = sub.add_parser("generate-trace")
    add_model_args(p_gen)
    p_gen.add_argument("--dataset", default="math500", help="math500, gsm8k, or local jsonl")
    p_gen.add_argument("--n_samples", type=int, default=20)
    p_gen.add_argument("--max_new_tokens", type=int, default=1024)
    p_gen.add_argument("--min_latent_steps", type=int, default=4)
    p_gen.add_argument("--top_k", type=int, default=5)
    p_gen.add_argument("--seed", type=int, default=42)
    p_gen.add_argument("--output_jsonl", required=True)
    p_gen.add_argument("--sample_output_jsonl", required=True)
    p_gen.add_argument("--output_csv", default=None)
    p_gen.add_argument("--save_full_text", action="store_true")
    p_gen.set_defaults(func=cmd_generate_trace)

    p_int = sub.add_parser("latent-intervention")
    add_model_args(p_int)
    p_int.add_argument("--dataset", default="math500", help="math500, gsm8k, or local jsonl")
    p_int.add_argument("--n_samples", type=int, default=20)
    p_int.add_argument("--max_new_tokens", type=int, default=1024)
    p_int.add_argument("--min_latent_steps", type=int, default=4)
    p_int.add_argument("--max_latent_steps_per_block", type=int, default=256)
    p_int.add_argument("--modes", default="normal,zero,random_norm,skip")
    p_int.add_argument("--seed", type=int, default=42)
    p_int.add_argument("--output_jsonl", required=True)
    p_int.add_argument("--output_csv", default=None)
    p_int.add_argument("--summary_json", default=None)
    p_int.add_argument("--summary_csv", default=None)
    p_int.add_argument("--save_full_text", action="store_true")
    p_int.set_defaults(func=cmd_latent_intervention)

    p_lens = sub.add_parser("latent-logit-lens")
    add_model_args(p_lens)
    p_lens.add_argument("--dataset", default="math500", help="math500, gsm8k, or local jsonl")
    p_lens.add_argument("--n_samples", type=int, default=10)
    p_lens.add_argument("--max_new_tokens", type=int, default=1024)
    p_lens.add_argument("--min_latent_steps", type=int, default=4)
    p_lens.add_argument("--top_k", type=int, default=10)
    p_lens.add_argument("--max_latent_records_per_sample", type=int, default=32)
    p_lens.add_argument("--seed", type=int, default=42)
    p_lens.add_argument("--output_jsonl", required=True)
    p_lens.add_argument("--sample_output_jsonl", required=True)
    p_lens.add_argument("--output_csv", default=None)
    p_lens.add_argument("--save_full_text", action="store_true")
    p_lens.set_defaults(func=cmd_latent_logit_lens)

    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("--inputs", nargs="+", required=True)
    p_cmp.add_argument("--output_json", default=None)
    p_cmp.add_argument("--output_csv", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_ga = sub.add_parser("generation-analysis")
    p_ga.add_argument("--events_jsonl", required=True)
    p_ga.add_argument("--samples_jsonl", required=True)
    p_ga.add_argument("--output_json", default=None)
    p_ga.add_argument("--sample_group_csv", default=None)
    p_ga.add_argument("--switch_group_csv", default=None)
    p_ga.add_argument("--exit_curve_csv", default=None)
    p_ga.set_defaults(func=cmd_generation_analysis)

    p_probe = sub.add_parser("probe")
    add_model_args(p_probe)
    p_probe.add_argument("--data_file", default=str(PROJECT_ROOT / "data_hf/data/math_openr1_all.jsonl"))
    p_probe.add_argument("--n_samples", type=int, default=50)
    p_probe.add_argument("--max_length", type=int, default=4096)
    p_probe.add_argument("--chunk_size", type=int, default=128)
    p_probe.add_argument("--top_k", type=int, default=5)
    p_probe.add_argument("--prompt_style", choices=["phase2", "chat"], default="phase2")
    p_probe.add_argument("--append_im_end", action="store_true")
    p_probe.add_argument("--layers", default="-1")
    p_probe.add_argument("--probe_epochs", type=int, default=100)
    p_probe.add_argument("--probe_lr", type=float, default=1e-3)
    p_probe.add_argument("--seed", type=int, default=42)
    p_probe.add_argument("--output_json", required=True)
    p_probe.set_defaults(func=cmd_probe)

    p_report = sub.add_parser("report")
    p_report.add_argument("--teacher_forced", default=None)
    p_report.add_argument("--generation_trace", default=None)
    p_report.add_argument("--probe_json", default=None)
    p_report.add_argument("--output_md", required=True)
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
