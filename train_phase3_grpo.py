"""
Phase 3 – Custom GRPO RL Training with Latent Inference
=========================================================
Trains the Phase-2 CoconutSwiModel with GRPO using latent-mode rollout.

Unlike the verl-based approach, this script uses ``CoconutSwiModel.generate_rl()``
for rollout, so the model performs actual hidden-state injection during RL.
This ensures training-evaluation alignment.

Algorithm (per optimizer step):
  1. For each question, sample G responses via ``generate_rl(min_latent_steps=K)``
  2. Score each with ``compute_score`` (correctness + efficiency + format)
  3. GRPO advantage: ``A_i = (r_i - mean(r)) / (std(r) + eps)``
  4. Re-compute log-probs under current policy via ``compute_response_log_probs``
  5. Clipped surrogate loss + optional KL penalty
  6. Backward + optimizer step

Usage:
    torchrun --standalone --nproc_per_node=8 train_phase3_grpo.py \\
        --model_path  outputs/phase2_2_v1_bf16/stage_008 \\
        --data_file   data_hf/data/math_openr1_all.jsonl \\
        --save_dir    outputs/phase3_grpo_v1
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

from src.model.coconut_swi_model import CoconutSwiModel
from src.rl.grpo import compute_grpo_advantages, grpo_clipped_loss, kl_penalty_loss
from src.rl.reward import compute_score


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3 – GRPO RL with latent inference")

    p.add_argument("--model_path", type=str, required=True,
                    help="Phase-2 LoRA adapter directory")
    p.add_argument("--data_file", type=str, required=True,
                    help="JSONL with 'instruction' and 'ground_truth_answer'")
    p.add_argument("--save_dir", type=str, required=True)

    # GRPO
    p.add_argument("--group_size", type=int, default=5,
                    help="G: number of rollout samples per question")
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--kl_coef", type=float, default=0.001,
                    help="KL penalty coefficient (0 = disable)")

    # Generation
    p.add_argument("--min_latent_steps", type=int, default=8,
                    help="Minimum latent steps per <swi> block during rollout")
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--latent_noise_scale", type=float, default=0.0,
                    help="Relative Gaussian noise scale for latent actions; 0 disables.")
    p.add_argument("--record_latent_actions", action="store_true",
                    help="Store sampled latent action tensors for dry-run debugging.")
    p.add_argument("--latent_replay", action="store_true",
                    help="Run a no-grad second pass per rollout that recomputes "
                         "new_log_prob(z | mu_current, sigma_old) for each saved "
                         "latent action and logs latent_log_ratio statistics. "
                         "Does NOT contribute to the loss.")

    # Optimiser
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--grad_accum", type=int, default=1,
                    help="Accumulate across this many questions before stepping")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--bf16", action="store_true", default=True)

    # Logging / saving
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--report_to", type=str, default="swanlab",
                    choices=["none", "swanlab"])
    p.add_argument("--run_name", type=str, default="swilatent_phase3_grpo")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class MathQuestionDataset(Dataset):
    def __init__(self, data, tokenizer, max_prompt_len=1024):
        self.items = []
        for d in data:
            q = d.get("instruction", "")
            inp = d.get("input", "")
            if inp:
                q = f"{q}\n{inp}"
            gold = d.get("ground_truth_answer", "")
            if not gold:
                continue

            content = q + "\nPlease reason step by step concisely, and put your final answer within \\boxed{}."
            messages = [{"role": "user", "content": content}]
            try:
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=True,
                )
            except (TypeError, ValueError):
                ids = tokenizer.apply_chat_template(
                    messages, tokenize=True,
                    add_generation_prompt=True,
                )
            if len(ids) > max_prompt_len:
                continue
            self.items.append({
                "prompt_ids": ids,
                "gold": gold,
                "question": q,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        ids = item["prompt_ids"]
        # Normalize various tokenizer return types to a flat List[int].
        if hasattr(ids, "ids"):
            ids = ids.ids
        elif isinstance(ids, dict) and "input_ids" in ids:
            ids = ids["input_ids"]
        elif hasattr(ids, "input_ids"):
            ids = ids.input_ids
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return {
            "prompt_ids": torch.tensor(ids, dtype=torch.long),
            "gold": item["gold"],
            "question": item["question"],
        }


def _collate(batch):
    """Custom collate: batch_size=1 already, just unwrap. Avoids default
    collate's attempt to stack variable-length prompts / Encoding objects."""
    return batch[0]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_log_file = None
_swan_run = None


def _log(data: dict):
    global _log_file, _swan_run
    if _log_file is not None:
        data["time"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _log_file.write(json.dumps(data, ensure_ascii=False) + "\n")
        _log_file.flush()
    if _swan_run is not None:
        try:
            numeric = {k: v for k, v in data.items()
                       if isinstance(v, (int, float, bool))}
            if numeric:
                _swan_run.log(numeric)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group("nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    is_main = (rank == 0)

    # ------------------------------------------------------------------
    # 1. Load model (same pattern as train_phase2.py)
    # ------------------------------------------------------------------
    if is_main:
        print(f"\n{'='*60}")
        print("Phase 3 – GRPO RL with Latent Inference")
        print(f"{'='*60}")
        print(f"  Model        : {args.model_path}")
        print(f"  Data         : {args.data_file}")
        print(f"  Group size   : {args.group_size}")
        print(f"  Min latent   : {args.min_latent_steps}")
        print(f"  LR           : {args.lr}")
        print(f"  KL coef      : {args.kl_coef}")
        print(f"  Epochs       : {args.epochs}")
        print(f"  GPUs         : {world_size}")
        print(f"{'='*60}\n")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_path = args.model_path
    if not os.path.isabs(adapter_path):
        adapter_path = os.path.join(os.getcwd(), adapter_path)

    adapter_cfg_path = os.path.join(adapter_path, "adapter_config.json")
    with open(adapter_cfg_path) as f:
        adapter_info = json.load(f)
    base_model_path = adapter_info.get("base_model_name_or_path", "")

    use_modelscope = os.environ.get("USE_MODELSCOPE", "0") == "1"
    if use_modelscope:
        from modelscope.hub.snapshot_download import snapshot_download
        ms_id = os.environ.get("MODELSCOPE_MODEL_ID", "Qwen/Qwen3-8B")
        base_model_path = snapshot_download(ms_id)
        if is_main:
            print(f"[Phase 3] ModelScope base: {base_model_path}")

    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    from src.setup_tokens import SwiTokenConfig, SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN
    token_cfg = SwiTokenConfig(
        swi_start_id=tokenizer.convert_tokens_to_ids(SWI_START_TOKEN),
        swi_end_id=tokenizer.convert_tokens_to_ids(SWI_END_TOKEN),
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        eos_token_id=tokenizer.eos_token_id,
    )
    if is_main:
        print(f"[Phase 3] Token IDs: swi_start={token_cfg.swi_start_id}, "
              f"swi_end={token_cfg.swi_end_id}, latent={token_cfg.latent_id}")

    dtype = torch.bfloat16 if args.bf16 else torch.float16

    def _load_model(is_trainable: bool):
        m = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=dtype, trust_remote_code=True
        )
        if len(tokenizer) != m.config.vocab_size:
            m.resize_token_embeddings(len(tokenizer))
            if is_main:
                print(f"[Phase 3] Resized embeddings: {m.config.vocab_size} -> {len(tokenizer)}")
        from peft import PeftModel as PM
        m = PM.from_pretrained(m, adapter_path, is_trainable=is_trainable)
        return m

    model = _load_model(is_trainable=True)
    if is_main:
        model.print_trainable_parameters()

    coconut_model = CoconutSwiModel(model, token_cfg)
    coconut_model = coconut_model.to(device)

    if is_main and args.kl_coef > 0:
        print(f"[Phase 3] KL penalty enabled (coef={args.kl_coef}), "
              f"using old_log_probs as anchor (no ref model needed)")

    # ------------------------------------------------------------------
    # 2. Data
    # ------------------------------------------------------------------
    with open(args.data_file) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]

    dataset = MathQuestionDataset(raw_data, tokenizer)
    if is_main:
        print(f"[Phase 3] Dataset: {len(dataset)} questions")

    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset, batch_size=1, shuffle=(sampler is None),
        sampler=sampler, num_workers=0,
        collate_fn=_collate,
    )

    # ------------------------------------------------------------------
    # 3. Optimizer + Scheduler
    # ------------------------------------------------------------------
    trainable_params = [p for p in coconut_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    total_steps = len(loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.1, 0.5 * (1 + __import__("math").cos(__import__("math").pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ------------------------------------------------------------------
    # 4. Logging
    # ------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    global _log_file, _swan_run
    if is_main:
        _log_file = open(os.path.join(args.save_dir, "trainer_log.jsonl"), "a")

        if args.report_to == "swanlab":
            try:
                import swanlab
                _swan_run = swanlab.init(
                    project="SwiLatent",
                    experiment_name=args.run_name,
                    config=vars(args),
                )
                print(f"[Phase 3] SwanLab enabled (run_name={args.run_name})")
            except Exception as exc:
                print(f"[Phase 3] SwanLab init failed: {exc}")
                _swan_run = None

    # ------------------------------------------------------------------
    # 5. Training loop
    # ------------------------------------------------------------------
    global_step = 0
    G = args.group_size

    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)

        coconut_model.train()
        optimizer.zero_grad()
        accum_loss = 0.0
        accum_reward = 0.0
        accum_correct = 0
        accum_total = 0
        accum_swi_count = 0
        accum_latent_correct = 0
        accum_latent_total = 0
        accum_nolatent_correct = 0
        accum_nolatent_total = 0
        accum_visible_tokens = 0
        accum_latent_blocks = 0
        accum_latent_action_count = 0
        accum_latent_old_lp = 0.0
        accum_latent_noise_ratio = 0.0
        accum_latent_mu_norm = 0.0
        accum_latent_z_norm = 0.0
        # Replay-only accumulators (filled when args.latent_replay is set).
        accum_latent_replayed = 0
        accum_latent_new_lp = 0.0
        accum_latent_log_ratio_sum = 0.0
        accum_latent_log_ratio_sq_sum = 0.0
        accum_latent_log_ratio_min = float("inf")
        accum_latent_log_ratio_max = float("-inf")
        accum_latent_replay_failed = 0
        accum_latent_mu_new_norm = 0.0

        pbar = tqdm(
            total=len(loader) // args.grad_accum,
            desc=f"Epoch {epoch+1}/{args.epochs}",
            disable=(not is_main),
            dynamic_ncols=True,
        )

        for step, item in enumerate(loader):
            prompt_ids = item["prompt_ids"]  # 1D tensor from __getitem__
            gold = item["gold"]

            prompt_tensor = prompt_ids.unsqueeze(0).to(device)

            # === Rollout (no grad) ===
            coconut_model.eval()
            rollouts = []
            for g in range(G):
                try:
                    result = coconut_model.generate_rl(
                        prompt_tensor,
                        max_new_tokens=args.max_new_tokens,
                        min_latent_steps=args.min_latent_steps,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        latent_noise_scale=args.latent_noise_scale,
                        record_latent_actions=args.record_latent_actions,
                    )
                    rollouts.append(result)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if is_main:
                        print(f"[OOM] Skipping rollout g={g} at step {step}")
                    continue

            if len(rollouts) < 2:
                continue

            # === Reward ===
            rewards = []
            for g_idx, r in enumerate(rollouts):
                resp_text = tokenizer.decode(r["response_ids"], skip_special_tokens=False)
                n_visible_tokens = sum(
                    1
                    for tok_id in r["response_ids"]
                    if tok_id not in {
                        token_cfg.swi_start_id,
                        token_cfg.swi_end_id,
                        token_cfg.eos_token_id,
                    }
                )
                reward = compute_score(
                    data_source="swilatent_math",
                    solution_str=resp_text,
                    ground_truth=gold,
                    extra_info={
                        "n_visible_tokens": n_visible_tokens,
                        "n_latent_blocks": len(r.get("latent_info", [])),
                    },
                )
                rewards.append(reward)

                # Debug: print first 2 rollouts of first 3 steps
                if is_main and step == 0 and g_idx == 0:
                    decoded_prompt = tokenizer.decode(prompt_tensor[0][:50], skip_special_tokens=False)
                    print(f"[DEBUG] prompt tokens (first 50 decoded): {decoded_prompt}", flush=True)
                    print(f"[DEBUG] prompt_tensor shape: {prompt_tensor.shape}, "
                          f"first 10 ids: {prompt_tensor[0][:10].tolist()}", flush=True)
                if is_main and step < 3 and g_idx < 2:
                    try:
                        q_text = str(item["question"][0])[:150]
                    except Exception:
                        q_text = "N/A"
                    print(f"\n[DEBUG] step={step} g={g_idx} reward={reward:.1f} "
                          f"gold={gold}", flush=True)
                    print(f"[DEBUG] question: {q_text}", flush=True)
                    print(f"[DEBUG] response (last 300): ...{resp_text[-300:]}", flush=True)
                    print(f"[DEBUG] latent_blocks={len(r['latent_info'])}, "
                          f"resp_len={len(r['response_ids'])}", flush=True)

                has_swi = any(t == token_cfg.swi_start_id for t in r["response_ids"])
                is_correct = reward > 0
                accum_swi_count += int(has_swi)
                accum_total += 1
                accum_visible_tokens += n_visible_tokens
                accum_latent_blocks += len(r["latent_info"])
                latent_stats = r.get("latent_action_stats", {})
                n_lat_actions = int(latent_stats.get("n_actions", 0))
                if n_lat_actions > 0:
                    accum_latent_action_count += n_lat_actions
                    accum_latent_old_lp += (
                        latent_stats.get("old_log_prob_mean", 0.0) * n_lat_actions
                    )
                    accum_latent_noise_ratio += (
                        latent_stats.get("noise_ratio_mean", 0.0) * n_lat_actions
                    )
                    accum_latent_mu_norm += (
                        latent_stats.get("mu_norm_mean", 0.0) * n_lat_actions
                    )
                    accum_latent_z_norm += (
                        latent_stats.get("z_norm_mean", 0.0) * n_lat_actions
                    )
                if has_swi:
                    accum_latent_total += 1
                    if is_correct:
                        accum_latent_correct += 1
                else:
                    accum_nolatent_total += 1
                    if is_correct:
                        accum_nolatent_correct += 1
                if is_correct:
                    accum_correct += 1

            accum_reward += sum(rewards) / len(rewards)

            # === GRPO Advantage ===
            advantages = compute_grpo_advantages(rewards).to(device)

            # === Policy Gradient (with grad) ===
            coconut_model.train()

            # Per-response forward + backward (micro-accumulation).
            # Only 1 autograd graph alive at a time to avoid OOM.
            n_survived = 0

            loss_scale = 1.0 / (len(rollouts) * args.grad_accum)

            for i, r in enumerate(rollouts):
                resp_full = r["response_ids_full"]
                resp_tensor = torch.tensor([resp_full], device=device)
                full_ids = torch.cat([prompt_tensor, resp_tensor], dim=1)

                old_lp = torch.tensor(r["old_log_probs"], device=device)
                mask = torch.tensor(r["sampled_mask"], device=device, dtype=torch.float32)
                adv_i = advantages[i].item()

                # === Optional latent-action replay (no grad, no loss) ===
                # Runs BEFORE the policy-gradient backward so it doesn't hold
                # an autograd graph; falls back silently on OOM.
                if (
                    args.latent_replay
                    and args.latent_noise_scale > 0
                    and len(r.get("latent_zs", [])) > 0
                ):
                    coconut_model.eval()
                    try:
                        replay = coconut_model.compute_latent_action_log_probs(
                            full_ids,
                            len(resp_full),
                            r["latent_zs"],
                            r["latent_sigmas"],
                            latent_old_log_probs=r["latent_action_old_log_probs"],
                        )
                        new_lps = replay["new_log_probs"]
                        ratios = replay["log_ratios"]
                        mu_new_norms = replay["mu_new_norms"]
                        accum_latent_replayed += len(new_lps)
                        accum_latent_new_lp += sum(new_lps)
                        accum_latent_log_ratio_sum += sum(ratios)
                        accum_latent_log_ratio_sq_sum += sum(x * x for x in ratios)
                        if ratios:
                            accum_latent_log_ratio_min = min(
                                accum_latent_log_ratio_min, min(ratios)
                            )
                            accum_latent_log_ratio_max = max(
                                accum_latent_log_ratio_max, max(ratios)
                            )
                        accum_latent_mu_new_norm += sum(mu_new_norms)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        accum_latent_replay_failed += 1
                        if is_main:
                            print(f"[OOM] Skipping latent_replay at step {step}, g={i}")
                    coconut_model.train()

                try:
                    resp_loss = coconut_model.segmented_grpo_backward(
                        full_ids, len(resp_full),
                        old_lp, mask, adv_i,
                        clip_eps=args.clip_eps,
                        kl_coef=args.kl_coef,
                        loss_scale=loss_scale,
                    )
                    accum_loss += resp_loss
                    n_survived += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if is_main:
                        print(f"[OOM] Skipping log_prob at step {step}, g={i}")
                    continue

            if n_survived < 2:
                optimizer.zero_grad()
                continue

            # === Step ===
            if (step + 1) % args.grad_accum == 0 or step == len(loader) - 1:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                pbar.update(1)

                if is_main:
                    avg_reward = accum_reward / max(args.grad_accum, 1)
                    accuracy = accum_correct / max(accum_total, 1) * 100
                    swi_rate = accum_swi_count / max(accum_total, 1) * 100
                    latent_acc = accum_latent_correct / max(accum_latent_total, 1) * 100
                    nolatent_acc = accum_nolatent_correct / max(accum_nolatent_total, 1) * 100
                    avg_vis_tok = accum_visible_tokens / max(accum_total, 1)
                    avg_lat_blk = accum_latent_blocks / max(accum_total, 1)
                    avg_lat_actions = accum_latent_action_count / max(accum_total, 1)
                    avg_lat_old_lp = accum_latent_old_lp / max(accum_latent_action_count, 1)
                    avg_lat_noise = (
                        accum_latent_noise_ratio / max(accum_latent_action_count, 1)
                    )
                    avg_lat_mu_norm = (
                        accum_latent_mu_norm / max(accum_latent_action_count, 1)
                    )
                    avg_lat_z_norm = (
                        accum_latent_z_norm / max(accum_latent_action_count, 1)
                    )
                    n_rep = accum_latent_replayed
                    avg_lat_new_lp = accum_latent_new_lp / max(n_rep, 1)
                    avg_lat_log_ratio = accum_latent_log_ratio_sum / max(n_rep, 1)
                    if n_rep > 0:
                        var_log_ratio = (
                            accum_latent_log_ratio_sq_sum / n_rep
                            - avg_lat_log_ratio * avg_lat_log_ratio
                        )
                        std_log_ratio = (max(var_log_ratio, 0.0)) ** 0.5
                        lat_log_ratio_min = accum_latent_log_ratio_min
                        lat_log_ratio_max = accum_latent_log_ratio_max
                    else:
                        std_log_ratio = 0.0
                        lat_log_ratio_min = 0.0
                        lat_log_ratio_max = 0.0
                    avg_lat_mu_new_norm = accum_latent_mu_new_norm / max(n_rep, 1)
                    cur_lr = scheduler.get_last_lr()[0]
                    pbar.set_postfix(
                        loss=f"{accum_loss:.4f}",
                        reward=f"{avg_reward:.2f}",
                        acc=f"{accuracy:.0f}%",
                        lat=f"{latent_acc:.0f}%",
                        swi=f"{swi_rate:.0f}%",
                        tok=f"{avg_vis_tok:.0f}",
                        blk=f"{avg_lat_blk:.1f}",
                    )

                    _log({
                        "event": "train_step",
                        "epoch": epoch,
                        "global_step": global_step,
                        "loss": round(accum_loss, 6),
                        "avg_reward": round(avg_reward, 4),
                        "accuracy": round(accuracy, 2),
                        "latent_accuracy": round(latent_acc, 2),
                        "nolatent_accuracy": round(nolatent_acc, 2),
                        "swi_rate": round(swi_rate, 2),
                        "avg_visible_tokens": round(avg_vis_tok, 1),
                        "avg_latent_blocks": round(avg_lat_blk, 2),
                        "avg_latent_actions": round(avg_lat_actions, 2),
                        "latent_action_old_log_prob": round(avg_lat_old_lp, 6),
                        "latent_noise_ratio": round(avg_lat_noise, 6),
                        "latent_mu_norm": round(avg_lat_mu_norm, 4),
                        "latent_z_norm": round(avg_lat_z_norm, 4),
                        "latent_samples": accum_latent_total,
                        "nolatent_samples": accum_nolatent_total,
                        "latent_replayed_actions": n_rep,
                        "latent_new_log_prob": round(avg_lat_new_lp, 6),
                        "latent_log_ratio_mean": round(avg_lat_log_ratio, 6),
                        "latent_log_ratio_std": round(std_log_ratio, 6),
                        "latent_log_ratio_min": round(
                            lat_log_ratio_min if n_rep > 0 else 0.0, 6
                        ),
                        "latent_log_ratio_max": round(
                            lat_log_ratio_max if n_rep > 0 else 0.0, 6
                        ),
                        "latent_mu_new_norm": round(avg_lat_mu_new_norm, 4),
                        "latent_replay_failed": accum_latent_replay_failed,
                        "lr": cur_lr,
                    })

                accum_loss = 0.0
                accum_reward = 0.0
                accum_correct = 0
                accum_total = 0
                accum_swi_count = 0
                accum_latent_correct = 0
                accum_latent_total = 0
                accum_nolatent_correct = 0
                accum_nolatent_total = 0
                accum_visible_tokens = 0
                accum_latent_blocks = 0
                accum_latent_action_count = 0
                accum_latent_old_lp = 0.0
                accum_latent_noise_ratio = 0.0
                accum_latent_mu_norm = 0.0
                accum_latent_z_norm = 0.0
                accum_latent_replayed = 0
                accum_latent_new_lp = 0.0
                accum_latent_log_ratio_sum = 0.0
                accum_latent_log_ratio_sq_sum = 0.0
                accum_latent_log_ratio_min = float("inf")
                accum_latent_log_ratio_max = float("-inf")
                accum_latent_replay_failed = 0
                accum_latent_mu_new_norm = 0.0

            # === Save checkpoint ===
            if is_main and global_step > 0 and global_step % args.save_steps == 0:
                ckpt_dir = os.path.join(args.save_dir, f"step_{global_step:06d}")
                os.makedirs(ckpt_dir, exist_ok=True)
                unwrapped = coconut_model.base_causallm
                if hasattr(unwrapped, "module"):
                    unwrapped = unwrapped.module
                unwrapped.save_pretrained(ckpt_dir)
                tokenizer.save_pretrained(ckpt_dir)
                print(f"\n[Phase 3] Checkpoint saved: {ckpt_dir}")

        pbar.close()
        if world_size > 1:
            dist.barrier()

        # Epoch checkpoint
        if is_main:
            ckpt_dir = os.path.join(args.save_dir, f"epoch_{epoch}")
            os.makedirs(ckpt_dir, exist_ok=True)
            unwrapped = coconut_model.base_causallm
            if hasattr(unwrapped, "module"):
                unwrapped = unwrapped.module
            unwrapped.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"\n[Phase 3] Epoch {epoch} checkpoint saved: {ckpt_dir}")
            _log({"event": "epoch_end", "epoch": epoch, "global_step": global_step})

    if is_main:
        print("\n[Phase 3] Training complete!")
        _log({"event": "train_complete", "global_step": global_step})
        _log_file.close()
        if _swan_run is not None:
            try:
                _swan_run.finish()
            except Exception:
                pass

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
