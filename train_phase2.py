"""
Phase 2 – Coconut Curriculum Training Entry Point
===================================================
Trains a :class:`~src.model.coconut_swi_model.CoconutSwiModel` using the
multi-stage curriculum defined in :mod:`src.data.coconut_swi_dataset`.

Training schedule
-----------------
  Stage 0:  SFT warm-up — ``<swi>text</swi>`` kept as-is (same as Phase 1),
            lets the model build a strong baseline before any latent injection.
  Stage 1:  Replace ``<swi>`` content with 1×c_thought ``<latent>`` tokens.
  …
  Stage K:  Replace with K×c_thought ``<latent>`` tokens (K = max_latent_stage).

Each stage runs for ``epochs_per_stage`` epochs.  The curriculum advances in
the outer loop; the inner loop is a standard PyTorch DDP/FSDP training loop
identical to coconut/run.py L200-400.

Usage
-----
    torchrun --standalone --nnodes=1 --nproc_per_node=4 train_phase2.py \\
        --model_path  /path/to/phase1_checkpoint \\
        --data_file   /path/to/openr1_swi_annotated.jsonl \\
        --save_dir    /path/to/phase2_checkpoints \\
        --max_latent_stage  8 \\
        --c_thought         1 \\
        --epochs_per_stage  1 \\
        --lr                1e-4 \\
        --batch_size        4

Reference
---------
  coconut/run.py        – full DDP training loop, FSDP setup, LR scheduling
  coconut/dataset.py    – MyCollator, get_cot_latent_dataset
  src/model/coconut_swi_model.py  – the model to train
  src/data/coconut_swi_dataset.py – dataset & collator for this phase
"""

from __future__ import annotations

import argparse
import contextlib
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 2 – Coconut curriculum training")

    # Model & data
    parser.add_argument("--model_path",  type=str, required=True,
                        help="Path to Phase-1 SFT checkpoint (with <swi>/</ swi> tokens).")
    parser.add_argument("--data_file",   type=str, required=True,
                        help="JSONL with 'problem' and 'solution' (containing <swi>text</swi>).")
    parser.add_argument("--save_dir",    type=str, required=True,
                        help="Directory for saving per-stage checkpoints.")

    # Curriculum
    parser.add_argument("--max_latent_stage", type=int, default=8,
                        help="Maximum curriculum stage index.")
    parser.add_argument("--c_thought",   type=int, default=1,
                        help=(
                            "phase2-1: fixed latent tokens per replaced span; "
                            "phase2-2: latent growth factor per stage."
                        ))
    parser.add_argument("--epochs_per_stage", type=int, default=1)
    parser.add_argument("--uniform_prob", type=float, default=0.1,
                        help="Prob of randomly sampling stage (curriculum mixing).")
    parser.add_argument(
        "--phase2_strategy",
        type=str,
        default="phase2-2",
        choices=["phase2-1", "phase2-2"],
        help=(
            "phase2-1: sequentially replace leftmost <swi> spans; "
            "phase2-2: replace all spans in parallel (original behavior)."
        ),
    )
    parser.add_argument("--max_total_latent", type=int, default=0,
                        help=(
                            "Hard cap on total <latent> tokens per sample. "
                            "Once the budget is exhausted, remaining spans keep text. "
                            "0 = no cap (default)."
                        ))

    # Optimiser
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--batch_size",  type=int,   default=4)
    parser.add_argument("--max_length",  type=int,   default=2048)
    parser.add_argument("--bf16",        action="store_true", default=True)
    parser.add_argument("--load_in_4bit", action="store_true",
                        help="Load base model in 4-bit NF4 (match Phase-1 QLoRA precision).")
    parser.add_argument(
        "--finetuning_type",
        type=str,
        default="lora",
        choices=["lora", "full"],
        help="Train with LoRA adapters (recommended) or full-parameter finetuning.",
    )
    parser.add_argument("--lora_r", type=int, default=32,
                        help="LoRA rank when finetuning_type=lora.")
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="LoRA alpha when finetuning_type=lora.")
    parser.add_argument("--lora_dropout", type=float, default=0.05,
                        help="LoRA dropout when finetuning_type=lora.")
    parser.add_argument(
        "--lora_target",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated target module names for LoRA.",
    )

    # Infrastructure
    parser.add_argument("--resume",      type=int,   default=0,
                        help="Resume from this stage index.")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--grad_accum",  type=int,   default=4,
                        help="Gradient accumulation steps.")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="Fraction of total steps for linear warmup.")
    parser.add_argument("--stage0_epochs", type=int, default=0,
                        help="Override epochs for stage 0 (SFT warm-up). 0 = use epochs_per_stage.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save GPU memory.")
    parser.add_argument(
        "--report_to",
        type=str,
        default="swanlab",
        choices=["none", "swanlab"],
        help="Logging backend. Use 'swanlab' to mirror Phase-1 style logging.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="swilatent_phase2",
        help="Run name used by logger backend.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log metrics every N optimizer steps.",
    )

    return parser.parse_args()


def main() -> None:
    """Phase 2 training entry point – Coconut curriculum over staged latent injection."""
    import gc
    import json
    import random
    from datetime import datetime

    # ── GPU isolation: MUST happen BEFORE importing torch ──────────
    # CUDA runtime reads CUDA_VISIBLE_DEVICES at first torch CUDA call.
    # Restrict each process to its own GPU so nothing leaks to GPU 0.
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    rank        = int(os.environ.get("RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))

    all_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(all_gpus) > 1 and local_rank < len(all_gpus):
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus[local_rank].strip()

    import torch
    import torch.nn as nn
    import torch.distributed as dist
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from tqdm import tqdm

    from src.model.coconut_swi_model import CoconutSwiModel
    from src.data.coconut_swi_dataset import get_phase2_latent_dataset, SwiCollator

    args = parse_args()

    # ----------------------------------------------------------------
    # 1. Distributed setup
    # ----------------------------------------------------------------
    torch.cuda.set_device(0)       # cuda:0 is the ONLY visible GPU
    device = torch.device("cuda:0")
    if world_size > 1:
        dist.init_process_group("nccl")

    print(
        f"[Phase 2][rank {rank}] LOCAL_RANK={local_rank}, "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}, "
        f"device_count={torch.cuda.device_count()}, "
        f"device_name={torch.cuda.get_device_name(0)}"
    )

    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)

    if rank == 0:
        os.makedirs(args.save_dir, exist_ok=True)

    swan_run = None
    log_fp = None
    if rank == 0:
        log_path = os.path.join(args.save_dir, "trainer_log.jsonl")
        log_fp = open(log_path, "a", encoding="utf-8")

        if args.report_to == "swanlab":
            try:
                import swanlab
                swan_run = swanlab.init(
                    project="SwiLatent",
                    experiment_name=args.run_name,
                    config=vars(args),
                )
                print(f"[Phase 2] SwanLab enabled (run_name={args.run_name})")
            except Exception as exc:
                print(f"[Phase 2] SwanLab init failed; continue without cloud logging: {exc}")
                swan_run = None

    def _log_metrics(payload: dict) -> None:
        if rank != 0:
            return
        record = {"time": datetime.utcnow().isoformat(), **payload}
        if log_fp is not None:
            log_fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            log_fp.flush()
        if swan_run is not None:
            try:
                numeric_payload = {
                    k: v for k, v in payload.items() if isinstance(v, (int, float, bool))
                }
                # Map string event names to numeric ids for charting.
                if "event" in payload and isinstance(payload["event"], str):
                    event_map = {
                        "train_start": 0,
                        "stage_start": 1,
                        "train_step": 2,
                        "stage_checkpoint": 3,
                        "train_complete": 4,
                    }
                    numeric_payload["event_id"] = event_map.get(payload["event"], -1)
                if numeric_payload:
                    swan_run.log(numeric_payload)
            except Exception:
                pass

    # ----------------------------------------------------------------
    # 2. Resolve model/tokenizer sources
    # ----------------------------------------------------------------

    def _recover_repo_id_from_cache_path(path_str: str) -> str | None:
        """Best-effort conversion from HF cache path to repo id.

        Example:
        /.../models--Qwen--Qwen3-8B/snapshots/... -> Qwen/Qwen3-8B
        """
        import re

        m = re.search(r"models--([^/]+)--([^/]+)", path_str)
        if m is None:
            return None
        return f"{m.group(1)}/{m.group(2)}"

    # Detect LoRA checkpoint: local adapter directory or remote HF adapter repo.
    lora_path   = None
    base_path   = args.model_path
    adapter_cfg = os.path.join(args.model_path, "adapter_config.json")
    has_adapter = os.path.isfile(adapter_cfg)
    adapter_info = None

    if has_adapter:
        import json as _json
        with open(adapter_cfg) as f:
            adapter_info = _json.load(f)
    else:
        # Try to treat model_path as a remote PEFT adapter repo id by reading
        # adapter_config.json directly from HF Hub.
        try:
            import json as _json
            from huggingface_hub import hf_hub_download

            remote_adapter_cfg = hf_hub_download(args.model_path, "adapter_config.json")
            with open(remote_adapter_cfg) as f:
                adapter_info = _json.load(f)
            has_adapter = True
        except Exception:
            has_adapter = False

    if has_adapter and adapter_info is not None:
        base_path_raw = adapter_info.get("base_model_name_or_path", args.model_path)

        # Some adapters store machine-local absolute paths that are invalid on
        # a new server. Recover gracefully via env override or repo-id parsing.
        base_path_invalid_local = (
            isinstance(base_path_raw, str)
            and os.path.isabs(base_path_raw)
            and (
                (not os.path.exists(base_path_raw))
                or (not os.path.isfile(os.path.join(base_path_raw, "config.json")))
            )
        )

        if base_path_invalid_local:
            env_base = os.environ.get("BASE_MODEL_PATH")
            recovered_repo = _recover_repo_id_from_cache_path(base_path_raw)
            base_path = env_base or recovered_repo or "Qwen/Qwen3-8B"
            if rank == 0:
                print(
                    "[Phase 2] Warning: adapter base model path does not exist on this machine: "
                    f"{base_path_raw}\n"
                    f"[Phase 2] Fallback base model = {base_path} "
                    "(set BASE_MODEL_PATH to override)."
                )
        else:
            base_path = base_path_raw

        lora_path = args.model_path
        if rank == 0:
            print(f"[Phase 2] LoRA adapter detected – base model: {base_path}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Optional acceleration path: load base model from ModelScope mirror.
    use_modelscope = os.environ.get("USE_MODELSCOPE", "0") == "1" or \
        os.environ.get("BASE_MODEL_SOURCE", "").lower() == "modelscope"
    if use_modelscope:
        modelscope_model_id = os.environ.get("MODELSCOPE_MODEL_ID", base_path)
        modelscope_cache_dir = os.environ.get("MODELSCOPE_CACHE_DIR")
        if rank == 0:
            print(f"[Phase 2] Resolving base model via ModelScope: {modelscope_model_id}")
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            base_path = snapshot_download(
                model_id=modelscope_model_id,
                cache_dir=modelscope_cache_dir,
            )
            if rank == 0:
                print(f"[Phase 2] ModelScope local path: {base_path}")
        except Exception as exc:
            raise RuntimeError(
                "ModelScope loading requested but failed. "
                "Install modelscope (`pip install modelscope`) or unset USE_MODELSCOPE. "
                f"Details: {exc}"
            )

    tokenizer_source = os.environ.get("TOKENIZER_SOURCE")
    if tokenizer_source is None:
        # Default policy: prefer adapter tokenizer when available.
        # This keeps tokenizer metadata consistent with the adapter repo,
        # while setup_special_tokens() still guarantees required token IDs.
        tokenizer_source = args.model_path if has_adapter else base_path

    if rank == 0:
        print(f"[Phase 2] Loading tokenizer from {tokenizer_source}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    except Exception as tok_exc:
        # Fallback chain: adapter tokenizer -> base tokenizer.
        fallback_sources = []
        if has_adapter and tokenizer_source != args.model_path:
            fallback_sources.append(args.model_path)
        if tokenizer_source != base_path:
            fallback_sources.append(base_path)

        last_exc = tok_exc
        tokenizer = None
        for src in fallback_sources:
            try:
                if rank == 0:
                    print(
                        f"[Phase 2] Tokenizer load from {tokenizer_source} failed: {last_exc}\n"
                        f"[Phase 2] Retrying tokenizer load from: {src}"
                    )
                tokenizer = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
                break
            except Exception as exc:
                last_exc = exc

        if tokenizer is None:
            raise RuntimeError(
                "Failed to load tokenizer from all candidate sources. "
                f"primary={tokenizer_source}, fallbacks={fallback_sources}"
            ) from last_exc

    load_kwargs = dict(trust_remote_code=True)
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        load_kwargs["device_map"] = {"": device}
        if rank == 0:
            print("[Phase 2] Loading base model in 4-bit NF4 (matching Phase-1 QLoRA)")
    else:
        load_kwargs["torch_dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(base_path, **load_kwargs)

    # Ensure model is on CPU for DDP — transformers 5.x with accelerate may
    # auto-dispatch to GPU 0, which causes all DDP ranks to share one GPU.
    if not args.load_in_4bit:
        if hasattr(model, "hf_device_map"):
            print(f"[Phase 2][rank {rank}] WARNING: model has hf_device_map={model.hf_device_map}")
            try:
                from accelerate.hooks import remove_hook_from_submodules
                remove_hook_from_submodules(model)
            except Exception:
                pass
        first_param = next(model.parameters())
        print(f"[Phase 2][rank {rank}] Model device after from_pretrained: {first_param.device}")
        model = model.to("cpu")
        print(f"[Phase 2][rank {rank}] Model device after .to('cpu'): {next(model.parameters()).device}")

    # ----------------------------------------------------------------
    # 3. Register special tokens in tokenizer (model resize deferred)
    # ----------------------------------------------------------------
    # Phase 1 only added <swi> and </swi>; the adapter was saved at that
    # vocab size.  We register all three tokens in the tokenizer NOW but
    # defer model.resize_token_embeddings until after the adapter is
    # loaded so we can match its saved weight dimensions first.
    from src.setup_tokens import (
        SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN, SwiTokenConfig,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    n_tok_added = tokenizer.add_tokens(
        [SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN], special_tokens=True,
    )
    if rank == 0:
        print(f"[Phase 2] Tokenizer: added {n_tok_added} new token(s), "
              f"total len={len(tokenizer)}")

    token_cfg = SwiTokenConfig(
        swi_start_id=tokenizer.convert_tokens_to_ids(SWI_START_TOKEN),
        swi_end_id=tokenizer.convert_tokens_to_ids(SWI_END_TOKEN),
        latent_id=tokenizer.convert_tokens_to_ids(LATENT_TOKEN),
        eos_token_id=tokenizer.eos_token_id,
    )
    if rank == 0:
        print(f"[Phase 2] Token IDs: swi_start={token_cfg.swi_start_id}, "
              f"swi_end={token_cfg.swi_end_id}, latent={token_cfg.latent_id}")

    def _init_new_token_embeddings(mdl, tok_ids):
        """Copy anchor embedding into newly added token positions."""
        anchor_id = tokenizer.convert_tokens_to_ids("<<")
        if anchor_id == tokenizer.unk_token_id:
            anchor_id = tokenizer.convert_tokens_to_ids("<")
        if anchor_id == tokenizer.unk_token_id:
            anchor_id = tokenizer.pad_token_id
        inp_emb = mdl.get_input_embeddings()
        anchor_vec = inp_emb.weight.data[anchor_id].clone()
        for tid in tok_ids:
            if tid < inp_emb.weight.shape[0]:
                inp_emb.weight.data[tid] = anchor_vec
        lm_head = getattr(mdl, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            if inp_emb.weight.data_ptr() != lm_head.weight.data_ptr():
                anchor_logit = lm_head.weight.data[anchor_id].clone()
                for tid in tok_ids:
                    if tid < lm_head.weight.shape[0]:
                        lm_head.weight.data[tid] = anchor_logit

    def _grow_peft_embeddings(peft_model, old_vocab, new_vocab):
        """Manually grow embed_tokens and lm_head inside a PeftModel.

        Works around transformers 5.x refusing to resize ModulesToSaveWrapper
        (not an nn.Embedding instance).
        """
        from peft.utils.other import ModulesToSaveWrapper

        anchor_id = tokenizer.convert_tokens_to_ids("<<")
        if anchor_id == tokenizer.unk_token_id:
            anchor_id = tokenizer.convert_tokens_to_ids("<")
        if anchor_id == tokenizer.unk_token_id:
            anchor_id = tokenizer.pad_token_id

        def _grow_wrapper(wrapper, is_embedding):
            for mod in [wrapper.original_module] + list(wrapper.modules_to_save.values()):
                w = mod.weight
                if w.shape[0] >= new_vocab:
                    continue
                w_dev, w_dt = w.device, w.dtype
                anchor_row = w.data[anchor_id].clone()
                new_w = torch.zeros(new_vocab, w.shape[1], dtype=w_dt, device=w_dev)
                new_w[:old_vocab] = w.data[:old_vocab]
                for i in range(old_vocab, new_vocab):
                    new_w[i] = anchor_row
                mod.weight = nn.Parameter(new_w)
                if is_embedding:
                    mod.num_embeddings = new_vocab
                else:
                    mod.out_features = new_vocab
                    if hasattr(mod, "bias") and mod.bias is not None:
                        old_bias = mod.bias.data
                        new_bias = torch.zeros(new_vocab, dtype=w_dt, device=w_dev)
                        new_bias[:old_vocab] = old_bias[:old_vocab]
                        mod.bias = nn.Parameter(new_bias)

        # Navigate through LoraModel → base HF model
        lora_model = peft_model.base_model  # LoraModel
        hf_model = lora_model.model          # Qwen3ForCausalLM

        embed = hf_model.model.embed_tokens
        if isinstance(embed, ModulesToSaveWrapper):
            _grow_wrapper(embed, is_embedding=True)

        head = hf_model.lm_head
        if isinstance(head, ModulesToSaveWrapper):
            _grow_wrapper(head, is_embedding=False)

        # Update model config so downstream code sees the right vocab size
        if hasattr(hf_model, "config"):
            hf_model.config.vocab_size = new_vocab

    # ----------------------------------------------------------------
    # 3.5 Apply finetuning strategy (LoRA or full)
    # ----------------------------------------------------------------
    if args.finetuning_type == "lora":
        from peft import LoraConfig, PeftModel, get_peft_model

        if lora_path is not None:
            if rank == 0:
                print("[Phase 2] Continue training existing LoRA adapter")

            # Read the adapter's actual vocab size from its saved weights
            # so we can resize the base model to match BEFORE loading.
            adapter_vocab = None
            if adapter_info is not None:
                mts = adapter_info.get("modules_to_save") or []
                if any(m in mts for m in ("embed_tokens", "lm_head")):
                    adapter_safetensors = os.path.join(lora_path, "adapter_model.safetensors")
                    if os.path.isfile(adapter_safetensors):
                        from safetensors import safe_open
                        with safe_open(adapter_safetensors, framework="pt") as sf:
                            for key in sf.keys():
                                if "embed_tokens" in key and "weight" in key:
                                    adapter_vocab = sf.get_slice(key).get_shape()[0]
                                    break

            if adapter_vocab is not None:
                current_vocab = model.get_input_embeddings().weight.shape[0]
                if adapter_vocab != current_vocab:
                    if rank == 0:
                        print(
                            f"[Phase 2] Resizing base model embeddings "
                            f"{current_vocab} -> {adapter_vocab} to match adapter"
                        )
                    model.resize_token_embeddings(adapter_vocab)
            else:
                # No modules_to_save vocab issue; safe to resize to full tokenizer
                model.resize_token_embeddings(len(tokenizer))

            try:
                model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
            except TypeError as exc:
                # Backward compatibility: older peft versions may not accept
                # newly added fields (e.g., `corda_config`) in adapter_config.
                if "unexpected keyword argument" not in str(exc):
                    raise

                import inspect
                import json as _json
                from huggingface_hub import hf_hub_download

                if rank == 0:
                    print(
                        "[Phase 2] PEFT config compatibility fallback triggered: "
                        f"{exc}"
                    )

                if os.path.isdir(lora_path):
                    adapter_cfg_path = os.path.join(lora_path, "adapter_config.json")
                else:
                    adapter_cfg_path = hf_hub_download(lora_path, "adapter_config.json")

                with open(adapter_cfg_path) as f:
                    raw_cfg = _json.load(f)

                allowed_keys = set(inspect.signature(LoraConfig.__init__).parameters.keys())
                compat_cfg_kwargs = {
                    k: v for k, v in raw_cfg.items() if k in allowed_keys
                }
                removed_keys = sorted(set(raw_cfg.keys()) - set(compat_cfg_kwargs.keys()))

                # Keep training defaults safe in case the adapter config omits them.
                compat_cfg_kwargs.setdefault("task_type", "CAUSAL_LM")
                compat_cfg_kwargs.setdefault("bias", "none")

                if rank == 0 and removed_keys:
                    print(
                        "[Phase 2] Removed unsupported adapter config keys for this peft version: "
                        f"{removed_keys}"
                    )

                compat_cfg = LoraConfig(**compat_cfg_kwargs)
                model = PeftModel.from_pretrained(
                    model,
                    lora_path,
                    is_trainable=True,
                    config=compat_cfg,
                )

            # After adapter is loaded, grow embedding to accommodate any
            # tokens that Phase 1 didn't include (e.g. <latent>).
            # Standard resize_token_embeddings fails on PeftModel because
            # transformers 5.x checks isinstance(old_emb, nn.Embedding) but
            # PEFT wraps it in ModulesToSaveWrapper.  Resize manually.
            full_vocab = len(tokenizer)
            loaded_vocab = model.get_input_embeddings().weight.shape[0]
            if full_vocab > loaded_vocab:
                if rank == 0:
                    print(
                        f"[Phase 2] Growing PeftModel embeddings "
                        f"{loaded_vocab} -> {full_vocab} for new tokens"
                    )
                _grow_peft_embeddings(model, loaded_vocab, full_vocab)
                new_tok_ids = [
                    tid for tid in [
                        token_cfg.swi_start_id,
                        token_cfg.swi_end_id,
                        token_cfg.latent_id,
                    ]
                    if tid >= loaded_vocab
                ]
                if new_tok_ids and rank == 0:
                    print(f"[Phase 2] New token IDs added to embedding: {new_tok_ids}")

        else:
            model.resize_token_embeddings(len(tokenizer))
            _init_new_token_embeddings(
                model,
                [token_cfg.swi_start_id, token_cfg.swi_end_id, token_cfg.latent_id],
            )
            target_modules = [m.strip() for m in args.lora_target.split(",") if m.strip()]
            peft_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_cfg)
            if rank == 0:
                print(
                    f"[Phase 2] New LoRA attached (r={args.lora_r}, alpha={args.lora_alpha}, "
                    f"dropout={args.lora_dropout})"
                )

        if rank == 0 and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()
    else:
        # Full-parameter training: resize first, then load/merge adapter.
        model.resize_token_embeddings(len(tokenizer))
        _init_new_token_embeddings(
            model,
            [token_cfg.swi_start_id, token_cfg.swi_end_id, token_cfg.latent_id],
        )
        if lora_path is not None:
            from peft import PeftModel
            if rank == 0:
                print("[Phase 2] Merging LoRA adapter for full-parameter training …")
            model = PeftModel.from_pretrained(model, lora_path)
            model = model.merge_and_unload()
            if rank == 0:
                print("[Phase 2] LoRA merged ✓")

    # ----------------------------------------------------------------
    # 4. Wrap with CoconutSwiModel
    # ----------------------------------------------------------------
    coconut_model = CoconutSwiModel(model, token_cfg)
    coconut_model = coconut_model.to(device)

    if args.gradient_checkpointing:
        allow_gc = os.environ.get("ALLOW_GC_WITH_COCONUT", "0") == "1"
        if allow_gc:
            coconut_model.base_causallm.gradient_checkpointing_enable()
            if rank == 0:
                print("[Phase 2] Gradient checkpointing enabled (ALLOW_GC_WITH_COCONUT=1).")
        elif rank == 0:
            print(
                "[Phase 2] gradient_checkpointing requested but disabled by default "
                "for Coconut forward (it conflicts with cache-based latent passes). "
                "Set ALLOW_GC_WITH_COCONUT=1 to force-enable."
            )

    # DDP wrap when running multi-GPU
    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        coconut_model = DDP(coconut_model, device_ids=[0],
                            find_unused_parameters=False)
        if rank == 0:
            print("[Phase 2] DDP enabled")

    # ----------------------------------------------------------------
    # 5. Load training data
    # ----------------------------------------------------------------
    if rank == 0:
        print(f"[Phase 2] Loading data from {args.data_file}")
    with open(args.data_file) as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    # Only keep correct samples for curriculum training
    raw_data = [d for d in raw_data if d.get("is_correct", True)]
    if rank == 0:
        print(f"[Phase 2] {len(raw_data)} training samples")
        print(f"[Phase 2] Strategy: {args.phase2_strategy}")
        print(f"[Phase 2] Finetuning: {args.finetuning_type}")
        _log_metrics(
            {
                "event": "train_start",
                "num_samples": len(raw_data),
                "strategy": args.phase2_strategy,
                "finetuning_type": args.finetuning_type,
            }
        )

    # ----------------------------------------------------------------
    # 6. Curriculum outer loop
    # ----------------------------------------------------------------
    trainable_params = [p for p in coconut_model.parameters() if p.requires_grad]
    if rank == 0:
        n_trainable = sum(p.numel() for p in trainable_params)
        print(f"[Phase 2] Trainable parameters: {n_trainable:,}")
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    # Estimate total optimizer steps for LR scheduler
    # With DDP, each GPU processes n_samples/world_size per epoch
    n_samples = len(raw_data)
    steps_per_epoch = max(n_samples // (args.batch_size * args.grad_accum * world_size), 1)
    num_stages = args.max_latent_stage + 1  # stages 0 … max_latent_stage
    stage0_eps = args.stage0_epochs if args.stage0_epochs > 0 else args.epochs_per_stage
    total_epochs = stage0_eps + (num_stages - 1) * args.epochs_per_stage
    total_opt_steps = steps_per_epoch * total_epochs
    warmup_steps = int(total_opt_steps * args.warmup_ratio)
    if rank == 0:
        print(f"[Phase 2] Steps/epoch={steps_per_epoch}, total_epochs={total_epochs}, "
              f"total_opt_steps≈{total_opt_steps}, warmup={warmup_steps}")

    from torch.optim.lr_scheduler import LambdaLR
    import math

    def _cosine_with_warmup(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, _cosine_with_warmup)

    resume_stage = args.resume
    global_step = 0

    for stage in range(resume_stage, num_stages):
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"[Phase 2] Stage {stage}/{args.max_latent_stage}  "
                f"({'SFT warm-up' if stage == 0 else f'{stage * args.c_thought} latent units'})")
            print(f"{'='*60}")
            _log_metrics({"event": "stage_start", "stage": stage})

        # Build dataset for this stage (rebuild every stage to refresh sampling)
        dataset = get_phase2_latent_dataset(
            raw_data=raw_data,
            tokenizer=tokenizer,
            token_cfg=token_cfg,
            stage=stage,
            c_thought=args.c_thought,
            max_latent_stage=args.max_latent_stage,
            uniform_prob=args.uniform_prob,
            max_length=args.max_length,
            phase2_strategy=args.phase2_strategy,
            max_total_latent=args.max_total_latent,
        )

        collator = SwiCollator(tokenizer=tokenizer, token_cfg=token_cfg)

        sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(sampler is None),
            sampler=sampler,
            collate_fn=collator,
            num_workers=2,
            pin_memory=True,
        )

        n_epochs = stage0_eps if stage == 0 else args.epochs_per_stage
        for epoch in range(n_epochs):
            if sampler is not None:
                sampler.set_epoch(stage * args.epochs_per_stage + epoch)

            coconut_model.train()

            total_steps = len(loader) // args.grad_accum
            pbar = tqdm(
                total=total_steps,
                desc=f"Stage {stage} Epoch {epoch+1}/{n_epochs}",
                disable=(rank != 0),
                dynamic_ncols=True,
            )

            optimizer.zero_grad()
            running_loss = 0.0

            oom_skip_count = 0

            for step, batch in enumerate(loader):
                batch = {k: v.to(device) for k, v in batch.items()}

                is_sync_step = (step + 1) % args.grad_accum == 0 or step == len(loader) - 1
                ctx = contextlib.nullcontext() if (world_size <= 1 or is_sync_step) \
                    else coconut_model.no_sync()

                try:
                    with ctx:
                        outputs = coconut_model(**batch)
                        loss    = outputs.loss / args.grad_accum
                        loss.backward()
                    running_loss += loss.item()
                except torch.cuda.OutOfMemoryError:
                    oom_skip_count += 1
                    if rank == 0:
                        n_latent = (batch["input_ids"] == coconut_model.module.latent_token_id
                                    if hasattr(coconut_model, "module")
                                    else batch["input_ids"] == coconut_model.latent_token_id
                                    ).sum().item()
                        print(
                            f"[OOM] Skipping batch at step {step} "
                            f"(seq_len={batch['input_ids'].shape[1]}, "
                            f"latent_tokens={n_latent}, "
                            f"oom_skips={oom_skip_count})"
                        )
                    torch.cuda.empty_cache()
                    optimizer.zero_grad()
                    continue

                if (step + 1) % args.grad_accum == 0 or step == len(loader) - 1:
                    torch.nn.utils.clip_grad_norm_(
                        coconut_model.parameters(), max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    pbar.update(1)
                    cur_lr = scheduler.get_last_lr()[0]
                    if rank == 0:
                        step_loss = running_loss * args.grad_accum
                        pbar.set_postfix(loss=f"{step_loss:.4f}", lr=f"{cur_lr:.2e}")
                        if global_step % args.logging_steps == 0:
                            _log_metrics(
                                {
                                    "event": "train_step",
                                    "stage": stage,
                                    "epoch": epoch,
                                    "global_step": global_step,
                                    "loss": float(step_loss),
                                    "lr": float(cur_lr),
                                }
                            )
                    running_loss = 0.0

            pbar.close()

            if world_size > 1:
                dist.barrier()

        # ----------------------------------------------------------------
        # Save stage checkpoint
        # ----------------------------------------------------------------
        if rank == 0:
            ckpt_dir = os.path.join(args.save_dir, f"stage_{stage:03d}")
            os.makedirs(ckpt_dir, exist_ok=True)

            _module = coconut_model.module if world_size > 1 else coconut_model
            # Save the inner base_causallm so it can be loaded standalone later
            _module.base_causallm.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"[Phase 2] Checkpoint saved → {ckpt_dir}")
            _log_metrics({"event": "stage_checkpoint", "stage": stage, "checkpoint": ckpt_dir})

        gc.collect()
        torch.cuda.empty_cache()
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        print("\n[Phase 2] Training complete ✓")
        _log_metrics({"event": "train_complete", "global_step": global_step})

    if rank == 0:
        if log_fp is not None:
            log_fp.close()
        if swan_run is not None:
            try:
                swan_run.finish()
            except Exception:
                pass

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
