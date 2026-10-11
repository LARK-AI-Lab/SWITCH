"""
Phase 2 – SwiLatent Dataset & Collator
=======================================
Two dataset builders for Phase-1 SFT and Phase-2 Coconut curriculum
training, plus the ``SwiCollator`` that pads each batch so that all
``<latent>`` tokens are aligned, enabling KV-cache reuse across passes.

Data format assumption
-----------------------
The input JSONL files (OpenR1 + SwiReasoning annotation) have the structure::

    {
        "problem":   "Natalia sold clips to 48 friends and then ...",
        "solution":  "Natalia sold 48/2 = <<48/2=24>>24 clips after ...
                     <swi>We add those together to get ...</swi>
                     Natalia sold 48+24 = <<48+24=72>>72 clips.
                     #### 72"
    }

The ``<swi>…</swi>`` tags wrap CoT segments that are candidates for
compression into latent hidden states.

Phase-1 SFT dataset (``get_phase1_sft_dataset``)
-------------------------------------------------
Keeps the ``<swi>…</swi>`` text as-is; labels mask only the *question/prompt*
portion (``-100``).  The model learns to produce ``<swi>…</swi>`` wrapped CoT.

Phase-2 curriculum dataset (``get_phase2_latent_dataset``)
-----------------------------------------------------------
Mirrors the multi-stage approach in coconut/dataset.py L225-328::

    stage = 0: <swi> [original text] </swi>   (no replacement, same as SFT)
    stage = 1: <swi> <latent>×c_thought </swi>
    stage = k: <swi> <latent>×(k*c_thought) </swi>
    stage = K: fully replaced (K = n_swi_tokens_in_original // c_thought)

Labels layout::

    [question tokens: -100] [<swi>: -100] [<latent>×N: -100]
    [</swi>: kept] [remaining text + answer: kept]

``</swi>`` is intentionally kept in the loss so the model learns *when* to
exit the latent region.

SwiCollator
-----------
Pads each batch so that the position of the first ``<latent>`` token is
identical across all items (left-padding with ``pad_token_id``).  This lets
the Coconut-style multi-pass forward reuse the KV cache built from the shared
question prefix – see coconut/dataset.py L75-180 for the original design.

Reference
---------
  coconut/dataset.py  L75-200  (MyCollator + get_question_latent_dataset)
  coconut/dataset.py  L225-328 (get_cot_latent_dataset, multi-stage logic)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase
from transformers.tokenization_utils_base import PaddingStrategy

from src.setup_tokens import SwiTokenConfig, SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN


# ---------------------------------------------------------------------------
# SwiCollator – KV-cache-aligned batch padding
# ---------------------------------------------------------------------------

@dataclass
class SwiCollator:
    """Custom data collator that left-pads sequences so the first
    ``<latent>`` token appears at the **same absolute position** in every
    batch item.

    This is required for the multi-pass KV-cache strategy in
    :class:`~src.model.coconut_swi_model.CoconutSwiModel`: the shared
    question prefix can be computed once and cached, but only if all items
    agree on when the latent region starts.

    When a batch contains no latent tokens (e.g. during Stage-0 SFT), the
    collator falls back to standard right-padding.

    Args:
        tokenizer:           Must have ``padding_side = 'right'`` (enforced
                             by :func:`~src.setup_tokens.setup_special_tokens`).
        token_cfg:           SwiTokenConfig with the ``latent_id``.
        label_pad_token_id:  Value used to pad label tensors (default: -100).

    Reference: coconut/dataset.py L82-180 (``MyCollator.__call__``).

    TODO (Phase 2): implement ``__call__``.
    """

    tokenizer: PreTrainedTokenizerBase
    token_cfg: SwiTokenConfig
    label_pad_token_id: int = -100

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Pad a list of dataset items into a batched tensor dict.

        Expected keys in each feature dict:
            ``input_ids``, ``attention_mask``, ``labels``, ``position_ids``
            (all as plain Python lists).

        Steps (reference: coconut/dataset.py L95-175):
        1. Find the position of the first ``<latent>`` in each item.
        2. Compute ``max_first_latent = max(first_latent_positions)``.
        3. Left-pad every item with ``n_pad = max_first_latent - item_first_latent``
           pad tokens; prepend ``0`` to attention_mask and ``-100`` to labels.
        4. Assign contiguous position_ids starting from 0 after padding.
        5. Right-pad to uniform length and return ``torch.Tensor`` batch.

        TODO (Phase 2): implement.
              Follow coconut/dataset.py L95-180 step by step.
        """
        from transformers.data.data_collator import pad_without_fast_tokenizer_warning

        assert self.tokenizer.padding_side == "right"

        latent_id = self.token_cfg.latent_id

        # ------------------------------------------------------------------
        # Step 1-4: left-pad to align first <latent> across batch
        # ------------------------------------------------------------------
        earliest_latent = [
            feature["input_ids"].index(latent_id)
            for feature in features
            if latent_id in feature["input_ids"]
        ]

        if len(earliest_latent) > 0:
            latest_earliest = max(earliest_latent)
            for feature in features:
                if latent_id in feature["input_ids"]:
                    n_pad = latest_earliest - feature["input_ids"].index(latent_id)
                else:
                    n_pad = 0
                feature["position_ids"] = [0] * n_pad + list(
                    range(len(feature["input_ids"]))
                )
                feature["input_ids"] = (
                    [self.tokenizer.pad_token_id] * n_pad + feature["input_ids"]
                )
                if "labels" in feature:
                    feature["labels"] = (
                        [self.label_pad_token_id] * n_pad + feature["labels"]
                    )
                feature["attention_mask"] = [0] * n_pad + feature["attention_mask"]
        else:
            # No latent tokens: assign vanilla position_ids so the key exists
            for feature in features:
                feature["position_ids"] = list(range(len(feature["input_ids"])))

        # ------------------------------------------------------------------
        # Step 5: right-pad to uniform length
        # ------------------------------------------------------------------
        label_name = "label" if "label" in features[0] else "labels"

        non_label_position_features = [
            {k: v for k, v in f.items() if k != label_name and k != "position_ids"}
            for f in features
        ]

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            non_label_position_features,
            padding=True,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

        # Manually pad labels (tokenizer.pad doesn't know about -100)
        labels = (
            [f[label_name] for f in features]
            if label_name in features[0]
            else None
        )
        if labels is not None:
            max_label_len = max(len(l) for l in labels)
            batch[label_name] = torch.tensor(
                [
                    l + [self.label_pad_token_id] * (max_label_len - len(l))
                    for l in labels
                ],
                dtype=torch.int64,
            )

        # Manually pad position_ids
        position_ids = [f["position_ids"] for f in features]
        max_pos_len = max(len(p) for p in position_ids)
        batch["position_ids"] = torch.tensor(
            [p + [0] * (max_pos_len - len(p)) for p in position_ids],
            dtype=torch.int64,
        )

        return batch


# ---------------------------------------------------------------------------
# Phase-1 SFT dataset
# ---------------------------------------------------------------------------

def get_phase1_sft_dataset(
    raw_data: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    token_cfg: SwiTokenConfig,
    max_length: int = 2048,
    problem_key: str = "instruction",
    solution_key: str = "output",
) -> Dataset:
    """Build a HuggingFace-compatible dataset for Phase-1 SFT.

    Each sample is formatted as::

        [chat template: role=user]  {problem}
        [chat template: role=assistant]  {solution with <swi>text</swi> intact}

    Labels mask the prompt/question portion with ``-100``; the ``solution``
    (including ``<swi>…</swi>`` text) participates in the loss.

    Args:
        raw_data:     List of dicts with at least ``problem_key`` and
                      ``solution_key``.
        tokenizer:    Tokenizer with registered ``<swi>``, ``</swi>`` tokens.
        token_cfg:    SwiTokenConfig (used to verify token IDs are present).
        max_length:   Truncate sequences longer than this.
        problem_key:  Key for the question string in raw_data dicts.
        solution_key: Key for the annotated solution string.

    Returns:
        A ``torch.utils.data.Dataset`` whose items contain
        ``input_ids``, ``attention_mask``, ``labels``, ``position_ids``.
    """
    from datasets import Dataset as HFDataset

    SYSTEM = "You are a helpful assistant."

    def process_sample(sample):
        problem  = sample.get(problem_key, "")
        inp_text = sample.get("input", "")
        solution = sample.get(solution_key, "")

        if not solution:
            return None

        # Build the user content  (include "input" field if non-empty, Alpaca style)
        user_content = problem
        if inp_text:
            user_content = f"{problem}\n{inp_text}"

        # Construct Qwen3-compatible prompt string manually so that we avoid
        # the auto-added <think>\n prefix that `add_generation_prompt=True` inserts.
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_text = prompt_text + solution + "<|im_end|>"

        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        full_ids   = tokenizer.encode(full_text,   add_special_tokens=False)

        if len(full_ids) > max_length:
            full_ids = full_ids[:max_length]

        n_prompt  = len(prompt_ids)
        labels    = [-100] * n_prompt + full_ids[n_prompt:]
        labels    = labels[:max_length]

        return {
            "input_ids":      full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels":         labels,
            "position_ids":   list(range(len(full_ids))),
        }

    processed = [r for s in raw_data if (r := process_sample(s)) is not None]
    return HFDataset.from_list(processed)


# ---------------------------------------------------------------------------
# Phase-2 curriculum dataset
# ---------------------------------------------------------------------------

def get_phase2_latent_dataset(
    raw_data: List[Dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    token_cfg: SwiTokenConfig,
    stage: int,
    c_thought: int = 1,
    max_latent_stage: int = 16,
    uniform_prob: float = 0.1,
    max_length: int = 2048,
    phase2_strategy: str = "phase2-2",
    max_total_latent: int = 0,
    problem_key: str = "instruction",
    solution_key: str = "output",
) -> Dataset:
    """Build the Phase-2 curriculum training dataset.

        Supports two curriculum strategies:

        ``phase2-2`` (default, parallel):
            At each ``stage`` k, the content inside every ``<swi>…</swi>`` block is
            replaced by ``k * c_thought`` ``<latent>`` tokens::

                    stage=0: <swi> [original CoT text] </swi>   # pure SFT warm-up
                    stage=1: <swi> <latent> </swi>               # c_thought=1
                    stage=2: <swi> <latent><latent> </swi>
                    …

        ``phase2-1`` (sequential):
            At stage ``k``, only the leftmost ``k`` ``<swi>`` spans are replaced;
            later spans remain plain text. Each replaced span uses a fixed
            ``c_thought`` ``<latent>`` tokens.

    With probability ``uniform_prob`` the stage is sampled uniformly from
    ``[0, K]`` (where K = #tokens in the <swi> block) to mix easy and hard
    examples (same trick as coconut/dataset.py L248-255).

    Labels::

        question tokens    → -100  (no loss)
        <swi>              → -100  (boundary, not predicted)
        <latent> tokens    → -100  (hidden-state placeholders, not predicted)
        </swi>             → loss  (model learns to close the latent region)
        remaining CoT      → loss
        answer             → loss

    Args:
        raw_data:         List of annotated dicts.
        tokenizer:        Tokenizer with all special tokens registered.
        token_cfg:        SwiTokenConfig.
        stage:            Current curriculum stage (0 = SFT, k = k latents).
        c_thought:        For ``phase2-1``, fixed latent count per replaced span.
                  For ``phase2-2``, latent growth factor per stage.
        max_latent_stage: Maximum curriculum stage index.
        uniform_prob:     Probability of randomly mixing stages.
        max_length:       Sequence length cap.
        phase2_strategy:  ``phase2-1`` (sequential span replacement) or
                  ``phase2-2`` (parallel replacement across all spans).
        max_total_latent: Hard cap on the total number of ``<latent>`` tokens in
                  a single sample.  Once the budget is exhausted, remaining
                  spans keep their original text.  0 = no cap.
        problem_key:      Key for problem text.
        solution_key:     Key for annotated solution (with ``<swi>…</swi>``).

    Returns:
        A ``torch.utils.data.Dataset``.
    """
    import re
    from datasets import Dataset as HFDataset

    SYSTEM  = "You are a helpful assistant."
    SWI_PAT = re.compile(r"<swi>(.*?)</swi>", re.DOTALL)

    latent_id    = token_cfg.latent_id
    swi_start_id = token_cfg.swi_start_id
    swi_end_id   = token_cfg.swi_end_id

    if phase2_strategy not in {"phase2-1", "phase2-2"}:
        raise ValueError(
            f"Unknown phase2_strategy={phase2_strategy!r}; expected 'phase2-1' or 'phase2-2'."
        )

    def process_sample(sample):
        problem  = sample.get(problem_key, "")
        inp_text = sample.get("input", "")
        solution = sample.get(solution_key, "")

        if not solution:
            return None

        user_content = problem
        if inp_text:
            user_content = f"{problem}\n{inp_text}"

        # Prompt (question) portion – always masked in labels
        prompt_text = (
            f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{user_content}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)

        # ----------------------------------------------------------------
        # Curriculum: decide actual stage for this sample
        # ----------------------------------------------------------------
        if random.random() < uniform_prob:
            actual_stage = random.randint(0, max_latent_stage)
        else:
            actual_stage = min(stage, max_latent_stage)

        # ----------------------------------------------------------------
        # Build solution token sequence with staged <latent> injection
        # ----------------------------------------------------------------
        solution_input_ids: List[int] = []
        solution_labels:    List[int] = []

        matches = list(SWI_PAT.finditer(solution))
        n_spans = len(matches)

        latent_budget_remaining = max_total_latent if max_total_latent > 0 else float("inf")

        pos = 0
        for span_idx, match in enumerate(matches):
            # --- text segment before this <swi> block ---
            before     = solution[pos : match.start()]
            before_ids = tokenizer.encode(before, add_special_tokens=False) if before else []
            solution_input_ids.extend(before_ids)
            solution_labels.extend(before_ids)          # kept in loss

            # --- <swi> boundary token ---
            solution_input_ids.append(swi_start_id)
            solution_labels.append(swi_start_id)        # <swi> kept in loss: model must learn WHEN to enter latent region

            content     = match.group(1)
            content_ids = tokenizer.encode(content, add_special_tokens=False) if content else []
            n_span      = len(content_ids)

            if actual_stage == 0:
                # Stage 0: SFT warm-up – keep original content tokens
                solution_input_ids.extend(content_ids)
                solution_labels.extend(content_ids)     # content kept in loss
            elif phase2_strategy == "phase2-1":
                # Sequential strategy: progressively replace leftmost spans.
                should_replace = span_idx < min(actual_stage, n_spans)
                if should_replace:
                    n_latent = c_thought if n_span > 0 else 0
                    n_latent = min(n_latent, int(latent_budget_remaining))
                    if n_latent > 0:
                        solution_input_ids.extend([latent_id] * n_latent)
                        solution_labels.extend([-100] * n_latent)
                        latent_budget_remaining -= n_latent
                    else:
                        solution_input_ids.extend(content_ids)
                        solution_labels.extend(content_ids)
                else:
                    solution_input_ids.extend(content_ids)
                    solution_labels.extend(content_ids)
            else:
                # Parallel strategy: replace every span with stage-dependent latent count.
                n_latent = min(actual_stage, n_span, max_latent_stage) * c_thought
                n_latent = min(n_latent, int(latent_budget_remaining))
                if n_latent == 0:
                    solution_input_ids.extend(content_ids)
                    solution_labels.extend(content_ids)
                else:
                    solution_input_ids.extend([latent_id] * n_latent)
                    solution_labels.extend([-100] * n_latent)
                    latent_budget_remaining -= n_latent

            # --- </swi> boundary token (ALWAYS kept in loss) ---
            solution_input_ids.append(swi_end_id)
            solution_labels.append(swi_end_id)          # model learns to close

            pos = match.end()

        # --- remaining text after the last </swi> ---
        tail     = solution[pos:]
        tail_ids = tokenizer.encode(tail, add_special_tokens=False) if tail else []
        solution_input_ids.extend(tail_ids)
        solution_labels.extend(tail_ids)

        # Closing assistant token + EOS
        end_token_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        solution_input_ids.extend(end_token_ids)
        solution_labels.extend(end_token_ids)

        # ----------------------------------------------------------------
        # Assemble full sequence and apply max_length truncation
        # ----------------------------------------------------------------
        full_input_ids = prompt_ids + solution_input_ids
        full_labels    = [-100] * len(prompt_ids) + solution_labels

        if len(full_input_ids) > max_length:
            full_input_ids = full_input_ids[:max_length]
            full_labels    = full_labels[:max_length]

        return {
            "input_ids":      full_input_ids,
            "attention_mask": [1] * len(full_input_ids),
            "labels":         full_labels,
            "position_ids":   list(range(len(full_input_ids))),
        }

    processed = [r for s in raw_data if (r := process_sample(s)) is not None]
    return HFDataset.from_list(processed)
