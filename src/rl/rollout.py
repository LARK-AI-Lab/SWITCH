"""
Phase 3 – Latent-Mode Rollout Utilities
========================================
Thin wrapper around ``CoconutSwiModel``'s generation methods for
evaluation and RL rollout.

RL rollout uses ``CoconutSwiModel.generate_rl()`` directly (called from
``train_phase3_grpo.py``). This module provides a convenience function
for standalone evaluation with latent inference.
"""

from __future__ import annotations

from typing import Dict

import torch

from src.model.coconut_swi_model import CoconutSwiModel


def generate_with_latent(
    model: CoconutSwiModel,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int = 4096,
    min_latent_steps: int = 8,
) -> Dict:
    """Run latent inference and return decoded results.

    Args:
        model:            CoconutSwiModel instance.
        tokenizer:        Tokenizer with special tokens registered.
        input_ids:        ``(1, prompt_len)`` tensor.
        attention_mask:   ``(1, prompt_len)`` tensor.
        max_new_tokens:   Max generation length.
        min_latent_steps: Minimum latent steps per ``<swi>`` block.

    Returns:
        Dict with ``output_ids`` (tensor), ``response_text`` (str),
        ``latent_info`` (list of dicts), ``total_latent_steps`` (int).
    """
    output_ids, latent_info = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        min_latent_steps=min_latent_steps,
    )

    prompt_len = input_ids.shape[1]
    response_ids = output_ids[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=False)
    total_latent_steps = sum(b["n_latent_steps"] for b in latent_info)

    return {
        "output_ids": output_ids,
        "response_text": response_text,
        "latent_info": latent_info,
        "total_latent_steps": total_latent_steps,
    }
