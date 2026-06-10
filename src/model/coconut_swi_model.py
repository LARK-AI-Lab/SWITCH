"""
Phase 2 – CoconutSwiModel
==========================
Adapts ``coconut.Coconut`` (coconut/coconut.py) to use ``<swi>``/``</swi>``
as latent-region boundaries and ``<latent>`` as the hidden-state placeholder.

Key differences vs. the original Coconut model
-----------------------------------------------
1. **Boundary tokens**: ``swi_start_id`` / ``swi_end_id`` instead of
   ``start_latent_id`` / ``end_latent_id``.
2. **``</swi>`` participates in loss**: the model must learn to *predict*
   ``</swi>`` to self-terminate latent reasoning (unlike Coconut where
   ``<|end-latent|>`` was always supplied from the data).  Labels on
   ``<latent>`` tokens are ``-100``; labels on ``</swi>`` are kept.
3. **Multiple disjoint regions**: a single sequence may contain several
   ``<swi>…</swi>`` blocks, each with its own run of ``<latent>`` tokens.
4. **Any decoder-only model**: not GPT-2 specific; tested for Qwen3 / LLaMA.

Forward-pass algorithm (same spirit as coconut/coconut.py L44-175)
-------------------------------------------------------------------
Given ``input_ids`` where some positions have ``<latent>`` (injected by the
dataset before being fed to the model):

  1.  Embed all tokens: ``inputs_embeds = embedding(input_ids)``
  2.  Find all ``<latent>`` positions in each batch item.
  3.  For each latent position *i* (in order):
        a. Run LM forward on tokens ``[prev_end : i]`` (with KV cache).
        b. ``h = last_hidden_state[:, i-1, :]``  (last token before latent).
        c. Replace ``inputs_embeds[:, i, :]`` with ``h``
           (using the list-of-tensors trick from coconut to avoid in-place ops
           that would break autograd).
  4.  Run a final LM forward on the remaining tokens.
  5.  Concatenate all partial logits; compute cross-entropy loss with the
      provided ``labels`` (``-100`` on question tokens and ``<latent>``
      positions; normal on ``</swi>`` and all text after the latent region).

Training curriculum (see data/coconut_swi_dataset.py)
------------------------------------------------------
  Stage 0: ``<swi> text </swi>``           — pure SFT, no latent tokens
  Stage 1: ``<swi> <latent> </swi>``       — 1 latent per ``<swi>`` block
  Stage k: ``<swi> <latent>×k </swi>``     — k latent tokens per block
  …
  Stage K: ``<swi> <latent>×K </swi>``     — fully compressed (K = max budget)

Reference
---------
  coconut/coconut.py  L1-220   (Coconut forward pass + generate)
  coconut/run.py      L148-200 (model construction + FSDP wrapping)
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

try:
    from transformers.cache_utils import Cache as HFCache
except Exception:  # pragma: no cover
    HFCache = None

from src.setup_tokens import SwiTokenConfig

Outputs = namedtuple("Outputs", ["loss", "inputs_embeds", "logits"])

# Hard upper bound on latent tokens per <swi> block to prevent runaway loops.
MAX_LATENT_PER_BLOCK = 256


class CoconutSwiModel(nn.Module):
    """Wrapper around a decoder-only causal LM that implements Coconut-style
    continuous-thought reasoning inside ``<swi>…</swi>`` regions.

    Args:
        base_causallm:  Pre-loaded ``AutoModelForCausalLM`` with the extended
                        vocabulary (after :func:`~src.setup_tokens.setup_special_tokens`).
        token_cfg:      :class:`~src.setup_tokens.SwiTokenConfig` returned by
                        :func:`~src.setup_tokens.setup_special_tokens`.
    """

    def __init__(
        self,
        base_causallm: nn.Module,
        token_cfg: SwiTokenConfig,
    ) -> None:
        super().__init__()
        self.base_causallm = base_causallm
        self.token_cfg = token_cfg
        self.latent_token_id = token_cfg.latent_id
        self.swi_start_id    = token_cfg.swi_start_id
        self.swi_end_id      = token_cfg.swi_end_id
        self.eos_token_id    = token_cfg.eos_token_id

        # Grab the embedding table (handles both GPT-2 and Qwen3/LLaMA layout)
        self.embedding = base_causallm.get_input_embeddings()

        model_type = getattr(getattr(base_causallm, "config", None), "model_type", "")
        # Qwen3 path in current transformers expects `past_key_values` to be
        # a Cache object (or None), not legacy tuple/list KV structures.
        self._requires_cache_object = str(model_type).lower() == "qwen3"

        self.gen_forward_cnt = 0  # diagnostic counter (same as coconut.py)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_past_key_values_for_prefix(kv_cache, prefix_len: int):
        """Prepare past_key_values for the next incremental forward.

        - Newer HF models (e.g. Qwen3) return a `Cache` object and expect
          a `Cache` object back. In this case, pass it through unchanged.
        - Legacy models may return tuple/list of (k, v); keep the previous
          slicing behavior for backward compatibility.
        """
        if kv_cache is None:
            return None
        if HFCache is not None and isinstance(kv_cache, HFCache):
            return kv_cache
        if isinstance(kv_cache, (list, tuple)):
            # Legacy cache path: list/tuple of per-layer (k, v) tensors.
            if all(
                isinstance(layer_kv, (list, tuple)) and len(layer_kv) == 2
                for layer_kv in kv_cache
            ):
                return [
                    (
                        k[:, :, :prefix_len, :],
                        v[:, :, :prefix_len, :],
                    )
                    for k, v in kv_cache
                ]
            # Unknown list/tuple cache format: disable cache reuse safely.
            return None
        # Unknown cache object type: disable cache reuse safely.
        return None

    def _get_past_for_model(self, kv_cache, prefix_len: int):
        """Return past_key_values in a format accepted by current base model."""
        prepared = self._prepare_past_key_values_for_prefix(kv_cache, prefix_len)
        if self._requires_cache_object:
            if HFCache is not None and isinstance(prepared, HFCache):
                return prepared
            return None
        return prepared

    def forward(
        self,
        input_ids: torch.Tensor,       # (B, L)
        attention_mask: torch.Tensor,  # (B, L)
        labels: torch.Tensor,          # (B, L)  -100 on ignored positions
        position_ids: torch.Tensor,    # (B, L)
        **kwargs,
    ) -> Outputs:
        """Multi-pass forward with hidden-state injection at <latent> positions.

        Identical in structure to ``coconut.Coconut.forward`` but uses
        ``self.latent_token_id`` (<latent>) as the injection target and keeps
        ``</swi>`` in the loss.

        TODO (Phase 2): implement this method.
              Follow coconut/coconut.py L44-220 closely:
              - latent_indices / latent_lists construction
              - KV-cache management across passes
              - list-of-tensors embedding replacement trick
              - final cross-entropy loss calculation
        """
        logits = []

        # ----------------------------------------------------------------
        # 1. Locate every <latent> position in the batch
        # ----------------------------------------------------------------
        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_positions = sorted({idx[1].item() for idx in latent_indices})
        n_latent_positions = len(latent_positions)

        # ----------------------------------------------------------------
        # 2. Initial embedding of the full sequence
        # ----------------------------------------------------------------
        inputs_embeds = self.embedding(input_ids)

        # Determine first compute range
        if n_latent_positions > 0:
            next_compute_range = (0, latent_positions[0])
        else:
            next_compute_range = (0, input_ids.shape[1])

        kv_cache = None

        # ----------------------------------------------------------------
        # 3. Iterative passes: one per absolute latent position
        # ----------------------------------------------------------------
        for pass_idx, latent_pos in enumerate(latent_positions):
            current_start, current_end = next_compute_range

            if current_end < current_start:
                raise RuntimeError(
                    f"Invalid compute range: start={current_start}, end={current_end}."
                )

            if kv_cache is None:
                # If cache is unavailable (e.g., gradient checkpointing forced
                # use_cache=False), recompute from sequence start and slice.
                if current_start == 0:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, :current_end, :],
                        attention_mask=attention_mask[:, :current_end],
                        position_ids=position_ids[:, :current_end],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    logits.append(outputs.logits)
                else:
                    outputs = self.base_causallm(
                        inputs_embeds=inputs_embeds[:, :current_end, :],
                        attention_mask=attention_mask[:, :current_end],
                        position_ids=position_ids[:, :current_end],
                        output_hidden_states=True,
                        use_cache=True,
                    )
                    logits.append(outputs.logits[:, current_start:current_end, :])
                hidden_states_offset = 0
            else:
                past_key_values = self._get_past_for_model(
                    kv_cache, current_start
                )
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[
                        :, current_start:current_end, :
                    ],
                    attention_mask=attention_mask[:, :current_end],
                    position_ids=position_ids[
                        :, current_start:current_end
                    ],
                    past_key_values=past_key_values,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = current_start
                logits.append(outputs.logits)

            next_compute_range = (
                current_end,
                input_ids.shape[1]
                if pass_idx + 1 >= n_latent_positions
                else latent_positions[pass_idx + 1],
            )

            hidden_states = outputs.hidden_states[-1]   # last transformer layer
            kv_cache      = outputs.past_key_values

            # ----------------------------------------------------------------
            # 4. Inject hidden state into <latent> positions (list-of-tensors
            #    trick to avoid in-place ops that would break autograd)
            # ----------------------------------------------------------------
            filling_indices = [
                (b_idx, latent_pos)
                for b_idx in range(input_ids.shape[0])
                if input_ids[b_idx, latent_pos].item() == self.latent_token_id
            ]

            # Decompose to list-of-lists-of-1d tensors
            tensor_list = [
                [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
                for b in range(inputs_embeds.shape[0])
            ]

            for b_idx, tok_idx in filling_indices:
                src_pos = tok_idx - 1 - hidden_states_offset
                if src_pos < 0 or src_pos >= hidden_states.shape[1]:
                    raise RuntimeError(
                        "Invalid latent injection index: "
                        f"tok_idx={tok_idx}, hidden_states_offset={hidden_states_offset}, "
                        f"hidden_len={hidden_states.shape[1]}. "
                        "This usually means the current compute range does not "
                        "cover the token preceding this <latent>."
                    )
                tensor_list[b_idx][tok_idx] = hidden_states[
                    b_idx, src_pos, :
                ]

            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[b]) for b in range(inputs_embeds.shape[0])]
            )

        # ----------------------------------------------------------------
        # 5. Final forward pass (remaining tokens after last latent)
        #    No hidden_states or cache needed — saves significant memory.
        # ----------------------------------------------------------------
        final_start, final_end = next_compute_range
        if kv_cache is None:
            if final_start == 0:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :final_end, :],
                    attention_mask=attention_mask[:, :final_end],
                    position_ids=position_ids[:, :final_end],
                    output_hidden_states=False,
                    use_cache=False,
                )
                logits.append(outputs.logits)
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :final_end, :],
                    attention_mask=attention_mask[:, :final_end],
                    position_ids=position_ids[:, :final_end],
                    output_hidden_states=False,
                    use_cache=False,
                )
                logits.append(outputs.logits[:, final_start:final_end, :])
        else:
            outputs = self.base_causallm(
                inputs_embeds=inputs_embeds[:, final_start:final_end, :],
                attention_mask=attention_mask[:, :final_end],
                position_ids=position_ids[:, final_start:final_end],
                past_key_values=self._get_past_for_model(
                    kv_cache, final_start
                ),
                output_hidden_states=False,
                use_cache=False,
            )
            logits.append(outputs.logits)

        self.gen_forward_cnt += n_latent_positions + 1

        # ----------------------------------------------------------------
        # 6. Compute cross-entropy loss (shift by 1)
        # ----------------------------------------------------------------
        logits = torch.cat(logits, dim=-2)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return Outputs(loss=loss, inputs_embeds=inputs_embeds, logits=logits)

    # ------------------------------------------------------------------
    # Generation (inference / RL rollout seed)
    # ------------------------------------------------------------------

    @staticmethod
    def _gaussian_latent_log_prob(
        z: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Mean-dimension Gaussian surrogate log-prob for latent actions.

        This is intentionally normalized by hidden dimension so the scale stays
        usable for logging and future policy losses.
        """
        sigma = sigma.clamp(min=1e-6)
        return -0.5 * (((z - mu) / sigma) ** 2).mean(dim=-1)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        min_latent_steps: int = 0,
        **kwargs,
    ) -> Tuple[torch.Tensor, List[dict]]:
        """Auto-regressive generation with Coconut-style latent inference.

        When the model outputs ``<swi>``, generation enters **latent mode**:
        the last-layer hidden state is fed back as the next input embedding
        (continuous thought), repeating until the model predicts ``</swi>``
        from the logits or a safety cap is reached.  No text tokens are
        emitted during latent steps — the model "thinks silently".

        Args:
            min_latent_steps: Minimum number of hidden-state steps per latent
                block before the model is allowed to exit by predicting
                ``</swi>``.  Set this to match the training curriculum
                (e.g. ``max_latent_stage * c_thought``) so that inference
                behaviour aligns with what the model was trained on.
                0 = no minimum (original behaviour).

        Returns:
            output_ids:  (1, prompt_len + generated_len) token IDs.
                         Latent steps are **not** represented as tokens;
                         the output contains ``<swi>`` then ``</swi>``
                         directly (the gap between them is latent thought).
            latent_info: List of dicts, one per ``<swi>`` block encountered,
                         each with ``{"position": int, "n_latent_steps": int}``.
        """
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1, "generate() only supports batch_size == 1"

        device = input_ids.device
        prompt_len = input_ids.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # ── 1. Prefill: process the full prompt ───────────────────────────
        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=False,
        )
        self.gen_forward_cnt += 1

        kv_cache = outputs.past_key_values
        cur_pos = prompt_len
        generated_tokens: List[int] = []
        latent_info: List[dict] = []

        next_token = torch.argmax(outputs.logits[0, -1]).item()

        # ── 2. Token-by-token decoding ────────────────────────────────────
        tokens_budget = max_new_tokens
        while tokens_budget > 0:
            if next_token == self.eos_token_id:
                generated_tokens.append(next_token)
                break

            if next_token == self.swi_start_id:
                # ── ENTER LATENT MODE ─────────────────────────────────
                generated_tokens.append(next_token)
                tokens_budget -= 1

                # Feed <swi> token to update KV cache and get hidden state
                swi_embed = self.embedding(
                    torch.tensor([[next_token]], device=device)
                )
                outputs = self.base_causallm(
                    inputs_embeds=swi_embed,
                    attention_mask=torch.ones((1, cur_pos + 1), device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                self.gen_forward_cnt += 1
                kv_cache = outputs.past_key_values
                cur_pos += 1

                # Latent thinking loop
                n_latent_steps = 0
                exited = False

                for _ in range(MAX_LATENT_PER_BLOCK):
                    h = outputs.hidden_states[-1][:, -1:, :]  # (1, 1, D)

                    outputs = self.base_causallm(
                        inputs_embeds=h,
                        attention_mask=torch.ones(
                            (1, cur_pos + 1), device=device
                        ),
                        position_ids=torch.tensor(
                            [[cur_pos]], device=device
                        ),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    self.gen_forward_cnt += 1
                    kv_cache = outputs.past_key_values
                    cur_pos += 1
                    n_latent_steps += 1

                    if n_latent_steps >= min_latent_steps:
                        predicted = torch.argmax(
                            outputs.logits[0, -1]
                        ).item()
                        if predicted == self.swi_end_id:
                            exited = True
                            break

                # Emit </swi> (whether model predicted it or we force it)
                generated_tokens.append(self.swi_end_id)
                tokens_budget -= 1

                latent_info.append({
                    "position": len(generated_tokens) - 1,
                    "n_latent_steps": n_latent_steps,
                    "natural_exit": exited,
                })

                # Feed </swi> token so KV cache covers it
                end_embed = self.embedding(
                    torch.tensor([[self.swi_end_id]], device=device)
                )
                outputs = self.base_causallm(
                    inputs_embeds=end_embed,
                    attention_mask=torch.ones(
                        (1, cur_pos + 1), device=device
                    ),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
                self.gen_forward_cnt += 1
                kv_cache = outputs.past_key_values
                cur_pos += 1

                next_token = torch.argmax(outputs.logits[0, -1]).item()
                continue

            # ── NORMAL TEXT TOKEN ─────────────────────────────────────
            generated_tokens.append(next_token)
            tokens_budget -= 1

            tok_embed = self.embedding(
                torch.tensor([[next_token]], device=device)
            )
            outputs = self.base_causallm(
                inputs_embeds=tok_embed,
                attention_mask=torch.ones((1, cur_pos + 1), device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=False,
            )
            self.gen_forward_cnt += 1
            kv_cache = outputs.past_key_values
            cur_pos += 1

            next_token = torch.argmax(outputs.logits[0, -1]).item()

        all_tokens = input_ids[0].tolist() + generated_tokens
        return (
            torch.tensor(all_tokens, device=device).view(1, -1),
            latent_info,
        )

    # ------------------------------------------------------------------
    # RL rollout: sampling + log-prob collection
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_rl(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 2048,
        min_latent_steps: int = 8,
        temperature: float = 1.0,
        top_p: float = 0.95,
        latent_noise_scale: float = 0.0,
        record_latent_actions: bool = False,
        **kwargs,
    ) -> dict:
        """Sample a response with latent inference for RL rollout.

        Returns a dict with:
            ``response_ids``:     List[int] — visible tokens (text + <swi>/<\\/swi>).
            ``response_ids_full``: List[int] — same but with <latent> tokens
                                   inserted between <swi>/<\\/swi>, suitable for
                                   second-pass log-prob computation via forward().
            ``old_log_probs``:    List[float] — per-position log-prob under the
                                  RAW policy (before temperature/top_p), aligned
                                  with ``response_ids_full``. Latent and forced
                                  positions are marked with 0.0.
            ``sampled_mask``:     List[int] — 1 for positions that were actually
                                  sampled (text tokens + <swi>), 0 for <latent>
                                  and forced <\\/swi>.
            ``latent_action_stats``: Summary stats for optional Gaussian latent
                                  perturbations. Present even when disabled.
            ``latent_info``:      List[dict].
        """
        self.gen_forward_cnt = 0
        assert input_ids.shape[0] == 1

        device = input_ids.device
        prompt_len = input_ids.shape[1]
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

        outputs = self.base_causallm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
            output_hidden_states=False,
        )
        self.gen_forward_cnt += 1

        kv_cache = outputs.past_key_values
        cur_pos = prompt_len

        response_ids: List[int] = []       # visible tokens
        response_ids_full: List[int] = []  # with <latent> inserted
        old_log_probs: List[float] = []    # aligned with response_ids_full
        sampled_mask: List[int] = []       # aligned with response_ids_full
        latent_info: List[dict] = []
        latent_actions: List[dict] = []
        latent_action_old_log_probs: List[float] = []
        latent_noise_ratios: List[float] = []
        latent_mu_norms: List[float] = []
        latent_z_norms: List[float] = []
        # Lightweight per-action storage so a future second-pass replay can
        # recompute new_log_prob(z | mu_current, sigma_old) without re-sampling.
        # Always populated when latent_noise_scale > 0, regardless of
        # ``record_latent_actions`` (which gates the richer dict in
        # ``latent_actions``). Stored on CPU to avoid GPU pressure.
        latent_zs: List[torch.Tensor] = []
        latent_sigmas: List[torch.Tensor] = []

        raw_logits = outputs.logits[0, -1]  # unscaled logits for raw log-prob

        def _raw_log_prob(logits_1d, token_id):
            """Log-prob under the raw policy (no temperature/top_p)."""
            return F.log_softmax(logits_1d, dim=-1)[token_id].item()

        def _sample(logits_1d):
            """Sample with temperature/top_p but return raw log-prob."""
            scaled = logits_1d / temperature if temperature != 1.0 else logits_1d
            probs = torch.softmax(scaled, dim=-1)
            if 0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
            token = torch.multinomial(probs, 1).item()
            raw_lp = _raw_log_prob(logits_1d, token)
            return token, raw_lp

        next_token, next_lp = _sample(raw_logits)

        tokens_budget = max_new_tokens
        while tokens_budget > 0:
            if next_token == self.eos_token_id:
                response_ids.append(next_token)
                response_ids_full.append(next_token)
                old_log_probs.append(next_lp)
                sampled_mask.append(1)
                break

            if next_token == self.swi_start_id:
                # <swi> was sampled
                response_ids.append(next_token)
                response_ids_full.append(next_token)
                old_log_probs.append(next_lp)
                sampled_mask.append(1)
                tokens_budget -= 1

                swi_embed = self.embedding(
                    torch.tensor([[next_token]], device=device)
                )
                outputs = self.base_causallm(
                    inputs_embeds=swi_embed,
                    attention_mask=torch.ones((1, cur_pos + 1), device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=True,
                )
                self.gen_forward_cnt += 1
                kv_cache = outputs.past_key_values
                cur_pos += 1

                n_latent_steps = 0
                exited = False
                for _ in range(MAX_LATENT_PER_BLOCK):
                    h = outputs.hidden_states[-1][:, -1:, :]
                    latent_input = h

                    if latent_noise_scale > 0:
                        eps = torch.randn_like(h)
                        eps = eps / (eps.norm(dim=-1, keepdim=True) + 1e-6)
                        sigma = (
                            latent_noise_scale
                            * h.norm(dim=-1, keepdim=True).detach()
                        ).clamp(min=1e-6)
                        latent_input = h + sigma * eps
                        old_lp = self._gaussian_latent_log_prob(
                            latent_input, h, sigma
                        )

                        mu_norm = h.norm(dim=-1).detach()
                        z_norm = latent_input.norm(dim=-1).detach()
                        noise_norm = (latent_input - h).norm(dim=-1).detach()
                        ratio = noise_norm / (mu_norm + 1e-6)

                        latent_action_old_log_probs.append(float(old_lp.item()))
                        latent_noise_ratios.append(float(ratio.item()))
                        latent_mu_norms.append(float(mu_norm.item()))
                        latent_z_norms.append(float(z_norm.item()))

                        # Save z and sigma for second-pass replay.
                        # z: (D,) on CPU; sigma: scalar on CPU (kept fp32 for
                        # numeric safety when re-evaluating Gaussian log-prob).
                        latent_zs.append(
                            latent_input[0, 0, :].detach().to("cpu")
                        )
                        latent_sigmas.append(
                            sigma[0, 0, 0].detach().to(device="cpu", dtype=torch.float32)
                        )

                        if record_latent_actions:
                            latent_actions.append({
                                "position": cur_pos,
                                "z": latent_input.detach().cpu(),
                                "old_log_prob": float(old_lp.item()),
                                "sigma": sigma.detach().cpu(),
                            })

                    outputs = self.base_causallm(
                        inputs_embeds=latent_input,
                        attention_mask=torch.ones((1, cur_pos + 1), device=device),
                        position_ids=torch.tensor([[cur_pos]], device=device),
                        past_key_values=kv_cache,
                        use_cache=True,
                        output_hidden_states=True,
                    )
                    self.gen_forward_cnt += 1
                    kv_cache = outputs.past_key_values
                    cur_pos += 1
                    n_latent_steps += 1

                    # Record <latent> placeholder in full sequence
                    response_ids_full.append(self.latent_token_id)
                    old_log_probs.append(0.0)
                    sampled_mask.append(0)  # not sampled

                    if n_latent_steps >= min_latent_steps:
                        predicted = torch.argmax(outputs.logits[0, -1]).item()
                        if predicted == self.swi_end_id:
                            exited = True
                            break

                # </swi> is forced (not sampled)
                response_ids.append(self.swi_end_id)
                response_ids_full.append(self.swi_end_id)
                old_log_probs.append(0.0)
                sampled_mask.append(0)  # forced, not sampled
                tokens_budget -= 1

                latent_info.append({
                    "position": len(response_ids) - 1,
                    "n_latent_steps": n_latent_steps,
                    "natural_exit": exited,
                })

                end_embed = self.embedding(
                    torch.tensor([[self.swi_end_id]], device=device)
                )
                outputs = self.base_causallm(
                    inputs_embeds=end_embed,
                    attention_mask=torch.ones((1, cur_pos + 1), device=device),
                    position_ids=torch.tensor([[cur_pos]], device=device),
                    past_key_values=kv_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
                self.gen_forward_cnt += 1
                kv_cache = outputs.past_key_values
                cur_pos += 1

                next_token, next_lp = _sample(outputs.logits[0, -1])
                continue

            # Normal text token (sampled)
            response_ids.append(next_token)
            response_ids_full.append(next_token)
            old_log_probs.append(next_lp)
            sampled_mask.append(1)
            tokens_budget -= 1

            tok_embed = self.embedding(
                torch.tensor([[next_token]], device=device)
            )
            outputs = self.base_causallm(
                inputs_embeds=tok_embed,
                attention_mask=torch.ones((1, cur_pos + 1), device=device),
                position_ids=torch.tensor([[cur_pos]], device=device),
                past_key_values=kv_cache,
                use_cache=True,
                output_hidden_states=False,
            )
            self.gen_forward_cnt += 1
            kv_cache = outputs.past_key_values
            cur_pos += 1

            next_token, next_lp = _sample(outputs.logits[0, -1])

        def _avg(xs: List[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        return {
            "response_ids": response_ids,
            "response_ids_full": response_ids_full,
            "old_log_probs": old_log_probs,
            "sampled_mask": sampled_mask,
            "latent_info": latent_info,
            "latent_actions": latent_actions,
            "latent_zs": latent_zs,
            "latent_sigmas": latent_sigmas,
            "latent_action_old_log_probs": latent_action_old_log_probs,
            "latent_action_stats": {
                "n_actions": len(latent_action_old_log_probs),
                "old_log_prob_mean": _avg(latent_action_old_log_probs),
                "noise_ratio_mean": _avg(latent_noise_ratios),
                "mu_norm_mean": _avg(latent_mu_norms),
                "z_norm_mean": _avg(latent_z_norms),
            },
        }

    def compute_response_log_probs(
        self,
        input_ids: torch.Tensor,
        response_len: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute per-token log-probs for the response portion under the
        current policy. Gradients flow through this call.

        ``input_ids`` must be the full sequence (prompt + response_ids_full)
        with ``<latent>`` tokens inserted so that ``forward()`` correctly
        replays the hidden-state injection passes.

        Args:
            input_ids:     ``(1, prompt_len + response_full_len)`` — includes
                           ``<latent>`` placeholders between ``<swi>``/``</swi>``.
            response_len:  Length of the response portion (= len(response_ids_full)).
            attention_mask: ``(1, total_len)``.

        Returns:
            ``(response_len,)`` tensor of raw-policy log-probs.
        """
        total_len = input_ids.shape[1]
        prompt_len = total_len - response_len

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        labels = torch.full_like(input_ids, -100)
        labels[:, prompt_len:] = input_ids[:, prompt_len:]

        position_ids = torch.arange(total_len, device=input_ids.device).unsqueeze(0)

        out = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            position_ids=position_ids,
        )

        shift_logits = out.logits[:, prompt_len - 1:-1, :]  # (1, response_len, V)
        log_probs = F.log_softmax(shift_logits, dim=-1)

        target_tokens = input_ids[0, prompt_len:]            # (response_len,)
        token_log_probs = log_probs[0].gather(
            1, target_tokens.unsqueeze(-1)
        ).squeeze(-1)

        return token_log_probs

    # ------------------------------------------------------------------
    # RL-specific: memory-efficient log-prob with hidden-state injection
    # ------------------------------------------------------------------

    def compute_response_log_probs_masked(
        self,
        input_ids: torch.Tensor,
        response_len: int,
        sampled_mask: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute log-probs only at sampled positions, with full multi-pass
        hidden-state injection. Gradients flow through this call.

        Unlike ``forward()`` which stores logits for every position (including
        latent passes), this method only materializes logits at text segments
        (``sampled_mask=1``). Latent injection passes only extract the
        last-layer hidden state — no logits, no vocab-size tensors stored.

        Memory saving: for 80 latent passes, this avoids storing
        ~80 × (1, L, V) logit tensors (~46 GB), replacing them with
        80 × (1, 1, D) hidden-state extractions (~0.6 MB).

        The ``forward()`` method is NOT modified.

        Args:
            input_ids:     ``(1, prompt_len + response_full_len)`` with
                           ``<latent>`` tokens inserted.
            response_len:  Length of the response portion.
            sampled_mask:  ``(response_len,)`` — 1 for sampled text tokens,
                           0 for ``<latent>`` and forced ``</swi>``.
            attention_mask: ``(1, total_len)``.

        Returns:
            ``(response_len,)`` tensor of log-probs. Positions with
            ``sampled_mask=0`` are filled with 0.0 (no gradient).
        """
        B = input_ids.shape[0]
        total_len = input_ids.shape[1]
        prompt_len = total_len - response_len
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        inputs_embeds = self.embedding(input_ids)

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_positions = sorted({idx[1].item() for idx in latent_indices})
        n_latent = len(latent_positions)

        if n_latent > 0:
            next_compute_range = (0, latent_positions[0])
        else:
            next_compute_range = (0, total_len)

        kv_cache = None
        text_logit_chunks: List[tuple] = []

        for pass_idx, latent_pos in enumerate(latent_positions):
            current_start, current_end = next_compute_range

            is_text_segment = current_end > current_start

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :current_end, :],
                    attention_mask=attention_mask[:, :current_end],
                    position_ids=position_ids[:, :current_end],
                    output_hidden_states=True,
                    use_cache=True,
                )
                if is_text_segment:
                    text_logit_chunks.append(
                        (current_start, current_end, outputs.logits[:, current_start:current_end, :])
                    )
                hidden_states_offset = 0
            else:
                past_kv = self._get_past_for_model(kv_cache, current_start)
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, current_start:current_end, :],
                    attention_mask=attention_mask[:, :current_end],
                    position_ids=position_ids[:, current_start:current_end],
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
                if is_text_segment:
                    text_logit_chunks.append(
                        (current_start, current_end, outputs.logits)
                    )
                hidden_states_offset = current_start

            next_compute_range = (
                current_end,
                total_len if pass_idx + 1 >= n_latent else latent_positions[pass_idx + 1],
            )

            hidden_states = outputs.hidden_states[-1]
            kv_cache = outputs.past_key_values

            filling_indices = [
                (b, latent_pos)
                for b in range(B)
                if input_ids[b, latent_pos].item() == self.latent_token_id
            ]

            tensor_list = [
                [inputs_embeds[b, p, :] for p in range(inputs_embeds.shape[1])]
                for b in range(B)
            ]
            for b_idx, tok_idx in filling_indices:
                src_pos = tok_idx - 1 - hidden_states_offset
                tensor_list[b_idx][tok_idx] = hidden_states[b_idx, src_pos, :]

            inputs_embeds = torch.stack(
                [torch.stack(tensor_list[b]) for b in range(B)]
            )

        # Final forward pass (text after last latent)
        final_start, final_end = next_compute_range
        if final_end > final_start:
            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :final_end, :],
                    attention_mask=attention_mask[:, :final_end],
                    position_ids=position_ids[:, :final_end],
                    output_hidden_states=False,
                    use_cache=False,
                )
                text_logit_chunks.append(
                    (final_start, final_end, outputs.logits[:, final_start:final_end, :])
                )
            else:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, final_start:final_end, :],
                    attention_mask=attention_mask[:, :final_end],
                    position_ids=position_ids[:, final_start:final_end],
                    past_key_values=self._get_past_for_model(kv_cache, final_start),
                    output_hidden_states=False,
                    use_cache=False,
                )
                text_logit_chunks.append(
                    (final_start, final_end, outputs.logits)
                )

        # Assemble log-probs only at response positions with sampled_mask=1
        result_log_probs = torch.zeros(response_len, device=device, dtype=inputs_embeds.dtype)

        for abs_start, abs_end, chunk_logits in text_logit_chunks:
            for pos_in_chunk in range(chunk_logits.shape[1]):
                abs_pos = abs_start + pos_in_chunk
                resp_idx = abs_pos - prompt_len  # index in response
                # log_prob at position abs_pos predicts token at abs_pos+1
                target_resp_idx = resp_idx + 1
                if target_resp_idx < 0 or target_resp_idx >= response_len:
                    continue
                if sampled_mask[target_resp_idx].item() == 0:
                    continue
                target_token = input_ids[0, abs_pos + 1]
                lp = F.log_softmax(chunk_logits[0, pos_in_chunk, :], dim=-1)
                result_log_probs[target_resp_idx] = lp[target_token]

        return result_log_probs

    # ------------------------------------------------------------------
    # RL-specific: latent-action replay (no-grad, no loss)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compute_latent_action_log_probs(
        self,
        input_ids: torch.Tensor,
        response_len: int,
        latent_zs: List[torch.Tensor],
        latent_sigmas: List[torch.Tensor],
        latent_old_log_probs: Optional[List[float]] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """Replay rollout with saved latent z's; return per-action
        ``new_log_prob = N(z; mu_current, sigma_old).log_prob`` and
        (optionally) ``log_ratio = new_lp - old_lp``.

        ``sigma`` is held fixed at the rollout-time value, so the ratio
        only reflects drift of ``mu`` between rollout and the current
        policy. The current policy's ``mu`` at each latent step is the
        last-layer hidden state at the position immediately preceding the
        ``<latent>`` token, in a forward pass that injects the SAME saved
        ``z`` values that rollout did. This keeps the KV trajectory
        identical to rollout, so downstream ``mu`` values are well defined.

        No gradients flow through this call; no loss is computed.

        Args:
            input_ids:           ``(1, prompt_len + response_full_len)`` with
                                 ``<latent>`` placeholders inserted between
                                 ``<swi>`` / ``</swi>``.
            response_len:        Length of the response portion.
            latent_zs:           Ordered list of saved ``z`` tensors, one per
                                 ``<latent>`` token in ``input_ids``. Each is
                                 a 1-D ``(D,)`` tensor (CPU is fine).
            latent_sigmas:       Ordered list of saved ``sigma`` scalars.
            latent_old_log_probs: Optional list of rollout-time ``old_log_prob``
                                  values, one per latent action. When provided,
                                  per-step ``log_ratio`` is also returned.
            attention_mask:      ``(1, total_len)``.

        Returns:
            ``{"new_log_probs": [...], "mu_new_norms": [...], "log_ratios":
              [...] (optional)}``
        """
        assert input_ids.shape[0] == 1, "expects batch_size=1"
        total_len = input_ids.shape[1]
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        inputs_embeds = self.embedding(input_ids)
        embed_dtype = inputs_embeds.dtype

        latent_indices = (input_ids == self.latent_token_id).nonzero()
        latent_positions = sorted({idx[1].item() for idx in latent_indices})
        n_latent = len(latent_positions)

        out = {
            "new_log_probs": [],
            "mu_new_norms": [],
        }
        if latent_old_log_probs is not None:
            out["log_ratios"] = []

        if n_latent == 0:
            return out

        if len(latent_zs) != n_latent or len(latent_sigmas) != n_latent:
            raise ValueError(
                f"latent_zs ({len(latent_zs)}) / latent_sigmas "
                f"({len(latent_sigmas)}) do not match number of <latent> "
                f"tokens in input_ids ({n_latent})."
            )
        if (
            latent_old_log_probs is not None
            and len(latent_old_log_probs) != n_latent
        ):
            raise ValueError(
                f"latent_old_log_probs ({len(latent_old_log_probs)}) does not "
                f"match number of <latent> tokens ({n_latent})."
            )

        next_compute_range = (0, latent_positions[0])
        kv_cache = None

        for pass_idx, latent_pos in enumerate(latent_positions):
            current_start, current_end = next_compute_range

            if kv_cache is None:
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :current_end, :],
                    attention_mask=attention_mask[:, :current_end],
                    position_ids=position_ids[:, :current_end],
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = 0
            else:
                past_kv = self._get_past_for_model(kv_cache, current_start)
                outputs = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, current_start:current_end, :],
                    attention_mask=attention_mask[:, :current_end],
                    position_ids=position_ids[:, current_start:current_end],
                    past_key_values=past_kv,
                    output_hidden_states=True,
                    use_cache=True,
                )
                hidden_states_offset = current_start

            kv_cache = outputs.past_key_values

            next_compute_range = (
                current_end,
                total_len if pass_idx + 1 >= n_latent else latent_positions[pass_idx + 1],
            )

            hidden_states = outputs.hidden_states[-1]
            src_pos = latent_pos - 1 - hidden_states_offset
            mu_new = hidden_states[0, src_pos, :]  # (D,)

            z = latent_zs[pass_idx].to(device=device, dtype=mu_new.dtype)
            sigma = latent_sigmas[pass_idx].to(device=device, dtype=mu_new.dtype)
            sigma = sigma.clamp(min=1e-6)

            new_lp = -0.5 * (((z - mu_new) / sigma) ** 2).mean()
            new_lp_val = float(new_lp.item())
            out["new_log_probs"].append(new_lp_val)
            out["mu_new_norms"].append(float(mu_new.norm().item()))
            if latent_old_log_probs is not None:
                out["log_ratios"].append(
                    new_lp_val - float(latent_old_log_probs[pass_idx])
                )

            # Inject the SAVED z (not mu_new) into this latent position so
            # subsequent forwards see the same trajectory as rollout did.
            seq_list = [inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])]
            seq_list[latent_pos] = z.to(embed_dtype)
            inputs_embeds = torch.stack(seq_list).unsqueeze(0)

            del outputs
        return out

    # ------------------------------------------------------------------
    # RL-specific: segmented GRPO backward (minimal memory)
    # ------------------------------------------------------------------

    def segmented_grpo_backward(
        self,
        input_ids: torch.Tensor,
        response_len: int,
        old_log_probs: torch.Tensor,
        sampled_mask: torch.Tensor,
        advantage: float,
        clip_eps: float = 0.2,
        kl_coef: float = 0.0,
        loss_scale: float = 1.0,
    ) -> float:
        """Compute GRPO loss and backward in segments to minimize memory.

        Splits the sequence into text segments and latent blocks. Text
        segments are forwarded with grad and immediately backwarded; latent
        blocks are forwarded without grad but with correct hidden-state
        injection. KV cache threads context across all segments.

        Only ONE text segment's autograd graph exists at a time, so peak
        memory is proportional to the longest text segment, not the full
        sequence.

        Gradients accumulate on model parameters via repeated backward()
        calls (standard gradient accumulation).

        Args:
            input_ids:     ``(1, total_len)`` with ``<latent>`` tokens.
            response_len:  Length of response portion.
            old_log_probs: ``(response_len,)`` from rollout.
            sampled_mask:  ``(response_len,)`` — 1 for sampled tokens.
            advantage:     Scalar GRPO advantage for this response.
            clip_eps:      Clipping range.
            kl_coef:       KL penalty coefficient.
            loss_scale:    Scale factor (e.g. 1 / (G * grad_accum)).

        Returns:
            Scalar loss value (for logging; gradients already accumulated).
        """
        total_len = input_ids.shape[1]
        prompt_len = total_len - response_len
        device = input_ids.device

        latent_set = set()
        for pos in range(prompt_len, total_len):
            if input_ids[0, pos].item() == self.latent_token_id:
                latent_set.add(pos)

        segments = []
        seg_start = prompt_len
        in_latent = False

        for pos in range(prompt_len, total_len):
            is_lat = pos in latent_set
            if is_lat and not in_latent:
                if pos > seg_start:
                    segments.append(("text", seg_start, pos))
                seg_start = pos
                in_latent = True
            elif not is_lat and in_latent:
                if pos > seg_start:
                    segments.append(("latent", seg_start, pos))
                seg_start = pos
                in_latent = False

        if seg_start < total_len:
            segments.append(("latent" if in_latent else "text", seg_start, total_len))

        inputs_embeds = self.embedding(input_ids)
        position_ids = torch.arange(total_len, device=device).unsqueeze(0)
        attention_mask = torch.ones(1, total_len, device=device, dtype=torch.long)

        kv_cache = None
        processed_up_to = 0
        total_loss = 0.0

        # Forward prompt (no grad, just build KV cache)
        if prompt_len > 0:
            with torch.no_grad():
                out = self.base_causallm(
                    inputs_embeds=inputs_embeds[:, :prompt_len, :],
                    attention_mask=attention_mask[:, :prompt_len],
                    position_ids=position_ids[:, :prompt_len],
                    use_cache=True,
                    output_hidden_states=False,
                )
                kv_cache = out.past_key_values
            processed_up_to = prompt_len

        for seg_type, seg_start, seg_end in segments:
            seg_len = seg_end - seg_start

            if seg_type == "latent":
                with torch.no_grad():
                    for lat_pos in range(seg_start, seg_end):
                        lat_embed = inputs_embeds[:, lat_pos:lat_pos + 1, :]

                        out = self.base_causallm(
                            inputs_embeds=lat_embed,
                            attention_mask=attention_mask[:, :lat_pos + 1],
                            position_ids=position_ids[:, lat_pos:lat_pos + 1],
                            past_key_values=kv_cache,
                            use_cache=True,
                            output_hidden_states=True,
                        )
                        kv_cache = out.past_key_values

                        h = out.hidden_states[-1][:, -1:, :]

                        next_pos = lat_pos + 1
                        if next_pos < total_len and input_ids[0, next_pos].item() == self.latent_token_id:
                            emb_list = [inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])]
                            emb_list[next_pos] = h[0, 0, :]
                            inputs_embeds = torch.stack(emb_list).unsqueeze(0)

                processed_up_to = seg_end

            else:
                past_kv_detached = kv_cache

                seg_embeds = inputs_embeds[:, seg_start:seg_end, :]
                out = self.base_causallm(
                    inputs_embeds=seg_embeds,
                    attention_mask=attention_mask[:, :seg_end],
                    position_ids=position_ids[:, seg_start:seg_end],
                    past_key_values=past_kv_detached,
                    use_cache=True,
                    output_hidden_states=True,
                )

                seg_logits = out.logits  # (1, seg_len, V)

                seg_loss = torch.tensor(0.0, device=device)
                n_tokens = 0

                for j in range(seg_len):
                    abs_pos = seg_start + j
                    target_pos = abs_pos + 1
                    if target_pos >= total_len:
                        continue
                    resp_idx = target_pos - prompt_len
                    if resp_idx < 0 or resp_idx >= response_len:
                        continue
                    if sampled_mask[resp_idx].item() == 0:
                        continue

                    target_token = input_ids[0, target_pos]
                    new_lp = F.log_softmax(seg_logits[0, j, :], dim=-1)[target_token]
                    old_lp = old_log_probs[resp_idx]

                    log_ratio = new_lp - old_lp
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * advantage
                    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
                    token_loss = -torch.min(surr1, surr2)

                    if kl_coef > 0:
                        kl = torch.exp(log_ratio) - log_ratio - 1.0
                        token_loss = token_loss + kl_coef * kl

                    seg_loss = seg_loss + token_loss
                    n_tokens += 1

                if n_tokens > 0:
                    seg_loss = seg_loss / max(sampled_mask.sum().item(), 1) * loss_scale
                    seg_loss.backward()
                    total_loss += seg_loss.item()

                with torch.no_grad():
                    kv_cache = out.past_key_values

                    if seg_end < total_len and input_ids[0, seg_end].item() == self.latent_token_id:
                        h = out.hidden_states[-1][:, -1:, :]
                        emb_list = [inputs_embeds[0, p, :] for p in range(inputs_embeds.shape[1])]
                        emb_list[seg_end] = h[0, 0, :]
                        inputs_embeds = torch.stack(emb_list).unsqueeze(0)

                processed_up_to = seg_end

        return total_loss

    # ------------------------------------------------------------------
    # train / eval passthrough
    # ------------------------------------------------------------------

    def train(self, mode: bool = True):
        self.base_causallm.train(mode)
        return self

    def eval(self):
        self.base_causallm.eval()
        return self
