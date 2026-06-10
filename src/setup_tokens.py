"""
Phase 0 – Special Token Registration
=====================================
Registers <swi>, </swi>, <latent> into the tokenizer + model and initialises
their embeddings from an "anchor" token to avoid random-init instability.

Reference: coconut/run.py L105-156 (analogous registration for
           <|start-latent|>, <|end-latent|>, <|latent|>).

Usage
-----
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.setup_tokens import setup_special_tokens, SwiTokenConfig

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    token_cfg = setup_special_tokens(model, tokenizer)
    # Save the extended tokenizer so downstream code loads the right vocab size
    tokenizer.save_pretrained("path/to/save")
    print(token_cfg)
    # SwiTokenConfig(swi_start_id=..., swi_end_id=..., latent_id=..., eos_token_id=...)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

# ---------------------------------------------------------------------------
# Token string constants — single source of truth used across all phases
# ---------------------------------------------------------------------------
SWI_START_TOKEN = "<swi>"    # enter latent reasoning region
SWI_END_TOKEN   = "</swi>"   # exit  latent reasoning region (predicted by model)
LATENT_TOKEN    = "<latent>" # placeholder whose embedding is replaced by hidden state


@dataclass
class SwiTokenConfig:
    """Token IDs for the three SwiLatent special tokens.

    Returned by :func:`setup_special_tokens` and threaded through all
    Phase-2 (CoconutSwiModel, SwiCollator) and Phase-3 (rollout, reward)
    components.
    """
    swi_start_id: int  # <swi>    – boundary: begin latent region
    swi_end_id:   int  # </swi>   – boundary: end   latent region
    latent_id:    int  # <latent> – hidden-state placeholder (Phase 2+)
    eos_token_id: int  # model's EOS token ID (convenience copy)

    def as_dict(self) -> dict:
        return {
            "swi_start_id": self.swi_start_id,
            "swi_end_id":   self.swi_end_id,
            "latent_id":    self.latent_id,
            "eos_token_id": self.eos_token_id,
        }


def setup_special_tokens(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    anchor_token: str = "<<",
) -> SwiTokenConfig:
    """Add the three SwiLatent tokens to *tokenizer* and *model*, initialising
    their embeddings from *anchor_token*.

    This function MUST be called before:
      * constructing :class:`~src.model.coconut_swi_model.CoconutSwiModel`
      * building any Phase-2 or Phase-3 datasets

    It is idempotent: calling it twice on the same (model, tokenizer) pair
    raises no error (``add_tokens`` skips already-registered tokens).

    Args:
        model:        HuggingFace causal-LM (e.g. Qwen3ForCausalLM).
        tokenizer:    Matching tokenizer.  ``padding_side`` is forced to
                      ``'right'`` and ``pad_token`` is set to ``eos_token``
                      when absent.
        anchor_token: An existing vocabulary token whose embedding is copied
                      into the new tokens.  Falls back to ``"<"`` then to the
                      pad token if *anchor_token* is OOV.

    Returns:
        :class:`SwiTokenConfig` carrying the three new token IDs.
    """
    # ------------------------------------------------------------------ #
    # 0. Tokenizer hygiene                                                #
    # ------------------------------------------------------------------ #
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # 1. Register the three tokens                                        #
    # ------------------------------------------------------------------ #
    new_tokens = [SWI_START_TOKEN, SWI_END_TOKEN, LATENT_TOKEN]
    n_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    print(f"[setup_tokens] Requested {len(new_tokens)} tokens, "
          f"added {n_added} new token(s): {new_tokens}")

    swi_start_id = tokenizer.convert_tokens_to_ids(SWI_START_TOKEN)
    swi_end_id   = tokenizer.convert_tokens_to_ids(SWI_END_TOKEN)
    latent_id    = tokenizer.convert_tokens_to_ids(LATENT_TOKEN)

    # Sanity checks
    for name, tok_id in [("<swi>", swi_start_id), ("</swi>", swi_end_id),
                          ("<latent>", latent_id)]:
        assert tok_id != tokenizer.unk_token_id, (
            f"Token '{name}' registration failed — still maps to unk_token_id "
            f"({tokenizer.unk_token_id}).  Check tokenizer compatibility."
        )

    print(f"[setup_tokens] IDs: "
          f"swi_start={swi_start_id}, swi_end={swi_end_id}, "
          f"latent={latent_id}, eos={tokenizer.eos_token_id}")

    # ------------------------------------------------------------------ #
    # 2. Resize model token embeddings                                    #
    # ------------------------------------------------------------------ #
    old_vocab = model.get_input_embeddings().weight.shape[0]
    # Never shrink model embeddings: some LoRA checkpoints save modules_to_save
    # for embed/lm_head at the base-model vocab size.
    target_vocab = max(old_vocab, len(tokenizer))
    model.resize_token_embeddings(target_vocab)
    new_vocab = model.get_input_embeddings().weight.shape[0]
    print(f"[setup_tokens] Vocab size: {old_vocab} → {new_vocab}")

    # ------------------------------------------------------------------ #
    # 3. Resolve the anchor token for embedding initialisation            #
    # ------------------------------------------------------------------ #
    anchor_id = tokenizer.convert_tokens_to_ids(anchor_token)
    if anchor_id == tokenizer.unk_token_id:
        anchor_id = tokenizer.convert_tokens_to_ids("<")
    if anchor_id == tokenizer.unk_token_id:
        anchor_id = tokenizer.pad_token_id
        print(f"[setup_tokens] Warning: anchor_token='{anchor_token}' is OOV. "
              f"Using pad_token_id={anchor_id} for embedding init.")
    else:
        print(f"[setup_tokens] Using anchor_token='{anchor_token}' "
              f"(id={anchor_id}) for embedding init.")

    # ------------------------------------------------------------------ #
    # 4. Copy anchor embedding → new tokens (input embeddings)           #
    # ------------------------------------------------------------------ #
    input_embeddings = model.get_input_embeddings()
    anchor_embed = input_embeddings.weight.data[anchor_id].clone()
    for tok_id in [swi_start_id, swi_end_id, latent_id]:
        input_embeddings.weight.data[tok_id] = anchor_embed

    # ------------------------------------------------------------------ #
    # 5. Update lm_head weight when NOT tied to input embeddings          #
    #    (GPT-2 ties them; Qwen3 / LLaMA3 do NOT → must update both)     #
    # ------------------------------------------------------------------ #
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        tied = (
            input_embeddings.weight.data_ptr() == lm_head.weight.data_ptr()
        )
        if not tied:
            anchor_logit = lm_head.weight.data[anchor_id].clone()
            for tok_id in [swi_start_id, swi_end_id, latent_id]:
                lm_head.weight.data[tok_id] = anchor_logit
            print("[setup_tokens] lm_head updated (weights are NOT tied).")
        else:
            print("[setup_tokens] lm_head shares weights with embeddings — "
                  "no separate update needed.")

    return SwiTokenConfig(
        swi_start_id=swi_start_id,
        swi_end_id=swi_end_id,
        latent_id=latent_id,
        eos_token_id=tokenizer.eos_token_id,
    )
