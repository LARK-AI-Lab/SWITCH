"""
SwiLatent src package
=====================
SFT+RL based Switch Latent Reasoning — combining SwiReasoning (entropy-based
switching) with Coconut (continuous hidden-state reasoning).

Directory layout
----------------
src/
├── setup_tokens.py          # Phase 0 – register <swi>, </swi>, <latent> tokens
├── model/
│   └── coconut_swi_model.py # Phase 2 – multi-pass hidden-state injection model
├── data/
│   └── coconut_swi_dataset.py  # Phase 2 – curriculum dataset & SwiCollator
└── rl/
    ├── rollout.py           # Phase 3 – latent-mode rollout for verl GRPO
    └── reward.py            # Phase 3 – correctness + efficiency + format reward
"""

from .setup_tokens import setup_special_tokens, SwiTokenConfig

__all__ = ["setup_special_tokens", "SwiTokenConfig"]
