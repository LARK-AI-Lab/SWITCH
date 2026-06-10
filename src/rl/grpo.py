"""
GRPO (Group Relative Policy Optimisation) core algorithm.

Value-free RL: advantages are estimated from group reward statistics
instead of a learned critic. For each question, G responses are sampled;
the advantage of response *i* is ``(r_i - mean(r)) / (std(r) + eps)``.

The policy gradient uses the standard clipped surrogate objective
(same as PPO but without a value baseline).
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F


def compute_grpo_advantages(
    rewards: List[float],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute GRPO advantages from a group of rewards.

    Args:
        rewards: List of G scalar rewards for one question.
        eps:     Small constant to avoid division by zero.

    Returns:
        Tensor of shape ``(G,)`` with normalized advantages.
    """
    r = torch.tensor(rewards, dtype=torch.float32)
    mean_r = r.mean()
    std_r = r.std(unbiased=False)
    return (r - mean_r) / (std_r + eps)


def grpo_clipped_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    clip_eps: float = 0.2,
) -> torch.Tensor:
    """Compute the GRPO clipped surrogate loss.

    Args:
        new_log_probs:  ``(G, T)`` log-probs under current policy.
        old_log_probs:  ``(G, T)`` log-probs from the rollout policy.
        advantages:     ``(G,)`` per-response advantages.
        response_mask:  ``(G, T)`` binary mask (1 for valid response tokens).
        clip_eps:       Clipping range for the probability ratio.

    Returns:
        Scalar loss (to be minimized).
    """
    log_ratio = new_log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    adv = advantages.unsqueeze(-1)  # (G, 1) broadcast over T

    surr1 = ratio * adv
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv

    per_token_loss = -torch.min(surr1, surr2)

    masked_loss = (per_token_loss * response_mask).sum() / response_mask.sum().clamp(min=1)
    return masked_loss


def kl_penalty_loss(
    new_log_probs: torch.Tensor,
    ref_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token KL divergence penalty.

    Uses the approximation ``KL ≈ exp(log_ratio) - log_ratio - 1`` which
    is numerically stable and matches verl's ``low_var_kl`` estimator.
    """
    log_ratio = new_log_probs - ref_log_probs
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (kl * response_mask).sum() / response_mask.sum().clamp(min=1)
