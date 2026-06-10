"""
Phase 3 – Reward Functions for verl GRPO
=========================================
Five complementary reward signals:

1. **Correctness** (dominant): +1 if answer is correct, -1 otherwise.
2. **Efficiency**: bonus for *moderate* latent usage when the answer is correct.
3. **Format**: small bonus/penalty for well-formed ``<swi>``/``</swi>`` tags.
4. **Visible length**: small penalty for *correct* responses that are too long
   in visible tokens.
5. **Latent overuse**: small penalty once the model exceeds a free latent-block
   budget, preventing reward hacking via many tiny ``<swi>`` blocks.

The top-level ``compute_score`` function follows verl's reward interface::

    compute_score(data_source, solution_str, ground_truth, extra_info=None)

and is registered in the verl config via
``reward.custom_reward_function.path`` / ``name``.
"""

from __future__ import annotations

import re
from typing import Optional

SWI_OPEN = "<swi>"
SWI_CLOSE = "</swi>"


# ---------------------------------------------------------------------------
# Grader (reuse SwiLatent/grader.py when available, fall back to simple match)
# ---------------------------------------------------------------------------

_HAS_GRADER = False
try:
    import sys, os

    _project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)

    from grader import gsm8k_grader, math500_grader
    _HAS_GRADER = True
except ImportError:
    pass


def _simple_grade(solution_str: str, ground_truth: str) -> bool:
    """Fallback grader: extract last number and compare."""
    gt = ground_truth.strip().replace(" ", "").replace(",", "")
    numbers = re.findall(r"[-+]?\d*\.?\d+", solution_str.replace(",", ""))
    if not numbers:
        return False
    pred = numbers[-1].strip()
    if gt == pred:
        return True
    try:
        return abs(float(pred) - float(gt)) < 1e-4
    except (ValueError, TypeError):
        return False


def _grade(solution_str: str, ground_truth: str, data_source: str) -> bool:
    if _HAS_GRADER:
        try:
            if "gsm8k" in data_source:
                ok, _ = gsm8k_grader(solution_str, ground_truth)
                return bool(ok)
            else:
                # math500_grader works for all math datasets (olympiads,
                # cn_contest, aops_forum, inequalities, swilatent_math, etc.)
                ok, _ = math500_grader(solution_str, ground_truth)
                return bool(ok)
        except Exception:
            pass
    return _simple_grade(solution_str, ground_truth)


# ---------------------------------------------------------------------------
# Individual reward components
# ---------------------------------------------------------------------------

def correctness_reward(solution_str: str, ground_truth: str,
                       data_source: str) -> float:
    return 1.0 if _grade(solution_str, ground_truth, data_source) else -1.0


def _count_swi_blocks(solution_str: str) -> int:
    return solution_str.count(SWI_OPEN)


def efficiency_reward(
    solution_str: str,
    is_correct: bool,
    extra_info: Optional[dict] = None,
) -> float:
    """Bonus for *moderate* well-formed latent usage on correct answers.

    The policy should use latent mode when it helps, but not spam many tiny
    ``<swi>`` blocks. We therefore reward a small band of latent usage rather
    than any positive count.

    Returns:
        +1.0  for 1-3 latent blocks on a correct, well-formed response
        +0.5  for 4-6 latent blocks
         0.0  otherwise
    """
    if not is_correct:
        return 0.0
    if format_reward(solution_str) <= 0:
        return 0.0
    if extra_info is not None and extra_info.get("n_latent_blocks") is not None:
        n_blocks = int(extra_info["n_latent_blocks"])
    else:
        n_blocks = _count_swi_blocks(solution_str)

    if n_blocks == 0:
        return 0.0
    if 1 <= n_blocks <= 3:
        return 1.0
    if 4 <= n_blocks <= 6:
        return 0.5
    return 0.0


def format_reward(solution_str: str) -> float:
    """Check ``<swi>``/``</swi>`` tag well-formedness.

    Returns:
        +1.0  if tags are well-formed (matched, non-nested)
        -1.0  if tags are malformed
         0.0  if no tags at all
    """
    opens = [m.start() for m in re.finditer(re.escape(SWI_OPEN), solution_str)]
    closes = [m.start() for m in re.finditer(re.escape(SWI_CLOSE), solution_str)]

    if not opens and not closes:
        return 0.0

    if len(opens) != len(closes):
        return -1.0

    depth = 0
    events = [(p, "open") for p in opens] + [(p, "close") for p in closes]
    events.sort()

    for _, kind in events:
        if kind == "open":
            depth += 1
            if depth > 1:
                return -1.0
        else:
            depth -= 1
            if depth < 0:
                return -1.0

    return 1.0 if depth == 0 else -1.0


def visible_length_penalty(
    is_correct: bool,
    extra_info: Optional[dict] = None,
) -> float:
    """Penalize overly long *visible* responses, but only when correct.

    This keeps correctness dominant while nudging the model toward better
    surface-level compression for evaluation-time efficiency comparisons.
    """
    if not is_correct:
        return 0.0
    if extra_info is None:
        return 0.0

    n_visible_tokens = extra_info.get("n_visible_tokens")
    if n_visible_tokens is None:
        return 0.0

    try:
        n_visible_tokens = float(n_visible_tokens)
    except (TypeError, ValueError):
        return 0.0

    no_penalty_tokens = 1200.0
    full_penalty_tokens = 2000.0
    if n_visible_tokens <= no_penalty_tokens:
        return 0.0
    if n_visible_tokens >= full_penalty_tokens:
        return 1.0
    return (n_visible_tokens - no_penalty_tokens) / (
        full_penalty_tokens - no_penalty_tokens
    )


def latent_overuse_penalty(extra_info: Optional[dict] = None) -> float:
    """Penalize latent-block spam after a small free budget.

    The first few latent blocks are free to preserve compression ability.
    Once the model keeps opening many blocks, the penalty ramps up.
    """
    if extra_info is None:
        return 0.0

    n_blocks = extra_info.get("n_latent_blocks")
    if n_blocks is None:
        return 0.0

    try:
        n_blocks = float(n_blocks)
    except (TypeError, ValueError):
        return 0.0

    free_blocks = 3.0
    full_penalty_blocks = 9.0
    if n_blocks <= free_blocks:
        return 0.0
    if n_blocks >= full_penalty_blocks:
        return 1.0
    return (n_blocks - free_blocks) / (full_penalty_blocks - free_blocks)


# ---------------------------------------------------------------------------
# verl-compatible compute_score
# ---------------------------------------------------------------------------

W_CORRECT = 1.0
W_EFFICIENCY = 0.2
W_FORMAT = 0.1
W_VISIBLE_LEN = 0.08
W_LATENT_OVERUSE = 0.05


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict] = None,
) -> float:
    """Combined reward following verl's ``compute_score`` interface.

    Called by verl's NaiveRewardManager for each rollout sample.
    """
    r_correct = correctness_reward(solution_str, ground_truth, data_source)
    is_correct = r_correct > 0
    r_eff = efficiency_reward(solution_str, is_correct, extra_info)
    r_fmt = format_reward(solution_str)
    r_len = visible_length_penalty(is_correct, extra_info)
    r_overuse = latent_overuse_penalty(extra_info)

    return (
        W_CORRECT * r_correct
        + W_EFFICIENCY * r_eff
        + W_FORMAT * r_fmt
        - W_VISIBLE_LEN * r_len
        - W_LATENT_OVERUSE * r_overuse
    )
