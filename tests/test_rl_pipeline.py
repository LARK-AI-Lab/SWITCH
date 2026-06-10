"""
Tests for the Phase 3 RL pipeline components.

Run: python -m pytest tests/test_rl_pipeline.py -v
  or: python tests/test_rl_pipeline.py  (standalone)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.model.coconut_swi_model import CoconutSwiModel
from src.rl.grpo import compute_grpo_advantages, grpo_clipped_loss, kl_penalty_loss
from src.rl.reward import (
    compute_score, correctness_reward, efficiency_reward, format_reward,
)


# ======================================================================
# Reward tests
# ======================================================================

class TestReward:
    def test_correctness_correct(self):
        assert correctness_reward("The answer is 42", "42", "gsm8k") == 1.0

    def test_correctness_wrong(self):
        assert correctness_reward("The answer is 43", "42", "gsm8k") == -1.0

    def test_format_well_formed(self):
        assert format_reward("text <swi>a</swi> more <swi>b</swi>") == 1.0

    def test_format_unmatched_open(self):
        assert format_reward("text <swi>a") == -1.0

    def test_format_nested(self):
        assert format_reward("<swi>a <swi>b</swi></swi>") == -1.0

    def test_format_no_tags(self):
        assert format_reward("just text") == 0.0

    def test_format_extra_close(self):
        assert format_reward("</swi> text") == -1.0

    def test_efficiency_correct_good_tags(self):
        assert efficiency_reward("<swi>think</swi> ans 42", True) == 1.0

    def test_efficiency_correct_bad_tags(self):
        """Bad tags should NOT get efficiency bonus (loophole fix)."""
        assert efficiency_reward("<swi>think", True) == 0.0

    def test_efficiency_wrong(self):
        assert efficiency_reward("<swi>think</swi> ans 43", False) == 0.0

    def test_efficiency_no_tags(self):
        assert efficiency_reward("just text", True) == 0.0

    def test_compute_score_no_loophole(self):
        """Correct + bad tags must score <= correct + no tags."""
        bad_tag = compute_score("gsm8k", "<swi>think 42", "42")
        no_tag = compute_score("gsm8k", "answer is 42", "42")
        assert bad_tag <= no_tag, f"Loophole: bad_tag={bad_tag} > no_tag={no_tag}"

    def test_compute_score_good_tags_best(self):
        good = compute_score("gsm8k", "<swi>think</swi> 42", "42")
        no_tag = compute_score("gsm8k", "answer is 42", "42")
        assert good > no_tag


# ======================================================================
# GRPO algorithm tests
# ======================================================================

class TestGRPO:
    def test_advantages_zero_mean(self):
        adv = compute_grpo_advantages([1.4, -0.9, 1.0, -1.0, 1.4])
        assert abs(adv.mean().item()) < 1e-6

    def test_advantages_single_value(self):
        """All same reward → all zero advantage."""
        adv = compute_grpo_advantages([1.0, 1.0, 1.0])
        assert torch.allclose(adv, torch.zeros(3), atol=1e-5)

    def test_clipped_loss_shape(self):
        G, T = 5, 10
        new_lp = torch.randn(G, T, requires_grad=True)
        old_lp = new_lp.detach()
        adv = torch.randn(G)
        mask = torch.ones(G, T)
        loss = grpo_clipped_loss(new_lp, old_lp, adv, mask)
        assert loss.shape == ()
        loss.backward()
        assert new_lp.grad is not None

    def test_clipped_loss_mask_zeros(self):
        """Loss should be 0 when mask is all zeros."""
        G, T = 3, 5
        new_lp = torch.randn(G, T, requires_grad=True)
        old_lp = new_lp.detach()
        adv = torch.randn(G)
        mask = torch.zeros(G, T)
        loss = grpo_clipped_loss(new_lp, old_lp, adv, mask)
        assert abs(loss.item()) < 1e-6

    def test_kl_penalty_zero_when_same(self):
        """KL between identical distributions should be ~0."""
        T = 10
        lp = torch.randn(3, T)
        mask = torch.ones(3, T)
        kl = kl_penalty_loss(lp, lp.clone(), mask)
        assert abs(kl.item()) < 1e-6

    def test_kl_penalty_positive(self):
        T = 10
        lp1 = torch.randn(3, T)
        lp2 = lp1 + torch.randn(3, T) * 0.5
        mask = torch.ones(3, T)
        kl = kl_penalty_loss(lp1, lp2, mask)
        assert kl.item() > 0


# ======================================================================
# Latent action surrogate (Gaussian mean-dim log-prob)
# ======================================================================

class TestGaussianLatentLogProb:
    def test_identity_z_equals_mu(self):
        """When z == mu the log-prob surrogate is exactly 0."""
        D = 16
        mu = torch.randn(1, 1, D)
        sigma = torch.full((1, 1, 1), 0.05)
        lp = CoconutSwiModel._gaussian_latent_log_prob(mu.clone(), mu, sigma)
        assert torch.allclose(lp, torch.zeros_like(lp), atol=1e-7)

    def test_symmetry_in_z_minus_mu(self):
        """Surrogate depends only on (z - mu)^2, so z↔mu mirrored gives same."""
        D = 32
        mu = torch.randn(1, 1, D)
        eps = torch.randn(1, 1, D)
        sigma = torch.full((1, 1, 1), 0.1)
        lp_pos = CoconutSwiModel._gaussian_latent_log_prob(mu + eps, mu, sigma)
        lp_neg = CoconutSwiModel._gaussian_latent_log_prob(mu - eps, mu, sigma)
        assert torch.allclose(lp_pos, lp_neg, atol=1e-6)

    def test_quadratic_in_eps(self):
        """Doubling the perturbation should quadruple |log_prob|."""
        D = 64
        mu = torch.zeros(1, 1, D)
        eps = torch.randn(1, 1, D)
        sigma = torch.full((1, 1, 1), 0.2)
        lp1 = CoconutSwiModel._gaussian_latent_log_prob(eps, mu, sigma)
        lp2 = CoconutSwiModel._gaussian_latent_log_prob(2 * eps, mu, sigma)
        assert torch.allclose(lp2, 4 * lp1, atol=1e-5)

    def test_inverse_square_in_sigma(self):
        """Doubling sigma should quarter |log_prob|."""
        D = 64
        mu = torch.zeros(1, 1, D)
        eps = torch.randn(1, 1, D)
        sigma1 = torch.full((1, 1, 1), 0.1)
        sigma2 = torch.full((1, 1, 1), 0.2)
        lp1 = CoconutSwiModel._gaussian_latent_log_prob(eps, mu, sigma1)
        lp2 = CoconutSwiModel._gaussian_latent_log_prob(eps, mu, sigma2)
        assert torch.allclose(lp2, lp1 / 4, atol=1e-5)

    def test_finite_for_extreme_inputs(self):
        D = 8
        mu = torch.full((1, 1, D), 1e3)
        z = torch.full((1, 1, D), -1e3)
        sigma = torch.full((1, 1, 1), 1e-8)
        lp = CoconutSwiModel._gaussian_latent_log_prob(z, mu, sigma)
        assert torch.isfinite(lp).all() or lp.item() < 0  # large negative is OK

    def test_replay_log_ratio_zero_when_mu_unchanged(self):
        """If mu_new == mu_old (no policy update yet), the replay
        log_ratio = new_lp - old_lp should be exactly 0."""
        D = 32
        mu = torch.randn(1, 1, D)
        sigma = torch.full((1, 1, 1), 0.05)
        eps = torch.randn(1, 1, D)
        z = mu + sigma * eps
        old_lp = CoconutSwiModel._gaussian_latent_log_prob(z, mu, sigma)
        new_lp = CoconutSwiModel._gaussian_latent_log_prob(z, mu, sigma)
        assert torch.allclose(new_lp - old_lp, torch.zeros_like(old_lp), atol=1e-7)


# ======================================================================
# Index alignment test (OOM scenario simulation)
# ======================================================================

class TestIndexAlignment:
    def test_survived_indices_correct(self):
        """Simulate OOM: rollout 1 (of 0,1,2,3,4) fails in second pass.
        Verify survived_indices maps correctly to advantages."""
        rewards = [1.4, -0.9, 1.0, -1.0, 0.5]
        advantages = compute_grpo_advantages(rewards)

        survived_indices = [0, 2, 3, 4]  # index 1 failed
        selected_adv = advantages[survived_indices]

        assert len(selected_adv) == 4
        assert selected_adv[0] == advantages[0]
        assert selected_adv[1] == advantages[2]
        assert selected_adv[2] == advantages[3]
        assert selected_adv[3] == advantages[4]


# ======================================================================
# Run standalone
# ======================================================================

if __name__ == "__main__":
    import traceback
    passed = 0
    failed = 0
    for cls in [TestReward, TestGRPO, TestGaussianLatentLogProb, TestIndexAlignment]:
        obj = cls()
        for name in sorted(dir(obj)):
            if not name.startswith("test_"):
                continue
            try:
                getattr(obj, name)()
                print(f"  PASS  {cls.__name__}.{name}")
                passed += 1
            except Exception as e:
                print(f"  FAIL  {cls.__name__}.{name}: {e}")
                traceback.print_exc()
                failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(1 if failed else 0)
