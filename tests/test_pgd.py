"""
Pure-logic tests for PGD attack math — no GPU, no model, no network.
We test pgd_step directly (the testable kernel) with random CPU tensors.
"""
import torch
from adversarial.attacks.pgd import pgd_step


def _rand(shape=(1, 3, 224, 224), low=0.1, high=0.9) -> torch.Tensor:
    return torch.empty(shape).uniform_(low, high)


def test_pgd_step_projects_onto_epsilon_ball():
    x_orig = _rand()
    gradient = torch.randn_like(x_orig)
    epsilon, alpha = 0.03, 0.01
    x_new = pgd_step(x_orig.clone(), gradient, epsilon, alpha, x_orig)
    assert (x_new - x_orig).abs().max().item() <= epsilon + 1e-6


def test_pgd_step_clamped_to_unit_range():
    x_orig = _rand(low=0.0, high=1.0)
    gradient = torch.ones_like(x_orig)
    x_new = pgd_step(x_orig.clone(), gradient, epsilon=0.5, alpha=1.0, x_orig=x_orig)
    assert x_new.min().item() >= 0.0
    assert x_new.max().item() <= 1.0


def test_pgd_step_zero_alpha_is_identity():
    x_orig = _rand()
    x_adv = x_orig.clone()
    gradient = torch.randn_like(x_orig)
    x_new = pgd_step(x_adv, gradient, epsilon=0.03, alpha=0.0, x_orig=x_orig)
    assert torch.allclose(x_new, x_adv)


def test_pgd_step_negative_gradient_pushes_down():
    x_orig = _rand(low=0.5, high=0.6)
    x_adv = x_orig.clone()
    gradient = -torch.ones_like(x_orig)
    x_new = pgd_step(x_adv, gradient, epsilon=0.03, alpha=0.01, x_orig=x_orig)
    assert (x_new < x_adv).all()


def test_pgd_step_output_shape_preserved():
    x_orig = _rand((1, 3, 224, 224))
    x_new = pgd_step(x_orig.clone(), torch.randn_like(x_orig), 0.03, 0.01, x_orig)
    assert x_new.shape == x_orig.shape


def test_pgd_step_does_not_modify_inputs():
    x_orig = _rand()
    x_adv = x_orig.clone()
    x_orig_copy = x_orig.clone()
    x_adv_copy = x_adv.clone()
    pgd_step(x_adv, torch.randn_like(x_orig), 0.03, 0.01, x_orig)
    assert torch.allclose(x_orig, x_orig_copy)
    assert torch.allclose(x_adv, x_adv_copy)


def test_pgd_step_large_alpha_still_clips_to_epsilon_ball():
    """Even with alpha >> epsilon, projection keeps deviation within epsilon."""
    x_orig = _rand(low=0.4, high=0.6)
    gradient = torch.ones_like(x_orig)
    x_new = pgd_step(x_orig.clone(), gradient, epsilon=0.03, alpha=10.0, x_orig=x_orig)
    assert (x_new - x_orig).abs().max().item() <= 0.03 + 1e-6
