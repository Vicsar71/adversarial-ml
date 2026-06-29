"""
Pure-logic tests for attack math — no GPU, no model, no network.
We test fgsm_step directly (the testable kernel) with random CPU tensors.
"""
import torch
from adversarial.attacks.fgsm import fgsm_step


def _rand(shape=(1, 3, 224, 224), low=0.1, high=0.9) -> torch.Tensor:
    """Random [0,1] tensor that won't be clipped by a small epsilon."""
    return torch.empty(shape).uniform_(low, high)


def test_fgsm_step_within_epsilon():
    image = _rand()
    gradient = torch.randn_like(image)
    epsilon = 0.03
    adv = fgsm_step(image, gradient, epsilon)
    diff = (adv - image).abs().max().item()
    assert diff <= epsilon + 1e-6


def test_fgsm_step_clamped_to_unit_range():
    image = _rand(low=0.0, high=1.0)
    gradient = torch.ones_like(image)  # all positive → push toward 1
    adv = fgsm_step(image, gradient, epsilon=0.5)
    assert adv.min().item() >= 0.0
    assert adv.max().item() <= 1.0


def test_fgsm_step_zero_epsilon_is_identity():
    image = _rand()
    gradient = torch.randn_like(image)
    adv = fgsm_step(image, gradient, epsilon=0.0)
    assert torch.allclose(adv, image)


def test_fgsm_step_large_epsilon_clips_to_bounds():
    image = _rand(low=0.4, high=0.6)
    gradient = torch.ones_like(image)   # all push toward 1
    adv = fgsm_step(image, gradient, epsilon=1.0)
    assert torch.allclose(adv, torch.ones_like(image))


def test_fgsm_step_negative_gradient_pushes_down():
    image = _rand(low=0.5, high=0.6)
    gradient = -torch.ones_like(image)  # all push toward 0
    adv = fgsm_step(image, gradient, epsilon=0.03)
    assert (adv < image).all()


def test_fgsm_step_output_shape_preserved():
    image = _rand((1, 3, 224, 224))
    gradient = torch.randn_like(image)
    adv = fgsm_step(image, gradient, epsilon=0.03)
    assert adv.shape == image.shape


def test_fgsm_step_does_not_modify_input():
    image = _rand()
    image_copy = image.clone()
    gradient = torch.randn_like(image)
    fgsm_step(image, gradient, epsilon=0.03)
    assert torch.allclose(image, image_copy)
