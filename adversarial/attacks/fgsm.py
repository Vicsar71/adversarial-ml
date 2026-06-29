from __future__ import annotations
import torch
import torch.nn.functional as F
from ..classifier import NORMALIZE


def fgsm_step(image_01: torch.Tensor, gradient: torch.Tensor, epsilon: float) -> torch.Tensor:
    """
    Pure FGSM step (no model needed — testable in isolation).
    Returns perturbed tensor clamped to [0, 1].
    """
    perturbed = image_01 + epsilon * gradient.sign()
    return perturbed.clamp(0, 1)


def fgsm_attack(
    model: torch.nn.Module,
    image_01: torch.Tensor,
    true_class: int,
    epsilon: float,
) -> torch.Tensor:
    """
    Untargeted FGSM attack (Goodfellow et al., 2014).

    Maximises cross-entropy loss w.r.t. true_class by taking one gradient step
    in the direction that increases the loss the most. Perturbation is bounded
    by epsilon in L-inf norm in [0, 1] pixel space.

    Returns adversarial tensor in [0, 1], detached from the computation graph.
    """
    x = image_01.clone().requires_grad_(True)
    logits = model(NORMALIZE(x))
    label = torch.tensor([true_class], device=x.device)
    loss = F.cross_entropy(logits, label)
    loss.backward()

    with torch.no_grad():
        adversarial = fgsm_step(x, x.grad, epsilon)

    return adversarial.detach()
