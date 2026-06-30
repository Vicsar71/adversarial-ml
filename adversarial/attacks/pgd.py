from __future__ import annotations
import torch
import torch.nn.functional as F
from ..classifier import NORMALIZE


def pgd_step(
    x_adv: torch.Tensor,
    gradient: torch.Tensor,
    epsilon: float,
    alpha: float,
    x_orig: torch.Tensor,
) -> torch.Tensor:
    """
    Single PGD update (pure kernel — no model needed, testable in isolation).

    Applies an FGSM micro-step of size alpha, then projects back onto the
    L-inf epsilon-ball centred at x_orig, then clamps to [0, 1].
    """
    stepped = x_adv + alpha * gradient.sign()
    projected = x_orig + (stepped - x_orig).clamp(-epsilon, epsilon)
    return projected.clamp(0, 1)


def pgd_attack(
    model: torch.nn.Module,
    image_01: torch.Tensor,
    true_class: int,
    epsilon: float,
    steps: int = 40,
    alpha: float | None = None,
) -> torch.Tensor:
    """
    Untargeted PGD attack (Madry et al., 2017).

    Starts from a random point inside the L-inf epsilon-ball, then takes
    `steps` gradient steps of size `alpha`, projecting back onto the ball
    after each step. Significantly stronger than single-step FGSM.

    Returns adversarial tensor in [0, 1], detached from the computation graph.
    """
    if alpha is None:
        alpha = epsilon / 4

    x = (image_01.clone() + torch.empty_like(image_01).uniform_(-epsilon, epsilon)).clamp(0, 1)

    for _ in range(steps):
        x = x.detach().requires_grad_(True)
        logits = model(NORMALIZE(x))
        label = torch.tensor([true_class], device=x.device)
        F.cross_entropy(logits, label).backward()
        with torch.no_grad():
            x = pgd_step(x, x.grad, epsilon, alpha, image_01)

    return x.detach()
