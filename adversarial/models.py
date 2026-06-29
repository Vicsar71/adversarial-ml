from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel


class Prediction(BaseModel):
    class_id: int
    class_name: str
    confidence: float  # 0–1


class AttackConfig(BaseModel):
    method: str = "fgsm"        # "fgsm" | "pgd"
    epsilon: float = 0.03       # max L-inf perturbation in [0,1] pixel space
    steps: int = 40             # PGD only
    alpha: float | None = None  # PGD step size; defaults to epsilon/10


class AttackResult(BaseModel):
    image_path: str
    original_top1: Prediction
    original_top5: list[Prediction]
    adversarial_top1: Prediction
    adversarial_top5: list[Prediction]
    config: AttackConfig
    attack_successful: bool     # True if top-1 class changed
    l_inf: float                # max absolute pixel perturbation
    l2: float                   # L2 norm of perturbation (normalized by pixels)
    adversarial_image_path: str = ""
    perturbation_image_path: str = ""
    timestamp: datetime
