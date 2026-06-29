from __future__ import annotations
import torch
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
from .models import Prediction

_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)

# Spatial preprocessing: resize + crop + to float tensor in [0, 1]
PREPROCESS = T.Compose([
    T.Resize(256, interpolation=T.InterpolationMode.BILINEAR),
    T.CenterCrop(224),
    T.ToTensor(),
])

# Normalization applied separately so attacks work in [0, 1] space
NORMALIZE = T.Normalize(mean=_MEAN, std=_STD)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval().to(device)
    categories: list[str] = weights.meta["categories"]
    return model, categories


def load_image(path: str, device: torch.device) -> torch.Tensor:
    """Load image as [0,1] tensor of shape (1, 3, 224, 224)."""
    img = Image.open(path).convert("RGB")
    return PREPROCESS(img).unsqueeze(0).to(device)


def classify(
    model: torch.nn.Module,
    image_01: torch.Tensor,
    categories: list[str],
    top_k: int = 5,
) -> list[Prediction]:
    """Classify a [0,1] tensor. Returns top-k predictions."""
    with torch.no_grad():
        logits = model(NORMALIZE(image_01))
    probs = torch.softmax(logits[0], dim=0)
    top_probs, top_ids = probs.topk(top_k)
    return [
        Prediction(
            class_id=idx.item(),
            class_name=categories[idx.item()],
            confidence=round(prob.item(), 6),
        )
        for prob, idx in zip(top_probs, top_ids)
    ]


def tensor_to_pil(image_01: torch.Tensor) -> Image.Image:
    """Convert (1,3,H,W) [0,1] tensor → PIL Image."""
    return T.ToPILImage()(image_01.squeeze(0).cpu().clamp(0, 1))


def perturbation_to_pil(original: torch.Tensor, adversarial: torch.Tensor, amplify: float = 10.0) -> Image.Image:
    """Visualize the perturbation amplified for human visibility."""
    diff = (adversarial - original).abs() * amplify
    return T.ToPILImage()(diff.squeeze(0).cpu().clamp(0, 1))
