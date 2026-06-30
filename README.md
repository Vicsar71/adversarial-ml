# adversarial-ml

**Adversarial attacks on image classifiers** — imperceptible pixel perturbations that fool ResNet-50 into misclassifying images with high confidence.

Implements **FGSM** (Goodfellow et al., 2014) and **PGD** (Madry et al., 2017) to demonstrate a core vulnerability in deep neural networks: the same model that achieves 76% top-1 accuracy on ImageNet can be systematically deceived by noise invisible to the human eye. Built as an AI/ML security project — the techniques used in academic adversarial robustness research, applied hands-on.

---

## Example

A single FGSM step with ε=0.03 nearly halves the model's confidence on a correctly classified image. With ε=0.1 or using PGD (iterative), the top-1 prediction changes entirely.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Adversarial ML  attacking perro.jpg  method=FGSM  ε=0.03  device=cuda     │
└─────────────────────────────────────────────────────────────────────────────┘

Original:     Border collie (47.9%)
Adversarial:  Border collie (22.1%)   confidence nearly halved with invisible noise

                       Top-5 Before vs After
 ─────────────────────────────────────────────────────────────────────
  1   Border collie   47.9%      Border collie   22.1%
  2   collie          10.3%      collie           6.0%
  3   kelpie           2.6%      Cardigan         2.3%
  4   Cardigan         0.6%      kelpie           1.4%
  5   Appenzeller      0.6%      tennis ball      0.6%
```

Three files are saved to `reports/` after each attack:

| File | Contents |
|---|---|
| `<name>_fgsm_adv_<ts>.png` | Adversarial image — visually identical to the original |
| `<name>_fgsm_perturbation_<ts>.png` | The noise amplified ×10 to make it visible |
| `<name>_fgsm_<ts>.json` | Full report: top-5 before/after, L∞ and L2 metrics |
| `<name>_fgsm_<ts>.md` | Human-readable Markdown report |

---

## Quickstart

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works, but slower for PGD)

### Install

```bash
git clone https://github.com/Vicsar71/adversarial-ml.git
cd adversarial-ml
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

# PyTorch with CUDA 13 (RTX 30xx / 40xx)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
# CPU-only alternative:
# pip install torch torchvision

pip install pillow pydantic "typer[all]" rich pytest
```

> ResNet-50 weights (~100 MB) are downloaded automatically from PyTorch Hub on first run.

### Run

```bash
# FGSM attack — one gradient step (default ε=0.03)
python -m adversarial photo.jpg

# stronger perturbation — more likely to fool the classifier
python -m adversarial photo.jpg --epsilon 0.1

# PGD attack — iterative, much stronger (Milestone 2)
python -m adversarial photo.jpg --method pgd --epsilon 0.03

# save only Markdown (JSON is always written)
python -m adversarial photo.jpg --format md

# custom output directory
python -m adversarial photo.jpg --output my_reports/

# compare FGSM vs PGD on the same image (Milestone 2)
python -m adversarial compare photo.jpg --epsilon 0.03
```

---

## How it works

### FGSM (Fast Gradient Sign Method)

Instead of computing the gradient of the loss with respect to the model's *weights* (as in training), FGSM computes it with respect to the *input pixels*. The sign of that gradient tells us which direction to nudge each pixel to maximally increase the model's error. A perturbation of just ε=0.03 (in [0,1] pixel space) is often enough to degrade confidence significantly — invisible to humans, disruptive to the classifier.

```
x_adv = clamp( x + ε · sign(∇ₓ L(f(x), y)) , 0, 1)
```

### PGD (Projected Gradient Descent)

PGD is FGSM repeated N times with a smaller step size α, starting from a random point inside the ε-ball. Each step increases the loss slightly; after N steps the perturbation is projected back inside the ε-ball. It almost always achieves full top-1 misclassification where single-step FGSM only degrades confidence.

### Design decision: attack in [0,1] space

ImageNet normalization (subtract mean, divide by std) is applied *inside* the model's forward pass, not to the input tensor. This means the attack always works in [0,1] pixel space, `clamp(0,1)` is meaningful, and ε values are directly comparable with the literature.

---

## Architecture

```
adversarial/
├── classifier.py        — ResNet-50 loader; PREPROCESS ([0,1]) separated from NORMALIZE
├── models.py            — Pydantic models: Prediction, AttackConfig, AttackResult
├── attacks/
│   ├── fgsm.py          — fgsm_step (pure testable kernel) + fgsm_attack
│   └── pgd.py           — PGD iterative attack with random init (Milestone 2)
├── reporter.py          — saves JSON + Markdown to reports/
├── html_reporter.py     — self-contained HTML with base64 images (Milestone 3)
└── cli.py               — Typer CLI; lazy torch imports so --help is instant
tests/
├── test_models.py       — Pydantic model and serialization tests
└── test_attacks.py      — pure math tests on CPU tensors (no GPU or model needed)
```

---

## Tests

```bash
pytest          # 14 tests, all pure logic — no GPU, no model download, no network
```

The mathematical kernel (`fgsm_step`) is tested in isolation with random CPU tensors, verifying: L∞ bound, [0,1] clamping, zero-epsilon identity, negative gradient direction, shape preservation, and input immutability.

---

## Roadmap

- [x] **Milestone 1** — ResNet-50 classifier + FGSM attack + CLI + JSON/Markdown reports + 14 tests
- [ ] **Milestone 2** — PGD attack (iterative, stronger) + `compare` command (FGSM vs PGD side by side)
- [ ] **Milestone 3** — Self-contained HTML report with base64-embedded images and top-5 confidence bar charts
- [ ] **Milestone 4** — Batch mode (attack a whole folder) + README with GIF demo

---

## License

MIT
