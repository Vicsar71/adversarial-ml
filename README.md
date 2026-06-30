# adversarial-ml

**Adversarial attacks on image classifiers** — imperceptible pixel perturbations that fool ResNet-50 into misclassifying images with high confidence.

Implements **FGSM** (Goodfellow et al., 2014) and **PGD** (Madry et al., 2017) to demonstrate a core vulnerability in deep neural networks: the same model that achieves 76% top-1 accuracy on ImageNet can be systematically deceived by noise invisible to the human eye. Built as an AI/ML security project — the techniques used in academic adversarial robustness research, applied hands-on.

---

## Example

Same image, same ε=0.03, perturbation invisible to the human eye. One gradient step vs forty:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Adversarial ML  comparing perro.jpg  FGSM vs PGD  ε=0.03  device=cuda     │
└─────────────────────────────────────────────────────────────────────────────┘

Original: Border collie (47.9%)

                        FGSM vs PGD  (ε=0.03)
 ─────────────────────────────────────────────────────────────────────────────
               FGSM (1 step)               PGD (40 steps)
 ─────────────────────────────────────────────────────────────────────────────
  Top-1        Border collie (22.2%)       manhole cover (99.6%)
  Status       RESISTED                    FOOLED
  L∞           0.03000                     0.03000
  L2           0.02642                     0.01937
```

FGSM halves the model's confidence but fails to change the prediction. PGD — same budget, same image — makes ResNet-50 classify a Border Collie as a manhole cover with 99.6% confidence.

Three files are saved to `reports/` after each run:

| File | Contents |
|---|---|
| `<name>_<method>_adv_<ts>.png` | Adversarial image — visually identical to the original |
| `<name>_<method>_perturbation_<ts>.png` | The noise amplified ×10 to make it visible |
| `<name>_<method>_<ts>.json` + `.md` | Full report: top-5 before/after and metrics |

The `compare` command additionally saves a side-by-side `<name>_compare_<ts>.json/md`.

---

## Quickstart

### Requirements

- Python 3.10+
- CUDA-capable GPU recommended (CPU works, but PGD with 40 steps is slow)

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
python -m adversarial attack photo.jpg

# stronger perturbation
python -m adversarial attack photo.jpg --epsilon 0.1

# PGD attack — 40 iterative steps, much stronger
python -m adversarial attack photo.jpg --method pgd --epsilon 0.03

# fewer PGD steps (faster, still effective)
python -m adversarial attack photo.jpg --method pgd --steps 20

# compare FGSM vs PGD side by side on the same image
python -m adversarial compare photo.jpg --epsilon 0.03

# save only Markdown (JSON is always written)
python -m adversarial attack photo.jpg --format md

# custom output directory
python -m adversarial attack photo.jpg --output my_reports/
```

---

## How it works

### FGSM (Fast Gradient Sign Method)

Instead of computing the gradient of the loss with respect to the model's *weights* (as in training), FGSM computes it with respect to the *input pixels*. The sign of that gradient tells us which direction to nudge each pixel to maximally increase the model's error. A perturbation of just ε=0.03 in [0,1] pixel space is often enough to degrade confidence significantly — invisible to humans, disruptive to the classifier.

```
x_adv = clamp( x + ε · sign(∇ₓ L(f(x), y)) , 0, 1)
```

### PGD (Projected Gradient Descent)

PGD is FGSM repeated N times with a smaller step size α, starting from a random point inside the ε-ball. After each step the iterate is projected back inside the ball. The random start avoids local optima; the projection guarantees the perturbation never exceeds ε. It almost always achieves full top-1 misclassification where single-step FGSM only degrades confidence.

```
x₀ = clamp(x + uniform(−ε, ε), 0, 1)
xₜ₊₁ = clamp( Proj_ε(xₜ + α · sign(∇ₓ L(f(xₜ), y))) , 0, 1)
```

### Design decision: attack in [0,1] space

ImageNet normalization (subtract mean, divide by std) is applied *inside* the model's forward pass, not to the input tensor. This means the attack always works in [0,1] pixel space, `clamp(0,1)` is meaningful, and ε values are directly comparable with the literature.

---

## Architecture

```
adversarial/
├── classifier.py        — ResNet-50 loader; PREPROCESS ([0,1]) separated from NORMALIZE
├── models.py            — Pydantic models: Prediction, AttackConfig, AttackResult, CompareResult
├── attacks/
│   ├── fgsm.py          — fgsm_step (pure testable kernel) + fgsm_attack
│   └── pgd.py           — pgd_step (pure testable kernel) + pgd_attack
├── reporter.py          — saves JSON + Markdown; save_reports and save_compare_reports
├── html_reporter.py     — self-contained HTML with base64 images (Milestone 3)
└── cli.py               — Typer CLI: attack and compare subcommands; lazy torch imports
tests/
├── test_models.py       — Pydantic model and serialization tests
├── test_attacks.py      — pure FGSM math tests (CPU tensors, no GPU or model needed)
└── test_pgd.py          — pure PGD math tests (CPU tensors, no GPU or model needed)
```

---

## Tests

```bash
pytest          # 21 tests, all pure logic — no GPU, no model download, no network
```

Both `fgsm_step` and `pgd_step` are tested in isolation with random CPU tensors, verifying: L∞ bound, [0,1] clamping, epsilon-ball projection, zero-step identity, gradient direction, shape preservation, and input immutability.

---

## Roadmap

- [x] **Milestone 1** — ResNet-50 classifier + FGSM attack + CLI + JSON/Markdown reports + 14 tests
- [x] **Milestone 2** — PGD attack (iterative, 40 steps) + `compare` command (FGSM vs PGD side by side) + 21 tests
- [ ] **Milestone 3** — Self-contained HTML report with base64-embedded images and top-5 confidence bar charts
- [ ] **Milestone 4** — Batch mode (attack a whole folder) + README with GIF demo

---

## License

MIT
