from __future__ import annotations
from pathlib import Path
from .models import AttackResult, CompareResult
from .html_reporter import render_html, render_compare_html


def save_reports(
    result: AttackResult,
    output_dir: Path = Path("reports"),
    fmt: str = "both",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{stem}_{result.config.method}_{ts}"

    paths: dict[str, Path] = {}

    json_path = base.with_suffix(".json")
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    paths["json"] = json_path

    if fmt in ("md", "both"):
        md_path = base.with_suffix(".md")
        md_path.write_text(_render_markdown(result), encoding="utf-8")
        paths["markdown"] = md_path

    if fmt in ("html", "both"):
        html_path = base.with_suffix(".html")
        html_path.write_text(render_html(result), encoding="utf-8")
        paths["html"] = html_path

    return paths


def _render_markdown(r: AttackResult) -> str:
    orig = r.original_top1
    adv  = r.adversarial_top1
    status = "SUCCESS" if r.attack_successful else "FAILED"

    lines = [
        f"# Adversarial Attack Report",
        f"",
        f"**Image:** `{r.image_path}`  ",
        f"**Method:** {r.config.method.upper()}  ",
        f"**Epsilon (ε):** {r.config.epsilon}  ",
        f"**Timestamp:** {r.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}  ",
        f"**Result:** {status}",
        f"",
        f"## Classification: Before → After",
        f"",
        f"| | Class | Confidence |",
        f"|--|-------|-----------|",
        f"| Original | {orig.class_name} | {orig.confidence:.1%} |",
        f"| Adversarial | {adv.class_name} | {adv.confidence:.1%} |",
        f"",
        f"## Perturbation",
        f"",
        f"- **L∞ norm:** {r.l_inf:.6f} (max pixel shift in [0,1])",
        f"- **L2 norm:** {r.l2:.6f}",
        f"",
        f"## Top-5 Before",
        f"",
        f"| Rank | Class | Confidence |",
        f"|------|-------|-----------|",
    ]
    for i, p in enumerate(r.original_top5, 1):
        lines.append(f"| {i} | {p.class_name} | {p.confidence:.1%} |")

    lines += [
        f"",
        f"## Top-5 After",
        f"",
        f"| Rank | Class | Confidence |",
        f"|------|-------|-----------|",
    ]
    for i, p in enumerate(r.adversarial_top5, 1):
        lines.append(f"| {i} | {p.class_name} | {p.confidence:.1%} |")

    if r.adversarial_image_path:
        lines += ["", f"## Output Images", "", f"- Adversarial: `{r.adversarial_image_path}`"]
    if r.perturbation_image_path:
        lines.append(f"- Perturbation (10× amplified): `{r.perturbation_image_path}`")

    return "\n".join(lines)


def save_compare_reports(
    result: CompareResult,
    output_dir: Path = Path("reports"),
    fmt: str = "both",
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(result.image_path).stem
    ts = result.timestamp.strftime("%Y%m%d_%H%M%S")
    base = output_dir / f"{stem}_compare_{ts}"

    paths: dict[str, Path] = {}

    json_path = base.with_suffix(".json")
    json_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    paths["json"] = json_path

    if fmt in ("md", "both"):
        md_path = base.with_suffix(".md")
        md_path.write_text(_render_compare_markdown(result), encoding="utf-8")
        paths["markdown"] = md_path

    if fmt in ("html", "both"):
        html_path = base.with_suffix(".html")
        html_path.write_text(render_compare_html(result), encoding="utf-8")
        paths["html"] = html_path

    return paths


def _render_compare_markdown(r: CompareResult) -> str:
    f = r.fgsm
    p = r.pgd
    f_status = "SUCCESS" if f.attack_successful else "FAILED"
    p_status = "SUCCESS" if p.attack_successful else "FAILED"

    lines = [
        "# Adversarial Attack Comparison: FGSM vs PGD",
        "",
        f"**Image:** `{r.image_path}`  ",
        f"**Epsilon (ε):** {r.epsilon}  ",
        f"**Timestamp:** {r.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Original Classification",
        "",
        f"**{r.original_top1.class_name}** ({r.original_top1.confidence:.1%})",
        "",
        "## Results",
        "",
        f"| | FGSM (1 step) | PGD ({p.config.steps} steps) |",
        "|--|--------------|--------------|",
        f"| Top-1 | {f.adversarial_top1.class_name} ({f.adversarial_top1.confidence:.1%}) | {p.adversarial_top1.class_name} ({p.adversarial_top1.confidence:.1%}) |",
        f"| Status | {f_status} | {p_status} |",
        f"| L∞ | {f.l_inf:.6f} | {p.l_inf:.6f} |",
        f"| L2 | {f.l2:.6f} | {p.l2:.6f} |",
        "",
        "## Top-5 After FGSM",
        "",
        "| Rank | Class | Confidence |",
        "|------|-------|-----------|",
    ]
    for i, pred in enumerate(f.adversarial_top5, 1):
        lines.append(f"| {i} | {pred.class_name} | {pred.confidence:.1%} |")

    lines += [
        "",
        "## Top-5 After PGD",
        "",
        "| Rank | Class | Confidence |",
        "|------|-------|-----------|",
    ]
    for i, pred in enumerate(p.adversarial_top5, 1):
        lines.append(f"| {i} | {pred.class_name} | {pred.confidence:.1%} |")

    return "\n".join(lines)
