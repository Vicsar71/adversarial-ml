from __future__ import annotations
import base64
import html
from pathlib import Path

from .models import AttackResult, CompareResult, Prediction

_CSS = """
body{margin:0;padding:24px;background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;font-size:14px;line-height:1.5}
h1{color:#e6edf3;border-bottom:1px solid #30363d;padding-bottom:12px;margin-top:0}
h2{color:#e6edf3;margin-top:32px;margin-bottom:12px}
h3{color:#e6edf3;margin:0 0 12px 0}
.meta{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px;margin:16px 0}
.meta p{margin:4px 0;color:#8b949e}
.meta strong{color:#c9d1d9}
.badge{display:inline-block;padding:2px 10px;border-radius:12px;font-size:0.85em;font-weight:600}
.badge-green{background:#1a4429;color:#3fb950}
.badge-yellow{background:#3d2b00;color:#d29922}
.images{display:flex;gap:24px;flex-wrap:wrap;margin:16px 0}
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:24px;margin:16px 0}
.card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:16px}
"""


def _load_b64(path: str) -> str:
    if not path:
        return ""
    p = Path(path)
    if not p.exists():
        return ""
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"


def _data_uri(path: str) -> str:
    b64 = _load_b64(path)
    return f"data:{_mime(path)};base64,{b64}" if b64 else ""


def _image_card(title: str, data_uri: str, label: str, label_color: str = "#c9d1d9") -> str:
    title_esc = html.escape(title)
    label_esc = html.escape(label)
    if data_uri:
        img = f'<img src="{data_uri}" style="width:224px;height:224px;object-fit:cover;border-radius:6px;display:block">'
    else:
        img = '<div style="width:224px;height:224px;background:#21262d;border-radius:6px;display:flex;align-items:center;justify-content:center;color:#8b949e">No image</div>'
    return (
        f'<div style="text-align:center">'
        f'<p style="font-weight:600;margin-bottom:8px;color:#e6edf3">{title_esc}</p>'
        f'{img}'
        f'<p style="margin-top:8px;color:{label_color};font-size:0.9em">{label_esc}</p>'
        f'</div>'
    )


def _bar_rows(preds: list[Prediction], highlight_id: int | None = None) -> str:
    rows = []
    for p in preds:
        pct = min(p.confidence * 100, 100.0)
        color = "#3fb950" if p.class_id == highlight_id else "#238636"
        name_esc = html.escape(p.class_name)
        rows.append(
            f'<div style="margin-bottom:8px">'
            f'<div style="display:flex;justify-content:space-between;margin-bottom:2px">'
            f'<span>{name_esc}</span>'
            f'<span style="color:#8b949e">{pct:.1f}%</span>'
            f'</div>'
            f'<div style="background:#21262d;border-radius:4px;height:14px">'
            f'<div style="width:{pct:.1f}%;background:{color};height:14px;border-radius:4px;min-width:2px"></div>'
            f'</div>'
            f'</div>'
        )
    return "\n".join(rows)


def render_html(result: AttackResult) -> str:
    orig_uri = _data_uri(result.image_path)
    adv_uri  = _data_uri(result.adversarial_image_path)
    pert_uri = _data_uri(result.perturbation_image_path)

    status = "FOOLED" if result.attack_successful else "RESISTED"
    badge_cls = "badge-green" if result.attack_successful else "badge-yellow"
    adv_color = "#3fb950" if result.attack_successful else "#d29922"

    orig_label = f"{result.original_top1.class_name} ({result.original_top1.confidence:.1%})"
    adv_label  = f"{result.adversarial_top1.class_name} ({result.adversarial_top1.confidence:.1%})"

    orig_card = _image_card("Original", orig_uri, orig_label)
    adv_card  = _image_card(f"Adversarial ({result.config.method.upper()})", adv_uri, adv_label, adv_color)
    pert_card = _image_card("Perturbation (10× amplified)", pert_uri, "Noise added to fool the model")

    method_esc   = html.escape(result.config.method.upper())
    img_path_esc = html.escape(result.image_path)
    ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    before_bars = _bar_rows(result.original_top5, result.original_top1.class_id)
    after_bars  = _bar_rows(result.adversarial_top5, result.adversarial_top1.class_id)

    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'  <meta charset="UTF-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>Adversarial Attack Report — {method_esc}</title>\n'
        f'  <style>{_CSS}</style>\n'
        f'</head>\n<body>\n'
        f'  <h1>Adversarial Attack Report</h1>\n'
        f'  <div class="meta">\n'
        f'    <p><strong>Image:</strong> {img_path_esc}</p>\n'
        f'    <p><strong>Method:</strong> {method_esc}'
        f'       &nbsp;&nbsp;<strong>ε:</strong> {result.config.epsilon}'
        f'       &nbsp;&nbsp;<strong>Timestamp:</strong> {ts}</p>\n'
        f'    <p><strong>Result:</strong> <span class="badge {badge_cls}">{status}</span>'
        f'       &nbsp;&nbsp;<strong>L∞:</strong> {result.l_inf:.6f}'
        f'       &nbsp;&nbsp;<strong>L2:</strong> {result.l2:.6f}</p>\n'
        f'  </div>\n\n'
        f'  <h2>Images</h2>\n'
        f'  <div class="images">\n'
        f'    {orig_card}\n'
        f'    {adv_card}\n'
        f'    {pert_card}\n'
        f'  </div>\n\n'
        f'  <h2>Top-5 Classifications</h2>\n'
        f'  <div class="two-col">\n'
        f'    <div class="card"><h3>Before</h3>{before_bars}</div>\n'
        f'    <div class="card"><h3>After</h3>{after_bars}</div>\n'
        f'  </div>\n'
        f'</body>\n</html>'
    )


def render_compare_html(result: CompareResult) -> str:
    f = result.fgsm
    p = result.pgd

    orig_uri      = _data_uri(result.image_path)
    fgsm_adv_uri  = _data_uri(f.adversarial_image_path)
    fgsm_pert_uri = _data_uri(f.perturbation_image_path)
    pgd_adv_uri   = _data_uri(p.adversarial_image_path)
    pgd_pert_uri  = _data_uri(p.perturbation_image_path)

    f_status = "FOOLED" if f.attack_successful else "RESISTED"
    p_status = "FOOLED" if p.attack_successful else "RESISTED"
    f_color  = "#3fb950" if f.attack_successful else "#d29922"
    p_color  = "#3fb950" if p.attack_successful else "#d29922"
    f_badge  = "badge-green" if f.attack_successful else "badge-yellow"
    p_badge  = "badge-green" if p.attack_successful else "badge-yellow"

    orig_label      = f"{result.original_top1.class_name} ({result.original_top1.confidence:.1%})"
    fgsm_adv_label  = f"{f.adversarial_top1.class_name} ({f.adversarial_top1.confidence:.1%}) — {f_status}"
    pgd_adv_label   = f"{p.adversarial_top1.class_name} ({p.adversarial_top1.confidence:.1%}) — {p_status}"

    orig_card      = _image_card("Original", orig_uri, orig_label)
    fgsm_adv_card  = _image_card("FGSM (1 step)", fgsm_adv_uri, fgsm_adv_label, f_color)
    pgd_adv_card   = _image_card(f"PGD ({p.config.steps} steps)", pgd_adv_uri, pgd_adv_label, p_color)
    fgsm_pert_card = _image_card("FGSM Perturbation (10×)", fgsm_pert_uri, "Noise amplified for visibility")
    pgd_pert_card  = _image_card("PGD Perturbation (10×)", pgd_pert_uri, "Noise amplified for visibility")

    img_path_esc = html.escape(result.image_path)
    ts = result.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC")

    fgsm_bars = _bar_rows(f.adversarial_top5, f.adversarial_top1.class_id)
    pgd_bars  = _bar_rows(p.adversarial_top5, p.adversarial_top1.class_id)

    return (
        f'<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        f'  <meta charset="UTF-8">\n'
        f'  <meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'  <title>Adversarial Attack Comparison — FGSM vs PGD</title>\n'
        f'  <style>{_CSS}</style>\n'
        f'</head>\n<body>\n'
        f'  <h1>Adversarial Attack Comparison: FGSM vs PGD</h1>\n'
        f'  <div class="meta">\n'
        f'    <p><strong>Image:</strong> {img_path_esc}</p>\n'
        f'    <p><strong>ε:</strong> {result.epsilon}'
        f'       &nbsp;&nbsp;<strong>FGSM steps:</strong> 1'
        f'       &nbsp;&nbsp;<strong>PGD steps:</strong> {p.config.steps}'
        f'       &nbsp;&nbsp;<strong>Timestamp:</strong> {ts}</p>\n'
        f'    <p>'
        f'<strong>FGSM:</strong> <span class="badge {f_badge}">{f_status}</span>'
        f'       &nbsp;&nbsp;'
        f'<strong>PGD:</strong> <span class="badge {p_badge}">{p_status}</span>'
        f'    </p>\n'
        f'  </div>\n\n'
        f'  <h2>Adversarial Images</h2>\n'
        f'  <div class="images">\n'
        f'    {orig_card}\n'
        f'    {fgsm_adv_card}\n'
        f'    {pgd_adv_card}\n'
        f'  </div>\n\n'
        f'  <h2>Perturbations</h2>\n'
        f'  <div class="images">\n'
        f'    {fgsm_pert_card}\n'
        f'    {pgd_pert_card}\n'
        f'  </div>\n\n'
        f'  <h2>Top-5 After FGSM</h2>\n'
        f'  <div class="card" style="max-width:440px">{fgsm_bars}</div>\n\n'
        f'  <h2>Top-5 After PGD</h2>\n'
        f'  <div class="card" style="max-width:440px">{pgd_bars}</div>\n'
        f'</body>\n</html>'
    )
