from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path

import pytest
from PIL import Image

from adversarial.models import AttackConfig, AttackResult, CompareResult, Prediction
from adversarial.html_reporter import render_html, render_compare_html


_TS = datetime(2025, 1, 1, tzinfo=timezone.utc)


def _pred(name: str, conf: float, class_id: int = 0) -> Prediction:
    return Prediction(class_id=class_id, class_name=name, confidence=conf)


def _make_image(path: Path) -> None:
    Image.new("RGB", (4, 4), color=(128, 0, 0)).save(path)


def _attack_result(
    image_path: str = "",
    adv_path: str = "",
    pert_path: str = "",
    fooled: bool = True,
) -> AttackResult:
    orig = _pred("Border Collie", 0.479, class_id=0)
    adv  = _pred("manhole cover", 0.996, class_id=1) if fooled else _pred("Border Collie", 0.3, class_id=0)
    return AttackResult(
        image_path=image_path,
        original_top1=orig,
        original_top5=[orig],
        adversarial_top1=adv,
        adversarial_top5=[adv],
        config=AttackConfig(method="fgsm", epsilon=0.03),
        attack_successful=fooled,
        l_inf=0.03,
        l2=0.01,
        adversarial_image_path=adv_path,
        perturbation_image_path=pert_path,
        timestamp=_TS,
    )


# ── render_html ──────────────────────────────────────────────────────────────

def test_render_html_escapes_class_name():
    malicious = "<script>alert(1)</script>"
    pred = _pred(malicious, 0.9, class_id=0)
    result = AttackResult(
        image_path="test.jpg",
        original_top1=pred,
        original_top5=[pred],
        adversarial_top1=pred,
        adversarial_top5=[pred],
        config=AttackConfig(method="fgsm", epsilon=0.03),
        attack_successful=False,
        l_inf=0.0,
        l2=0.0,
        timestamp=_TS,
    )
    out = render_html(result)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_html_embeds_three_images(tmp_path):
    orig_path = tmp_path / "orig.jpg"
    adv_path  = tmp_path / "adv.png"
    pert_path = tmp_path / "pert.png"
    _make_image(orig_path)
    _make_image(adv_path)
    _make_image(pert_path)

    result = _attack_result(str(orig_path), str(adv_path), str(pert_path))
    out = render_html(result)

    assert "data:image/jpeg;base64," in out
    assert out.count("data:image/png;base64,") >= 2


def test_render_html_confidence_bars_present():
    result = _attack_result()
    out = render_html(result)
    assert "height:14px" in out


def test_render_html_no_image_files_shows_placeholder():
    result = _attack_result()  # all paths empty
    out = render_html(result)
    assert "No image" in out


def test_render_html_fooled_badge():
    out = render_html(_attack_result(fooled=True))
    assert "FOOLED" in out
    assert "badge-green" in out


def test_render_html_resisted_badge():
    out = render_html(_attack_result(fooled=False))
    assert "RESISTED" in out
    assert "badge-yellow" in out


def test_render_html_escapes_image_path():
    result = _attack_result(image_path='<b>evil</b>.jpg')
    out = render_html(result)
    assert "<b>evil</b>" not in out
    assert "&lt;b&gt;evil&lt;/b&gt;" in out


# ── render_compare_html ───────────────────────────────────────────────────────

def test_render_compare_html_embeds_five_images(tmp_path):
    orig_path  = tmp_path / "orig.jpg"
    fadv_path  = tmp_path / "fadv.png"
    fpert_path = tmp_path / "fpert.png"
    padv_path  = tmp_path / "padv.png"
    ppert_path = tmp_path / "ppert.png"
    for p in (orig_path, fadv_path, fpert_path, padv_path, ppert_path):
        _make_image(p)

    orig = _pred("Border Collie", 0.479, class_id=0)
    fgsm = _attack_result(str(orig_path), str(fadv_path), str(fpert_path), fooled=False)
    pgd  = AttackResult(
        image_path=str(orig_path),
        original_top1=orig, original_top5=[orig],
        adversarial_top1=_pred("manhole cover", 0.996, class_id=1),
        adversarial_top5=[_pred("manhole cover", 0.996, class_id=1)],
        config=AttackConfig(method="pgd", epsilon=0.03, steps=40),
        attack_successful=True,
        l_inf=0.03, l2=0.01,
        adversarial_image_path=str(padv_path),
        perturbation_image_path=str(ppert_path),
        timestamp=_TS,
    )
    result = CompareResult(
        image_path=str(orig_path),
        epsilon=0.03,
        original_top1=orig,
        original_top5=[orig],
        fgsm=fgsm,
        pgd=pgd,
        timestamp=_TS,
    )
    out = render_compare_html(result)

    assert "data:image/jpeg;base64," in out      # original (orig.jpg)
    assert out.count("data:image/png;base64,") >= 4  # fadv, fpert, padv, ppert


def test_render_compare_html_escapes_image_path():
    orig = _pred("dog", 0.9, class_id=0)
    fgsm = _attack_result(fooled=False)
    pgd  = AttackResult(
        image_path="",
        original_top1=orig, original_top5=[orig],
        adversarial_top1=_pred("cat", 0.8, class_id=1),
        adversarial_top5=[_pred("cat", 0.8, class_id=1)],
        config=AttackConfig(method="pgd", epsilon=0.03, steps=40),
        attack_successful=True,
        l_inf=0.03, l2=0.01,
        timestamp=_TS,
    )
    result = CompareResult(
        image_path='<script>xss</script>.jpg',
        epsilon=0.03,
        original_top1=orig,
        original_top5=[orig],
        fgsm=fgsm,
        pgd=pgd,
        timestamp=_TS,
    )
    out = render_compare_html(result)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


def test_render_compare_html_shows_both_statuses():
    orig = _pred("dog", 0.9, class_id=0)
    fgsm = _attack_result(fooled=False)
    pgd  = AttackResult(
        image_path="",
        original_top1=orig, original_top5=[orig],
        adversarial_top1=_pred("cat", 0.8, class_id=1),
        adversarial_top5=[_pred("cat", 0.8, class_id=1)],
        config=AttackConfig(method="pgd", epsilon=0.03, steps=40),
        attack_successful=True,
        l_inf=0.03, l2=0.01,
        timestamp=_TS,
    )
    result = CompareResult(
        image_path="test.jpg",
        epsilon=0.03,
        original_top1=orig,
        original_top5=[orig],
        fgsm=fgsm,
        pgd=pgd,
        timestamp=_TS,
    )
    out = render_compare_html(result)
    assert "RESISTED" in out
    assert "FOOLED" in out
    assert "badge-yellow" in out
    assert "badge-green" in out
