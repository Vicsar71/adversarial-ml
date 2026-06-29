from datetime import datetime, timezone
from adversarial.models import Prediction, AttackConfig, AttackResult


def _pred(class_id=0, class_name="cat", confidence=0.9) -> Prediction:
    return Prediction(class_id=class_id, class_name=class_name, confidence=confidence)


def _result(**kwargs) -> AttackResult:
    defaults = dict(
        image_path="test.jpg",
        original_top1=_pred(0, "cat", 0.95),
        original_top5=[_pred(0, "cat", 0.95), _pred(1, "dog", 0.03)],
        adversarial_top1=_pred(2, "rifle", 0.87),
        adversarial_top5=[_pred(2, "rifle", 0.87)],
        config=AttackConfig(method="fgsm", epsilon=0.03),
        attack_successful=True,
        l_inf=0.03,
        l2=0.005,
        timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    defaults.update(kwargs)
    return AttackResult(**defaults)


def test_prediction_stores_fields():
    p = _pred(42, "golden retriever", 0.987)
    assert p.class_id == 42
    assert p.class_name == "golden retriever"
    assert p.confidence == 0.987


def test_attack_config_defaults():
    cfg = AttackConfig()
    assert cfg.method == "fgsm"
    assert cfg.epsilon == 0.03
    assert cfg.steps == 40
    assert cfg.alpha is None


def test_attack_config_custom():
    cfg = AttackConfig(method="pgd", epsilon=0.05, steps=20, alpha=0.005)
    assert cfg.method == "pgd"
    assert cfg.alpha == 0.005


def test_attack_result_successful():
    r = _result(attack_successful=True)
    assert r.attack_successful is True
    assert r.adversarial_top1.class_name == "rifle"


def test_attack_result_failed():
    r = _result(
        adversarial_top1=_pred(0, "cat", 0.91),
        attack_successful=False,
    )
    assert r.attack_successful is False


def test_attack_result_optional_paths_default_empty():
    r = _result()
    assert r.adversarial_image_path == ""
    assert r.perturbation_image_path == ""


def test_attack_result_json_roundtrip():
    import json
    r = _result()
    data = json.loads(r.model_dump_json())
    assert data["image_path"] == "test.jpg"
    assert data["attack_successful"] is True
    assert data["config"]["method"] == "fgsm"
