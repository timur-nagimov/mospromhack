"""Правило без явной severity получает default_severity плагина, а явная severity не меняется."""

from guard.contracts import CameraConfig, RuleConfig, Severity, SourceConfig
from guard.pipeline.config import load_cameras, save_cameras
from guard.rules.base import with_default_severity


def _camera(*rules: RuleConfig) -> CameraConfig:
    return CameraConfig(camera_id="c1", name="c1", site="s", source=SourceConfig(type="file", uri="x.mp4"), rules=list(rules))


def test_unset_severity_takes_plugin_default():
    cam = with_default_severity(_camera(RuleConfig(rule_id="r1", type="fall_immobile")))
    rule = cam.rules[0]
    assert rule.severity is Severity.critical
    assert "severity" in rule.model_fields_set  # API отдаст critical, и UI сохранит именно его


def test_explicit_severity_is_kept():
    cam = with_default_severity(_camera(RuleConfig(rule_id="r1", type="fall_immobile", severity=Severity.medium)))
    assert cam.rules[0].severity is Severity.medium


def test_unknown_type_is_left_for_validation():
    cam = _camera(RuleConfig(rule_id="r1", type="no_such_rule"))
    assert with_default_severity(cam) is cam


def test_get_put_roundtrip_keeps_plugin_default(tmp_path):
    path = tmp_path / "cameras.yaml"
    path.write_text("cameras:\n- {camera_id: c1, name: c1, site: s, source: {type: file, uri: x.mp4},\n   rules: [{rule_id: r1, type: vehicle_proximity}]}\n", encoding="utf-8")
    loaded = load_cameras(path)
    assert loaded[0].rules[0].severity is Severity.critical
    # UI: GET -> JSON -> PUT
    again = with_default_severity(CameraConfig.model_validate(loaded[0].model_dump(mode="json")))
    save_cameras(path, [again])
    assert load_cameras(path)[0].rules[0].severity is Severity.critical
