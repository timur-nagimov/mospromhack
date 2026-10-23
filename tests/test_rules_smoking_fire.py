from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from guard.contracts import (
    CameraConfig,
    Detection,
    FrameContext,
    RuleConfig,
    Severity,
    SourceConfig,
    Zone,
)
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.rules.plugins.smoking_fire import SmokingFireRule


@pytest.fixture
def ctx() -> FrameContext:
    ts = datetime(2026, 10, 24, 12, tzinfo=ZoneInfo("Europe/Moscow")).timestamp()
    camera = CameraConfig(
        camera_id="camera-1",
        name="Камера",
        site="Участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
        zones=[
            Zone(
                zone_id="no-smoking",
                name="Курение запрещено",
                kind="no_smoking",
                polygon=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.9), (0.1, 0.9)],
            )
        ],
    )
    detection = Detection(
        model="fire",
        cls="fire",
        cls_id=0,
        conf=0.9,
        bbox=(60.0, 80.0, 100.0, 180.0),
    )
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((300, 400, 3), dtype=np.uint8),
        detections={"fire": [detection]},
    )


def _rule(ctx: FrameContext, **overrides) -> SmokingFireRule:
    cfg = RuleConfig(rule_id="fire-test", type="smoking_fire", **overrides)
    return SmokingFireRule(cfg, ctx.camera)


def test_fire_in_no_smoking_zone(ctx):
    violation, = _rule(ctx).evaluate(ctx)
    assert violation.rule_id == "fire-test"
    assert violation.type == "smoking_fire"
    assert violation.track_id is None
    assert violation.zone_id == "no-smoking"
    assert violation.bbox == ctx.detections["fire"][0].bbox
    assert violation.severity == Severity.critical
    assert violation.score == pytest.approx(0.9)
    assert violation.details == {"cls": "fire", "conf": 0.9}


@pytest.mark.parametrize(
    "cls, include_smoke, expected",
    [
        ("fire", False, 1),
        ("fire", True, 1),
        ("smoke", False, 0),
        ("smoke", True, 1),
        ("person", True, 0),
        ("cell phone", True, 0),
    ],
)
def test_class_selection(ctx, cls, include_smoke, expected):
    ctx.detections["fire"][0].cls = cls
    rule = _rule(ctx, params={"include_smoke": include_smoke})
    violations = rule.evaluate(ctx)
    assert len(violations) == expected
    if violations:
        assert violations[0].details["cls"] == cls


def test_smoke_is_disabled_by_default(ctx):
    ctx.detections["fire"][0].cls = "smoke"
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("cls", ["fire", "smoke"])
@pytest.mark.parametrize(
    "confidence, expected",
    [
        (0.499, 0),
        (0.5, 1),
        (0.501, 1),
        (float("nan"), 0),
        (float("inf"), 0),
    ],
)
def test_confidence_threshold(ctx, cls, confidence, expected):
    detection = ctx.detections["fire"][0]
    detection.cls = cls
    detection.conf = confidence
    rule = _rule(ctx, params={"include_smoke": True})
    assert len(rule.evaluate(ctx)) == expected


def test_custom_confidence_threshold(ctx):
    assert _rule(ctx, params={"min_conf": 0.95}).evaluate(ctx) == []


@pytest.mark.parametrize("cls", ["fire", "smoke"])
@pytest.mark.parametrize("width, expected", [(19.0, 0), (20.0, 1), (21.0, 1)])
def test_area_threshold(ctx, cls, width, expected):
    detection = ctx.detections["fire"][0]
    detection.cls = cls
    detection.bbox = (60.0, 80.0, 60.0 + width, 100.0)
    rule = _rule(ctx, params={"include_smoke": True})
    assert len(rule.evaluate(ctx)) == expected


def test_custom_area_threshold(ctx):
    assert _rule(ctx, params={"min_area_px": 4001}).evaluate(ctx) == []


@pytest.mark.parametrize(
    "box",
    [
        (60.0, 80.0, 60.0, 180.0),
        (100.0, 80.0, 60.0, 180.0),
        (100.0, 180.0, 60.0, 80.0),
        (float("nan"), 80.0, 100.0, 180.0),
        (60.0, 80.0, 100.0, float("inf")),
    ],
)
def test_invalid_box(ctx, box):
    ctx.detections["fire"][0].bbox = box
    assert _rule(ctx).evaluate(ctx) == []


def test_outside_no_smoking_zone(ctx):
    ctx.detections["fire"][0].bbox = (240.0, 80.0, 280.0, 180.0)
    assert _rule(ctx).evaluate(ctx) == []


def test_zone_uses_center_even_when_bottom_is_outside(ctx):
    ctx.camera.zones[0].polygon = [
        (0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5),
    ]
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_bottom_inside_does_not_replace_center_check(ctx):
    ctx.camera.zones[0].polygon = [
        (0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.1, 0.9),
    ]
    assert _rule(ctx).evaluate(ctx) == []


def test_zone_boundary_is_included(ctx):
    ctx.detections["fire"][0].bbox = (180.0, 80.0, 220.0, 180.0)
    assert len(_rule(ctx).evaluate(ctx)) == 1


@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize(
    "hours, expected",
    [(["08:00-10:00"], 0), (["11:00-13:00"], 1)],
)
def test_zone_schedule(ctx, explicit, hours, expected):
    ctx.camera.zones[0].active_hours = hours
    zones = ["no-smoking"] if explicit else None
    assert len(_rule(ctx, zones=zones).evaluate(ctx)) == expected


def test_explicit_zone_can_have_another_kind(ctx):
    ctx.camera.zones[0].kind = "danger"
    violation, = _rule(ctx, zones=["no-smoking"]).evaluate(ctx)
    assert violation.zone_id == "no-smoking"


def test_explicit_selection_excludes_other_no_smoking_zones(ctx):
    ctx.camera.zones.append(Zone(
        zone_id="other",
        name="Другой участок",
        kind="danger",
        polygon=[(0.6, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9)],
    ))
    assert _rule(ctx, zones=["other"]).evaluate(ctx) == []


def test_unknown_explicit_zone_does_not_enable_whole_frame(ctx):
    assert _rule(ctx, zones=["unknown"]).evaluate(ctx) == []


@pytest.mark.parametrize("zones", [None, []])
def test_no_zones_checks_whole_frame(ctx, zones):
    ctx.camera.zones = []
    violation, = _rule(ctx, zones=zones).evaluate(ctx)
    assert violation.zone_id is None


def test_unrelated_zone_does_not_limit_whole_frame(ctx):
    ctx.camera.zones[0].kind = "danger"
    ctx.detections["fire"][0].bbox = (240.0, 80.0, 280.0, 180.0)
    violation, = _rule(ctx).evaluate(ctx)
    assert violation.zone_id is None


@pytest.mark.parametrize("cls", ["fire", "smoke"])
def test_whole_frame_excludes_smoking_area(ctx, cls):
    ctx.camera.zones[0].kind = "smoking_area"
    ctx.detections["fire"][0].cls = cls
    rule = _rule(ctx, params={"include_smoke": True})
    assert rule.evaluate(ctx) == []


def test_outside_smoking_area_is_checked(ctx):
    ctx.camera.zones[0].kind = "smoking_area"
    ctx.detections["fire"][0].bbox = (240.0, 80.0, 280.0, 180.0)
    violation, = _rule(ctx).evaluate(ctx)
    assert violation.zone_id is None


def test_inactive_smoking_area_does_not_exclude_detection(ctx):
    ctx.camera.zones[0].kind = "smoking_area"
    ctx.camera.zones[0].active_hours = ["08:00-10:00"]
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_smoking_area_exclusion_uses_center(ctx):
    ctx.camera.zones[0].kind = "smoking_area"
    ctx.camera.zones[0].polygon = [
        (0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5),
    ]
    assert _rule(ctx).evaluate(ctx) == []


def test_explicit_zone_selection_takes_precedence_over_fallback_exclusion(ctx):
    ctx.camera.zones[0].kind = "smoking_area"
    violation, = _rule(ctx, zones=["no-smoking"]).evaluate(ctx)
    assert violation.zone_id == "no-smoking"


def test_overlapping_zones_return_one_violation_per_detection(ctx):
    ctx.camera.zones.append(
        ctx.camera.zones[0].model_copy(update={"zone_id": "overlap"})
    )
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_spatial_dedup_is_left_to_event_manager(ctx):
    first = ctx.detections["fire"][0]
    ctx.detections["fire"].append(Detection(
        model="fire",
        cls="smoke",
        cls_id=1,
        conf=0.8,
        bbox=first.bbox,
    ))
    violations = _rule(ctx, params={"include_smoke": True}).evaluate(ctx)
    assert len(violations) == 2
    assert all(violation.track_id is None for violation in violations)
    assert [violation.details["cls"] for violation in violations] == [
        "fire", "smoke",
    ]


def test_detections_from_other_model_are_not_used(ctx):
    detection = ctx.detections["fire"][0]
    detection.model = "det"
    ctx.detections = {"det": [detection]}
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("detections", [{}, {"fire": []}])
def test_no_detections(ctx, detections):
    ctx.detections = detections
    assert _rule(ctx).evaluate(ctx) == []


def test_each_frame_is_evaluated_without_confirmation_or_dedup(ctx):
    rule = _rule(ctx, min_duration_s=100)
    assert len(rule.evaluate(ctx)) == 1
    assert len(rule.evaluate(ctx)) == 1
    ctx.detections.clear()
    assert rule.evaluate(ctx) == []


@pytest.mark.parametrize("severity", list(Severity))
def test_explicit_severity(ctx, severity):
    violation, = _rule(ctx, severity=severity).evaluate(ctx)
    assert violation.severity == severity


def test_default_severity_does_not_mutate_config(ctx):
    cfg = RuleConfig(rule_id="fire-test", type="smoking_fire")
    rule = SmokingFireRule(cfg, ctx.camera)
    assert rule.evaluate(ctx)[0].severity == Severity.critical
    assert cfg.severity == Severity.high
    assert "severity" not in cfg.model_fields_set


def test_plugin_is_available():
    load_plugins()
    rules = {rule["type"]: rule for rule in available_rules()}
    assert {"phone_usage", "smoking_fire"} <= rules.keys()
    assert get_rule("smoking_fire") is SmokingFireRule
    assert rules["smoking_fire"]["required_models"] == ["fire"]
    assert rules["smoking_fire"]["default_severity"] == "critical"
    assert rules["smoking_fire"]["params_schema"] == SmokingFireRule.params_schema
    properties = SmokingFireRule.params_schema["properties"]
    assert properties["include_smoke"]["default"] is False
    assert properties["min_conf"]["default"] == 0.5
    assert properties["min_area_px"]["default"] == 400
    for schema in properties.values():
        assert any("а" <= char.lower() <= "я" for char in schema["title"])
