from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from guard.contracts import (
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    SourceConfig,
    Track,
    Zone,
)
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.rules.plugins.phone_usage import PhoneUsageRule


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
                zone_id="workshop",
                name="Цех",
                kind="danger",
                polygon=[(0.1, 0.1), (0.5, 0.1), (0.5, 0.9), (0.1, 0.9)],
            )
        ],
    )
    person = Track(
        track_id=1,
        cls="person",
        bbox=(60.0, 80.0, 100.0, 180.0),
        conf=0.95,
        first_seen=ts,
        last_seen=ts,
        attributes={"phone": True},
    )
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((300, 400, 3), dtype=np.uint8),
        people=[person],
    )


def _rule(ctx: FrameContext, **overrides) -> PhoneUsageRule:
    cfg = RuleConfig(rule_id="phone-test", type="phone_usage", **overrides)
    return PhoneUsageRule(cfg, ctx.camera)


def test_phone_in_selected_zone(ctx):
    violation, = _rule(ctx, zones=["workshop"]).evaluate(ctx)
    assert violation.rule_id == "phone-test"
    assert violation.type == "phone_usage"
    assert violation.track_id == 1
    assert violation.zone_id == "workshop"
    assert violation.bbox == ctx.people[0].bbox
    assert violation.severity == Severity.medium
    assert violation.score == pytest.approx(0.95)
    assert violation.details == {"phone": True}


@pytest.mark.parametrize("value", [False, None, 0, 1, "true", np.bool_(True)])
def test_only_builtin_true_means_phone(ctx, value):
    ctx.people[0].attributes["phone"] = value
    assert _rule(ctx).evaluate(ctx) == []


def test_missing_phone_attribute(ctx):
    ctx.people[0].attributes.clear()
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("zones", [None, []])
def test_unspecified_zones_check_whole_frame(ctx, zones):
    ctx.people[0].bbox = (240.0, 80.0, 280.0, 180.0)
    violation, = _rule(ctx, zones=zones).evaluate(ctx)
    assert violation.zone_id is None


def test_no_camera_zones(ctx):
    ctx.camera.zones = []
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_outside_selected_zone(ctx):
    ctx.people[0].bbox = (240.0, 80.0, 280.0, 180.0)
    assert _rule(ctx, zones=["workshop"]).evaluate(ctx) == []


def test_zone_uses_foot_point(ctx):
    ctx.camera.zones[0].polygon = [
        (0.1, 0.5), (0.5, 0.5), (0.5, 0.9), (0.1, 0.9),
    ]
    assert len(_rule(ctx, zones=["workshop"]).evaluate(ctx)) == 1


def test_center_inside_but_foot_outside_zone(ctx):
    ctx.camera.zones[0].polygon = [
        (0.1, 0.1), (0.5, 0.1), (0.5, 0.5), (0.1, 0.5),
    ]
    assert _rule(ctx, zones=["workshop"]).evaluate(ctx) == []


def test_zone_boundary_is_included(ctx):
    ctx.people[0].bbox = (180.0, 80.0, 220.0, 180.0)
    assert len(_rule(ctx, zones=["workshop"]).evaluate(ctx)) == 1


def test_unknown_zone_does_not_enable_whole_frame(ctx):
    assert _rule(ctx, zones=["unknown"]).evaluate(ctx) == []


@pytest.mark.parametrize(
    "hours, expected",
    [(["08:00-10:00"], 0), (["11:00-13:00"], 1)],
)
def test_zone_schedule(ctx, hours, expected):
    ctx.camera.zones[0].active_hours = hours
    assert len(_rule(ctx, zones=["workshop"]).evaluate(ctx)) == expected


def test_overlapping_zones_return_one_violation(ctx):
    ctx.camera.zones.append(
        ctx.camera.zones[0].model_copy(update={"zone_id": "overlap"})
    )
    rule = _rule(ctx, zones=["workshop", "overlap"])
    assert len(rule.evaluate(ctx)) == 1


@pytest.mark.parametrize("height, expected", [(79.0, 0), (80.0, 1), (81.0, 1)])
def test_box_height_threshold(ctx, height, expected):
    ctx.people[0].bbox = (60.0, 180.0 - height, 100.0, 180.0)
    assert len(_rule(ctx).evaluate(ctx)) == expected


def test_custom_box_height_threshold(ctx):
    assert _rule(ctx, params={"min_box_h_px": 101}).evaluate(ctx) == []


@pytest.mark.parametrize(
    "box",
    [
        (60.0, 80.0, 60.0, 180.0),
        (100.0, 80.0, 60.0, 180.0),
        (60.0, 180.0, 100.0, 80.0),
        (float("nan"), 80.0, 100.0, 180.0),
        (60.0, 80.0, 100.0, float("inf")),
    ],
)
def test_invalid_box(ctx, box):
    ctx.people[0].bbox = box
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize(
    "speed, expected",
    [
        (0.0, 0),
        (0.299, 0),
        (0.3, 1),
        (0.301, 1),
        (float("nan"), 0),
        (float("inf"), 0),
    ],
)
def test_only_when_moving(ctx, speed, expected):
    ctx.people[0].speed_bh = speed
    rule = _rule(ctx, params={"only_when_moving": True})
    assert len(rule.evaluate(ctx)) == expected


def test_custom_speed_threshold(ctx):
    ctx.people[0].speed_bh = 0.5
    rule = _rule(
        ctx,
        params={"only_when_moving": True, "min_speed_bh": 0.6},
    )
    assert rule.evaluate(ctx) == []
    ctx.people[0].speed_bh = 0.6
    assert len(rule.evaluate(ctx)) == 1


def test_speed_threshold_is_ignored_without_movement_filter(ctx):
    ctx.people[0].speed_bh = 0.0
    assert len(_rule(ctx, params={"min_speed_bh": 10}).evaluate(ctx)) == 1


def test_empty_people(ctx):
    ctx.people = []
    assert _rule(ctx).evaluate(ctx) == []


def test_each_frame_is_evaluated_without_confirmation_or_dedup(ctx):
    rule = _rule(ctx, min_duration_s=100)
    assert len(rule.evaluate(ctx)) == 1
    assert len(rule.evaluate(ctx)) == 1
    ctx.people[0].attributes["phone"] = False
    assert rule.evaluate(ctx) == []


@pytest.mark.parametrize("severity", list(Severity))
def test_explicit_severity(ctx, severity):
    violation, = _rule(ctx, severity=severity).evaluate(ctx)
    assert violation.severity == severity


def test_default_severity_does_not_mutate_config(ctx):
    cfg = RuleConfig(rule_id="phone-test", type="phone_usage")
    rule = PhoneUsageRule(cfg, ctx.camera)
    assert rule.evaluate(ctx)[0].severity == Severity.medium
    assert cfg.severity == Severity.high
    assert "severity" not in cfg.model_fields_set


def test_plugin_is_available():
    load_plugins()
    rules = {rule["type"]: rule for rule in available_rules()}
    assert {"phone_usage", "smoking_fire"} <= rules.keys()
    assert get_rule("phone_usage") is PhoneUsageRule
    assert rules["phone_usage"]["required_models"] == ["det", "pose"]
    assert rules["phone_usage"]["default_severity"] == "medium"
    assert rules["phone_usage"]["params_schema"] == PhoneUsageRule.params_schema
    properties = PhoneUsageRule.params_schema["properties"]
    assert properties["only_when_moving"]["default"] is False
    assert properties["min_speed_bh"]["default"] == 0.3
    assert properties["min_box_h_px"]["default"] == 80
    for schema in properties.values():
        assert any("а" <= char.lower() <= "я" for char in schema["title"])
