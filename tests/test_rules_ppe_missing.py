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
from guard.rules.base import available_rules
from guard.rules.plugins.ppe_missing import PpeMissingRule


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
                zone_id="ppe-zone",
                name="Зона СИЗ",
                kind="ppe_required",
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
        attributes={"helmet": False, "vest": True},
    )
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((300, 400, 3), dtype=np.uint8),
        people=[person],
    )


def _rule(ctx: FrameContext, **overrides) -> PpeMissingRule:
    cfg = RuleConfig(rule_id="ppe-test", type="ppe_missing", **overrides)
    return PpeMissingRule(cfg, ctx.camera)


def test_missing_helmet_in_zone(ctx):
    violations = _rule(ctx).evaluate(ctx)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.rule_id == "ppe-test"
    assert violation.type == "ppe_missing"
    assert violation.track_id == 1
    assert violation.zone_id == "ppe-zone"
    assert violation.bbox == ctx.people[0].bbox
    assert violation.severity == Severity.high
    assert violation.details == {"missing": ["helmet"]}


@pytest.mark.parametrize("value", [True, None, 0, "false", np.bool_(False)])
def test_only_builtin_false_means_missing(ctx, value):
    ctx.people[0].attributes["helmet"] = value
    assert _rule(ctx).evaluate(ctx) == []


def test_absent_attribute_is_unknown(ctx):
    del ctx.people[0].attributes["helmet"]
    assert _rule(ctx).evaluate(ctx) == []


def test_required_equipment_and_missing_details(ctx):
    ctx.people[0].attributes.update(vest=False, respirator=False)
    rule = _rule(ctx, params={"required": ["vest", "respirator"]})
    violations = rule.evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].details == {"missing": ["vest", "respirator"]}


def test_unknown_helmet_does_not_hide_missing_vest(ctx):
    ctx.people[0].attributes.update(helmet=None, vest=False)
    assert _rule(ctx).evaluate(ctx)[0].details == {"missing": ["vest"]}


def test_empty_required_list(ctx):
    assert _rule(ctx, params={"required": []}).evaluate(ctx) == []


def test_person_outside_zone(ctx):
    ctx.people[0].bbox = (240.0, 80.0, 280.0, 180.0)
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("height, expected", [(59.0, 0), (60.0, 1)])
def test_minimum_box_height_boundary(ctx, height, expected):
    ctx.people[0].bbox = (60.0, 180.0 - height, 100.0, 180.0)
    assert len(_rule(ctx).evaluate(ctx)) == expected


def test_custom_minimum_box_height(ctx):
    assert _rule(ctx, params={"min_box_h_px": 101}).evaluate(ctx) == []


def test_inactive_zone_does_not_fall_back_to_whole_frame(ctx):
    ctx.camera.zones[0].active_hours = ["08:00-10:00"]
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("zone_kind", [None, "danger"])
def test_no_ppe_zones_checks_whole_frame(ctx, zone_kind):
    if zone_kind is None:
        ctx.camera.zones = []
    else:
        ctx.camera.zones[0].kind = zone_kind
    ctx.people[0].bbox = (240.0, 80.0, 280.0, 180.0)
    violations = _rule(ctx).evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].zone_id is None


def test_explicit_zone_can_have_another_kind(ctx):
    ctx.camera.zones[0].kind = "danger"
    violations = _rule(ctx, zones=["ppe-zone"]).evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].zone_id == "ppe-zone"


def test_explicit_selection_excludes_other_ppe_zones(ctx):
    ctx.camera.zones.append(Zone(
        zone_id="other",
        name="Другая зона",
        kind="ppe_required",
        polygon=[(0.6, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9)],
    ))
    assert _rule(ctx, zones=["other"]).evaluate(ctx) == []


def test_unknown_explicit_zone_does_not_enable_whole_frame(ctx):
    assert _rule(ctx, zones=["unknown"]).evaluate(ctx) == []


def test_overlapping_zones_return_one_violation_per_person(ctx):
    duplicate = ctx.camera.zones[0].model_copy(update={"zone_id": "overlap"})
    ctx.camera.zones.append(duplicate)
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_each_frame_is_evaluated_without_confirmation_or_dedup(ctx):
    rule = _rule(ctx)
    assert len(rule.evaluate(ctx)) == 1
    assert len(rule.evaluate(ctx)) == 1
    ctx.people[0].attributes["helmet"] = True
    assert rule.evaluate(ctx) == []


def test_explicit_severity(ctx):
    assert _rule(ctx, severity=Severity.low).evaluate(ctx)[0].severity == Severity.low


def test_plugin_is_available():
    rules = {rule["type"]: rule for rule in available_rules()}
    assert rules["ppe_missing"]["required_models"] == ["det", "ppe"]
    assert rules["ppe_missing"]["default_severity"] == "high"
    assert rules["ppe_missing"]["params_schema"] == PpeMissingRule.params_schema
