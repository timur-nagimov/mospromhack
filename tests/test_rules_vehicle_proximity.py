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
from guard.rules.base import available_rules, load_plugins
from guard.rules.plugins.vehicle_proximity import VehicleProximityRule


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
                zone_id="path",
                name="Проезд техники",
                kind="vehicle_path",
                polygon=[(0.1, 0.1), (0.35, 0.1), (0.35, 0.9), (0.1, 0.9)],
            )
        ],
    )
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((300, 400, 3), dtype=np.uint8),
        people=[
            Track(
                track_id=1,
                cls="person",
                bbox=(80.0, 80.0, 120.0, 180.0),
                conf=0.95,
                first_seen=ts,
                last_seen=ts,
            )
        ],
        objects=[
            Track(
                track_id=10,
                cls="forklift",
                bbox=(120.0, 120.0, 220.0, 200.0),
                conf=0.9,
                first_seen=ts - 1,
                last_seen=ts,
                velocity=(24.0, 0.0),
                speed_bh=0.3,
            )
        ],
    )


def _rule(ctx: FrameContext, **overrides) -> VehicleProximityRule:
    cfg = RuleConfig(rule_id="vehicle-test", type="vehicle_proximity", **overrides)
    return VehicleProximityRule(cfg, ctx.camera)


def test_stationary_vehicle_nearby(ctx):
    ctx.objects[0].speed_bh = 0.0
    ctx.objects[0].velocity = (0.0, 0.0)
    assert _rule(ctx).evaluate(ctx) == []


def test_moving_vehicle_nearby(ctx):
    violations = _rule(ctx).evaluate(ctx)
    assert len(violations) == 1
    violation = violations[0]
    assert violation.rule_id == "vehicle-test"
    assert violation.type == "vehicle_proximity"
    assert violation.track_id == 1
    assert violation.zone_id is None
    assert violation.bbox == ctx.people[0].bbox
    assert violation.severity == Severity.critical
    assert violation.details["vehicle_track_id"] == 10
    assert violation.details["distance_px"] == pytest.approx(20.0)


def test_moving_vehicle_far_away(ctx):
    ctx.objects[0].bbox = (240.0, 120.0, 340.0, 200.0)
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize("speed, expected", [(0.149, 0), (0.15, 1), (0.151, 1)])
def test_speed_threshold_is_inclusive(ctx, speed, expected):
    ctx.objects[0].speed_bh = speed
    assert len(_rule(ctx).evaluate(ctx)) == expected


def test_custom_speed_threshold(ctx):
    assert _rule(ctx, params={"min_vehicle_speed_bh": 0.4}).evaluate(ctx) == []


def test_zero_threshold_still_requires_movement(ctx):
    ctx.objects[0].speed_bh = 0.0
    assert _rule(ctx, params={"min_vehicle_speed_bh": 0.0}).evaluate(ctx) == []


@pytest.mark.parametrize("speed", [float("nan"), float("inf"), -0.3])
def test_invalid_speed_is_not_movement_evidence(ctx, speed):
    ctx.objects[0].speed_bh = speed
    assert _rule(ctx).evaluate(ctx) == []


@pytest.mark.parametrize(
    "box, expected",
    [
        ((149.0, 120.0, 249.0, 200.0), 1),
        ((150.0, 120.0, 250.0, 200.0), 0),
        ((151.0, 120.0, 251.0, 200.0), 0),
    ],
)
def test_distance_threshold_is_strict(ctx, box, expected):
    ctx.objects[0].bbox = box
    assert len(_rule(ctx).evaluate(ctx)) == expected


def test_distance_to_corner_is_euclidean(ctx):
    # Точка опоры (100, 180); ближайший угол (130, 220): расстояние 50.
    ctx.objects[0].bbox = (130.0, 220.0, 230.0, 280.0)
    assert _rule(ctx).evaluate(ctx) == []
    violations = _rule(ctx, params={"distance_factor": 0.51}).evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].details["distance_px"] == pytest.approx(50.0)


def test_foot_point_inside_vehicle_box_has_zero_distance(ctx):
    ctx.objects[0].bbox = (80.0, 120.0, 180.0, 200.0)
    assert _rule(ctx).evaluate(ctx)[0].details["distance_px"] == 0.0


def test_distance_scales_with_person_height(ctx):
    # Высота человека 40: расстояние 20 равно порогу, нарушения нет.
    ctx.people[0].bbox = (80.0, 140.0, 120.0, 180.0)
    assert _rule(ctx).evaluate(ctx) == []


def test_custom_distance_factor(ctx):
    assert _rule(ctx, params={"distance_factor": 0.2}).evaluate(ctx) == []
    assert len(_rule(ctx, params={"distance_factor": 0.21}).evaluate(ctx)) == 1


@pytest.mark.parametrize("cls", ["car", "truck", "bus", "forklift", "motorcycle"])
def test_default_vehicle_classes(ctx, cls):
    ctx.objects[0].cls = cls
    assert len(_rule(ctx).evaluate(ctx)) == 1


def test_custom_vehicle_classes(ctx):
    ctx.objects[0].cls = "crane"
    assert _rule(ctx).evaluate(ctx) == []
    assert len(_rule(ctx, params={"vehicle_classes": ["crane"]}).evaluate(ctx)) == 1


def test_empty_vehicle_classes(ctx):
    assert _rule(ctx, params={"vehicle_classes": []}).evaluate(ctx) == []


def test_explicit_zone_checks_person_foot_point(ctx):
    violations = _rule(ctx, zones=["path"]).evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].zone_id == "path"


def test_zones_do_not_restrict_rule_unless_selected(ctx):
    ctx.camera.zones[0].polygon = [
        (0.6, 0.1), (0.9, 0.1), (0.9, 0.9), (0.6, 0.9)
    ]
    assert len(_rule(ctx).evaluate(ctx)) == 1
    assert _rule(ctx, zones=["path"]).evaluate(ctx) == []


def test_inactive_selected_zone(ctx):
    ctx.camera.zones[0].active_hours = ["08:00-10:00"]
    assert _rule(ctx, zones=["path"]).evaluate(ctx) == []


def test_unknown_selected_zone(ctx):
    assert _rule(ctx, zones=["unknown"]).evaluate(ctx) == []


def test_selected_zone_must_be_vehicle_path(ctx):
    ctx.camera.zones[0].kind = "danger"
    assert _rule(ctx, zones=["path"]).evaluate(ctx) == []


def test_nearest_vehicle_returns_one_violation_per_person(ctx):
    ctx.objects.append(Track(
        track_id=11,
        cls="truck",
        bbox=(110.0, 120.0, 210.0, 200.0),
        conf=0.9,
        first_seen=ctx.ts - 1,
        last_seen=ctx.ts,
        speed_bh=0.2,
    ))
    rule = _rule(ctx)
    violations = rule.evaluate(ctx)
    assert len(violations) == 1
    assert violations[0].details["vehicle_track_id"] == 11
    assert violations[0].details["distance_px"] == pytest.approx(10.0)
    ctx.objects.reverse()
    assert rule.evaluate(ctx)[0].details == violations[0].details


def test_evaluation_does_not_deduplicate_or_remember_movement(ctx):
    rule = _rule(ctx)
    assert len(rule.evaluate(ctx)) == 1
    assert len(rule.evaluate(ctx)) == 1
    ctx.objects[0].speed_bh = 0.0
    assert rule.evaluate(ctx) == []


@pytest.mark.parametrize("severity", list(Severity))
def test_explicit_severity_is_preserved(ctx, severity):
    cfg = RuleConfig(
        rule_id="custom",
        type="vehicle_proximity",
        severity=severity,
    )
    rule = VehicleProximityRule(cfg, ctx.camera)
    assert rule.evaluate(ctx)[0].severity == severity
    assert cfg.severity == severity


def test_both_plugins_are_discoverable():
    load_plugins()
    rules = {rule["type"]: rule for rule in available_rules()}
    assert {"ppe_missing", "vehicle_proximity"} <= rules.keys()
    assert rules["vehicle_proximity"]["required_models"] == ["det"]
    assert rules["vehicle_proximity"]["default_severity"] == "critical"
    assert (
        rules["vehicle_proximity"]["params_schema"]
        == VehicleProximityRule.params_schema
    )
