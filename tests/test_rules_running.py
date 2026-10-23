from datetime import datetime, timedelta, timezone

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


@pytest.fixture
def camera():
    return CameraConfig(
        camera_id="camera-1",
        name="Тестовая камера",
        site="Тестовый участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
    )


@pytest.fixture
def zone():
    # Точка опоры (30, 80) внутри; центр рамки (30, 45) снаружи.
    return Zone(
        zone_id="zone-1",
        name="Зона запрета бега",
        polygon=[(0.0, 0.7), (0.5, 0.7), (0.5, 1.0), (0.0, 1.0)],
    )


def _track(
    *,
    speed=2.5,
    first_seen=0.0,
    bbox=(20.0, 10.0, 40.0, 80.0),
):
    return Track(
        track_id=1,
        cls="person",
        bbox=bbox,
        conf=0.95,
        first_seen=first_seen,
        last_seen=10.0,
        speed_bh=speed,
    )


def _context(camera, track, ts=10.0):
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((100, 100, 3), dtype=np.uint8),
        people=[] if track is None else [track],
    )


def _rule(camera, **overrides):
    load_plugins()
    cfg = RuleConfig(
        rule_id="running-1",
        type="running",
        **overrides,
    )
    return get_rule("running")(cfg, camera)


def test_running_speed_violates(camera):
    track = _track(speed=2.5)
    result = _rule(camera).evaluate(_context(camera, track))

    assert len(result) == 1
    violation = result[0]
    assert (
        violation.rule_id,
        violation.type,
        violation.track_id,
        violation.zone_id,
        violation.severity,
    ) == ("running-1", "running", 1, None, Severity.medium)
    assert violation.bbox == track.bbox
    assert violation.score == pytest.approx(0.95)
    assert violation.details == {"speed_bh": 2.5}


@pytest.mark.parametrize(
    ("speed", "expected"),
    [
        (1.0, False),
        (1.799, False),
        (1.8, True),
        (float("nan"), False),
        (float("inf"), False),
    ],
)
def test_speed_threshold(camera, speed, expected):
    result = _rule(camera).evaluate(
        _context(camera, _track(speed=speed))
    )
    assert bool(result) is expected


@pytest.mark.parametrize(
    ("first_seen", "expected"),
    [(9.5, False), (9.0, True), (10.5, False)],
)
def test_track_age(camera, first_seen, expected):
    result = _rule(camera).evaluate(
        _context(camera, _track(first_seen=first_seen))
    )
    assert bool(result) is expected


@pytest.mark.parametrize(
    ("box_height", "expected"),
    [(49.0, False), (50.0, True), (0.0, False)],
)
def test_box_height(camera, box_height, expected):
    track = _track(bbox=(20.0, 80.0 - box_height, 40.0, 80.0))
    result = _rule(camera).evaluate(_context(camera, track))
    assert bool(result) is expected


def test_configurable_thresholds(camera):
    track = _track(speed=1.0, bbox=(20.0, 40.0, 40.0, 80.0))
    ctx = _context(camera, track)
    assert _rule(camera).evaluate(ctx) == []
    rule = _rule(
        camera,
        params={"min_speed_bh": 1.0, "min_box_h_px": 40},
    )
    assert len(rule.evaluate(ctx)) == 1


def test_selected_zone_uses_foot_point(camera, zone):
    camera.zones = [zone]
    result = _rule(camera, zones=["zone-1"]).evaluate(
        _context(camera, _track())
    )
    assert len(result) == 1
    assert result[0].zone_id == "zone-1"


def test_outside_selected_zone(camera, zone):
    camera.zones = [zone]
    track = _track(bbox=(60.0, 10.0, 80.0, 80.0))
    assert _rule(camera, zones=["zone-1"]).evaluate(
        _context(camera, track)
    ) == []


@pytest.mark.parametrize("selected", [[], ["missing-zone"]])
def test_no_matching_zones_does_not_expand_to_entire_frame(camera, zone, selected):
    camera.zones = [zone]
    assert _rule(camera, zones=selected).evaluate(
        _context(camera, _track())
    ) == []


def test_no_zone_restriction_means_entire_frame(camera, zone):
    camera.zones = [zone]
    track = _track(bbox=(60.0, 10.0, 80.0, 80.0))
    result = _rule(camera).evaluate(_context(camera, track))
    assert len(result) == 1
    assert result[0].zone_id is None


@pytest.mark.parametrize(
    ("hours", "expected"),
    [(["11:00-13:00"], True), (["13:00-14:00"], False)],
)
def test_zone_active_hours(camera, zone, hours, expected):
    zone.active_hours = hours
    camera.zones = [zone]
    ts = datetime(
        2026, 1, 1, 12, 0, tzinfo=timezone(timedelta(hours=3))
    ).timestamp()
    track = _track(first_seen=ts - 2.0)
    track.last_seen = ts

    result = _rule(camera, zones=["zone-1"]).evaluate(
        _context(camera, track, ts=ts)
    )
    assert bool(result) is expected


def test_overlapping_zones_produce_one_violation(camera, zone):
    second = zone.model_copy(update={"zone_id": "zone-2"})
    camera.zones = [zone, second]
    result = _rule(camera, zones=["zone-1", "zone-2"]).evaluate(
        _context(camera, _track())
    )
    assert len(result) == 1
    assert result[0].zone_id == "zone-1"


@pytest.mark.parametrize("severity", [Severity.high, Severity.critical])
def test_explicit_severity_preserved(camera, severity):
    result = _rule(camera, severity=severity).evaluate(
        _context(camera, _track())
    )
    assert result[0].severity == severity


def test_each_frame_reports_violation_without_confirmation(camera):
    rule = _rule(camera)
    ctx = _context(camera, _track())
    first = rule.evaluate(ctx)
    assert len(first) == 1
    assert rule.evaluate(ctx) == first
    assert rule.cfg.severity == Severity.high


def test_empty_frame(camera):
    assert _rule(camera).evaluate(_context(camera, None)) == []


def test_running_plugin_discovered():
    load_plugins()
    metadata = {
        item["type"]: item for item in available_rules()
    }["running"]
    assert metadata["required_models"] == ["det"]
    assert metadata["default_severity"] == "medium"
    assert metadata["params_schema"]["properties"]["min_speed_bh"]["default"] == 1.8
