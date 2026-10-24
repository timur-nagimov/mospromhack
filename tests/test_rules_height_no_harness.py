from copy import deepcopy

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
        camera_id="height-camera",
        name="Камера",
        site="Участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
        zones=[
            Zone(
                zone_id="height-1",
                name="Площадка",
                kind="height",
                polygon=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.8), (0.4, 0.8)],
            ),
        ],
    )


def _track(attributes=None, bbox=(450.0, 100.0, 550.0, 350.0)):
    return Track(
        track_id=1,
        cls="person",
        bbox=bbox,
        conf=0.95,
        first_seen=0.0,
        last_seen=10.0,
        attributes={} if attributes is None else attributes,
    )


def _context(camera, track, ts=10.0):
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((500, 1000, 3), dtype=np.uint8),
        people=[] if track is None else [track],
    )


def _rule(camera, **kwargs):
    load_plugins()
    cfg = RuleConfig(rule_id="height-rule", type="height_no_harness", **kwargs)
    return get_rule("height_no_harness")(cfg, camera)


@pytest.mark.parametrize("mode", ["off", "strict"])
def test_explicit_absence(camera, mode):
    track = _track({"harness": False})
    result = _rule(camera, params={"mode": mode}).evaluate(_context(camera, track))
    assert len(result) == 1
    violation = result[0]
    assert (
        violation.rule_id,
        violation.type,
        violation.track_id,
        violation.zone_id,
        violation.severity,
    ) == ("height-rule", "height_no_harness", 1, "height-1", Severity.high)
    assert violation.bbox == track.bbox
    assert violation.score == pytest.approx(0.95)
    assert violation.details["missing"] == ["harness"]
    assert violation.details["evidence"] == "attribute"


@pytest.mark.parametrize("attributes", [{}, {"harness": None}, {"harness": 0}])
def test_default_off_requires_explicit_false(camera, attributes):
    assert _rule(camera).evaluate(_context(camera, _track(attributes))) == []


@pytest.mark.parametrize("attributes", [{}, {"harness": None}, {"harness": 0}])
def test_strict_is_access_control_for_unknown_harness(camera, attributes):
    result = _rule(camera, params={"mode": "strict"}).evaluate(
        _context(camera, _track(attributes))
    )
    assert len(result) == 1
    assert result[0].severity == Severity.high
    assert result[0].details["condition"] == "harness_unconfirmed"
    assert result[0].details["evidence"] == "access_control"
    assert "missing" not in result[0].details


@pytest.mark.parametrize("mode", ["off", "strict"])
def test_confirmed_harness(camera, mode):
    assert _rule(camera, params={"mode": mode}).evaluate(
        _context(camera, _track({"harness": True}))
    ) == []


def test_uses_foot_not_bbox_overlap(camera):
    track = _track({"harness": False}, bbox=(590.0, 100.0, 690.0, 350.0))
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_polygon_boundary_counts_as_inside(camera):
    track = _track({"harness": False}, bbox=(350.0, 100.0, 450.0, 400.0))
    assert len(_rule(camera).evaluate(_context(camera, track))) == 1


def test_default_requires_height_zone(camera):
    camera.zones[0].kind = "danger"
    ctx = _context(camera, _track({"harness": False}))
    assert _rule(camera).evaluate(ctx) == []
    assert len(_rule(camera, zones=["height-1"]).evaluate(ctx)) == 1


@pytest.mark.parametrize("selection", [[], ["unknown"]])
def test_explicit_selection_does_not_expand_scope(camera, selection):
    ctx = _context(camera, _track({"harness": False}))
    assert _rule(camera, zones=selection).evaluate(ctx) == []


def test_no_zones_does_not_mean_entire_frame(camera):
    camera.zones = []
    assert _rule(camera, params={"mode": "strict"}).evaluate(
        _context(camera, _track())
    ) == []


def test_inactive_zone(camera):
    camera.zones[0].active_hours = ["12:00-13:00"]
    assert _rule(camera).evaluate(
        _context(camera, _track({"harness": False}))
    ) == []


@pytest.mark.parametrize(
    "polygon",
    [[], [(0.4, 0.4), (0.6, 0.4)], [(float("nan"), 0.4), (0.6, 0.4), (0.6, 0.8)]],
)
def test_invalid_polygon(camera, polygon):
    camera.zones[0].polygon = polygon
    assert _rule(camera).evaluate(
        _context(camera, _track({"harness": False}))
    ) == []


@pytest.mark.parametrize(
    "bbox",
    [
        (450.0, 350.0, 550.0, 350.0),
        (550.0, 100.0, 450.0, 350.0),
        (float("nan"), 100.0, 550.0, 350.0),
    ],
)
def test_invalid_bbox(camera, bbox):
    assert _rule(camera).evaluate(
        _context(camera, _track({"harness": False}, bbox=bbox))
    ) == []


def test_one_result_per_person_and_no_mutation(camera):
    camera.zones.append(camera.zones[0].model_copy(update={"zone_id": "height-2"}))
    track = _track({"harness": False, "other": {"value": 1}})
    before = deepcopy(track.attributes)
    history = list(track.history)
    rule = _rule(camera)
    ctx = _context(camera, track)
    result = rule.evaluate(ctx)
    assert len(result) == 1
    assert rule.evaluate(ctx) == result
    assert track.attributes == before
    assert list(track.history) == history


def test_empty_frame_and_invalid_time(camera):
    rule = _rule(camera)
    assert rule.evaluate(_context(camera, None)) == []
    assert rule.evaluate(
        _context(camera, _track({"harness": False}), ts=float("nan"))
    ) == []


def test_invalid_mode(camera):
    with pytest.raises(ValueError, match="mode"):
        _rule(camera, params={"mode": "guess"})


def test_plugin_discovered():
    load_plugins()
    metadata = {item["type"]: item for item in available_rules()}["height_no_harness"]
    assert metadata["required_models"] == ["det"]
    assert metadata["default_severity"] == "high"
    schema = metadata["params_schema"]
    assert schema["properties"]["mode"]["default"] == "off"
    assert "контроль доступа" in schema["description"]
