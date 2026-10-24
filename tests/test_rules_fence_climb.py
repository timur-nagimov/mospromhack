from collections import deque
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
        camera_id="camera-1",
        name="Тестовая камера",
        site="Тестовый участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
        zones=[
            Zone(
                zone_id="fence-1",
                name="Ограждение",
                kind="fence",
                polygon=[(0.4, 0.4), (0.6, 0.4), (0.6, 0.8), (0.4, 0.8)],
            )
        ],
    )


def _points(index, x=500.0, y=270.0, conf=0.9):
    points = np.zeros((17, 3), dtype=float)
    points[index] = (x, y, conf)
    return points


def _track(
    *,
    bbox=(450.0, 100.0, 550.0, 350.0),
    points=None,
    history=None,
):
    return Track(
        track_id=1,
        cls="person",
        bbox=bbox,
        conf=0.95,
        first_seen=0.0,
        last_seen=10.0,
        keypoints=points,
        attributes={"pose_state": "unknown"},
        history=deque([] if history is None else history, maxlen=64),
    )


def _history(times=(7.0, 8.0, 9.0, 10.0), x=500.0, y=350.0, h=250.0):
    return [(ts, x, y, h) for ts in times]


def _context(camera, track, ts=10.0):
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((500, 1000, 3), dtype=np.uint8),
        people=[] if track is None else [track],
    )


def _rule(camera, **overrides):
    load_plugins()
    cfg = RuleConfig(
        rule_id="fence-rule",
        type="fence_climb",
        **overrides,
    )
    return get_rule("fence_climb")(cfg, camera)


@pytest.mark.parametrize("index", [13, 14, 15, 16])
def test_climbing_with_each_knee_or_ankle(camera, index):
    track = _track(points=_points(index, conf=0.4))
    result = _rule(camera).evaluate(_context(camera, track))

    assert len(result) == 1
    violation = result[0]
    assert (
        violation.rule_id,
        violation.type,
        violation.track_id,
        violation.zone_id,
        violation.severity,
    ) == ("fence-rule", "fence_climb", 1, "fence-1", Severity.high)
    assert violation.bbox == track.bbox
    assert violation.score == pytest.approx(0.95)
    assert violation.details["condition"] == "climb"
    assert violation.details["keypoint_index"] == index
    assert violation.details["lift_bh"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("foot_y", "expected"),
    [(369.0, True), (370.0, False), (371.0, False)],
)
def test_lift_threshold_is_strict(camera, foot_y, expected):
    track = _track(
        bbox=(450.0, foot_y - 200.0, 550.0, foot_y),
        points=_points(13, y=300.0),
    )
    assert bool(_rule(camera).evaluate(_context(camera, track))) is expected


def test_climbing_does_not_require_foot_inside(camera):
    track = _track(
        bbox=(570.0, 100.0, 670.0, 350.0),
        points=_points(15, x=590.0),
    )
    result = _rule(camera).evaluate(_context(camera, track))
    assert result[0].details["condition"] == "climb"


def test_lift_parameter(camera):
    ctx = _context(camera, _track(points=_points(13)))
    assert _rule(camera, params={"lift_bh": 0.25}).evaluate(ctx) == []


def test_crawling_from_median_height_without_pose(camera):
    track = _track(
        bbox=(450.0, 300.0, 550.0, 390.0),
        history=[
            (9.0, 500.0, 390.0, 1000.0),
            (9.25, 500.0, 390.0, 200.0),
            (9.5, 500.0, 390.0, 200.0),
            (10.0, 500.0, 390.0, 90.0),
        ],
    )
    result = _rule(camera).evaluate(_context(camera, track))
    assert len(result) == 1
    assert result[0].details["condition"] == "crawl"
    assert result[0].details["evidence"] == "height_ratio"
    assert result[0].details["median_height_px"] == 200.0
    assert result[0].details["height_ratio"] == pytest.approx(0.45)


@pytest.mark.parametrize(
    ("body_height", "expected"),
    [(119.0, True), (120.0, False), (121.0, False)],
)
def test_crouch_threshold_is_strict(camera, body_height, expected):
    track = _track(
        bbox=(450.0, 390.0 - body_height, 550.0, 390.0),
        history=[
            (9.0, 500.0, 390.0, 200.0),
            (9.25, 500.0, 390.0, 200.0),
            (9.5, 500.0, 390.0, 200.0),
            (10.0, 500.0, 390.0, body_height),
        ],
    )
    assert bool(_rule(camera).evaluate(_context(camera, track))) is expected


def test_crouch_parameter(camera):
    track = _track(
        bbox=(450.0, 260.0, 550.0, 390.0),
        history=_history(times=(9.0, 9.5), y=390.0, h=200.0),
    )
    ctx = _context(camera, track)
    assert _rule(camera).evaluate(ctx) == []
    result = _rule(camera, params={"crouch_ratio": 0.7}).evaluate(ctx)
    assert result[0].details["evidence"] == "height_ratio"


@pytest.mark.parametrize(
    ("nose_y", "expected"),
    [(299.0, False), (300.0, False), (301.0, True)],
)
def test_crawling_from_head_position(camera, nose_y, expected):
    track = _track(points=_points(0, y=nose_y))
    result = _rule(camera).evaluate(_context(camera, track))
    assert bool(result) is expected
    if expected:
        assert result[0].details["condition"] == "crawl"
        assert result[0].details["evidence"] == "head_below_midpoint"
        assert result[0].details["polygon_mid_y"] == 300.0


def test_dwell_without_pose(camera):
    track = _track(history=_history())
    result = _rule(camera).evaluate(_context(camera, track))
    assert len(result) == 1
    assert result[0].details["condition"] == "dwell"
    assert result[0].details["dwell_s"] == 3.0


@pytest.mark.parametrize(
    ("start", "expected"),
    [(7.9, True), (8.0, False), (8.1, False)],
)
def test_dwell_threshold_is_strict(camera, start, expected):
    track = _track(history=_history(times=(start, 10.0)))
    assert bool(_rule(camera).evaluate(_context(camera, track))) is expected


def test_dwell_resets_after_exit_and_respects_parameter(camera):
    track = _track(
        history=[
            (6.0, 500.0, 350.0, 250.0),
            (7.0, 700.0, 350.0, 250.0),
            (9.0, 500.0, 350.0, 250.0),
            (10.0, 500.0, 350.0, 250.0),
        ],
    )
    ctx = _context(camera, track)
    assert _rule(camera).evaluate(ctx) == []
    result = _rule(camera, params={"max_dwell_s": 0.5}).evaluate(ctx)
    assert result[0].details["dwell_s"] == 1.0


@pytest.mark.parametrize("times", [(), (10.0,), (11.0,)])
def test_track_age_does_not_replace_history(camera, times):
    track = _track(history=_history(times=times))
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_dwell_since_zero(camera):
    track = _track(history=_history(times=(0.0, 3.0)))
    track.last_seen = 3.0
    result = _rule(camera).evaluate(_context(camera, track, ts=3.0))
    assert result[0].details["dwell_s"] == 3.0


def test_future_history_does_not_change_dwell(camera):
    track = _track(
        history=_history() + [(11.0, 700.0, 350.0, 1000.0)]
    )
    result = _rule(camera).evaluate(_context(camera, track))
    assert result[0].details["dwell_s"] == 3.0


def test_invalid_history_position_breaks_dwell(camera):
    track = _track(
        history=[
            (6.0, 500.0, 350.0, 250.0),
            (8.0, float("nan"), 350.0, 250.0),
            (9.0, 500.0, 350.0, 250.0),
            (10.0, 500.0, 350.0, 250.0),
        ]
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_dwell_not_used_when_pose_is_present(camera):
    track = _track(
        points=_points(5, y=150.0),
        history=_history(),
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


@pytest.mark.parametrize("with_pose", [False, True])
def test_nearby_person_with_overlapping_box_is_outside(camera, with_pose):
    points = None
    if with_pose:
        points = _points(13, x=620.0)
        points[0] = (620.0, 320.0, 0.9)
    track = _track(
        bbox=(590.0, 100.0, 690.0, 350.0),
        points=points,
        history=_history(x=640.0),
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


@pytest.mark.parametrize("index", [0, 13, 14, 15, 16])
def test_low_confidence_keypoints_do_not_trigger(camera, index):
    track = _track(
        points=_points(index, y=320.0, conf=0.399),
        history=_history(),
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_keypoint_confidence_parameter(camera):
    ctx = _context(camera, _track(points=_points(13, conf=0.7)))
    assert len(_rule(camera).evaluate(ctx)) == 1
    assert _rule(camera, params={"kp_conf": 0.8}).evaluate(ctx) == []


def test_invalid_keypoints_are_not_pose_evidence(camera):
    points = _points(13)
    points[13, 0] = float("nan")
    points[0] = (500.0, 320.0, float("nan"))
    ctx = _context(camera, _track(points=points))
    assert _rule(camera).evaluate(ctx) == []


def test_future_and_invalid_heights_do_not_define_median(camera):
    track = _track(
        history=[
            (9.0, 500.0, 350.0, 0.0),
            (9.25, 500.0, 350.0, float("nan")),
            (10.0, 500.0, 350.0, 250.0),
            (11.0, 500.0, 350.0, 1000.0),
            (12.0, 500.0, 350.0, 1000.0),
        ]
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_explicit_zone_selection_allows_other_kinds(camera):
    camera.zones[0].kind = "danger"
    ctx = _context(camera, _track(points=_points(13)))

    assert _rule(camera).evaluate(ctx) == []
    result = _rule(camera, zones=["fence-1"]).evaluate(ctx)
    assert result[0].zone_id == "fence-1"
    assert _rule(camera, zones=["missing"]).evaluate(ctx) == []


def test_inactive_zone(camera):
    camera.zones[0].active_hours = ["12:00-13:00"]
    # ts=10 соответствует 03:00:10 в часовом поясе geometry.LOCAL_TZ.
    ctx = _context(camera, _track(points=_points(13)))
    assert _rule(camera).evaluate(ctx) == []


def test_empty_polygon(camera):
    camera.zones[0].polygon = []
    ctx = _context(camera, _track(points=_points(13)))
    assert _rule(camera).evaluate(ctx) == []


@pytest.mark.parametrize(
    "bbox",
    [
        (450.0, 350.0, 550.0, 350.0),
        (550.0, 100.0, 450.0, 350.0),
        (float("nan"), 100.0, 550.0, 350.0),
    ],
)
def test_invalid_bbox(camera, bbox):
    track = _track(bbox=bbox, points=_points(13))
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_explicit_severity(camera):
    ctx = _context(camera, _track(points=_points(13)))
    result = _rule(camera, severity=Severity.low).evaluate(ctx)
    assert result[0].severity == Severity.low


def test_one_result_per_person_and_no_mutation_or_confirmation(camera):
    camera.zones.append(
        camera.zones[0].model_copy(update={"zone_id": "fence-2"})
    )
    points = _points(13)
    points[0] = (500.0, 320.0, 0.9)
    track = _track(points=points, history=_history())
    before_points = points.copy()
    before_history = list(track.history)
    before_attributes = deepcopy(track.attributes)
    rule = _rule(camera)
    ctx = _context(camera, track)

    first = rule.evaluate(ctx)
    assert len(first) == 1
    assert first[0].details["condition"] == "climb"
    assert rule.evaluate(ctx) == first
    np.testing.assert_array_equal(track.keypoints, before_points)
    assert list(track.history) == before_history
    assert track.attributes == before_attributes


def test_empty_frame(camera):
    assert _rule(camera).evaluate(_context(camera, None)) == []


def test_plugin_discovered():
    load_plugins()
    metadata = {
        item["type"]: item for item in available_rules()
    }["fence_climb"]
    assert metadata["required_models"] == ["det", "pose"]
    assert metadata["default_severity"] == "high"
    properties = metadata["params_schema"]["properties"]
    assert {
        name: spec["default"] for name, spec in properties.items()
    } == {
        "kp_conf": 0.4,
        "lift_bh": 0.15,
        "crouch_ratio": 0.6,
        "max_dwell_s": 2.0,
    }
    assert all(spec["title"] for spec in properties.values())
