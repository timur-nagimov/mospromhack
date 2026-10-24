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


def _track(
    *,
    state="lying",
    since=7.0,
    history=None,
    bbox=(100.0, 100.0, 230.0, 200.0),
):
    if history is None:
        history = [
            (ts, 165.0, 200.0, 100.0) for ts in (7.0, 8.0, 9.0, 10.0)
        ]
    return Track(
        track_id=1,
        cls="person",
        bbox=bbox,
        conf=0.95,
        first_seen=0.0,
        last_seen=10.0,
        attributes={"pose_state": state, "lying_since": since},
        history=deque(history, maxlen=64),
    )


def _context(camera, track, ts=10.0):
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((240, 320, 3), dtype=np.uint8),
        people=[] if track is None else [track],
    )


def _rule(camera, **overrides):
    load_plugins()
    cfg = RuleConfig(
        rule_id="fall-1",
        type="fall_immobile",
        **overrides,
    )
    return get_rule("fall_immobile")(cfg, camera)


def test_lying_three_seconds_immobile(camera):
    track = _track()
    violations = _rule(camera).evaluate(_context(camera, track))

    assert len(violations) == 1
    violation = violations[0]
    assert (
        violation.rule_id,
        violation.type,
        violation.track_id,
        violation.zone_id,
        violation.severity,
    ) == ("fall-1", "fall_immobile", 1, None, Severity.critical)
    assert violation.bbox == track.bbox
    assert violation.score == pytest.approx(0.95)
    assert violation.details == {"lying_s": 3.0, "shift_bh": 0.0}


@pytest.mark.parametrize(
    ("since", "expected"),
    [(9.0, False), (8.0, True), (10.1, False), (None, False)],
)
def test_lying_duration(camera, since, expected):
    result = _rule(camera).evaluate(
        _context(camera, _track(since=since))
    )
    assert bool(result) is expected


def test_lying_since_zero_is_valid(camera):
    track = _track(
        since=0.0,
        history=[
            (0.0, 165.0, 200.0, 100.0),
            (3.0, 165.0, 200.0, 100.0),
        ],
    )
    track.last_seen = 3.0
    result = _rule(camera).evaluate(_context(camera, track, ts=3.0))
    assert result[0].details["lying_s"] == 3.0


@pytest.mark.parametrize(
    "history",
    [
        [
            (7.0, 105.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
        [
            (7.0, 165.0, 200.0, 100.0),
            (8.0, 205.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
        [
            (7.0, 165.0, 150.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
        [
            (7.0, 135.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
    ],
)
def test_movement_and_exact_shift_threshold_rejected(camera, history):
    assert _rule(camera).evaluate(
        _context(camera, _track(history=history))
    ) == []


def test_standing_is_not_overridden_by_wide_bbox(camera):
    assert _rule(camera).evaluate(
        _context(camera, _track(state="standing", since=None))
    ) == []


@pytest.mark.parametrize("enabled", [True, False])
def test_bbox_fallback_without_pose_timestamp(camera, enabled):
    track = _track(state="unknown", since=None)
    rule = _rule(camera, params={"allow_bbox_fallback": enabled})
    result = rule.evaluate(_context(camera, track))

    assert bool(result) is enabled
    if enabled:
        assert result[0].details == {"lying_s": None, "shift_bh": 0.0}


def test_bbox_fallback_enabled_by_default(camera):
    result = _rule(camera).evaluate(
        _context(camera, _track(state="unknown", since=None))
    )
    assert len(result) == 1


def test_bbox_fallback_requires_aspect_ratio(camera):
    track = _track(
        state="unknown",
        since=None,
        bbox=(100.0, 100.0, 229.0, 200.0),
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_bbox_fallback_still_requires_immobility(camera):
    track = _track(
        state="unknown",
        since=None,
        history=[
            (7.0, 100.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


@pytest.mark.parametrize(
    "history",
    [
        [],
        [(10.0, 165.0, 200.0, 100.0)],
        [
            (10.0, 165.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
        [
            (7.0, 165.0, 200.0, 0.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
        [
            (7.0, float("nan"), 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
    ],
)
def test_insufficient_or_invalid_history_is_not_immobility(camera, history):
    assert _rule(camera).evaluate(
        _context(camera, _track(history=history))
    ) == []


def test_history_outside_window_is_ignored(camera):
    track = _track(
        history=[
            (6.9, 500.0, 200.0, 100.0),
            (7.0, 165.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
            (10.1, 600.0, 200.0, 100.0),
        ]
    )
    result = _rule(camera).evaluate(_context(camera, track))
    assert result[0].details["shift_bh"] == 0.0


def test_window_parameter(camera):
    track = _track(
        history=[
            (7.0, 100.0, 200.0, 100.0),
            (9.0, 165.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ]
    )
    ctx = _context(camera, track)
    assert _rule(camera).evaluate(ctx) == []
    assert len(
        _rule(camera, params={"immobile_window_s": 1.0}).evaluate(ctx)
    ) == 1


def test_shift_normalized_by_history_height(camera):
    track = _track(
        bbox=(100.0, 0.0, 230.0, 200.0),
        history=[
            (7.0, 125.0, 200.0, 200.0),
            (10.0, 165.0, 200.0, 200.0),
        ],
    )
    ctx = _context(camera, track)
    result = _rule(camera).evaluate(ctx)
    assert result[0].details["shift_bh"] == pytest.approx(0.2)
    assert _rule(camera, params={"max_shift_bh": 0.1}).evaluate(ctx) == []


def test_min_lying_parameter(camera):
    ctx = _context(camera, _track(since=9.0))
    assert len(_rule(camera, params={"min_lying_s": 1.0}).evaluate(ctx)) == 1


@pytest.mark.parametrize("severity", [Severity.high, Severity.low])
def test_explicit_severity_preserved(camera, severity):
    rule = _rule(camera, severity=severity)
    assert rule.evaluate(_context(camera, _track()))[0].severity == severity


def test_evaluation_does_not_confirm_deduplicate_or_mutate(camera):
    track = _track()
    attributes = deepcopy(track.attributes)
    history = list(track.history)
    rule = _rule(camera)
    ctx = _context(camera, track)

    first = rule.evaluate(ctx)
    assert len(first) == 1
    assert rule.evaluate(ctx) == first
    assert track.attributes == attributes
    assert list(track.history) == history
    assert rule.cfg.severity == Severity.high


def test_empty_frame(camera):
    assert _rule(camera).evaluate(_context(camera, None)) == []


def test_fall_plugin_discovered():
    load_plugins()
    metadata = {
        item["type"]: item for item in available_rules()
    }["fall_immobile"]
    assert metadata["required_models"] == ["det", "pose"]
    assert metadata["default_severity"] == "critical"
    assert metadata["params_schema"]["properties"]["min_lying_s"]["default"] == 2.0
