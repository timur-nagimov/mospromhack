from collections import deque
from copy import deepcopy

import numpy as np
import pytest

from guard.contracts import (
    CameraConfig,
    Detection,
    FrameContext,
    RuleConfig,
    Severity,
    SourceConfig,
    Track,
)
from guard.events.manager import EventManager
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.tracking import Tracker, assign_attributes


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


def _observation(
    ts,
    *,
    state="lying",
    since=1.0,
    bbox=(100.0, 100.0, 230.0, 200.0),
    history=None,
    track_id=1,
    first_seen=0.0,
):
    if history is None:
        x1, y1, x2, y2 = bbox
        history = [(ts, (x1 + x2) / 2, y2, y2 - y1)]
    track = _track(state=state, since=since, bbox=bbox, history=history)
    track.last_seen = ts
    track.track_id = track_id
    track.first_seen = first_seen
    return track


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
    [
        (9.0, False),
        (8.5, True),
        (8.5001, False),
        (8.0, True),
        (10.1, False),
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        ("invalid", False),
    ],
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
    properties = metadata["params_schema"]["properties"]
    assert properties["min_lying_s"]["default"] == 1.5
    assert properties["immobile_window_s"]["default"] == 3.0
    assert properties["count_lost_after_fall"]["default"] is True
    assert properties["track_buffer_s"]["default"] == 3.0


def test_three_second_window_uses_real_tracker_history(camera):
    tracker = Tracker({"person"}, min_hits=1)
    for frame in range(31):
        ts = frame / 5.0
        x = 0.0 if ts < 3.4 else 50.0
        track = tracker.update(
            [
                Detection(
                    model="det",
                    cls="person",
                    cls_id=0,
                    conf=0.95,
                    bbox=(x, 100.0, x + 130.0, 200.0),
                )
            ],
            ts,
        )[0]
    track.attributes.update(pose_state="lying", lying_since=0.0)

    ctx = _context(camera, track, ts=6.0)
    # Motion 2.8 seconds ago must still be considered by the default window.
    assert _rule(camera).evaluate(ctx) == []
    assert len(
        _rule(camera, params={"immobile_window_s": 2.0}).evaluate(ctx)
    ) == 1


def test_five_second_window_includes_older_motion(camera):
    track = _track(
        since=0.0,
        history=[
            (5.0, 100.0, 200.0, 100.0),
            (6.0, 100.0, 200.0, 100.0),
            (7.0, 165.0, 200.0, 100.0),
            (8.0, 165.0, 200.0, 100.0),
            (10.0, 165.0, 200.0, 100.0),
        ],
    )
    ctx = _context(camera, track)
    assert len(_rule(camera).evaluate(ctx)) == 1
    assert _rule(camera, params={"immobile_window_s": 5.0}).evaluate(ctx) == []


@pytest.mark.parametrize("enabled", [True, False])
def test_lost_fall_with_tracker_attributes_and_event_manager(camera, enabled):
    rule = _rule(camera, params={"count_lost_after_fall": enabled})
    camera.rules = [rule.cfg]
    tracker = Tracker({"person"})
    manager = EventManager()
    created = []
    emitted = []
    original_id = None
    last_box = None

    for frame in range(25):
        ts = frame / 5.0
        detections = {"det": [], "pose": []}
        if frame <= 8:
            box = (
                (160.0, 80.0, 240.0, 240.0)
                if ts < 1.0
                else (110.0, 150.0, 290.0, 240.0)
            )
            for model in detections:
                detections[model].append(
                    Detection(
                        model=model,
                        cls="person",
                        cls_id=0,
                        conf=0.95,
                        bbox=box,
                    )
                )

        people = tracker.update(detections["det"], ts)
        assign_attributes(people, detections, ts)
        if people:
            if original_id is None:
                original_id = people[0].track_id
            assert people[0].track_id == original_id
            last_box = people[0].bbox

        ctx = _context(camera, None, ts=ts)
        ctx.people = people
        ctx.detections = detections
        violations = rule.evaluate(ctx)
        events, _ = manager.process(camera, ts, violations)
        created.extend(events)
        emitted.extend(violations)

        if frame <= 8:
            # The detector disappears before min_lying_s has elapsed.
            assert violations == []
        if frame == 24:
            assert violations == []

    if enabled:
        assert emitted
        assert all(v.track_id == original_id for v in emitted)
        assert all(v.bbox == last_box for v in emitted)
        assert all(v.details["lost_after_fall"] is True for v in emitted)
        assert all(v.details["lost_since"] == pytest.approx(1.8) for v in emitted)
        assert len(created) == 1
        assert created[0].track_id == original_id
        assert created[0].severity == Severity.critical
    else:
        assert emitted == []
        assert created == []


def test_loss_memory_preserves_snapshot_and_expires_from_last_seen(camera):
    rule = _rule(camera)
    track = _observation(1.4)
    original_box = track.bbox
    original_history = list(track.history)
    rule.evaluate(_context(camera, track, ts=1.4))
    assert rule.evaluate(_context(camera, None, ts=1.6)) == []

    # Later mutation of the original Track must not mutate the saved evidence.
    track.bbox = (0.0, 0.0, 10.0, 20.0)
    track.attributes.update(pose_state="standing", lying_since=None)
    track.history.clear()

    result = rule.evaluate(_context(camera, None, ts=2.5))
    assert len(result) == 1
    assert result[0].track_id == 1
    assert result[0].bbox == original_box
    assert result[0].details["lying_s"] == pytest.approx(1.5)
    assert result[0].details["lost_since"] == 1.6
    assert rule.evaluate(_context(camera, None, ts=2.5)) == result
    assert original_history

    assert len(rule.evaluate(_context(camera, None, ts=4.4))) == 1
    assert rule.evaluate(_context(camera, None, ts=4.4001)) == []
    assert rule.evaluate(_context(camera, None, ts=20.0)) == []


def test_loss_buffer_parameter(camera):
    rule = _rule(camera, params={"track_buffer_s": 1.5})
    rule.evaluate(_context(camera, _observation(1.4), ts=1.4))
    rule.evaluate(_context(camera, None, ts=1.6))
    assert len(rule.evaluate(_context(camera, None, ts=2.5))) == 1
    assert len(rule.evaluate(_context(camera, None, ts=2.9))) == 1
    assert rule.evaluate(_context(camera, None, ts=2.9001)) == []


@pytest.mark.parametrize(
    ("loss_ts", "expected"),
    [(1.99, True), (2.0, False), (2.2, False)],
)
def test_first_loss_must_be_less_than_one_second_after_fall(
    camera, loss_ts, expected
):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.8), ts=1.8))
    rule.evaluate(_context(camera, None, ts=loss_ts))
    assert bool(rule.evaluate(_context(camera, None, ts=2.6))) is expected


@pytest.mark.parametrize("enabled", [True, False])
def test_abrupt_aspect_change_without_pose_can_establish_fall(camera, enabled):
    rule = _rule(camera, params={"allow_bbox_fallback": enabled})
    observations = [
        _observation(
            0.6,
            state="unknown",
            since=None,
            bbox=(125.0, 40.0, 205.0, 200.0),
        ),
        _observation(
            0.8,
            state="unknown",
            since=None,
            bbox=(105.0, 70.0, 225.0, 200.0),
        ),
        _observation(
            1.0,
            state="unknown",
            since=None,
            bbox=(115.0, 100.0, 215.0, 200.0),
        ),
    ]
    for track in observations:
        assert rule.evaluate(_context(camera, track, ts=track.last_seen)) == []
    assert rule.evaluate(_context(camera, None, ts=1.2)) == []
    result = rule.evaluate(_context(camera, None, ts=2.5))
    assert bool(result) is enabled
    if enabled:
        assert result[0].bbox == observations[-1].bbox
        assert result[0].track_id == 1
        assert result[0].details["lying_s"] == pytest.approx(1.5)


def test_slow_aspect_change_is_not_a_lost_fall(camera):
    rule = _rule(camera)
    upright = _observation(
        0.0,
        state="unknown",
        since=None,
        bbox=(125.0, 40.0, 205.0, 200.0),
    )
    horizontal = _observation(
        1.2,
        state="unknown",
        since=None,
        bbox=(115.0, 100.0, 215.0, 200.0),
    )
    rule.evaluate(_context(camera, upright, ts=0.0))
    rule.evaluate(_context(camera, horizontal, ts=1.2))
    assert rule.evaluate(_context(camera, None, ts=1.4)) == []
    assert rule.evaluate(_context(camera, None, ts=3.0)) == []


def test_horizontal_observations_do_not_restart_fall_clock(camera):
    rule = _rule(camera)
    upright = _observation(
        0.6,
        state="unknown",
        since=None,
        bbox=(125.0, 40.0, 205.0, 200.0),
    )
    rule.evaluate(_context(camera, upright, ts=0.6))
    for ts in (1.0, 1.4, 1.8, 2.2):
        horizontal = _observation(
            ts,
            state="unknown",
            since=None,
            bbox=(115.0, 100.0, 215.0, 200.0),
        )
        rule.evaluate(_context(camera, horizontal, ts=ts))
    assert rule.evaluate(_context(camera, None, ts=2.4)) == []
    assert rule.evaluate(_context(camera, None, ts=3.0)) == []


def test_wide_unknown_track_without_transition_is_not_retained(camera):
    rule = _rule(camera)
    track = _observation(1.0, state="unknown", since=None)
    rule.evaluate(_context(camera, track, ts=1.0))
    assert rule.evaluate(_context(camera, None, ts=1.2)) == []
    assert rule.evaluate(_context(camera, None, ts=2.6)) == []


def test_crouching_then_standing_does_not_produce_violation(camera):
    rule = _rule(camera)
    observations = [
        _observation(
            0.0,
            state="standing",
            since=None,
            bbox=(125.0, 40.0, 205.0, 200.0),
        ),
        _observation(
            0.4,
            state="standing",
            since=None,
            bbox=(105.0, 90.0, 225.0, 200.0),
        ),
        _observation(
            0.8,
            state="standing",
            since=None,
            bbox=(125.0, 40.0, 205.0, 200.0),
        ),
    ]
    for track in observations:
        assert rule.evaluate(_context(camera, track, ts=track.last_seen)) == []
    for ts in (1.0, 1.6, 2.6, 4.0):
        assert rule.evaluate(_context(camera, None, ts=ts)) == []


def test_recovery_standing_cancels_pending_lost_fall(camera):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.0), ts=1.0))
    rule.evaluate(_context(camera, None, ts=1.2))
    standing = _observation(
        1.4,
        state="standing",
        since=None,
        bbox=(125.0, 40.0, 205.0, 200.0),
    )
    assert rule.evaluate(_context(camera, standing, ts=1.4)) == []
    for ts in (1.6, 2.6, 3.0):
        assert rule.evaluate(_context(camera, None, ts=ts)) == []


def test_recovered_lying_track_is_emitted_once(camera):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.4), ts=1.4))
    rule.evaluate(_context(camera, None, ts=1.6))
    recovered = _observation(
        2.6,
        history=[
            (1.0, 165.0, 200.0, 100.0),
            (1.4, 165.0, 200.0, 100.0),
            (2.6, 165.0, 200.0, 100.0),
        ],
    )
    result = rule.evaluate(_context(camera, recovered, ts=2.6))
    assert len(result) == 1
    assert result[0].track_id == 1
    assert "lost_after_fall" not in result[0].details
    # This new loss is too late after the original fall.
    assert rule.evaluate(_context(camera, None, ts=2.8)) == []


def test_crawling_while_lying_is_not_immobility(camera):
    history = [
        (7.0, 165.0, 200.0, 100.0),
        (8.0, 180.0, 200.0, 100.0),
        (9.0, 195.0, 200.0, 100.0),
        (10.0, 210.0, 200.0, 100.0),
    ]
    track = _track(
        history=history,
        bbox=(145.0, 100.0, 275.0, 200.0),
    )
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_crawling_evidence_does_not_age_out_during_loss(camera):
    rule = _rule(camera)
    track = _observation(
        1.4,
        bbox=(140.0, 100.0, 270.0, 200.0),
        history=[
            (1.0, 165.0, 200.0, 100.0),
            (1.2, 185.0, 200.0, 100.0),
            (1.4, 205.0, 200.0, 100.0),
        ],
    )
    assert rule.evaluate(_context(camera, track, ts=1.4)) == []
    for ts in (1.6, 2.6, 3.4, 4.2, 4.4, 4.6):
        assert rule.evaluate(_context(camera, None, ts=ts)) == []


@pytest.mark.parametrize(
    "history",
    [
        [],
        [(1.4, 165.0, 200.0, 0.0)],
        [(1.4, float("nan"), 200.0, 100.0)],
    ],
)
def test_lost_fall_requires_valid_position_history(camera, history):
    rule = _rule(camera)
    track = _observation(1.4, history=history)
    rule.evaluate(_context(camera, track, ts=1.4))
    rule.evaluate(_context(camera, None, ts=1.6))
    assert rule.evaluate(_context(camera, None, ts=2.6)) == []


def test_memory_is_isolated_between_rule_instances(camera):
    first = _rule(camera)
    second = _rule(camera)
    first.evaluate(_context(camera, _observation(1.4), ts=1.4))
    first.evaluate(_context(camera, None, ts=1.6))
    assert len(first.evaluate(_context(camera, None, ts=2.6))) == 1
    assert second.evaluate(_context(camera, None, ts=2.6)) == []


def test_reused_id_with_new_first_seen_does_not_inherit_fall(camera):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.0), ts=1.0))
    replacement = _observation(
        1.2, state="unknown", since=None, first_seen=1.2
    )
    rule.evaluate(_context(camera, replacement, ts=1.2))
    rule.evaluate(_context(camera, None, ts=1.4))
    assert rule.evaluate(_context(camera, None, ts=2.6)) == []


def test_rewind_clears_loss_memory(camera):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.4), ts=1.4))
    rule.evaluate(_context(camera, None, ts=1.6))
    assert rule.evaluate(_context(camera, None, ts=0.0)) == []
    assert rule.evaluate(_context(camera, None, ts=2.6)) == []


@pytest.mark.parametrize("ts", [float("nan"), float("inf")])
def test_invalid_timestamp_does_not_change_loss_memory(camera, ts):
    rule = _rule(camera)
    rule.evaluate(_context(camera, _observation(1.4), ts=1.4))
    assert rule.evaluate(_context(camera, None, ts=ts)) == []
    rule.evaluate(_context(camera, None, ts=1.6))
    assert len(rule.evaluate(_context(camera, None, ts=2.6))) == 1


@pytest.mark.parametrize(
    "params",
    [
        {"min_lying_s": -1.0},
        {"immobile_window_s": 0.0},
        {"immobile_window_s": float("inf")},
        {"max_shift_bh": 0.0},
        {"track_buffer_s": -1.0},
        {"track_buffer_s": float("nan")},
    ],
)
def test_invalid_rule_parameters(camera, params):
    with pytest.raises(ValueError):
        _rule(camera, params=params)
