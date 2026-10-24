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
    Zone,
)
from guard.rules.base import available_rules, get_rule, load_plugins


@pytest.fixture
def camera():
    return CameraConfig(
        camera_id="carry-camera",
        name="Камера",
        site="Участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
    )


def _points(bent=False):
    points = np.zeros((17, 3), dtype=float)
    points[0] = (500.0, 140.0, 0.9)
    points[9] = (480.0, 220.0, 0.9)
    points[10] = (520.0, 220.0, 0.9)
    points[5] = (480.0, 180.0, 0.9)
    points[6] = (520.0, 180.0, 0.9)
    points[11] = (480.0, 290.0, 0.9)
    points[12] = (520.0, 290.0, 0.9)
    if bent:
        points[5] = (430.0, 180.0, 0.9)
        points[6] = (470.0, 180.0, 0.9)
        points[11] = (500.0, 210.0, 0.9)
        points[12] = (540.0, 210.0, 0.9)
    return points


def _track(bent=False):
    return Track(
        track_id=1,
        cls="person",
        bbox=(400.0, 100.0, 600.0, 400.0),
        conf=0.95,
        first_seen=0.0,
        last_seen=10.0,
        keypoints=_points(bent),
    )


def _load(cls="suitcase", bbox=(470.0, 200.0, 530.0, 260.0), conf=0.9):
    return Detection(model="det", cls=cls, cls_id=28, conf=conf, bbox=bbox)


def _context(camera, track, loads=None, ts=10.0):
    return FrameContext(
        camera=camera,
        ts=ts,
        frame_idx=1,
        frame=np.zeros((500, 1000, 3), dtype=np.uint8),
        people=[] if track is None else [track],
        detections={"det": [_load()] if loads is None else loads},
    )


def _rule(camera, params=None, **kwargs):
    load_plugins()
    cfg = RuleConfig(
        rule_id="carry-rule",
        type="unsafe_carry",
        params={} if params is None else params,
        **kwargs,
    )
    return get_rule("unsafe_carry")(cfg, camera)


def _zone(kind="danger", zone_id="zone-1"):
    return Zone(
        zone_id=zone_id,
        name="Проезд",
        kind=kind,
        polygon=[(0.4, 0.6), (0.6, 0.6), (0.6, 0.9), (0.4, 0.9)],
    )


def _advance(rule, camera, track, ts, loads=None):
    track.last_seen = ts
    x1, y1, x2, y2 = track.bbox
    track.history.append((ts, (x1 + x2) / 2.0, y2, y2 - y1))
    return rule.evaluate(_context(camera, track, loads=loads, ts=ts))


def test_view_obstructed(camera):
    track = _track()
    load = _load(bbox=(470.0, 120.0, 530.0, 260.0))
    result = _rule(camera).evaluate(_context(camera, track, [load]))
    assert len(result) == 1
    violation = result[0]
    assert (
        violation.rule_id,
        violation.type,
        violation.track_id,
        violation.zone_id,
        violation.severity,
    ) == ("carry-rule", "unsafe_carry", 1, None, Severity.medium)
    assert violation.bbox == track.bbox
    assert violation.score == pytest.approx(0.9)
    assert violation.details["conditions"] == ["view_obstructed"]
    assert violation.details["loads"][0]["cls"] == "suitcase"


@pytest.mark.parametrize(
    ("top", "expected"),
    [(139.0, True), (140.0, False), (141.0, False)],
)
def test_view_threshold_is_strict(camera, top, expected):
    load = _load(bbox=(470.0, top, 530.0, 260.0))
    assert bool(_rule(camera).evaluate(_context(camera, _track(), [load]))) is expected


def test_tall_object_beside_face_does_not_block_view(camera):
    load = _load(bbox=(515.0, 110.0, 550.0, 250.0))
    assert _rule(camera).evaluate(_context(camera, _track(), [load])) == []


def test_view_check_can_be_disabled(camera):
    ctx = _context(camera, _track(), [_load(bbox=(470.0, 120.0, 530.0, 260.0))])
    assert _rule(camera, {"check_view": False}).evaluate(ctx) == []


def test_bend_requires_more_than_one_second_of_observations(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8, 1.0):
        assert _advance(rule, camera, track, ts) == []
    result = _advance(rule, camera, track, 1.2)
    assert len(result) == 1
    assert result[0].details["conditions"] == ["bent_back"]
    assert result[0].details["torso_angle_deg"] > 45.0
    assert result[0].details["bent_duration_s"] == pytest.approx(1.2)
    # Повторное чтение кадра не подтверждает и не подавляет событие.
    assert rule.evaluate(_context(camera, track, ts=1.2)) == result


def test_old_track_and_geometry_history_do_not_prove_past_carry(camera):
    track = _track(bent=True)
    track.history.extend((ts, 500.0, 400.0, 300.0) for ts in (0.0, 1.0, 10.0))
    assert _rule(camera).evaluate(_context(camera, track)) == []


@pytest.mark.parametrize("torso_dx", [0.0, 40.0])
def test_upright_or_exactly_45_degrees_is_not_bent(camera, torso_dx):
    rule = _rule(camera)
    track = _track()
    track.keypoints[5] = (430.0, 180.0, 0.9)
    track.keypoints[6] = (470.0, 180.0, 0.9)
    track.keypoints[11] = (430.0 + torso_dx, 220.0, 0.9)
    track.keypoints[12] = (470.0 + torso_dx, 220.0, 0.9)
    for ts in (0.0, 0.4, 0.8, 1.2, 1.6):
        assert _advance(rule, camera, track, ts) == []


def test_bend_parameters(camera):
    track = _track(bent=True)
    rule = _rule(camera, {"bend_duration_s": 0.6})
    assert _advance(rule, camera, track, 0.0) == []
    assert _advance(rule, camera, track, 0.4) == []
    assert _advance(rule, camera, track, 0.8)[0].details["conditions"] == ["bent_back"]

    for params in ({"check_bend": False}, {"torso_angle_deg": 80.0}):
        rule = _rule(camera, params)
        track = _track(bent=True)
        for ts in (0.0, 0.4, 0.8, 1.2):
            assert _advance(rule, camera, track, ts) == []


@pytest.mark.parametrize("interruption", ["load", "pose", "upright"])
def test_bend_duration_restarts_after_missing_evidence(camera, interruption):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4):
        assert _advance(rule, camera, track, ts) == []
    loads = [] if interruption == "load" else None
    if interruption == "pose":
        track.keypoints = None
    elif interruption == "upright":
        track.keypoints = _points()
    assert _advance(rule, camera, track, 0.8, loads=loads) == []
    track.keypoints = _points(bent=True)
    for ts in (1.2, 1.6, 2.0):
        assert _advance(rule, camera, track, ts) == []
    result = _advance(rule, camera, track, 2.4)
    assert result[0].details["bent_duration_s"] == pytest.approx(1.2)


def test_unobserved_history_sample_interrupts_bend(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4):
        _advance(rule, camera, track, ts)
    # Трек обновлялся, но правило не наблюдало груз/позу в этом кадре.
    track.history.append((0.8, 500.0, 400.0, 300.0))
    for ts in (1.2, 1.6, 2.0):
        assert _advance(rule, camera, track, ts) == []
    assert _advance(rule, camera, track, 2.4)


def test_large_gap_interrupts_bend(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 2.0, 2.4, 2.8):
        assert _advance(rule, camera, track, ts) == []
    assert _advance(rule, camera, track, 3.2)


def test_history_is_required_even_with_repeated_pose_observations(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8, 1.2):
        track.last_seen = ts
        assert rule.evaluate(_context(camera, track, ts=ts)) == []


def test_invalid_history_breaks_duration(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8):
        _advance(rule, camera, track, ts)
    track.history[-1] = (0.8, float("nan"), 400.0, 300.0)
    assert _advance(rule, camera, track, 1.2) == []


def test_future_history_does_not_extend_duration(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8, 1.2):
        result = _advance(rule, camera, track, ts)
    track.history.append((100.0, 500.0, 400.0, 300.0))
    repeated = rule.evaluate(_context(camera, track, ts=1.2))
    assert repeated == result


def test_disappearance_and_reused_id_do_not_reuse_evidence(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8, 1.2):
        _advance(rule, camera, track, ts)
    assert rule.evaluate(_context(camera, None, ts=1.3)) == []
    assert _advance(rule, camera, track, 1.4) == []

    track.first_seen = 1.5
    assert _advance(rule, camera, track, 1.6) == []


def test_backwards_time_restarts_bend(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    for ts in (0.0, 0.4, 0.8, 1.2):
        _advance(rule, camera, track, ts)
    track.history = deque([(0.5, 500.0, 400.0, 300.0)], maxlen=64)
    track.last_seen = 0.5
    assert rule.evaluate(_context(camera, track, ts=0.5)) == []


@pytest.mark.parametrize("kind", ["danger", "vehicle_path"])
def test_carry_in_active_danger_zone(camera, kind):
    camera.zones = [_zone(kind)]
    result = _rule(camera).evaluate(_context(camera, _track()))
    assert len(result) == 1
    assert result[0].zone_id == "zone-1"
    assert result[0].details["conditions"] == ["danger_zone"]
    assert result[0].details["zone_kind"] == kind


def test_zone_condition_requires_correct_kind_and_can_be_disabled(camera):
    camera.zones = [_zone("height")]
    assert _rule(camera).evaluate(_context(camera, _track())) == []
    camera.zones = [_zone()]
    assert _rule(camera, {"check_zone": False}).evaluate(
        _context(camera, _track())
    ) == []


def test_zone_uses_foot_and_active_hours(camera):
    camera.zones = [_zone()]
    camera.zones[0].polygon = [(0.4, 0.2), (0.6, 0.2), (0.6, 0.6), (0.4, 0.6)]
    assert _rule(camera).evaluate(_context(camera, _track())) == []
    camera.zones = [_zone()]
    camera.zones[0].active_hours = ["12:00-13:00"]
    assert _rule(camera).evaluate(_context(camera, _track())) == []


@pytest.mark.parametrize("selection", [[], ["missing"]])
def test_explicit_zone_scope_does_not_expand(camera, selection):
    camera.zones = [_zone()]
    load = _load(bbox=(470.0, 120.0, 530.0, 260.0))
    assert _rule(camera, zones=selection).evaluate(
        _context(camera, _track(), [load])
    ) == []


def test_explicit_scope_applies_to_view_condition(camera):
    camera.zones = [_zone("height")]
    ctx = _context(camera, _track(), [_load(bbox=(470.0, 120.0, 530.0, 260.0))])
    assert _rule(camera, zones=["zone-1"]).evaluate(ctx)
    camera.zones[0].active_hours = ["12:00-13:00"]
    assert _rule(camera, zones=["zone-1"]).evaluate(ctx) == []


@pytest.mark.parametrize("name", ["suitcase", "backpack", "handbag", "box", "load"])
def test_default_load_classes(camera, name):
    camera.zones = [_zone()]
    assert _rule(camera).evaluate(_context(camera, _track(), [_load(cls=name)]))


def test_custom_classes_keep_box_and_load(camera):
    camera.zones = [_zone()]
    rule = _rule(camera, {"load_classes": ["crate"]})
    for name in ("crate", "box", "load"):
        assert rule.evaluate(_context(camera, _track(), [_load(cls=name)]))
    for name in ("suitcase", "chair"):
        assert rule.evaluate(_context(camera, _track(), [_load(cls=name)])) == []


@pytest.mark.parametrize("index", [9, 10])
def test_one_confident_wrist_is_sufficient(camera, index):
    camera.zones = [_zone()]
    track = _track()
    track.keypoints[9, 2] = 0.0
    track.keypoints[10, 2] = 0.0
    track.keypoints[index, 2] = 0.5
    assert _rule(camera).evaluate(_context(camera, track))


def test_overlap_without_wrist_near_load_is_not_carry(camera):
    camera.zones = [_zone()]
    track = _track()
    track.keypoints[9] = (410.0, 150.0, 0.9)
    track.keypoints[10] = (590.0, 150.0, 0.9)
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_wrist_near_nonoverlapping_load_is_not_carry(camera):
    camera.zones = [_zone()]
    track = _track()
    track.keypoints[10] = (600.0, 220.0, 0.9)
    load = _load(bbox=(601.0, 200.0, 640.0, 260.0))
    assert _rule(camera).evaluate(_context(camera, track, [load])) == []


def test_hand_distance_parameter(camera):
    camera.zones = [_zone()]
    track = _track()
    track.keypoints[9] = (460.0, 220.0, 0.9)
    track.keypoints[10, 2] = 0.0
    ctx = _context(camera, track)
    assert _rule(camera).evaluate(ctx)
    assert _rule(camera, {"hand_distance_bh": 0.0}).evaluate(ctx) == []


@pytest.mark.parametrize("kind", ["none", "shape", "low", "nan"])
def test_unreliable_pose_does_not_trigger(camera, kind):
    camera.zones = [_zone()]
    track = _track()
    if kind == "none":
        track.keypoints = None
    elif kind == "shape":
        track.keypoints = np.zeros((16, 3))
    elif kind == "low":
        track.keypoints[:, 2] = 0.49
    else:
        track.keypoints[[9, 10], 0] = float("nan")
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_missing_nose_only_disables_view(camera):
    track = _track()
    track.keypoints[0, 2] = 0.0
    load = _load(bbox=(470.0, 120.0, 530.0, 260.0))
    assert _rule(camera).evaluate(_context(camera, track, [load])) == []
    camera.zones = [_zone()]
    result = _rule(camera).evaluate(_context(camera, track, [load]))
    assert result[0].details["conditions"] == ["danger_zone"]


def test_missing_hip_prevents_bend(camera):
    rule = _rule(camera)
    track = _track(bent=True)
    track.keypoints[11, 2] = 0.0
    for ts in (0.0, 0.4, 0.8, 1.2):
        assert _advance(rule, camera, track, ts) == []


@pytest.mark.parametrize("conf", [0.49, float("nan")])
def test_unreliable_load(camera, conf):
    camera.zones = [_zone()]
    assert _rule(camera).evaluate(
        _context(camera, _track(), [_load(conf=conf)])
    ) == []


def test_invalid_load_and_person_boxes(camera):
    camera.zones = [_zone()]
    load = _load(bbox=(530.0, 200.0, 470.0, 260.0))
    assert _rule(camera).evaluate(_context(camera, _track(), [load])) == []
    track = _track()
    track.bbox = (400.0, 100.0, float("nan"), 400.0)
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_tracked_object_and_duplicate_detection(camera):
    camera.zones = [_zone()]
    load = _load()
    obj = Track(
        track_id=2,
        cls=load.cls,
        bbox=load.bbox,
        conf=load.conf,
        first_seen=0.0,
        last_seen=10.0,
    )
    ctx = _context(camera, _track(), loads=[])
    ctx.objects = [obj]
    rule = _rule(camera)
    assert len(rule.evaluate(ctx)) == 1
    ctx.detections["det"] = [load, load]
    result = rule.evaluate(ctx)
    assert len(result) == 1
    assert len(result[0].details["loads"]) == 1

    ctx.detections["det"] = []
    obj.last_seen = 9.0
    assert rule.evaluate(ctx) == []


def test_ambiguous_owner_is_skipped(camera):
    camera.zones = [_zone()]
    ctx = _context(camera, _track())
    second = _track()
    second.track_id = 2
    ctx.people.append(second)
    assert _rule(camera).evaluate(ctx) == []


def test_combined_conditions_are_one_violation(camera):
    camera.zones = [_zone()]
    rule = _rule(camera)
    track = _track(bent=True)
    load = _load(bbox=(470.0, 120.0, 530.0, 260.0))
    for ts in (0.0, 0.4, 0.8, 1.2):
        result = _advance(rule, camera, track, ts, loads=[load])
    assert len(result) == 1
    assert result[0].details["conditions"] == [
        "view_obstructed", "bent_back", "danger_zone",
    ]


def test_no_load_safe_carry_and_stale_track(camera):
    rule = _rule(camera)
    track = _track()
    assert rule.evaluate(_context(camera, track, loads=[])) == []
    assert rule.evaluate(_context(camera, track)) == []
    camera.zones = [_zone()]
    track.last_seen = 9.0
    assert _rule(camera).evaluate(_context(camera, track)) == []


def test_no_input_mutation(camera):
    camera.zones = [_zone()]
    track = _track()
    track.history.append((10.0, 500.0, 400.0, 300.0))
    track.attributes = {"test": {"value": 1}}
    ctx = _context(camera, track)
    points = track.keypoints.copy()
    history = list(track.history)
    attributes = deepcopy(track.attributes)
    detection_box = ctx.detections["det"][0].bbox
    rule = _rule(camera)
    result = rule.evaluate(ctx)
    assert rule.evaluate(ctx) == result
    np.testing.assert_array_equal(track.keypoints, points)
    assert list(track.history) == history
    assert track.attributes == attributes
    assert ctx.detections["det"][0].bbox == detection_box


def test_empty_frame_and_invalid_timestamp(camera):
    rule = _rule(camera)
    assert rule.evaluate(_context(camera, None)) == []
    assert rule.evaluate(_context(camera, _track(), ts=float("nan"))) == []


@pytest.mark.parametrize(
    "params",
    [
        {"load_classes": "suitcase"},
        {"load_classes": [""]},
        {"check_view": "false"},
        {"kp_conf": 1.1},
        {"min_load_conf": -0.1},
        {"hand_distance_bh": -1},
        {"torso_angle_deg": 91},
        {"bend_duration_s": float("nan")},
        {"max_history_gap_s": 0},
    ],
)
def test_invalid_parameters(camera, params):
    with pytest.raises(ValueError):
        _rule(camera, params)


def test_plugin_discovered():
    load_plugins()
    metadata = {item["type"]: item for item in available_rules()}["unsafe_carry"]
    assert metadata["required_models"] == ["det", "pose"]
    assert metadata["default_severity"] == "medium"
    properties = metadata["params_schema"]["properties"]
    assert properties["torso_angle_deg"]["default"] == 45.0
    assert properties["bend_duration_s"]["default"] == 1.0
    for name in ("check_view", "check_bend", "check_zone"):
        assert properties[name]["default"] is True
    assert all(spec["title"] for spec in properties.values())
