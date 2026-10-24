"""S7: курение по детектору сигареты у головы или кисти (синтетика, без моделей)."""

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
    Track,
    Zone,
)
from guard.rules.base import get_rule
from guard.rules.plugins.smoking_fire import SmokingFireRule

TS0 = datetime(2026, 10, 24, 12, tzinfo=ZoneInfo("Europe/Moscow")).timestamp()
# Человек 100×300 px: голова около (150, 130), кисти (120, 260) и (180, 260).
PERSON_BOX = (100.0, 100.0, 200.0, 400.0)


def _camera(zones=()) -> CameraConfig:
    return CameraConfig(
        camera_id="camera-1",
        name="Камера",
        site="Участок",
        source=SourceConfig(type="file", uri="synthetic.mp4"),
        zones=list(zones),
    )


def _keypoints(conf: float = 0.9) -> np.ndarray:
    points = np.zeros((17, 3), dtype=float)
    for index, (x, y) in {0: (150, 130), 1: (145, 125), 2: (155, 125), 3: (140, 128), 4: (160, 128),
                          9: (120, 260), 10: (180, 260)}.items():
        points[index] = (x, y, conf)
    return points


def _person(track_id: int = 1, box=PERSON_BOX, keypoints="default") -> Track:
    return Track(
        track_id=track_id,
        cls="person",
        bbox=box,
        conf=0.9,
        first_seen=TS0,
        last_seen=TS0,
        keypoints=_keypoints() if isinstance(keypoints, str) else keypoints,
    )


def _cig(box, conf: float = 0.8, model: str = "cigarette") -> Detection:
    return Detection(model=model, cls="cigarette", cls_id=0, conf=conf, bbox=box)


# Сигарета у рта: 10 px ниже носа.
AT_MOUTH = (148.0, 138.0, 162.0, 142.0)
# Сигарета в правой кисти.
IN_HAND = (182.0, 262.0, 196.0, 266.0)
# Сигарета на столе в стороне: 150 px от ближайшей точки.
ON_TABLE = (330.0, 380.0, 344.0, 384.0)


def _ctx(dt: float, people, cigarettes, camera=None, fire=()) -> FrameContext:
    return FrameContext(
        camera=camera or _camera(),
        ts=TS0 + dt,
        frame_idx=int(dt * 5),
        frame=np.zeros((480, 640, 3), dtype=np.uint8),
        detections={"cigarette": list(cigarettes), "fire": list(fire)},
        people=list(people),
    )


def _rule(camera=None, zones=None, **params) -> SmokingFireRule:
    params.setdefault("cigarette_model", "cigarette")
    cfg = RuleConfig(rule_id="smoking", type="smoking_fire", params=params, **({"zones": zones} if zones else {}))
    return SmokingFireRule(cfg, camera or _camera())


def _run(rule, frames, camera=None):
    """frames: [(dt, people, cigarettes)] → нарушения по кадрам."""
    return [rule.evaluate(_ctx(dt, people, cigs, camera)) for dt, people, cigs in frames]


def test_without_cigarette_model_rule_is_fire_only():
    rule = SmokingFireRule(RuleConfig(rule_id="s", type="smoking_fire"), _camera())
    assert rule.required_models == frozenset({"fire"})
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)])) == []
    # Метаданные класса для UI не меняются.
    assert get_rule("smoking_fire").required_models == frozenset({"fire"})


def test_cigarette_model_adds_det_pose_and_model_on_instance():
    rule = _rule(cigarette_model="cig_v1")
    assert rule.required_models == frozenset({"fire", "det", "pose", "cig_v1"})
    assert SmokingFireRule.required_models == frozenset({"fire"})


def test_smoking_without_fire_model():
    rule = _rule(detect_fire=False)
    assert rule.required_models == frozenset({"det", "pose", "cigarette"})


def test_nothing_to_detect_is_rejected():
    with pytest.raises(ValueError):
        _rule(detect_fire=False, cigarette_model="")


@pytest.mark.parametrize("params", [{"window_s": -1}, {"near_factor": float("nan")}, {"min_hits": 0},
                                    {"cigarette_min_conf": -0.1}])
def test_invalid_params_are_rejected(params):
    with pytest.raises(ValueError):
        _rule(**params)


def test_cigarette_at_mouth_needs_min_hits():
    rule = _rule()
    first, second = _run(rule, [(0.0, [_person()], [_cig(AT_MOUTH)]), (0.2, [_person()], [_cig(AT_MOUTH)])])
    assert first == []
    violation, = second
    assert violation.type == "smoking_fire"
    assert violation.track_id == 1
    assert violation.bbox == PERSON_BOX
    assert violation.severity == Severity.critical
    assert violation.score == pytest.approx(0.8)
    assert violation.details == {"cls": "cigarette", "conf": 0.8, "near": "head", "hits": 2}


def test_cigarette_in_hand():
    rule = _rule(min_hits=1)
    violation, = rule.evaluate(_ctx(0.0, [_person()], [_cig(IN_HAND)]))
    assert violation.details["near"] == "hand"


def test_cigarette_far_from_head_and_hands_is_ignored():
    rule = _rule(min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(ON_TABLE)])) == []


def test_near_factor_controls_radius():
    # Кисть (180, 260), сигарета в 40 px правее. Радиус 0.15 × 300 = 45 px — да; 0.1 × 300 = 30 px — нет.
    box = (220.0, 258.0, 230.0, 262.0)
    assert _rule(min_hits=1).evaluate(_ctx(0.0, [_person()], [_cig(box)]))
    assert _rule(min_hits=1, near_factor=0.1).evaluate(_ctx(0.0, [_person()], [_cig(box)])) == []


def test_person_without_pose_is_not_matched():
    rule = _rule(min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person(keypoints=None)], [_cig(AT_MOUTH)])) == []


def test_low_confidence_keypoints_are_ignored():
    rule = _rule(min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person(keypoints=_keypoints(conf=0.1))], [_cig(AT_MOUTH)])) == []


def test_low_confidence_cigarette_is_ignored():
    rule = _rule(min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH, conf=0.3)])) == []
    assert _rule(min_hits=1, cigarette_min_conf=0.25).evaluate(
        _ctx(0.0, [_person()], [_cig(AT_MOUTH, conf=0.3)]))


def test_hits_expire_outside_window():
    rule = _rule(window_s=1.0)
    frames = _run(rule, [
        (0.0, [_person()], [_cig(AT_MOUTH)]),
        (2.0, [_person()], [_cig(AT_MOUTH)]),  # первое попадание уже вне окна
        (2.4, [_person()], []),                # два попадания в окне 1 с: 2.0 и… только одно
    ])
    assert frames == [[], [], []]


def test_violation_holds_between_detections_within_window():
    rule = _rule(window_s=3.0)
    frames = _run(rule, [
        (0.0, [_person()], [_cig(AT_MOUTH)]),
        (0.4, [_person()], [_cig(AT_MOUTH)]),
        (1.0, [_person()], []),  # сигарету закрыла рука — нарушение держится
        (2.8, [_person()], []),
        (3.5, [_person()], []),  # оба попадания старше 3 с
    ])
    assert [len(v) for v in frames] == [0, 1, 1, 1, 0]


def test_several_cigarettes_in_one_frame_count_once():
    rule = _rule()
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH), _cig(IN_HAND)])) == []


def test_same_frame_evaluated_twice_counts_once():
    rule = _rule()
    ctx = _ctx(0.0, [_person()], [_cig(AT_MOUTH)])
    assert rule.evaluate(ctx) == []
    assert rule.evaluate(ctx) == []


def test_cigarette_goes_to_nearest_person_only():
    rule = _rule(min_hits=1)
    near = _person(track_id=1)
    # Второй человек правее на 60 px: его кисть (120+60, 260) тоже в радиусе сигареты в руке первого.
    shifted = _keypoints()
    shifted[:, 0] += 60
    other = _person(track_id=2, box=(160.0, 100.0, 260.0, 400.0), keypoints=shifted)
    violation, = rule.evaluate(_ctx(0.0, [near, other], [_cig(IN_HAND)]))
    assert violation.track_id == 1


def test_small_person_is_ignored():
    rule = _rule(min_hits=1, min_box_h_px=400)
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)])) == []


def test_track_disappears_and_state_is_pruned():
    rule = _rule(window_s=1.0)
    _run(rule, [(0.0, [_person()], [_cig(AT_MOUTH)])])
    assert 1 in rule._hits
    _run(rule, [(5.0, [], [])])
    assert rule._hits == {}


def test_smoking_area_excludes_by_person_foot_point():
    area = Zone(zone_id="area", name="Место для курения", kind="smoking_area",
                polygon=[(0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)])  # ноги (150, 400) — внутри
    camera = _camera([area])
    rule = _rule(camera=camera, min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)], camera)) == []


def test_no_smoking_zone_limits_to_people_inside():
    zone = Zone(zone_id="nz", name="Склад", kind="no_smoking",
                polygon=[(0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)])  # правая половина: ноги вне
    camera = _camera([zone])
    rule = _rule(camera=camera, min_hits=1)
    assert rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)], camera)) == []
    inside = SmokingFireRule(
        RuleConfig(rule_id="s", type="smoking_fire", params={"cigarette_model": "cigarette", "min_hits": 1}),
        _camera([zone.model_copy(update={"polygon": [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]})]),
    )
    violation, = inside.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)], inside.camera))
    assert violation.zone_id == "nz"


def test_fire_still_reported_with_cigarette_model():
    rule = _rule(min_hits=1)
    fire = Detection(model="fire", cls="fire", cls_id=0, conf=0.9, bbox=(400.0, 300.0, 460.0, 380.0))
    out = rule.evaluate(_ctx(0.0, [_person()], [_cig(AT_MOUTH)], fire=[fire]))
    assert sorted(v.details["cls"] for v in out) == ["cigarette", "fire"]


def test_fire_disabled_ignores_fire_detections():
    rule = _rule(min_hits=1, detect_fire=False)
    fire = Detection(model="fire", cls="fire", cls_id=0, conf=0.9, bbox=(400.0, 300.0, 460.0, 380.0))
    out = rule.evaluate(_ctx(0.0, [], [], fire=[fire]))
    assert out == []
