"""S1 zone_intrusion и геометрия зон на синтетике."""

from datetime import datetime

import numpy as np
import pytest

from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Track, Zone
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.rules.geometry import (
    LOCAL_TZ,
    foot_point,
    in_time_ranges,
    is_zone_active,
    parse_time_range,
    point_in_polygon,
    to_pixels,
    validate_active_hours,
)

load_plugins()  # реестр мог быть частично заполнен прямыми импортами плагинов в других тестах

W, H = 1000, 500
SQUARE = [(0.2, 0.4), (0.6, 0.4), (0.6, 0.9), (0.2, 0.9)]


def msk(h: int, m: int = 0) -> float:
    return datetime(2026, 10, 24, h, m, tzinfo=LOCAL_TZ).timestamp()


def camera(zones=None, rules=None) -> CameraConfig:
    return CameraConfig(
        camera_id="cam1",
        name="Цех",
        site="Цех 1",
        source=SourceConfig(type="file", uri="x.mp4"),
        zones=zones if zones is not None else [Zone(zone_id="z1", name="Опасная", kind="danger", polygon=SQUARE)],
        rules=rules or [RuleConfig(rule_id="r1", type="zone_intrusion", zones=["z1"])],
    )


def track(tid: int, bbox, conf: float = 0.9) -> Track:
    return Track(track_id=tid, cls="person", bbox=bbox, conf=conf, first_seen=0.0, last_seen=0.0)


def ctx(cam: CameraConfig, people, ts: float | None = None) -> FrameContext:
    return FrameContext(camera=cam, ts=msk(12) if ts is None else ts, frame_idx=0, frame=np.zeros((H, W, 3), np.uint8), people=people)


# ---------- геометрия ----------


def test_point_in_polygon_inside_outside_boundary():
    poly = to_pixels(SQUARE, W, H)
    assert poly[0] == (200.0, 200.0)
    assert point_in_polygon((400, 300), poly)
    assert not point_in_polygon((100, 300), poly)
    assert not point_in_polygon((400, 480), poly)
    assert point_in_polygon((200, 300), poly)  # на границе — внутри
    assert not point_in_polygon((1, 1), [(0, 0), (1, 1)])  # вырожденный


def test_point_in_concave_polygon():
    l_shape = [(0, 0), (10, 0), (10, 2), (2, 2), (2, 10), (0, 10)]
    assert point_in_polygon((1, 5), l_shape)
    assert not point_in_polygon((5, 5), l_shape)


def test_foot_point_is_bottom_center():
    assert foot_point((10, 20, 30, 100)) == (20.0, 100.0)


def test_time_ranges_moscow_and_overnight():
    assert parse_time_range("07:55-12:00") == (475, 720)
    assert in_time_ranges(["08:00-20:00"], msk(12))
    assert not in_time_ranges(["08:00-20:00"], msk(21))
    assert in_time_ranges(["22:00-06:00"], msk(23, 30))
    assert in_time_ranges(["22:00-06:00"], msk(5, 59))
    assert not in_time_ranges(["22:00-06:00"], msk(6, 0))
    assert in_time_ranges(None, msk(3)) and in_time_ranges([], msk(3))
    assert in_time_ranges(["00:00-24:00"], msk(23, 59))


def test_active_hours_validation_and_failsafe():
    validate_active_hours(["07:55-12:00", "22:00-06:00"])
    for bad in ("7-12", "25:00-26:00", "10:61-11:00"):
        with pytest.raises(ValueError):
            validate_active_hours([bad])
    z = Zone(zone_id="z", name="z", polygon=SQUARE, active_hours=["garbage"])
    assert is_zone_active(z, msk(3))  # битое расписание не выключает зону


# ---------- плагин ----------


def test_plugin_registered_with_schema():
    info = {r["type"]: r for r in available_rules()}["zone_intrusion"]
    assert info["required_models"] == ["det"]
    assert info["default_severity"] == "high"
    props = info["params_schema"]["properties"]
    assert {"min_conf", "min_box_h_px"} <= set(props)


def test_person_foot_in_zone_gives_violation():
    cam = camera()
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    inside = track(7, (350, 100, 450, 400))  # опора (400, 400) в зоне
    outside = track(8, (700, 100, 800, 400))
    out = rule.evaluate(ctx(cam, [inside, outside]))
    assert len(out) == 1
    v = out[0]
    assert (v.track_id, v.zone_id, v.rule_id, v.type) == (7, "z1", "r1", "zone_intrusion")
    assert v.bbox == (350.0, 100.0, 450.0, 400.0)
    assert v.details["zone_name"] == "Опасная"


def test_head_in_zone_but_feet_outside_is_not_violation():
    cam = camera()
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    assert rule.evaluate(ctx(cam, [track(1, (350, 210, 450, 480))])) == []  # опора y=480 ниже зоны


def test_inactive_zone_by_hours():
    zone = Zone(zone_id="z1", name="Ночная", polygon=SQUARE, active_hours=["22:00-06:00"])
    cam = camera(zones=[zone])
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    p = [track(1, (350, 100, 450, 400))]
    assert rule.evaluate(ctx(cam, p, ts=msk(12))) == []
    assert len(rule.evaluate(ctx(cam, p, ts=msk(23)))) == 1


def test_rule_without_zones_uses_danger_zones_only():
    zones = [
        Zone(zone_id="d", name="danger", kind="danger", polygon=SQUARE),
        Zone(zone_id="s", name="smoking", kind="smoking_area", polygon=[(0.0, 0.0), (0.1, 0.0), (0.1, 0.1)]),
    ]
    cam = camera(zones=zones, rules=[RuleConfig(rule_id="r", type="zone_intrusion")])
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    assert [z.zone_id for z in rule.zones] == ["d"]
    out = rule.evaluate(ctx(cam, [track(1, (350, 100, 450, 400)), track(2, (10, 5, 60, 40))]))
    assert [(v.track_id, v.zone_id) for v in out] == [(1, "d")]


def test_min_conf_and_min_box_height_filters():
    cam = camera(rules=[RuleConfig(rule_id="r1", type="zone_intrusion", zones=["z1"], params={"min_conf": 0.5, "min_box_h_px": 50})])
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    low_conf = track(1, (350, 100, 450, 400), conf=0.3)
    tiny = track(2, (390, 370, 400, 400))
    ok = track(3, (300, 100, 400, 400))
    assert [v.track_id for v in rule.evaluate(ctx(cam, [low_conf, tiny, ok]))] == [3]


def test_one_violation_per_person_with_overlapping_zones():
    zones = [Zone(zone_id="a", name="a", polygon=SQUARE), Zone(zone_id="b", name="b", polygon=SQUARE)]
    cam = camera(zones=zones, rules=[RuleConfig(rule_id="r", type="zone_intrusion", zones=["a", "b"])])
    rule = get_rule("zone_intrusion")(cam.rules[0], cam)
    assert len(rule.evaluate(ctx(cam, [track(1, (350, 100, 450, 400))]))) == 1
