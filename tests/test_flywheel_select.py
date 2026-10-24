"""Маховик: отбор кадров — полоса уверенности, период и суточный лимит, мигание трека, расхождение TPU и CPU."""

from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection, FrameContext, RuleConfig, SourceConfig, Track
from guard.flywheel.collector import FlywheelCollector
from guard.flywheel.select import (
    FlickerState,
    boxes_near,
    compare_detections,
    decision_threshold,
    find_uncertain,
    rule_targets,
)
from guard.rules.base import get_rule


def _cam(rules, uri="videos/new_site.mp4", camera_id="cam1"):
    return CameraConfig(camera_id=camera_id, name="Цех", site="Цех 1", source=SourceConfig(type="file", uri=uri),
                        rules=rules)


def _rules(cam):
    return [get_rule(r.type)(r, cam) for r in cam.rules]


def det(cls, conf, bbox=(10, 10, 60, 160), model="det"):
    return Detection(model=model, cls=cls, cls_id=0, conf=conf, bbox=bbox)


def track(tid, bbox):
    return Track(track_id=tid, cls="person", bbox=bbox, conf=0.9, first_seen=0.0, last_seen=0.0)


class ListOutput:
    def __init__(self, accept=True):
        self.jobs = []
        self.accept = accept

    def put(self, job):
        if not self.accept:
            return False
        self.jobs.append(job)
        return True


def _ctx(cam, ts, dets=None, people=(), frame=None):
    return FrameContext(camera=cam, ts=ts, frame_idx=0, frame=frame if frame is not None else np.zeros((36, 64, 3), np.uint8),
                        detections=dets or {}, people=list(people), objects=[])


def _worker(cam, thresholds):
    return SimpleNamespace(cfg=cam, rules=_rules(cam),
                           detectors={m: SimpleNamespace(spec=SimpleNamespace(conf_threshold=t)) for m, t in thresholds.items()})


# ---------------------------------------------------------------- полоса уверенности


def test_band_uses_rule_min_conf_and_tracker_threshold():
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion", params={"min_conf": 0.5}),
                RuleConfig(rule_id="f", type="fall_immobile")])
    zone, fall = _rules(cam)
    assert decision_threshold(zone, "person", 0.25, 0.4) == 0.5  # min_conf правила
    assert decision_threshold(fall, "person", 0.25, 0.4) == 0.4  # порог трекера для людей
    ppe = get_rule("ppe_missing")(RuleConfig(rule_id="p", type="ppe_missing", params={"required": ["helmet"]}), cam)
    assert rule_targets(ppe) == {"helmet", "no_helmet"}
    assert decision_threshold(ppe, "no_helmet", 0.4, 0.4) == 0.4  # порог модели
    vehicle = get_rule("vehicle_proximity")(RuleConfig(rule_id="v", type="vehicle_proximity"), cam)
    assert decision_threshold(vehicle, "truck", 0.25, 0.4) == 0.4  # технику тоже ведёт трекер
    phone = get_rule("phone_usage")(RuleConfig(rule_id="t", type="phone_usage"), cam)
    assert decision_threshold(phone, "cell phone", 0.25, 0.4) == 0.25  # телефон трекер не ведёт


@pytest.mark.parametrize(
    ("conf", "hit"),
    [(0.39, False), (0.40, True), (0.47, True), (0.55, True), (0.56, False), (0.9, False)],
)
def test_band_edges_for_zone_intrusion(conf, hit):
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion", params={"min_conf": 0.5})])
    found = find_uncertain({"det": [det("person", conf)]}, _rules(cam), {"det": 0.25}, 0.4)
    assert (found is not None) is hit
    if hit:
        assert found.cls == "person" and (found.lo, found.hi) == (0.4, 0.55)
        assert "person conf" in found.reason and "∈ [0.40, 0.55]" in found.reason


def test_band_ignores_other_classes_and_unknown_models_and_picks_closest():
    cam = _cam([RuleConfig(rule_id="p", type="ppe_missing", params={"required": ["helmet"]})])
    dets = {
        "det": [det("person", 0.41)],  # не целевой класс для СИЗ
        "ppe": [det("no_helmet", 0.44, model="ppe"), det("helmet", 0.405, model="ppe"), det("no_vest", 0.41, model="ppe")],
        "other": [det("no_helmet", 0.40, model="other")],  # модели нет в порогах
    }
    found = find_uncertain(dets, _rules(cam), {"det": 0.25, "ppe": 0.4}, 0.4)
    assert found is not None and found.cls == "helmet" and found.model_id == "ppe"  # ближе всех к порогу 0.4


def test_collector_period_and_daily_cap():
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion", params={"min_conf": 0.5})])
    out = ListOutput()
    col = FlywheelCollector(out, period_s=10.0, daily_cap=2, holdout=frozenset())
    worker = _worker(cam, {"det": 0.25})
    t0 = 1_700_000_000.0
    for i in range(100):  # 20 с при 5 к/с, каждая рамка в полосе
        col.observe(worker, _ctx(cam, t0 + i * 0.2, {"det": [det("person", 0.48)]}))
    assert [j.sample["source"] for j in out.jobs] == ["uncertain", "uncertain"]  # 1 кадр в 10 с
    assert out.jobs[1].sample["created_at"] - out.jobs[0].sample["created_at"] == pytest.approx(10.0)
    for i in range(100, 300):  # ещё 40 с: суточный лимит 2 исчерпан
        col.observe(worker, _ctx(cam, t0 + i * 0.2, {"det": [det("person", 0.48)]}))
    assert len(out.jobs) == 2 and col.dropped["cap"] >= 1
    job = out.jobs[0]
    assert job.sample["reason"].startswith("person conf 0.48") and job.sample["meta"]["band"] == [0.4, 0.55]
    assert job.frame is not None and job.detections["det"][0].conf == 0.48  # ссылки на кадр и детекции, без копий


def test_collector_events_are_not_capped_and_queue_overflow_counts_drop():
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion")])
    out = ListOutput()
    col = FlywheelCollector(out, period_s=10.0, daily_cap=0, holdout=frozenset())
    ev = SimpleNamespace(id="ev1", type="zone_intrusion", rule_id="z", zone_id=None, track_id=3, bbox=(1, 2, 3, 4))
    for _ in range(3):
        col.on_event(_ctx(cam, 1.0), ev)
    assert len(out.jobs) == 3 and all(j.sample["source"] == "event" for j in out.jobs)
    assert out.jobs[0].sample["event_id"] == "ev1" and out.jobs[0].sample["meta"]["event_bbox"] == [1.0, 2.0, 3.0, 4.0]
    full = FlywheelCollector(ListOutput(accept=False), holdout=frozenset())
    full.on_event(_ctx(cam, 1.0), ev)
    assert full.dropped["queue"] == 1 and full.stats()["dropped_total"] == 1


def test_disabled_collector_does_nothing():
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion")])
    out = ListOutput()
    col = FlywheelCollector(out, enabled=False, holdout=frozenset())
    col.observe(_worker(cam, {"det": 0.25}), _ctx(cam, 1.0, {"det": [det("person", 0.45)]}))
    col.on_event(_ctx(cam, 1.0), SimpleNamespace(id="e", type="t", rule_id="r", zone_id=None, track_id=None, bbox=(0, 0, 1, 1)))
    assert out.jobs == [] and not col.collecting


# ---------------------------------------------------------------- мигание трека


def test_flicker_new_id_near_lost_track_within_one_second():
    st = FlickerState()
    assert st.update([(1, (100, 100, 150, 250))], 0.0) is None
    assert st.update([(1, (102, 100, 152, 250))], 0.2) is None
    assert st.update([], 0.4) is None  # трек #1 пропал (последний раз в 0.2 с)
    hit = st.update([(2, (110, 105, 160, 255))], 0.8)
    assert hit is not None and (hit.old_id, hit.new_id) == (1, 2) and hit.gap_s == pytest.approx(0.6)
    assert "трек #1 пропал" in hit.reason
    assert st.update([(2, (110, 105, 160, 255))], 1.0) is None  # тот же новый ID — не повтор


def test_flicker_ignores_far_late_and_returning_tracks():
    far = FlickerState()
    far.update([(1, (0, 0, 50, 100))], 0.0)
    far.update([], 0.2)
    assert far.update([(2, (500, 300, 550, 400))], 0.4) is None  # далеко
    late = FlickerState()
    late.update([(1, (0, 0, 50, 100))], 0.0)
    late.update([], 0.2)
    assert late.update([(2, (5, 0, 55, 100))], 1.3) is None  # через 1.3 с
    back = FlickerState()
    back.update([(1, (0, 0, 50, 100))], 0.0)
    back.update([], 0.2)
    assert back.update([(1, (5, 0, 55, 100))], 0.4) is None  # вернулся тот же ID
    # центры ближе 0.75 · max(h, w) — рядом даже без пересечения
    assert boxes_near((0, 0, 50, 100), (60, 0, 110, 100)) and not boxes_near((0, 0, 50, 100), (90, 0, 140, 100))


def test_collector_flicker_sample_with_period():
    cam = _cam([RuleConfig(rule_id="z", type="zone_intrusion")])
    out = ListOutput()
    col = FlywheelCollector(out, period_s=10.0, holdout=frozenset())
    worker = _worker(cam, {"det": 0.25})
    seq = [[track(1, (100, 100, 150, 250))], [], [track(2, (105, 100, 155, 250))],  # мигание в 0.4 с
           [], [track(3, (110, 100, 160, 250))]]  # второе мигание в 0.8 с — внутри периода, без сэмпла
    for i, people in enumerate(seq):
        col.observe(worker, _ctx(cam, 100.0 + i * 0.2, people=people))
    flicker = [j for j in out.jobs if j.sample["source"] == "flicker"]
    assert len(flicker) == 1 and flicker[0].sample["meta"]["new_track_id"] == 2


# ---------------------------------------------------------------- расхождение TPU и CPU


def test_compare_detections_pairs_by_iou_and_class():
    tpu = [det("person", 0.38, (10, 10, 60, 160)), det("person", 0.7, (300, 10, 350, 160))]
    cpu = [det("person", 0.45, (12, 10, 62, 160)), det("forklift", 0.8, (100, 100, 200, 200)),
           det("person", 0.36, (500, 10, 550, 160))]
    out = compare_detections("det", tpu, cpu, threshold=0.4, classes=frozenset({"person", "forklift"}))
    found = {(m.cls, m.only_on) for m in out}
    # CPU 0.45 спарен с TPU 0.38 (int8 занижает conf); TPU 0.7 без пары; forklift только на CPU; 0.36 ниже порога
    assert found == {("forklift", "cpu"), ("person", "tpu")}
    assert out[0].conf == 0.8 and "только на CPU" in out[0].reason
    assert compare_detections("det", tpu, cpu, threshold=0.4, classes=frozenset({"car"})) == []
