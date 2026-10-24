"""Wrong-way evidence: synthetic trajectories, real tracker/manager/SQLite."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Track, Zone
from guard.events.journal import Journal
from guard.events.manager import EventManager
from guard.pipeline.testing import person
from guard.rules.base import available_rules, get_rule
from guard.rules.plugins.wrong_way import WrongWayRule
from guard.tracking.bytetrack import Tracker


def camera(camera_id="a", **params):
    return CameraConfig(
        camera_id=camera_id, name=camera_id, site="test", source=SourceConfig(type="file", uri="unused"),
        zones=[Zone(zone_id="flow", name="Проход", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])],
        rules=[RuleConfig(rule_id="wrong-1", type="wrong_way", zones=["flow"],
                          params={"direction": [[0, .5], [1, .5]], **params},
                          min_duration_s=.4, min_ratio=1, close_after_s=.5, cooldown_s=30)],
    )


def build(cam=None, **params):
    cam = cam or camera(**params)
    return WrongWayRule(cam.rules[0], cam)


def track(ts, x=250., y=150., height=80., **fields):
    values = dict(track_id=1, cls="person", bbox=(x - 15, y - height, x + 15, y),
                  conf=.95, first_seen=0., last_seen=ts)
    values.update(fields)
    return Track(**values)


def context(rule, ts, people, *, status="observed", size=(400, 240), cam=None):
    return FrameContext(camera=cam or rule.camera, ts=ts, frame_idx=0,
                        frame=np.zeros((size[1], size[0], 3), np.uint8), people=people,
                        model_observations={} if status is None else {"det": status})


def trajectory(rule, *, start=0., xs=None, ys=None, size=(400, 240), first_seen=0.):
    xs = xs if xs is not None else [250 - 6 * i for i in range(7)]
    ys = ys if ys is not None else [150] * len(xs)
    return [rule.evaluate(context(rule, start + i / 5,
            [track(start + i / 5, x, y, first_seen=first_seen)], size=size))
            for i, (x, y) in enumerate(zip(xs, ys))]


def test_registry_exposes_required_direction_and_det():
    assert get_rule("wrong_way") is WrongWayRule
    meta = next(row for row in available_rules() if row["type"] == "wrong_way")
    assert meta["required_models"] == ["det"]
    assert meta["params_schema"]["required"] == ["direction"]
    assert meta["params_schema"]["additionalProperties"] is False


@pytest.mark.parametrize("direction", [None, [], [[0, 0]], [[0, 0], [0, 0]],
    [[0, 0], [1, 2]], [[False, 0], [1, 1]], [[0, 0], [float("nan"), 1]], [[0, 0], [1, 1, 1]]])
def test_direction_is_required_and_strict(direction):
    with pytest.raises(ValueError, match="direction"):
        build(direction=direction)


@pytest.mark.parametrize("selection", [None, [], ["missing"], ["flow", "flow"]])
def test_requires_one_existing_zone(selection):
    cam = camera()
    cam.rules[0].zones = selection
    with pytest.raises(ValueError, match="zone"):
        build(cam)


@pytest.mark.parametrize("params", [{"window_s": 0}, {"window_s": 11}, {"max_gap_s": 3},
    {"min_observed_s": 3}, {"min_conf": float("nan")}, {"min_conf": 1.1},
    {"min_box_h_px": True}, {"max_angle_deg": 90}, {"min_progress_ratio": 0},
    {"min_path_bh": float("inf")}, {"min_displacement_bh": -1}, {"typo": 1}])
def test_numeric_params_are_finite_and_bounded(params):
    with pytest.raises(ValueError, match="wrong_way"):
        build(**params)


def test_invalid_zone_and_schedule_fail_at_construction():
    cam = camera()
    cam.zones[0].polygon = [(0, 0), (0, 0), (1, 1)]
    with pytest.raises(ValueError, match="area"):
        build(cam)
    cam = camera()
    cam.zones[0].active_hours = ["malformed"]
    with pytest.raises(ValueError, match="active_hours"):
        build(cam)


def test_sustained_opposite_motion_records_recomputable_evidence():
    rule = build()
    result = trajectory(rule)
    assert all(not row for row in result[:4])
    assert all(len(row) == 1 for row in result[4:])
    violation = result[4][0]
    assert (violation.type, violation.track_id, violation.zone_id) == ("wrong_way", 1, "flow")
    evidence = violation.details
    assert evidence["observed_s"] == pytest.approx(.8)
    assert evidence["net_opposite_projection_bh"] == pytest.approx(.3)
    assert evidence["net_displacement_bh"] == pytest.approx(.3)
    assert evidence["path_bh"] == pytest.approx(.3)
    assert evidence["progress_ratio"] == pytest.approx(1)
    assert evidence["opposite_angle_deg"] == pytest.approx(0)
    assert evidence["positions"][0] == {"ts": 0., "foot_px": [250., 150.], "box_height_px": 80.}
    assert evidence["distance_unit"] == "mean_bbox_height"
    assert "speed" not in evidence


@pytest.mark.parametrize(("xs", "ys"), [
    ([150 + 6 * i for i in range(12)], [150] * 12),  # allowed
    ([150] * 12, [100 + 6 * i for i in range(12)]),  # crosswise
    ([150] * 12, [150] * 12),  # still
    ([150 + (-1) ** i for i in range(12)], [150] * 12),  # bbox jitter
    ([150 - .2 * i for i in range(12)], [150] * 12),  # tiny monotonic drift
])
def test_nonviolating_paths_do_not_accumulate(xs, ys):
    assert not any(trajectory(build(), xs=xs, ys=ys))


def test_progress_ratio_can_reject_winding_but_consistently_opposite_path():
    rule = build(min_progress_ratio=.99)
    assert not any(trajectory(rule, xs=[250 - 6 * i for i in range(10)], ys=[150 + 3 * (i % 2) for i in range(10)]))


def test_non_square_frame_uses_arrow_pixel_angle():
    rule = build(direction=[[0, 0], [1, 1]], max_angle_deg=1)
    results = trajectory(rule, xs=[800 - 10 * i for i in range(7)],
                         ys=[600 - 5.625 * i for i in range(7)], size=(1280, 720))
    assert results[-1]
    assert results[-1][0].details["allowed_unit_vector_px"] == pytest.approx([.8715755371, .4902612396])


@pytest.mark.parametrize("next_xy", [(220, 150), (214, 156), (214, 150)])
def test_turn_crosswise_or_stop_immediately_removes_old_evidence(next_xy):
    rule = build()
    assert trajectory(rule)[-1]
    assert rule.evaluate(context(rule, 1.4, [track(1.4, *next_xy)])) == []
    # New opposite movement must earn the entire interval again.
    assert rule.evaluate(context(rule, 1.6, [track(1.6, next_xy[0] - 6, next_xy[1])])) == []


@pytest.mark.parametrize("status", [None, "not_observable", "unsupported", "error", "invalid"])
def test_missing_or_failed_model_observation_clears_history(status):
    rule = build()
    assert trajectory(rule)[-1]
    ctx = context(rule, 1.4, [track(1.4, 208)], status=status)
    assert rule.evaluate(ctx) == [] and rule._history == {}
    expected = "not_observable" if status is None else "error" if status == "invalid" else status
    assert ctx.rule_observations == {"wrong-1": expected}
    assert rule.evaluate(context(rule, 1.6, [track(1.6, 202)])) == []


@pytest.mark.parametrize("bad_fields", [{"last_seen": 1.2}, {"first_seen": 2}, {"conf": .1},
    {"conf": float("nan")}, {"bbox": (10, 10, 20, 20)}, {"bbox": (20, 10, 10, 150)},
    {"bbox": (0, 0, float("inf"), 150)}, {"cls": "car"}])
def test_invalid_or_stale_track_breaks_interval(bad_fields):
    rule = build()
    assert trajectory(rule)[-1]
    ctx = context(rule, 1.4, [track(1.4, 208, **bad_fields)])
    assert rule.evaluate(ctx) == [] and not rule._history
    assert ctx.rule_observations["wrong-1"] == "not_observable"
    assert rule.evaluate(context(rule, 1.6, [track(1.6, 202)])) == []


def test_occlusion_empty_frame_gap_and_reused_identity_reset_evidence():
    for interrupted in ("empty", "gap", "new_identity"):
        rule = build()
        assert trajectory(rule)[-1]
        if interrupted == "empty":
            ctx = context(rule, 1.4, [])
            assert not rule.evaluate(ctx)
            assert ctx.rule_observations["wrong-1"] == "observed"
            assert not rule.evaluate(context(rule, 1.6, [track(1.6, 202)]))
        elif interrupted == "gap":
            assert not rule.evaluate(context(rule, 2., [track(2., 202)]))
        else:
            assert not rule.evaluate(context(rule, 1.4, [track(1.4, 208, first_seen=1.4)]))


@pytest.mark.parametrize("ts", [1.2, .8, float("nan"), float("inf")])
def test_duplicate_backward_and_invalid_clock_do_not_count_twice(ts):
    rule = build()
    assert trajectory(rule)[-1]
    assert not rule.evaluate(context(rule, ts, [track(ts, 208)]))
    assert not rule._history
    assert not rule.evaluate(context(rule, 1.6, [track(1.6, 202)]))


def test_frame_resize_and_foreign_camera_clear_history():
    rule = build()
    assert trajectory(rule)[-1]
    assert not rule.evaluate(context(rule, 1.4, [track(1.4, 208)], size=(800, 480)))
    assert not rule.evaluate(context(rule, 1.6, [track(1.6, 202)], cam=camera("b")))
    assert rule._history == {}


def test_zone_exit_and_inactive_schedule_clear_history():
    rule = build()
    assert trajectory(rule)[-1]
    assert not rule.evaluate(context(rule, 1.4, [track(1.4, -20)]))
    assert not rule.evaluate(context(rule, 1.6, [track(1.6, 202)]))
    cam = camera()
    cam.zones[0].active_hours = ["12:00-13:00"]
    rule = build(cam)
    start = datetime(2026, 1, 1, 12, 59, 58, tzinfo=ZoneInfo("Europe/Moscow")).timestamp()
    assert trajectory(rule, start=start, first_seen=start)[-1]
    ctx = context(rule, start + 2, [track(start + 2, 202, first_seen=start)])
    assert not rule.evaluate(ctx) and not rule._history
    assert ctx.rule_observations["wrong-1"] == "not_observable"


def test_snapshots_do_not_reuse_mutable_track_history():
    rule = build()
    current = track(0)
    current.history.extend([(i / 5, 300 - i * 10, 150, 80) for i in range(10)])
    assert not rule.evaluate(context(rule, 0, [current]))
    current.bbox = (0, 0, 1000, 1000)
    current.history.clear()
    assert list(rule._history[(1, 0)])[0].x == 250


def test_state_is_bounded_by_current_people_window_and_sample_cap():
    rule = build(window_s=10, min_observed_s=.8)
    for i in range(600):
        ts = i / 1000
        rule.evaluate(context(rule, ts, [track(ts, 250 - i / 100, track_id=1)]))
    assert len(rule._history[(1, 0)]) == rule.MAX_POINTS
    ctx = context(rule, 1., [track(1., track_id=i) for i in range(rule.MAX_TRACKS + 1)])
    assert not rule.evaluate(ctx) and not rule._history
    assert ctx.rule_observations["wrong-1"] == "not_observable"
    for i in range(300):
        ts = 2 + i / 5
        rule.evaluate(context(rule, ts, [track(ts, track_id=i, first_seen=ts)]))
        assert len(rule._history) == 1
    rule.evaluate(context(rule, 100, []))
    assert rule._history == {}


def test_small_drift_cannot_accumulate_across_expired_window():
    rule = build()
    results = trajectory(rule, xs=[300 - .3 * i for i in range(250)])
    assert not any(results)
    history = rule._history[(1, 0)]
    assert history[-1].ts - history[0].ts <= rule.window_s + 1e-9


def test_duplicate_ids_fail_closed_without_a_second_vote():
    rule = build()
    ctx = context(rule, 0, [track(0), track(0, 244)])
    assert not rule.evaluate(ctx) and not rule._history
    assert ctx.rule_observations["wrong-1"] == "error"


def test_real_tracker_manager_and_journal_confirm_once_and_close_after_turn(tmp_path):
    cam, manager = camera(), EventManager()
    rule = build(cam)
    tracker = Tracker({"person"}, min_hits=1)
    created_ids, closed_ids = [], []
    with Journal(tmp_path / "events.db", replication_edge_id="test") as journal:
        for i, x in enumerate([250 - i * 6 for i in range(12)] + [190, 196, 202, 208]):
            ts = i / 5
            people = tracker.update([person((x - 15, 70, x + 15, 150))], ts)
            violations = rule.evaluate(context(rule, ts, people))
            new, closed, reopened = manager.process_with_updates(cam, ts, violations)
            assert not reopened
            for event in new:
                journal.add(event)
                journal.update(event.id, status="confirmed", operator_note="reviewed")
                created_ids.append(event.id)
            for event in closed:
                journal.update(event.id, ended_at=event.ended_at,
                               details_patch={key: event.details[key] for key in ("active_duration_s", "reopened_count") if key in event.details})
                closed_ids.append(event.id)
        assert len(created_ids) == 1 and closed_ids == created_ids
        event = journal.get(created_ids[0])
        assert event.ended_at == pytest.approx(2.2)
        assert event.status == "confirmed" and event.operator_note == "reviewed"
        assert event.details["positions"] and event.details["distance_unit"] == "mean_bbox_height"
        assert journal.query()[1] == 1
        assert len(journal.pending_messages()) == 3  # create, operator decision, close


def test_real_tracker_occlusion_does_not_bridge_old_motion():
    rule, tracker = build(), Tracker({"person"}, min_hits=1)
    for i in range(7):
        ts, x = i / 5, 250 - 6 * i
        people = tracker.update([person((x - 15, 70, x + 15, 150))], ts)
        result = rule.evaluate(context(rule, ts, people))
    assert result and people[0].track_id == 1
    assert not rule.evaluate(context(rule, 1.4, tracker.update([], 1.4)))
    recovered = tracker.update([person((193, 70, 223, 150))], 1.6)
    assert recovered[0].track_id == 1
    assert not rule.evaluate(context(rule, 1.6, recovered))


def test_rule_instances_isolate_cameras_and_fresh_capture_state():
    first, second = build(camera("a")), build(camera("b"))
    assert trajectory(first)[-1]
    assert not second.evaluate(context(second, 1.4, [track(1.4, 208)]))
    # Production capture handling reconstructs each plugin with the same config.
    fresh = type(first)(first.cfg, first.camera)
    assert not fresh.evaluate(context(fresh, 1.4, [track(1.4, 208, first_seen=1.4)]))
    assert first._history and fresh._history is not first._history


@pytest.fixture
def worker_system(tmp_path):
    from types import SimpleNamespace
    from guard.pipeline.engine import PipelineEngine
    from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings
    from guard.pipeline.worker import CameraWorker

    settings = make_settings(tmp_path, autostart=False, clips_enabled=False, tracker_min_hits=1, edge_id="test")
    journal = Journal(settings.db_path, replication_edge_id="test")
    engine = PipelineEngine(settings, backend=FakeBackend(), journal=journal)
    published = []
    engine.bus.publish = published.append
    cam = camera()
    worker = CameraWorker(cam, engine)
    inputs = {"x": 250., "error": False}

    def detect(index, frame):
        if inputs["error"]:
            raise RuntimeError("synthetic failed inference")
        x = inputs["x"]
        return [person((x - 15, 70, x + 15, 150))]

    worker.detectors = {"det": ScriptedDetector(detect)}
    engine.cameras[cam.camera_id], engine.workers[cam.camera_id] = cam, worker

    def emit(capture, ts, x, *, error=False):
        inputs.update(x=x, error=error)
        worker.source.capture_id = capture
        worker.source._publish(np.zeros((240, 400, 3), np.uint8), 0, ts=ts, pos_s=ts)
        return worker.process(worker.source.latest())

    yield SimpleNamespace(worker=worker, journal=journal, engine=engine, emit=emit, published=published)
    engine.stop()


def test_production_capture_closes_durably_then_rebuilds_wrong_way_evidence(worker_system):
    from uuid import uuid4

    system = worker_system
    first, second = str(uuid4()), str(uuid4())
    for i in range(10):
        ctx = system.emit(first, i / 5, 250 - i * 6)
    assert ctx.model_observations == {"det": "observed"}
    assert ctx.rule_observations == {"wrong-1": "observed"}
    events, count = system.journal.query()
    assert count == 1
    event = events[0]
    system.engine.patch_event(event.id, status="confirmed", operator_note="keep capture A decision")
    old_rule, old_tracker = system.worker.rules[0], system.worker.person_tracker
    system.emit(second, 2., 250)
    assert system.worker.rules[0] is not old_rule
    assert system.worker.person_tracker is not old_tracker
    assert len(system.worker.rules[0]._history[(1, 2.)]) == 1
    assert system.journal.query()[1] == 1
    stored = system.journal.get(event.id)
    assert stored.ended_at == pytest.approx(1.8)
    assert stored.status == "confirmed" and stored.operator_note == "keep capture A decision"
    assert stored.details["origin"]["capture_id"] == first
    assert len(system.journal.pending_messages()) == 3  # A, operator update, durable boundary
    for i in range(1, 10):
        system.emit(second, 2. + i / 5, 250 - i * 6)
    events, count = system.journal.query()
    assert count == 2
    new = next(row for row in events if row.id != event.id)
    assert new.track_id == event.track_id == 1
    assert new.details["origin"]["capture_id"] == second
    assert new.details["positions"][0]["ts"] == 2.
    assert "reopened_count" not in new.details
    assert len(system.journal.pending_messages()) == 4
    assert [message["kind"] for message in system.published].count("event") == 2


def test_production_inference_error_is_unknown_and_cannot_bridge_motion(worker_system):
    from uuid import uuid4

    system, capture = worker_system, str(uuid4())
    for i in range(7):
        system.emit(capture, i / 5, 250 - i * 6)
    assert system.worker.rules[0]._history
    ctx = system.emit(capture, 1.4, 208, error=True)
    assert ctx.model_observations == {"det": "error"}
    assert ctx.rule_observations == {"wrong-1": "error"}
    assert system.worker.rules[0]._history == {}
    recovered = system.emit(capture, 1.6, 202)
    assert recovered.people[0].track_id == 1
    assert len(system.worker.rules[0]._history[(1, 0.)]) == 1
