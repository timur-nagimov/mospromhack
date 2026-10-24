"""Continuous observed dwell: synthetic inputs and real production state machines."""

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Track, Zone
from guard.rules.base import available_rules, build_rules, get_rule, load_plugins
from guard.rules.geometry import LOCAL_TZ
from guard.rules.plugins.zone_dwell import ZoneDwellRule

BOX = (30., 10., 50., 80.)
POLYGON = [(.2, .6), (.8, .6), (.8, .9), (.2, .9)]


def camera(camera_id="a", *, params=None, zones=None, selected=("z",), **rule_options):
    return CameraConfig(camera_id=camera_id, name=camera_id, site="test",
        source=SourceConfig(type="file", uri="unused"),
        zones=zones if zones is not None else [Zone(zone_id="z", name="Рабочая зона", polygon=POLYGON)],
        rules=[RuleConfig(rule_id="dwell", type="zone_dwell", zones=selected,
                          params={"dwell_s": 1., **(params or {})}, **rule_options)])


def rule(cam=None):
    cam = cam or camera()
    return ZoneDwellRule(cam.rules[0], cam)


def person(ts, *, tid=1, first=0., **options):
    return Track(track_id=tid, cls="person", bbox=BOX, conf=.9, first_seen=first, last_seen=ts, **options)


def context(cam, ts, people=None, *, status="observed", size=100):
    return FrameContext(camera=cam, ts=ts, frame_idx=0, frame=np.zeros((size, size, 3), np.uint8),
        people=[person(ts)] if people is None else people,
        model_observations={} if status is None else {"det": status})


def observe(plugin, ts, people=None, **kwargs):
    ctx = context(plugin.camera, ts, people, **kwargs)
    return plugin.evaluate(ctx), ctx


def test_registry_and_api_publish_required_schema_without_enabling_defaults(tmp_path):
    from fastapi.testclient import TestClient
    from guard.api.app import create_app
    from guard.pipeline.engine import PipelineEngine
    from guard.pipeline.testing import FakeBackend, make_settings

    load_plugins()
    assert get_rule("zone_dwell") is ZoneDwellRule
    info = next(item for item in available_rules() if item["type"] == "zone_dwell")
    assert info["required_models"] == ["det"] and info["default_severity"] == "high"
    schema = info["params_schema"]
    assert schema["required"] == ["dwell_s"] and schema["additionalProperties"] is False
    assert schema["properties"]["min_box_h_px"]["default"] == 40
    assert schema["properties"]["max_gap_s"]["maximum"] == 60
    settings = make_settings(tmp_path, autostart=False)
    engine = PipelineEngine(settings, backend=FakeBackend())
    with TestClient(create_app(settings, engine=engine)) as client:
        exposed = next(item for item in client.get("/api/rules/types").json() if item["type"] == "zone_dwell")
        assert exposed == info
        assert client.get("/api/cameras").json() == []
        configured = camera()
        configured.enabled = False  # Exercise real CRUD without starting a media source.
        response = client.post("/api/cameras", json=configured.model_dump(mode="json"))
        assert response.status_code == 201, response.text
        assert response.json()["rules"][0]["params"] == {"dwell_s": 1.}
    cam = camera()
    cam.rules[0].enabled = False
    assert build_rules(cam) == []


@pytest.mark.parametrize("params", [
    {}, {"dwell_s": 0}, {"dwell_s": -1}, {"dwell_s": float("nan")},
    {"dwell_s": float("inf")}, {"dwell_s": True}, {"dwell_s": "1"},
    {"dwell_s": 1, "max_gap_s": 0}, {"dwell_s": 1, "max_gap_s": 61},
    {"dwell_s": 1, "max_gap_s": float("inf")}, {"dwell_s": 1, "min_conf": -1},
    {"dwell_s": 1, "min_conf": 1.01}, {"dwell_s": 1, "min_conf": False},
    {"dwell_s": 1, "min_box_h_px": -1}, {"dwell_s": 1, "min_box_h_px": 1.5},
    {"dwell_s": 1, "extra": 2},
])
def test_invalid_params_fail_closed(params):
    cam = camera()
    cam.rules[0].params = params
    with pytest.raises(ValueError):
        rule(cam)


@pytest.mark.parametrize("selected", [None, [], ["z", "z"], ["missing"]])
def test_exactly_one_existing_explicit_zone_is_required(selected):
    with pytest.raises(ValueError):
        rule(camera(selected=selected))


@pytest.mark.parametrize("polygon", [[], [(0, 0), (1, 1), (.5, .5)],
    [(0, 0), (0, 0), (1, 1)], [(0, 0), (1.1, 0), (1, 1)],
    [(0, 0), (float("nan"), 0), (1, 1)]])
def test_bad_zone_geometry_is_rejected(polygon):
    with pytest.raises(ValueError):
        rule(camera(zones=[Zone(zone_id="z", name="z", polygon=polygon)]))


def test_duplicate_zone_and_malformed_schedule_are_rejected():
    cam = camera()
    cam.zones.append(cam.zones[0].model_copy())
    with pytest.raises(ValueError):
        rule(cam)
    cam = camera()
    cam.zones[0].active_hours = ["invalid"]
    with pytest.raises(ValueError):
        rule(cam)


def test_threshold_uses_observed_zone_entry_not_track_age_and_is_inclusive():
    plugin = rule()
    assert observe(plugin, 10)[0] == []  # Track existed since zero, but entry is ten.
    assert observe(plugin, 10.999)[0] == []
    violations, ctx = observe(plugin, 11)
    assert ctx.rule_observations == {"dwell": "observed"}
    assert len(violations) == 1
    item = violations[0]
    assert (item.track_id, item.rule_id, item.zone_id, item.type) == (1, "dwell", "z", "zone_dwell")
    assert item.bbox == BOX and item.score == .9
    assert item.details["observed_inside_since"] == 10 and item.details["dwell_s"] == 1
    assert observe(plugin, 11.5)[0][0].details["dwell_s"] == 1.5


def test_footpoint_boundary_policy_and_leave_reentry_are_continuous_only():
    plugin = rule()
    edge = (30., 0., 50., 60.)  # Center outside, feet on the included zone boundary.
    for ts in (0., 1.):
        out, _ = observe(plugin, ts, [replace(person(ts), bbox=edge)])
    assert len(out) == 1
    outside = replace(person(1.5), bbox=(30., 20., 50., 95.))
    assert observe(plugin, 1.5, [outside])[0] == []  # Center inside, feet outside.
    assert observe(plugin, 2)[0] == []
    assert observe(plugin, 2.999)[0] == []
    assert len(observe(plugin, 3)[0]) == 1


@pytest.mark.parametrize("status, expected", [(None, "not_observable"), ("not_observable", "not_observable"),
    ("unsupported", "unsupported"), ("error", "error"), ("unrecognized", "error")])
def test_unknown_unavailable_and_error_frames_break_proof(status, expected):
    plugin = rule()
    observe(plugin, 0)
    out, ctx = observe(plugin, .5, status=status)
    assert out == [] and ctx.rule_observations == {"dwell": expected}
    assert observe(plugin, 1)[0] == []
    assert observe(plugin, 1.5)[0] == []
    assert len(observe(plugin, 2)[0]) == 1


@pytest.mark.parametrize("changes", [
    {"last_seen": 0}, {"conf": .349}, {"conf": float("nan")}, {"conf": 1.1},
    {"bbox": (30, 50, 50, 80)}, {"bbox": (50, 10, 30, 80)},
    {"bbox": (-1, 10, 50, 80)}, {"bbox": (30, 10, 101, 80)},
    {"bbox": (30, 10, 50, float("inf"))}, {"bbox": (1, 2, 3)},
    {"bbox": None}, {"cls": "car"}, {"first_seen": 2},
    {"first_seen": float("nan")}, {"track_id": True}, {"track_id": -1},
])
def test_ineligible_track_drops_its_continuity(changes):
    plugin = rule()
    observe(plugin, 0)
    assert observe(plugin, .5, [replace(person(.5), **changes)])[0] == []
    assert observe(plugin, 1)[0] == []
    assert len(observe(plugin, 2)[0]) == 1


def test_empty_frame_duplicate_track_id_and_recycled_id_break_continuity():
    plugin = rule()
    observe(plugin, 0)
    assert observe(plugin, .5, [person(.5), person(.5)])[0] == []
    assert observe(plugin, 1)[0] == []
    assert observe(plugin, 1.5, [person(1.5, first=1.5)])[0] == []
    assert observe(plugin, 2, [person(2, first=1.5)])[0] == []
    assert len(observe(plugin, 2.5, [person(2.5, first=1.5)])[0]) == 1
    assert observe(plugin, 3, [])[0] == []
    assert observe(plugin, 3.5, [person(3.5, first=1.5)])[0] == []
    assert len(plugin._inside_since) == 1
    observe(plugin, 4, [])
    assert plugin._inside_since == {}


@pytest.mark.parametrize("bad_ts", [0, -.1, float("nan"), float("inf"), 1e30])
def test_bad_or_duplicate_clock_cannot_add_time(bad_ts):
    plugin = rule()
    observe(plugin, 0)
    out, ctx = observe(plugin, bad_ts)
    assert out == [] and ctx.rule_observations == {"dwell": "not_observable"}
    assert observe(plugin, 1)[0] == []
    assert len(observe(plugin, 2)[0]) == 1


def test_large_source_gap_and_resolution_change_restart_interval():
    plugin = rule()
    observe(plugin, 0)
    assert observe(plugin, 2)[0] == []
    assert observe(plugin, 2.5, size=200)[0] == []
    # The same pixel bbox now has its feet outside the normalized zone.
    scaled = (60., 20., 100., 160.)
    assert observe(plugin, 3, [replace(person(3), bbox=scaled)], size=200)[0] == []
    assert len(observe(plugin, 4, [replace(person(4), bbox=scaled)], size=200)[0]) == 1


def test_configured_visibility_and_exact_gap_limits_are_inclusive():
    plugin = rule(camera(params={"min_conf": .6, "min_box_h_px": 40, "max_gap_s": 1}))
    for ts in (0, 1):
        out, _ = observe(plugin, ts, [replace(person(ts), conf=.6, bbox=(30., 40., 50., 80.))])
    assert len(out) == 1
    ctx = context(plugin.camera, 1.5, size=0)
    assert plugin.evaluate(ctx) == [] and ctx.rule_observations == {"dwell": "not_observable"}
    assert observe(plugin, 2)[0] == []
    assert len(observe(plugin, 3)[0]) == 1


def test_two_people_zones_and_cameras_have_independent_state():
    a, b = rule(), rule(camera("b"))
    cam = camera(zones=[Zone(zone_id="z", name="left", polygon=[(0, 0), (.5, 0), (.5, 1), (0, 1)]),
                        Zone(zone_id="r", name="right", polygon=[(.5, 0), (1, 0), (1, 1), (.5, 1)])])
    left = rule(cam)
    right_cfg = cam.rules[0].model_copy(update={"rule_id": "right", "zones": ["r"]}, deep=True)
    right = ZoneDwellRule(right_cfg, cam)
    observe(a, 0)
    assert a.evaluate(context(b.camera, .5)) == []  # Does not reset camera a.
    assert len(observe(a, 1)[0]) == 1 and observe(b, 1)[0] == []
    for ts in (10, 11):
        people = [person(ts), replace(person(ts, tid=2), bbox=(60., 10., 80., 80.))]
        l = left.evaluate(context(cam, ts, people))
        r = right.evaluate(context(cam, ts, people))
    assert [v.track_id for v in l] == [1] and [v.track_id for v in r] == [2]
    observe(a, 1.5, [person(1.5), person(1.5, tid=2)])
    assert [v.track_id for v in observe(a, 2)[0]] == [1]


def test_active_hours_use_capture_epoch_moscow_and_do_not_accumulate_inactive_time():
    cam = camera()
    cam.zones[0].active_hours = ["22:00-06:00"]
    plugin = rule(cam)
    start = datetime(2026, 10, 24, 21, 59, 59, tzinfo=LOCAL_TZ).timestamp()
    assert observe(plugin, start)[0] == []
    assert observe(plugin, start + 1)[0] == []
    assert len(observe(plugin, start + 2)[0]) == 1
    end = datetime(2026, 10, 25, 6, tzinfo=LOCAL_TZ).timestamp()
    assert observe(plugin, end - 1)[0] == []
    assert observe(plugin, end)[0] == []


@pytest.fixture
def system(tmp_path):
    from guard.events.journal import Journal
    from guard.pipeline.engine import PipelineEngine
    from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person as detection
    from guard.pipeline.worker import CameraWorker

    load_plugins()
    settings = make_settings(tmp_path, autostart=False, clips_enabled=False, tracker_min_hits=1, edge_id="dwell-edge")
    journal = Journal(settings.db_path, replication_edge_id="dwell-edge")
    engine = PipelineEngine(settings, backend=FakeBackend(), journal=journal)
    published = []
    engine.bus.publish = published.append

    def make_worker(camera_id="a"):
        cam = camera(camera_id, min_duration_s=.5, min_ratio=1, close_after_s=.5, cooldown_s=30)
        worker = CameraWorker(cam, engine)
        worker.test_boxes = [BOX]
        worker.test_error = False

        def script(index, frame):
            if worker.test_error:
                raise RuntimeError("test model unavailable")
            return [detection(box) for box in worker.test_boxes]

        worker.detectors = {"det": ScriptedDetector(script)}
        engine.cameras[camera_id], engine.workers[camera_id] = cam, worker
        assert len(worker.rules) == 1 and isinstance(worker.rules[0], ZoneDwellRule)
        return worker

    def emit(worker, capture, ts):
        worker.source.capture_id = capture
        worker.source._publish(np.zeros((100, 100, 3), np.uint8), 0, ts=ts, pos_s=ts)
        return worker.process(worker.source.latest())

    yield SimpleNamespace(engine=engine, journal=journal, published=published,
                          make_worker=make_worker, emit=emit, worker=make_worker())
    engine.stop()


def test_real_tracker_rule_manager_journal_threshold_reopen_preserves_operator_outbox(system):
    worker, engine, journal = system.worker, system.engine, system.journal
    capture = str(uuid4())
    for ts in (100., 100.5, 101.):
        ctx = system.emit(worker, capture, ts)
        assert journal.query()[1] == 0
        assert ctx.model_observations == {"det": "observed"}
        assert ctx.rule_observations == {"dwell": "observed"}
    system.emit(worker, capture, 101.5)
    event = journal.query()[0][0]
    assert event.started_at == 101 and event.confirmed_at == 101.5
    assert event.details["observed_inside_since"] == 100
    assert event.details["dwell_s"] == 1.5 and engine.events_created == 1
    journal.update(event.id, status="confirmed", operator_note="Проверено", clip_path="keep.mp4",
                   details={**event.details, "operator_evidence": "keep"})
    system.emit(worker, capture, 102.)
    worker.test_boxes = []
    system.emit(worker, capture, 104.)
    assert journal.get(event.id).ended_at == 102
    worker.test_boxes = [BOX]
    for ts in (104.5, 105.):
        system.emit(worker, capture, ts)
        assert journal.get(event.id).ended_at == 102
    system.emit(worker, capture, 105.5)
    reopened = journal.get(event.id)
    assert reopened.ended_at is None and reopened.details["reopened_count"] == 1
    assert reopened.started_at == 101 and reopened.confirmed_at == 101.5
    assert reopened.status == "confirmed" and reopened.operator_note == "Проверено"
    assert reopened.clip_path == "keep.mp4" and reopened.details["operator_evidence"] == "keep"
    assert reopened.snapshot_path == event.snapshot_path and reopened.details["origin"] == event.details["origin"]
    assert journal.query()[1] == engine.events_created == 1
    update = journal.pending_messages()[-1].payload["event"]
    assert update["id"] == event.id and update["ended_at"] is None
    assert update["operator_note"] == "Проверено" and update["details"]["reopened_count"] == 1
    assert [message["kind"] for message in system.published].count("event") == 1


def test_real_worker_error_and_occlusion_do_not_bridge_dwell(system):
    worker, capture = system.worker, str(uuid4())
    system.emit(worker, capture, 100)
    worker.test_error = True
    failed = system.emit(worker, capture, 100.5)
    assert failed.rule_observations == {"dwell": "error"}
    worker.test_error = False
    for ts in (101, 101.5):
        system.emit(worker, capture, ts)
        assert system.journal.query()[1] == 0
    worker.test_boxes = []
    system.emit(worker, capture, 102)
    worker.test_boxes = [BOX]
    for ts in (102.5, 103, 103.5):
        system.emit(worker, capture, ts)
        assert system.journal.query()[1] == 0
    system.emit(worker, capture, 104)
    assert system.journal.query()[0][0].started_at == 103.5


def test_real_worker_capture_reset_closes_last_violation_and_is_camera_local(system):
    first, second, neighbour_capture = str(uuid4()), str(uuid4()), str(uuid4())
    worker, neighbour = system.worker, system.make_worker("b")
    for ts in (100, 100.5, 101, 101.5):
        system.emit(worker, first, ts)
        system.emit(neighbour, neighbour_capture, ts)
    old = system.journal.query(camera_id="a")[0][0]
    b_event = system.journal.query(camera_id="b")[0][0]
    old_rule, old_tracker = worker.rules[0], worker.person_tracker
    ctx = system.emit(worker, second, 102)
    assert worker.rules[0] is not old_rule and worker.person_tracker is not old_tracker
    assert ctx.people[0].first_seen == 102
    assert system.journal.get(old.id).ended_at == 101.5
    assert system.journal.get(b_event.id).ended_at is None
    for ts in (102.5, 103):
        system.emit(worker, second, ts)
        assert system.journal.query(camera_id="a")[1] == 1
    system.emit(worker, second, 103.5)
    events, total = system.journal.query(camera_id="a")
    assert total == 2
    new = next(item for item in events if item.id != old.id)
    assert new.started_at == 103 and new.confirmed_at == 103.5
    assert new.details["origin"]["capture_id"] == second
    assert "reopened_count" not in new.details


def test_real_tracker_old_person_entering_zone_needs_full_new_dwell(system):
    worker, capture = system.worker, str(uuid4())
    worker.test_boxes = [(30., 0., 50., 55.)]  # Same tracked person, feet initially above zone.
    first = system.emit(worker, capture, 100).people[0]
    for ts in (100.5, 101, 101.5):
        assert system.emit(worker, capture, ts).people[0].track_id == first.track_id
    worker.test_boxes = [BOX]
    for ts in (102, 102.5, 103):
        ctx = system.emit(worker, capture, ts)
        assert ctx.people[0].track_id == first.track_id and ctx.people[0].first_seen == 100
        assert system.journal.query()[1] == 0
    system.emit(worker, capture, 103.5)
    event = system.journal.query()[0][0]
    assert event.started_at == 103 and event.details["observed_inside_since"] == 102
