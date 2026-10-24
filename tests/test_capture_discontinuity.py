"""Capture boundaries use real state machines and isolated SQLite, no models."""

import pytest

from guard.contracts import CameraConfig, RuleConfig, SourceConfig, Violation
from guard.events.manager import EventManager
from guard.pipeline._fallbacks import SimpleEventManager


def camera(camera_id="a", **options):
    return CameraConfig(camera_id=camera_id, name=camera_id, site="unit", source=SourceConfig(type="file", uri="unused"),
        rules=[RuleConfig(rule_id="zone", type="zone_intrusion", min_duration_s=options.pop("min_duration_s", 0),
                          close_after_s=2, cooldown_s=60, min_ratio=1, **options)])


def violation(track_id=1):
    return Violation(rule_id="zone", type="zone_intrusion", severity="high", track_id=track_id, zone_id="danger",
                     bbox=(10, 10, 50, 90), details={"evidence": "keep"})


@pytest.mark.parametrize("manager_type", [EventManager, SimpleEventManager])
def test_capture_closures_are_frozen_at_last_violation_and_non_destructive(manager_type):
    manager, cfg = manager_type(), camera()
    created, _ = manager.process(cfg, 100, [violation()])
    manager.process(cfg, 101, [violation()])
    manager.process(cfg, 102, [])  # A later frame is not a later observation.
    closed = manager.capture_closures("a")
    assert len(closed) == 1 and closed[0].id == created[0].id
    assert closed[0].ended_at == 101
    assert closed[0].started_at == closed[0].confirmed_at == 100
    assert closed[0].details["evidence"] == "keep"
    closed[0].details["evidence"] = "mutated caller copy"
    assert manager.capture_closures("a")[0].details["evidence"] == "keep"
    assert manager.open_events("a")[0].ended_at is None
    manager.reset("a")
    assert manager.open_events("a") == manager.capture_closures("a") == []


@pytest.mark.parametrize("manager_type", [EventManager, SimpleEventManager])
def test_capture_reset_drops_candidates_and_prevents_reopen_at_same_position(manager_type):
    manager, cfg = manager_type(), camera(min_duration_s=2)
    assert manager.process(cfg, 0, [violation()])[0] == []
    assert manager.process(cfg, .2, [violation()])[0] == []
    assert manager.capture_closures("a") == []
    manager.reset("a")
    assert manager.process(cfg, 1.8, [violation()])[0] == []
    assert manager.process(cfg, 2, [violation()])[0] == []
    created, _ = manager.process(cfg, 3.8, [violation()])
    assert len(created) == 1 and created[0].started_at == 1.8
    old_id = created[0].id
    assert manager.capture_closures("a")[0].ended_at == 3.8
    manager.reset("a")
    assert manager.process(cfg, 4, [violation()])[0] == []
    created, _ = manager.process(cfg, 6, [violation()])
    assert len(created) == 1 and created[0].id != old_id and created[0].started_at == 4


@pytest.mark.parametrize("manager_type", [EventManager, SimpleEventManager])
def test_reset_is_camera_local_and_already_closed_events_need_no_second_closure(manager_type):
    manager = manager_type()
    first, second = camera("a"), camera("b")
    a = manager.process(first, 0, [violation()])[0][0]
    b = manager.process(second, 0, [violation()])[0][0]
    manager.process(first, 3, [])
    assert manager.capture_closures("a") == []
    manager.reset("a")
    again = manager.process(first, 0, [violation()])[0][0]
    assert again.id != a.id
    assert manager.open_events("b")[0].id == b.id
    assert manager.capture_closures("b")[0].ended_at == 0



def test_reopened_capture_closure_keeps_accumulated_active_duration():
    manager, cfg = EventManager(), camera()
    manager.process(cfg, 100, [violation()])
    manager.process(cfg, 101, [violation()])
    manager.process(cfg, 104, [])
    manager.process(cfg, 105, [violation()])
    manager.process(cfg, 106, [violation()])
    closed = manager.capture_closures("a")[0]
    assert closed.ended_at == 106
    assert closed.details["active_duration_s"] == 2 and closed.details["reopened_count"] == 1
    assert manager.open_events("a")[0].ended_at is None


@pytest.fixture
def system(tmp_path):
    import numpy as np
    from types import SimpleNamespace
    from guard.contracts import Detection, Zone
    from guard.events.journal import Journal
    from guard.pipeline.engine import PipelineEngine
    from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings, person
    from guard.pipeline.worker import CameraWorker
    from guard.rules.plugins.zone_intrusion import ZoneIntrusionRule

    class ObservedRule(ZoneIntrusionRule):
        def __init__(self, cfg, cam):
            super().__init__(cfg, cam)
            self.seen = []

        def evaluate(self, context):
            self.seen.append(context.ts)
            return super().evaluate(context)

    settings = make_settings(tmp_path, autostart=False, clips_enabled=False, tracker_min_hits=1, edge_id="unit-edge")
    journal = Journal(settings.db_path, replication_edge_id="unit-edge")
    engine = PipelineEngine(settings, backend=FakeBackend(), journal=journal)
    published = []
    engine.bus.publish = published.append

    def make_worker(camera_id="a", duration=1, two_people=False):
        cfg = camera(camera_id, min_duration_s=duration)
        cfg.zones = [Zone(zone_id="danger", name="danger", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])]
        cfg.rules[0].zones = ["danger"]
        worker = CameraWorker(cfg, engine)
        worker.rules = [ObservedRule(cfg.rules[0], cfg)]
        boxes = [(10, 10, 30, 90), (60, 10, 80, 90)] if two_people else [(10, 10, 50, 90)]
        detector = ScriptedDetector(lambda index, frame: [*[person(box) for box in boxes],
            Detection(model="det", cls="car", cls_id=2, conf=.9, bbox=(60, 10, 90, 40))])
        worker.detectors = {"det": detector}
        engine.cameras[camera_id], engine.workers[camera_id] = cfg, worker
        return worker

    def emit(worker, capture, ts, process=True, value=0):
        worker.source.capture_id = capture
        worker.source._publish(np.full((100, 100, 3), value, np.uint8), 0, ts=ts, pos_s=ts)
        packet = worker.source.latest()
        return worker.process(packet) if process else packet

    yield SimpleNamespace(engine=engine, journal=journal, published=published, worker=make_worker(),
                          make_worker=make_worker, emit=emit, base=100.)
    engine.stop()


def confirm(system, worker, capture, start=100.):
    for index in range(6):
        context = system.emit(worker, capture, start + index / 5)
    events, total = system.journal.query(camera_id=worker.cfg.camera_id)
    assert total >= 1
    return events[0], context


def test_worker_boundary_preserves_operator_evidence_but_clears_temporal_state(system):
    from uuid import uuid4

    worker, engine = system.worker, system.engine
    first, second = str(uuid4()), str(uuid4())
    old_event, old_ctx = confirm(system, worker, first)
    engine.patch_event(old_event.id, status="confirmed", operator_note="keep operator")
    old_people, old_objects, old_rule = worker.person_tracker, worker.object_tracker, worker.rules[0]
    old_ctx.people[0].attributes["old_person"] = True
    old_ctx.objects[0].attributes["old_object"] = True
    detector = worker.detectors["det"]
    worker.frame_jpeg()
    worker.overlay_jpeg()
    assert worker._jpeg_cache
    before_messages = len(system.published)
    context = system.emit(worker, second, 101.2)
    assert worker.person_tracker is not old_people and worker.object_tracker is not old_objects
    assert worker.rules[0] is not old_rule and worker.rules[0].seen == [101.2]
    assert context.people[0].hits == context.objects[0].hits == 1
    assert context.people[0].first_seen == context.objects[0].first_seen == 101.2
    assert "old_person" not in context.people[0].attributes and "old_object" not in context.objects[0].attributes
    assert not worker._jpeg_cache
    stored = system.journal.get(old_event.id)
    assert stored.ended_at == 101 and stored.details["active_duration_s"] == pytest.approx(1)
    assert stored.status == "confirmed" and stored.operator_note == "keep operator"
    assert stored.details["origin"]["capture_id"] == first
    assert stored.snapshot_path == old_event.snapshot_path and stored.bbox == old_event.bbox
    updates = system.published[before_messages:]
    assert len(updates) == 1 and updates[0]["kind"] == "event_update" and updates[0]["event"]["id"] == old_event.id
    for index in range(1, 6):
        system.emit(worker, second, 101.2 + index / 5)
    events, total = system.journal.query()
    assert total == engine.events_created == 2
    new = next(event for event in events if event.id != old_event.id)
    assert new.started_at == 101.2 and new.confirmed_at == pytest.approx(102.2)
    assert new.details["origin"]["capture_id"] == second and "reopened_count" not in new.details
    assert worker.detectors["det"] is detector and len(engine.backend.specs) == 1
    assert worker.frame_idx == worker.source.latest().seq == 12
    assert worker.metrics.snapshot()["frames_processed"] == 12


def test_same_capture_and_empty_legacy_packets_do_not_reset_state(system):
    worker = system.worker
    first, _ = confirm(system, worker, "")
    tracker, rule = worker.person_tracker, worker.rules[0]
    system.emit(worker, "first-known", 101.2)
    system.emit(worker, "", 101.4)
    system.emit(worker, "first-known", 101.6)
    assert worker.person_tracker is tracker and worker.rules[0] is rule
    assert system.journal.query()[1] == 1 and system.journal.get(first.id).ended_at is None
    assert worker.status()["capture_id"] == "first-known" and not worker.status()["capture_boundary_pending"]


def test_pending_candidate_does_not_cross_capture(system):
    worker = system.make_worker("candidate", duration=2)
    system.emit(worker, "old", 100)
    system.emit(worker, "old", 100.2)
    for index in range(10):
        system.emit(worker, "new", 100.4 + index / 5)
    assert system.journal.query(camera_id="candidate")[1] == 0
    system.emit(worker, "new", 102.4)
    created = system.journal.query(camera_id="candidate")[0][0]
    assert created.started_at == 100.4 and created.confirmed_at == 102.4


def test_capture_boundary_isolated_between_two_workers(system):
    first, second = system.worker, system.make_worker("b")
    a, _ = confirm(system, first, "capture-a")
    b, _ = confirm(system, second, "capture-b")
    tracker, rule = second.person_tracker, second.rules[0]
    system.emit(first, "capture-a-next", 101.2)
    system.emit(second, "capture-b", 101.2)
    assert system.journal.get(a.id).ended_at == 101
    assert system.journal.get(b.id).ended_at is None
    assert second.person_tracker is tracker and second.rules[0] is rule
    assert system.engine.events.open_events("b")[0].id == b.id


@pytest.mark.parametrize("failure", ["before_commit", "after_commit_and_failed_readback"])
def test_closure_retry_is_frozen_idempotent_and_accepts_latest_capture(system, monkeypatch, failure):
    import sqlite3

    worker = system.worker
    event, _ = confirm(system, worker, "old")
    system.engine.patch_event(event.id, status="confirmed", operator_note="operator")
    baseline = len(system.journal.pending_messages())
    system.published.clear()
    original_update, original_get = system.journal.update, system.journal.get
    attempts, fail_read = [], [False]

    def update(event_id, **fields):
        if fields.get("ended_at") is not None:
            attempts.append(event_id)
            if len(attempts) == 1:
                if failure == "before_commit":
                    raise sqlite3.OperationalError("write failed before commit")
                original_update(event_id, **fields)
                fail_read[0] = True
                raise sqlite3.OperationalError("commit outcome uncertain")
        return original_update(event_id, **fields)

    def get(event_id):
        if fail_read[0]:
            fail_read[0] = False
            raise sqlite3.OperationalError("readback temporarily unavailable")
        return original_get(event_id)

    monkeypatch.setattr(system.journal, "update", update)
    monkeypatch.setattr(system.journal, "get", get)
    tracker, calls = worker.person_tracker, worker.detectors["det"].n
    with pytest.raises((RuntimeError, sqlite3.OperationalError)):
        system.emit(worker, "intermediate", 101.2)
    assert worker.capture_boundary_pending and "retryable" in worker.status()["errors"]["capture"]
    assert worker.person_tracker is tracker and worker.detectors["det"].n == calls
    assert system.engine.events.open_events("a")[0].id == event.id and not system.published
    system.emit(worker, "latest", 101.4)
    assert not worker.capture_boundary_pending and worker.status()["capture_id"] == "latest"
    stored = system.journal.get(event.id)
    assert stored.ended_at == 101 and stored.status == "confirmed" and stored.operator_note == "operator"
    assert stored.details["origin"]["capture_id"] == "old"
    assert len(system.journal.pending_messages()) == baseline + 1
    assert len(system.published) == 1 and system.published[0]["event"]["id"] == event.id
    assert len(attempts) == (2 if failure == "before_commit" else 1)
    assert worker.rules[0].seen == [101.4]
    assert system.engine.events_created == 1 and "capture" not in worker.errors


def test_partial_closure_batch_does_not_duplicate_acknowledged_revision_or_notification(system, monkeypatch):
    import sqlite3

    worker = system.make_worker("multiple", two_people=True)
    confirm(system, worker, "old")
    events = system.journal.query(camera_id="multiple")[0]
    assert len(events) == 2
    baseline = len(system.journal.pending_messages())
    system.published.clear()
    original = system.journal.update
    calls = []

    def update(event_id, **fields):
        if fields.get("ended_at") is not None:
            calls.append(event_id)
            if len(calls) == 2:
                raise sqlite3.OperationalError("second closure unavailable")
        return original(event_id, **fields)

    monkeypatch.setattr(system.journal, "update", update)
    with pytest.raises(RuntimeError, match="not persisted"):
        system.emit(worker, "new", 101.2)
    assert worker.capture_boundary_pending and len(worker._pending_capture_closures) == 1
    assert len(system.published) == 1
    system.emit(worker, "new", 101.4)
    assert len(system.published) == 2 and len({message["event"]["id"] for message in system.published}) == 2
    assert len(system.journal.pending_messages()) == baseline + 2
    assert all(system.journal.get(event.id).ended_at == 101 for event in events)


def test_stop_failure_keeps_worker_journal_and_backend_available_for_recovery(system, monkeypatch):
    import sqlite3

    worker, engine = system.worker, system.engine
    event, _ = confirm(system, worker, "old")
    original = system.journal.update

    def fail(event_id, **fields):
        raise sqlite3.OperationalError("storage unavailable")

    monkeypatch.setattr(system.journal, "update", fail)
    with pytest.raises(RuntimeError):
        system.emit(worker, "new", 101.2)
    closed_backend = []
    monkeypatch.setattr(engine.backend, "close", lambda: closed_backend.append(True))
    with pytest.raises(RuntimeError, match="not persisted"):
        engine.stop()
    assert engine.workers["a"] is worker and worker.capture_boundary_pending
    assert system.journal.get(event.id).ended_at is None and not closed_backend
    monkeypatch.setattr(system.journal, "update", original)
    engine._stop_worker("a")
    assert system.journal.get(event.id).ended_at == 101 and not worker.capture_boundary_pending
    assert "a" not in engine.workers
    engine.stop()
    assert closed_backend == [True]



@pytest.mark.parametrize("failure", ["before_commit", "after_commit"])
def test_real_edge_capture_retry_has_one_outbox_revision_and_clears_pending(tmp_path, monkeypatch, failure):
    import sqlite3
    import numpy as np
    from guard.edge.runner import EdgeEngine
    from guard.pipeline.testing import FakeBackend, ScriptedDetector, make_settings
    from guard.pipeline.worker import CameraWorker
    from guard.transport.fake import FakeBroker

    broker = FakeBroker()
    edge = EdgeEngine(make_settings(tmp_path, mode="edge", autostart=False, clips_enabled=False),
                      backend=FakeBackend(), publisher=broker, telemetry_publisher=broker)
    cfg = camera()
    worker = CameraWorker(cfg, edge)
    worker.detectors = {"det": ScriptedDetector(lambda index, frame: [])}
    edge.workers["a"] = worker
    published = []
    edge.bus.publish = published.append
    try:
        created = edge.events.process(cfg, 100, [violation()])[0][0]
        created.details["origin"] = {"capture_id": "old"}
        edge.journal.add(created)
        edge.patch_event(created.id, status="confirmed", operator_note="operator")
        published.clear()
        baseline = len(edge.journal.pending_messages())
        original, attempts = edge.journal.update, []

        def update(event_id, **fields):
            if fields.get("ended_at") is not None:
                attempts.append(event_id)
                if len(attempts) == 1:
                    if failure == "after_commit":
                        original(event_id, **fields)
                    raise sqlite3.OperationalError("transient/uncertain write")
            return original(event_id, **fields)

        monkeypatch.setattr(edge.journal, "update", update)
        worker.source.capture_id = "old"
        worker.source._publish(np.zeros((100, 100, 3), np.uint8), 0, ts=100)
        worker.process(worker.source.latest())
        worker.source.capture_id = "new"
        worker.source._publish(np.zeros((100, 100, 3), np.uint8), 0, ts=101)
        worker.process(worker.source.latest())
        stored = edge.journal.get(created.id)
        assert stored.ended_at == 100 and stored.status == "confirmed" and stored.operator_note == "operator"
        assert len(edge.journal.pending_messages()) == baseline + 1
        assert edge.persistence_status() == {"pending": 0, "last_error": None}
        assert len(published) == 1 and published[0]["event"]["id"] == created.id
        assert len(attempts) == (2 if failure == "before_commit" else 1)
    finally:
        edge.stop()


def test_join_timeout_keeps_pending_boundary_and_does_not_drain_a_running_worker(system, monkeypatch):
    worker, engine = system.worker, system.engine
    event, _ = confirm(system, worker, "old")
    original = engine.close_capture
    calls = []

    def unavailable(*args):
        calls.append(True)
        raise RuntimeError("pending closure")

    monkeypatch.setattr(engine, "close_capture", unavailable)
    with pytest.raises(RuntimeError, match="pending closure"):
        system.emit(worker, "new", 101.2)

    class Thread:
        alive = True

        def join(self, timeout):
            pass

        def is_alive(self):
            return self.alive

    thread = worker._thread = Thread()
    try:
        with pytest.raises(RuntimeError, match="did not stop"):
            engine.stop()
        assert calls == [True] and engine.workers["a"] is worker
        assert worker.capture_boundary_pending and system.journal.get(event.id).ended_at is None
    finally:
        thread.alive = False
        monkeypatch.setattr(engine, "close_capture", original)
        engine._stop_worker("a")


def test_source_callback_and_live_overlay_are_independent_of_pending_journal_boundary(system, monkeypatch):
    import threading
    from dataclasses import replace
    from guard.pipeline import compat

    worker, engine = system.worker, system.engine
    confirm(system, worker, "old")
    engine.settings = replace(engine.settings, clips_enabled=True)
    system.emit(worker, "old", 101.1, process=False, value=10)
    old_ring = engine.media.ring("a")
    packet = system.emit(worker, "intermediate", 101.2, process=False, value=100)
    entered, resume = threading.Event(), threading.Event()
    original, errors = engine.close_capture, []

    def blocked(*args):
        entered.set()
        assert resume.wait(3), "Boundary test was not resumed"
        return original(*args)

    def process():
        try:
            worker.process(packet)
        except Exception as exc:
            errors.append(exc)

    monkeypatch.setattr(engine, "close_capture", blocked)
    thread = threading.Thread(target=process)
    thread.start()
    try:
        assert entered.wait(2)
        latest = system.emit(worker, "latest", 101.4, process=False, value=200)
        assert latest.capture_id == "latest" and old_ring.sealed
        assert engine.media._capture_ids["a"] == "latest"
        drawn = []

        def draw(image, people, objects, zones, highlight):
            drawn.append((int(image[0, 0, 0]), people, objects, highlight))
            return image

        monkeypatch.setattr(compat, "safe_draw_overlay", draw)
        worker.overlay_jpeg(mode="live")
        assert drawn == [(200, [], [], {})]
        resume.set()
        thread.join(timeout=2)
        assert not thread.is_alive() and not errors
        worker.process(latest)
        assert worker.last.capture_id == "latest"
    finally:
        resume.set()
        thread.join(timeout=3)
