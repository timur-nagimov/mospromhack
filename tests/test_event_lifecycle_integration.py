"""Real manager/SQLite lifecycle integration without models, cameras or RabbitMQ."""

from __future__ import annotations

import copy
import sqlite3
import time
from dataclasses import dataclass
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from guard.contracts import CameraConfig, FrameContext, RuleConfig, SourceConfig, Violation
from guard.edge.runner import EdgeEngine
from guard.events.journal import Journal
from guard.events.manager import EventManager
from guard.gateway.engine import GatewayEngine
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.settings import Settings
from guard.transport.protocol import Envelope, decode, encode, event_payload


class NoInference:
    name = "synthetic-lifecycle"

    def __init__(self):
        self.closed = 0

    def load(self, spec):
        raise AssertionError("Lifecycle tests must not load models")

    def infer(self, *args):
        raise AssertionError("Lifecycle tests must not infer")

    def stats(self):
        return {"backend": self.name, "models": {}}

    def close(self):
        self.closed += 1


class OfflineChannel:
    def __init__(self):
        self.messages = []
        self.receive = None

    def publish(self, message, *, durable=False):
        assert durable
        wire = encode(message)
        self.messages.append(wire)
        if self.receive:
            self.receive(wire)

    def close(self):
        pass

    def status(self):
        return {"connected": False}


class RetrySignal:
    def __init__(self):
        self.interrupt = False
        self.waits = []

    def wait(self, delay):
        self.waits.append(delay)
        return self.interrupt


def settings(root, mode):
    root.mkdir(parents=True, exist_ok=True)
    (root / "models.yaml").write_text("models: []\n")
    return Settings(data_dir=root / "data", models_dir=root / "no-models", cameras_config=root / "no-cameras.yaml",
                    models_config=root / "models.yaml", db_path=root / "data/events.db", media_dir=root / "data/media",
                    frontend_dist=root / "no-frontend", mode=mode, edge_id="lifecycle-edge" if mode != "local" else None,
                    autostart=False, clips_enabled=True)


def violation(*, track=7, bbox=(2., 2., 22., 22.)):
    return Violation(rule_id="zone", type="zone_intrusion", severity="high", track_id=track,
                     zone_id="zone-1", bbox=bbox, details={"evidence": {"labels": ["original"]}})


def origin(journal, event_id):
    with journal._lock:
        row = journal._replication.origin(event_id)
        return None if row is None else dict(row)


@dataclass
class Runtime:
    engine: PipelineEngine
    worker: SimpleNamespace
    camera: CameraConfig
    messages: list
    snapshots: list
    clips: list
    publisher: OfflineChannel

    def handle(self, ts, outputs):
        new, closed, reopened = outputs
        ctx = FrameContext(camera=self.camera, ts=ts, frame_idx=round(ts * 5), frame=np.zeros((24, 32, 3), np.uint8))
        self.engine.handle_events(self.worker, ctx, new, closed, time.perf_counter(), reopened=reopened)

    def process(self, ts, observations):
        outputs = self.engine.events.process_with_updates(self.camera, ts, observations)
        self.handle(ts, outputs)
        return outputs


@pytest.fixture
def runtime_factory(tmp_path, monkeypatch):
    runtimes = []

    def make(mode, **rule_options):
        cfg = settings(tmp_path / f"{mode}-{len(runtimes)}", mode)
        manager, backend, publisher = EventManager(), NoInference(), OfflineChannel()
        rule = RuleConfig(rule_id="zone", type="zone_intrusion", **{
            "min_duration_s": 1., "close_after_s": 1., "cooldown_s": 30., **rule_options})
        camera = CameraConfig(camera_id="cam", name="Synthetic camera", site="Synthetic site",
                              source=SourceConfig(type="images", uri="unused"), rules=[rule])
        journal = Journal(cfg.db_path, replication_edge_id=cfg.edge_id if mode == "edge" else None)
        kwargs = {"backend": backend, "journal": journal, "event_manager": manager}
        engine = (EdgeEngine(cfg, publisher=publisher, telemetry_publisher=OfflineChannel(), **kwargs)
                  if mode == "edge" else PipelineEngine(cfg, **kwargs))
        worker = SimpleNamespace(cfg=camera, _stop=RetrySignal(), source=SimpleNamespace(native_fps=5.))
        messages, snapshots, clips = [], [], []
        publish, save = engine.bus.publish, engine.media.save_snapshot

        def record(message):
            messages.append(copy.deepcopy(message))
            publish(message)

        def snapshot(event, frame):
            snapshots.append(event.id)
            return save(event, frame)

        monkeypatch.setattr(engine.bus, "publish", record)
        monkeypatch.setattr(engine, "_render_snapshot", lambda context, event: context.frame.copy())
        monkeypatch.setattr(engine.media, "save_snapshot", snapshot)
        monkeypatch.setattr(engine.media, "schedule_clip", lambda event, **kwargs: clips.append(event.id))
        runtime = Runtime(engine, worker, camera, messages, snapshots, clips, publisher)
        runtimes.append(runtime)
        return runtime

    yield make
    for runtime in runtimes:
        runtime.engine.media.stop()
        runtime.engine.journal.close()


def confirmed(runtime):
    assert runtime.process(100., [violation()]) == ([], [], [])
    new, closed, reopened = runtime.process(101., [violation()])
    assert len(new) == 1 and closed == reopened == []
    event = runtime.engine.journal.get(new[0].id)
    runtime.engine.journal.update(event.id, status="confirmed", operator_note="Keep operator decision",
                                  clip_path="media/clips/original.mp4",
                                  details={**event.details, "operator_custom": {"labels": ["keep"]}, "clip_error": True})
    runtime.process(102., [violation()])
    return runtime.engine.journal.get(event.id)


def prior_closed(runtime):
    original = confirmed(runtime)
    created, closed, reopened = runtime.process(104.1, [])
    assert created == reopened == [] and len(closed) == 1
    assert runtime.engine.journal.get(original.id).ended_at == 102.
    return original


def assert_preserved(current, original):
    for field in ("id", "camera_id", "site", "rule_id", "type", "severity", "track_id", "zone_id",
                  "started_at", "confirmed_at", "bbox", "status", "operator_note", "clip_path", "snapshot_path", "latency_ms"):
        assert getattr(current, field) == getattr(original, field), field
    assert current.details["operator_custom"] == {"labels": ["keep"]}
    assert current.details["evidence"] == {"labels": ["original"]}
    assert current.details["clip_error"] is True


@pytest.mark.parametrize("mode", ["local", "edge"])
@pytest.mark.parametrize("same_frame", [False, True])
def test_reopen_preserves_durable_evidence_and_operator_fields_without_new_alarm(runtime_factory, mode, same_frame):
    runtime = runtime_factory(mode)
    original = confirmed(runtime) if same_frame else prior_closed(runtime)
    engine = runtime.engine
    before = len(engine.journal.pending_messages()) if mode == "edge" else 0
    count, alarms, snapshots, clips = engine.events_created, len(runtime.messages), len(runtime.snapshots), len(runtime.clips)
    producer = engine.journal.producer_id
    outputs = engine.events.process_with_updates(runtime.camera, 104.1 if same_frame else 105.,
                                                  [violation(bbox=(8., 3., 29., 23.))])
    new, closed, reopened = outputs
    assert not new and len(reopened) == 1 and len(closed) == int(same_frame)
    saved_closed = [event.model_dump() for event in closed]
    runtime.handle(104.1 if same_frame else 105., outputs)
    current = engine.journal.get(original.id)
    assert_preserved(current, original)
    assert current.ended_at is None
    assert current.details["reopened_count"] == 1 and current.details["active_duration_s"] == 2.
    assert engine.events_created == count == 1 and len(runtime.snapshots) == snapshots == 1
    assert len(runtime.clips) == clips == 1
    updates = runtime.messages[alarms:]
    assert len(updates) == 1 and updates[0]["kind"] == "event_update" and updates[0]["event"]["ended_at"] is None
    assert [event.model_dump() for event in closed] == saved_closed
    assert engine.events.open_events(runtime.camera.camera_id)[0].ended_at is None
    # Returned snapshots and nested details are caller-owned, not manager/store aliases.
    reopened[0].details["evidence"]["labels"].append("caller mutation")
    reopened[0].ended_at = 999.
    assert engine.events.open_events()[0].details["evidence"] == {"labels": ["original"]}
    assert engine.journal.get(original.id).ended_at is None
    if mode == "edge":
        pending = engine.journal.pending_messages()
        assert len(pending) == before + 1
        assert pending[-1].payload["event"]["ended_at"] is None
        assert pending[-1].payload["event"]["bbox"] == list(original.bbox)
        assert {message.producer_id for message in pending} == {producer}


@pytest.mark.parametrize("after_commit", [False, True])
def test_edge_uncertain_reopen_is_one_revision_and_announces_only_after_readback(runtime_factory, monkeypatch, after_commit):
    runtime = runtime_factory("edge")
    original = prior_closed(runtime)
    engine = runtime.engine
    outputs = engine.events.process_with_updates(runtime.camera, 105., [violation()])
    real_update, calls = engine.journal.update, []
    before, bus_before = len(engine.journal.pending_messages()), len(runtime.messages)

    def flaky(event_id, **fields):
        calls.append(copy.deepcopy(fields))
        assert len(runtime.messages) == bus_before and engine.events_created == 1
        assert runtime.snapshots == runtime.clips == [original.id]
        assert engine.persistence_status()["pending"] == 1
        if len(calls) == 1:
            if after_commit:
                real_update(event_id, **fields)
            raise sqlite3.OperationalError("uncertain reopen")
        return real_update(event_id, **fields)

    monkeypatch.setattr(engine.journal, "update", flaky)
    runtime.handle(105., outputs)
    assert len(calls) == (1 if after_commit else 2)
    assert all(call == {"ended_at": None, "details_patch": {"reopened_count": 1, "active_duration_s": 2.}} for call in calls)
    assert len(engine.journal.pending_messages()) == before + 1
    assert len(runtime.messages) == bus_before + 1 and runtime.messages[-1]["kind"] == "event_update"
    assert engine.persistence_status() == {"pending": 0, "last_error": None}
    assert_preserved(engine.journal.get(original.id), original)


def test_shutdown_retains_interrupted_reopen_copy_until_durable_recovery(runtime_factory, monkeypatch):
    runtime = runtime_factory("edge")
    original = prior_closed(runtime)
    engine = runtime.engine
    outputs = engine.events.process_with_updates(runtime.camera, 105., [violation()])
    before, bus_before = len(engine.journal.pending_messages()), len(runtime.messages)
    producer, real_update = engine.journal.producer_id, engine.journal.update

    def failed(*args, **kwargs):
        raise sqlite3.OperationalError("reopen storage unavailable")

    monkeypatch.setattr(engine.journal, "update", failed)
    runtime.worker._stop.interrupt = True
    with pytest.raises(RuntimeError, match="pending revision"):
        runtime.handle(105., outputs)
    assert engine.persistence_status() == {"pending": 1, "last_error": "OperationalError"}
    assert engine.journal.get(original.id).ended_at == 102.
    assert len(engine.journal.pending_messages()) == before and len(runtime.messages) == bus_before
    assert engine.events_created == 1 and runtime.clips == runtime.snapshots == [original.id]
    outputs[2][0].ended_at = 999.
    outputs[2][0].details["reopened_count"] = 999
    with pytest.raises(sqlite3.OperationalError):
        engine.stop()
    assert not engine._stopped and engine.backend.closed == 0
    assert engine.persistence_status()["pending"] == 1
    assert engine.journal.get(original.id).ended_at == 102.
    monkeypatch.setattr(engine.journal, "update", real_update)
    engine.stop()
    assert engine._stopped and engine.backend.closed == 1
    assert engine.persistence_status() == {"pending": 0, "last_error": None}
    assert len(runtime.messages) == bus_before + 1 and runtime.messages[-1]["kind"] == "event_update"
    assert engine.events_created == 1 and runtime.clips == runtime.snapshots == [original.id]
    with Journal(engine.settings.db_path, replication_edge_id=engine.settings.edge_id) as restarted:
        current = restarted.get(original.id)
        assert_preserved(current, original)
        assert current.ended_at is None and current.details["reopened_count"] == 1
        assert current.details["active_duration_s"] == 2.
        assert restarted.producer_id == producer and len(restarted.pending_messages()) == before + 1
        assert restarted.pending_messages()[-1].payload["event"]["ended_at"] is None


def test_gateway_commits_fresh_reopen_and_stale_close_replay_cannot_reclose(runtime_factory, tmp_path):
    runtime = runtime_factory("edge")
    cfg = settings(tmp_path / "gateway", "gateway")
    gateway = GatewayEngine(cfg, consumer=OfflineChannel())
    runtime.publisher.receive = gateway.receive
    try:
        original = prior_closed(runtime)
        assert runtime.engine.replication.publish_pending_once() >= 3
        closed_wire = runtime.publisher.messages[-1]
        closed = decode(closed_wire)
        assert gateway.journal.get(original.id).ended_at == 102.
        gateway.journal.update(original.id, status="confirmed", operator_note="Gateway operator decision")
        old_origin = origin(gateway.journal, original.id)
        bus_before = gateway.bus.published
        runtime.process(105., [violation(bbox=(8., 3., 29., 23.))])
        assert runtime.engine.journal.pending_messages()[-1].sequence > closed.sequence
        assert runtime.engine.replication.publish_pending_once() == 1
        current = gateway.journal.get(original.id)
        assert current.ended_at is None and current.status == "confirmed" and current.operator_note == "Gateway operator decision"
        assert current.bbox == original.bbox and current.started_at == original.started_at and current.confirmed_at == original.confirmed_at
        assert current.details["reopened_count"] == 1 and current.details["active_duration_s"] == 2.
        new_origin = origin(gateway.journal, original.id)
        assert {key: new_origin[key] for key in ("event_id", "edge_id", "producer_id")} == {
            key: old_origin[key] for key in ("event_id", "edge_id", "producer_id")}
        assert new_origin["sequence"] > old_origin["sequence"]
        assert gateway.bus.published == bus_before + 1
        gateway.receive(closed_wire)
        assert gateway.journal.get(original.id) == current and origin(gateway.journal, original.id) == new_origin
        assert gateway.bus.published == bus_before + 1
        # Receipts and latest open state survive an actual SQLite reopen.
        gateway.stop()
        gateway = GatewayEngine(cfg, consumer=OfflineChannel())
        gateway.receive(closed_wire)
        assert gateway.journal.get(original.id) == current and gateway.bus.published == 0
    finally:
        gateway.stop()


@pytest.mark.parametrize("mode", ["local", "edge"])
def test_empty_frame_candidate_reopen_then_reclose_exposes_only_final_closed_state(runtime_factory, mode):
    runtime = runtime_factory(mode, min_duration_s=1., min_ratio=.5, close_after_s=.1, cooldown_s=10.)
    assert runtime.process(0., [violation(track=1)]) == ([], [], [])
    created, _, _ = runtime.process(1., [violation(track=1), violation(track=2)])
    assert len(created) == 1
    event_id = created[0].id
    new, closed, reopened = runtime.engine.events.process_with_updates(runtime.camera, 2., [])
    assert not new and not reopened and len(closed) == 1
    assert closed[0].id == event_id and closed[0].ended_at == 1.
    assert closed[0].details["reopened_count"] == 1
    assert runtime.engine.events.open_events() == []
    runtime.handle(2., (new, closed, reopened))
    stored = runtime.engine.journal.get(event_id)
    assert stored.ended_at == closed[0].ended_at and stored.details["reopened_count"] == 1
    assert runtime.engine.events_created == 1
    assert runtime.messages[-1]["kind"] == "event_update" and runtime.messages[-1]["event"]["ended_at"] == 1.


def test_clip_error_atomic_patch_preserves_concurrent_reopen_details_and_origin(runtime_factory, monkeypatch):
    runtime = runtime_factory("local")
    manager = runtime.engine.events
    manager.process_with_updates(runtime.camera, 100., [violation()])
    created, _, _ = manager.process_with_updates(runtime.camera, 101., [violation()])
    _, closed, _ = manager.process_with_updates(runtime.camera, 104., [])
    assert len(created) == len(closed) == 1
    event = closed[0]
    producer = str(uuid4())
    message = Envelope(edge_id="lifecycle-edge", producer_id=producer, run_id=str(uuid4()), sequence=1,
                       message_id=f"{producer}:1", kind="event", sent_at=104., payload=event_payload(event))
    journal = runtime.engine.journal
    journal.apply_remote_event(message, expected_edge_id="lifecycle-edge")
    journal.update(event.id, status="confirmed", operator_note="Keep", clip_path="media/clips/operator.mp4",
                   details={**event.details, "operator_custom": {"keep": True}})
    initial = journal.get(event.id)
    original_origin = origin(journal, event.id)
    real_update, injected = journal.update, []

    def concurrent_reopen(event_id, **fields):
        assert fields == {"details_patch": {"clip_error": True}}
        if not injected:
            injected.append(True)
            with Journal(runtime.engine.settings.db_path) as second_connection:
                second_connection.update(event_id, ended_at=None, details_patch={"reopened_count": 1, "active_duration_s": 2.})
        return real_update(event_id, **fields)

    monkeypatch.setattr(journal, "update", concurrent_reopen)
    runtime.engine._on_clip(event, None)
    current = journal.get(event.id)
    assert injected == [True] and current.ended_at is None
    assert current.details == {**initial.details, "reopened_count": 1, "active_duration_s": 2., "clip_error": True}
    assert current.status == initial.status and current.operator_note == initial.operator_note
    assert current.clip_path == initial.clip_path and current.bbox == initial.bbox
    assert origin(journal, event.id) == original_origin
    assert runtime.engine.events_created == 0 and runtime.clips == runtime.snapshots == []
    assert runtime.messages[-1]["kind"] == "event_update" and runtime.messages[-1]["event"]["ended_at"] is None
