"""One-shot EventManager output must survive transient/uncertain SQLite writes."""
import sqlite3
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from guard.contracts import CameraConfig, Event, FrameContext, SourceConfig
from guard.edge.runner import EdgeEngine
from guard.events.journal import Journal
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.settings import Settings


class Publisher:
    def close(self):
        pass

    def status(self):
        return {}


class RetrySignal:
    def __init__(self, interrupt=False):
        self.waits = []
        self.interrupt = interrupt

    def wait(self, delay):
        self.waits.append(delay)
        return self.interrupt


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, models_dir=tmp_path / 'models', cameras_config=tmp_path / 'cameras.yaml',
                        models_config=tmp_path / 'models.yaml', db_path=tmp_path / 'guard.db',
                        media_dir=tmp_path / 'media', frontend_dist=tmp_path / 'frontend', mode='edge', edge_id='test', autostart=False)
    backend = SimpleNamespace(name='synthetic', stats=lambda: {}, close=lambda: None)
    engine = EdgeEngine(settings, backend=backend, publisher=Publisher(), telemetry_publisher=Publisher())
    camera = CameraConfig(camera_id='cam', name='Camera', site='site', source=SourceConfig(type='file', uri='unused'))
    worker = SimpleNamespace(cfg=camera, _stop=RetrySignal())
    context = FrameContext(camera=camera, ts=102, frame_idx=1, frame=np.zeros((24, 32, 3), np.uint8))
    clips, snapshots = [], []
    save = engine.media.save_snapshot
    def snapshot(*args):
        snapshots.append(args[0].id)
        return save(*args)
    monkeypatch.setattr(engine.media, 'save_snapshot', snapshot)
    monkeypatch.setattr(engine.media, 'schedule_clip', lambda ev, **kwargs: clips.append(ev.id))
    yield engine, worker, context, clips, snapshots
    engine.journal.close()


def event(**kwargs):
    return Event(id='one', camera_id='cam', site='site', rule_id='zone', type='zone_intrusion', severity='high',
                 started_at=100, confirmed_at=102, bbox=(2, 2, 20, 20), **kwargs)


@pytest.mark.parametrize('after_commit', [False, True])
def test_new_write_retries_same_event_once_without_premature_success(runtime, monkeypatch, after_commit):
    engine, worker, context, clips, snapshots = runtime
    real_add = engine.journal.add
    calls = []
    def flaky(ev):
        calls.append(ev.model_dump())
        assert engine.events_created == engine.bus.published == 0 and clips == []
        if len(calls) == 1:
            if after_commit:
                real_add(ev)
            raise sqlite3.OperationalError('uncertain write')
        return real_add(ev)
    monkeypatch.setattr(engine.journal, 'add', flaky)
    engine.handle_events(worker, context, [event()], [], time.perf_counter())
    assert len(calls) == (1 if after_commit else 2)
    assert all(item == calls[0] for item in calls)
    assert snapshots == clips == ['one']
    assert engine.events_created == engine.bus.published == 1
    assert len(engine.journal.pending_messages()) == 1
    assert engine.journal.get('one').snapshot_path
    assert engine.persistence_status() == {'pending': 0, 'last_error': None}


@pytest.mark.parametrize('after_commit', [False, True])
def test_closed_write_retry_preserves_operator_and_one_revision(runtime, monkeypatch, after_commit):
    engine, worker, context, clips, snapshots = runtime
    engine.journal.add(event())
    engine.journal.update('one', status='confirmed', operator_note='Keep this')
    before = len(engine.journal.pending_messages())
    real_update = engine.journal.update
    calls = []
    def flaky(event_id, **fields):
        calls.append(fields)
        assert engine.bus.published == 0
        if len(calls) == 1:
            if after_commit:
                real_update(event_id, **fields)
            raise sqlite3.OperationalError('uncertain closure')
        return real_update(event_id, **fields)
    monkeypatch.setattr(engine.journal, 'update', flaky)
    engine.handle_events(worker, context, [], [event(ended_at=105)], time.perf_counter())
    assert len(calls) == (1 if after_commit else 2)
    assert all(fields == {'ended_at': 105} for fields in calls)
    current = engine.journal.get('one')
    assert current.ended_at == 105 and current.status == 'confirmed' and current.operator_note == 'Keep this'
    assert len(engine.journal.pending_messages()) == before + 1
    assert engine.bus.published == 1 and clips == snapshots == []


def test_interruption_is_visible_and_stop_can_retry_after_recovery(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    real_add = engine.journal.add
    def failed(_):
        raise sqlite3.OperationalError('disk unavailable')
    monkeypatch.setattr(engine.journal, 'add', failed)
    worker._stop.interrupt = True
    with pytest.raises(RuntimeError, match='pending revision'):
        engine.handle_events(worker, context, [event()], [], time.perf_counter())
    assert engine.persistence_status() == {'pending': 1, 'last_error': 'OperationalError'}
    assert engine.health()['status'] == 'error'
    assert engine.events_created == engine.bus.published == 0 and clips == []
    with pytest.raises(sqlite3.OperationalError):
        engine.stop()
    assert not engine._stopped and engine.journal.get('one') is None
    monkeypatch.setattr(engine.journal, 'add', real_add)
    engine.stop()
    assert engine._stopped and engine.events_created == engine.bus.published == 1
    assert snapshots == ['one'] and clips == []  # no new background media on shutdown
    with Journal(engine.settings.db_path, replication_edge_id='test') as reopened:
        assert reopened.get('one') and len(reopened.pending_messages()) == 1


def test_immutable_conflict_remains_visible_without_overwriting_operator(runtime):
    engine, worker, context, clips, snapshots = runtime
    engine.journal.add(event())
    engine.journal.update('one', status='confirmed', operator_note='Keep')
    conflicting = event().model_copy(update={'camera_id': 'different'})
    with pytest.raises(ValueError, match='identity'):
        engine.handle_events(worker, context, [conflicting], [], time.perf_counter())
    assert engine.persistence_status()['pending'] == 1
    assert engine.events_created == engine.bus.published == 0 and clips == []
    assert engine.journal.get('one').operator_note == 'Keep'
    assert len(engine.journal.pending_messages()) == 2


def test_shutdown_close_failure_keeps_worker_and_manager_until_recovery(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    original = event()
    engine.journal.add(original)
    resets = []
    engine.events = SimpleNamespace(open_events=lambda cid: [original], reset=lambda cid: resets.append(cid))
    worker.stop = lambda: None
    worker._thread = None
    engine.workers['cam'] = worker
    real_update = engine.journal.update
    def fail(*args, **kwargs):
        raise sqlite3.OperationalError('disk unavailable')
    monkeypatch.setattr(engine.journal, 'update', fail)
    with pytest.raises(sqlite3.OperationalError):
        engine.stop()
    assert engine.workers['cam'] is worker and resets == []
    assert engine.persistence_status()['pending'] == 1
    monkeypatch.setattr(engine.journal, 'update', real_update)
    engine.stop()
    assert resets == ['cam'] and not engine.workers
    with Journal(engine.settings.db_path, replication_edge_id='test') as reopened:
        assert reopened.get('one').ended_at is not None
        assert len(reopened.pending_messages()) == 2


def test_local_failed_add_has_no_success_effects(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    monkeypatch.setattr(engine, '_persist_new_event', lambda w, ev: PipelineEngine._persist_new_event(engine, w, ev))
    monkeypatch.setattr(engine.journal, 'add', lambda _: (_ for _ in ()).throw(sqlite3.OperationalError()))
    engine.handle_events(worker, context, [event()], [], time.perf_counter())
    assert engine.events_created == engine.bus.published == 0 and clips == []


def test_interrupted_first_write_retains_whole_one_shot_batch(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    original_add = engine.journal.add
    monkeypatch.setattr(engine.journal, 'add', lambda _: (_ for _ in ()).throw(sqlite3.OperationalError()))
    worker._stop.interrupt = True
    second = event().model_copy(update={'id': 'two'})
    with pytest.raises(RuntimeError):
        engine.handle_events(worker, context, [event(), second], [], time.perf_counter())
    assert engine.persistence_status()['pending'] == 2
    monkeypatch.setattr(engine.journal, 'add', original_add)
    engine.stop()
    assert engine.events_created == engine.bus.published == 2
    with Journal(engine.settings.db_path, replication_edge_id='test') as reopened:
        assert reopened.get('one') and reopened.get('two')
        assert len(reopened.pending_messages()) == 2


def test_live_worker_timeout_keeps_journal_backend_and_pending_open(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    engine._remember_pending('new', event())
    alive = [True]
    worker._thread = SimpleNamespace(is_alive=lambda: alive[0])
    worker.stop = lambda: None
    engine.workers['cam'] = worker
    closed = []
    monkeypatch.setattr(engine.backend, 'close', lambda: closed.append('backend'))
    with pytest.raises(RuntimeError, match='did not stop'):
        engine.stop()
    assert not engine._stopped and closed == []
    assert engine.journal.get('one') is None and engine.persistence_status()['pending'] == 1
    alive[0] = False
    engine.stop()
    assert engine._stopped and closed == ['backend']
    with Journal(engine.settings.db_path, replication_edge_id='test') as reopened:
        assert reopened.get('one') and len(reopened.pending_messages()) == 1


def test_closed_identity_conflict_does_not_modify_existing_event(runtime):
    engine, worker, context, clips, snapshots = runtime
    engine.journal.add(event())
    conflicting = event(ended_at=105).model_copy(update={'rule_id': 'different'})
    with pytest.raises(ValueError, match='identity'):
        engine.handle_events(worker, context, [], [conflicting], time.perf_counter())
    assert engine.journal.get('one').ended_at is None
    assert engine.bus.published == 0 and len(engine.journal.pending_messages()) == 1
    assert engine.persistence_status()['pending'] == 1


def test_moving_track_closes_without_changing_confirmation_bbox(runtime):
    engine, worker, context, clips, snapshots = runtime
    original = event()
    engine.journal.add(original)
    engine.journal.update('one', status='confirmed', operator_note='Keep')
    moved = event(ended_at=105).model_copy(update={'bbox': (8, 8, 30, 22)})
    engine.handle_events(worker, context, [], [moved], time.perf_counter())
    current = engine.journal.get('one')
    assert current.ended_at == 105 and current.bbox == original.bbox
    assert current.status == 'confirmed' and current.operator_note == 'Keep'
    assert engine.bus.published == 1 and len(engine.journal.pending_messages()) == 3


def test_fail_once_shutdown_close_does_not_close_resources_before_retry(runtime, monkeypatch):
    engine, worker, context, clips, snapshots = runtime
    original = event()
    engine.journal.add(original)
    resets, backend_closes = [], []
    engine.events = SimpleNamespace(open_events=lambda cid: [original], reset=lambda cid: resets.append(cid))
    worker.stop, worker._thread = lambda: None, None
    engine.workers['cam'] = worker
    monkeypatch.setattr(engine.backend, 'close', lambda: backend_closes.append(True))
    real_update = engine.journal.update
    calls = []
    def fail_once(*args, **kwargs):
        calls.append(True)
        if len(calls) == 1:
            raise sqlite3.OperationalError('transient shutdown error')
        return real_update(*args, **kwargs)
    monkeypatch.setattr(engine.journal, 'update', fail_once)
    with pytest.raises(sqlite3.OperationalError):
        engine.stop()
    assert not engine._stopped and engine.workers and resets == [] and backend_closes == []
    assert engine.journal.get('one').ended_at is None
    engine.stop()
    assert engine._stopped and not engine.workers and resets == ['cam'] and backend_closes == [True]
    with Journal(engine.settings.db_path, replication_edge_id='test') as reopened:
        assert reopened.get('one').ended_at is not None and len(reopened.pending_messages()) == 2
