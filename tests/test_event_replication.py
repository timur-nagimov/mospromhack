"""Transactional durability, replay/reordering and concurrent operator edits."""

import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest

from guard.contracts import Event, EventStatus, Severity
from guard.events.journal import Journal
from guard.transport.protocol import Envelope, decode, encode, event_payload


def make_event(event_id="event-1", **changes):
    data = dict(id=event_id, camera_id="camera-1", site="Цех", rule_id="s1",
                type="zone_intrusion", severity=Severity.critical, started_at=100.,
                confirmed_at=102., bbox=(1., 2., 30., 40.))
    data.update(changes)
    return Event(**data)


def message(event=None, *, sequence=1, producer_id=None, edge_id="edge-1", kind="event", **changes):
    producer_id = producer_id or str(uuid4())
    data = dict(edge_id=edge_id, producer_id=producer_id, run_id=str(uuid4()),
                message_id=f"{producer_id}:{sequence}", kind=kind, sequence=sequence,
                sent_at=200., payload=event_payload(event or make_event()))
    data.update(changes)
    return Envelope(**data)


def test_plain_journal_preserves_local_behavior(tmp_path):
    with Journal(tmp_path / "local.db") as journal:
        original = make_event(status="confirmed", operator_note="Локальное событие")
        journal.add(original)
        assert journal.get(original.id) == original
        assert journal.pending_messages() == []
        assert journal.producer_id is None
        with pytest.raises(ValueError, match="not enabled"):
            journal.make_message("metrics", {"data": {}})


def test_event_and_updates_recover_after_reopen_and_confirm(tmp_path):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as edge:
        edge.add(make_event())
        edge.update("event-1", snapshot_path="snapshots/local.jpg")
        edge.update("event-1", ended_at=120.)
        before = edge.pending_messages()
        assert [item.kind for item in before] == ["event", "event_update", "event_update"]
        assert [item.sequence for item in before] == [1, 2, 3]
        assert before[1].payload["event"]["snapshot_path"] == "snapshots/local.jpg"
        assert before[2].payload["event"]["ended_at"] == 120.
        producer_id = edge.producer_id
    with Journal(path, replication_edge_id="edge-1") as edge:
        assert edge.producer_id == producer_id
        assert edge.pending_messages() == before
        edge.mark_published(before[0].message_id)
        edge.mark_published(before[0].message_id)
        assert edge.pending_messages(limit=1) == before[1:2]
        edge.add(make_event("event-2"))
        assert edge.pending_messages()[-1].sequence == 4
        assert edge.replication_stats()["pending"] == 3
        assert edge.replication_stats()["next_sequence"] == 5
    with pytest.raises(ValueError, match="another replication edge"):
        Journal(path, replication_edge_id="other-edge")


@pytest.mark.parametrize("operation", ["add", "update"])
def test_outbox_failure_rolls_back_event_and_sequence(tmp_path, operation):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as edge:
        if operation == "update":
            edge.add(make_event())
        previous = edge.replication_stats()
        # Inject an actual SQLite failure after event SQL and sequence allocation.
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER fail_outbox BEFORE INSERT ON transport_outbox "
                       "BEGIN SELECT RAISE(ABORT, 'injected outbox failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            if operation == "add":
                edge.add(make_event())
            else:
                edge.update("event-1", ended_at=120.)
        assert edge.replication_stats() == previous
        assert edge.get("event-1") == (None if operation == "add" else make_event())
        with sqlite3.connect(path) as db:
            db.execute("DROP TRIGGER fail_outbox")
        if operation == "add":
            edge.add(make_event())
        else:
            edge.update("event-1", ended_at=120.)
        assert edge.pending_messages()[-1].sequence == previous["next_sequence"]


def test_validation_failure_rolls_back_edge_event(tmp_path):
    with Journal(tmp_path / "edge.db", replication_edge_id="edge-1") as edge:
        with pytest.raises(ValueError):
            edge.add(make_event(camera_id="../not-a-camera"))
        assert edge.query()[1] == 0
        assert edge.replication_stats()["next_sequence"] == 1
        edge.add(make_event())
        with pytest.raises(ValueError):
            edge.update("event-1", details={"oversized": "x" * 3_000_000})
        with pytest.raises(ValueError):
            edge.update("event-1", details={"nonfinite": float("inf")})
        assert edge.get("event-1").details == {}
        assert len(edge.pending_messages()) == 1


def test_prepare_schema_migrates_older_outbox_without_losing_messages(tmp_path):
    path = tmp_path / "edge.db"
    original = message()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE transport_outbox(sequence INTEGER PRIMARY KEY, message_id TEXT UNIQUE NOT NULL, envelope BLOB NOT NULL)")
        db.execute("INSERT INTO transport_outbox VALUES (?, ?, ?)", (1, original.message_id, encode(original)))
    with Journal(path) as journal:
        assert journal.pending_messages() == [original]
        assert journal.prepare_message(original.message_id, snapshot_jpeg=None) == original


def test_transient_messages_allocate_global_sequences_without_outbox(tmp_path):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as edge:
        edge.add(make_event())
        transient = edge.make_message("metrics", {"data": {"frames": 3}})
        assert transient.sequence == 2
        assert transient.producer_id == edge.producer_id
        assert transient.run_id == edge.run_id
        with pytest.raises(ValueError):
            edge.make_message("metrics", {"data": []})
        edge.update("event-1", ended_at=120.)
        assert [item.sequence for item in edge.pending_messages()] == [1, 3]
    with Journal(path, replication_edge_id="edge-1") as edge:
        assert edge.make_message("metrics", {"data": {}}).sequence == 4


@pytest.mark.parametrize("first_snapshot", [None, b"\xff\xd8first\xff\xd9"])
def test_prepare_message_freezes_identical_wire_after_restart(tmp_path, first_snapshot):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as edge:
        edge.add(make_event(snapshot_path="snapshots/a.jpg"))
        message_id = edge.pending_messages()[0].message_id
        prepared = edge.prepare_message(message_id, snapshot_jpeg=first_snapshot)
        original_wire = encode(prepared)
        assert encode(edge.prepare_message(message_id, snapshot_jpeg=b"\xff\xd8other\xff\xd9")) == original_wire
    with Journal(path, replication_edge_id="edge-1") as edge:
        assert encode(edge.prepare_message(message_id, snapshot_jpeg=None)) == original_wire
        assert encode(edge.pending_messages()[0]) == original_wire
        edge.mark_published(message_id)
        with pytest.raises(KeyError):
            edge.prepare_message(message_id, snapshot_jpeg=None)


def test_invalid_preparation_rolls_back_and_can_retry(tmp_path):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as edge:
        edge.add(make_event())
        original = edge.pending_messages()[0]
        with pytest.raises(ValueError):
            edge.prepare_message(original.message_id, snapshot_jpeg=b"not jpeg")
        assert edge.pending_messages() == [original]
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT prepared FROM transport_outbox").fetchone()[0] == 0
        assert "snapshot_jpeg" in edge.prepare_message(original.message_id, snapshot_jpeg=b"\xff\xd8valid\xff\xd9").payload


@pytest.mark.parametrize("limit", [-1, True, 1.5, 10001])
def test_pending_limit_is_bounded(tmp_path, limit):
    with Journal(tmp_path / "edge.db") as journal:
        with pytest.raises(ValueError):
            journal.pending_messages(limit)


def test_gateway_replay_restart_reorder_and_operator_ownership(tmp_path):
    producer = str(uuid4())
    old = message(make_event(status="confirmed", operator_note="edge note",
                             snapshot_path="../../outside.jpg", clip_path="/etc/passwd"),
                  producer_id=producer)
    closed = message(make_event(ended_at=120., details={"closed": True},
                                status="false", operator_note="wrong"),
                     producer_id=producer, sequence=3, kind="event_update")
    path = tmp_path / "gateway.db"
    with Journal(path) as gateway:
        first = gateway.apply_remote_event(closed, snapshot_path="snapshots/gateway.jpg", expected_edge_id="edge-1")
        assert first.created and first.changed
        assert first.event.ended_at == 120.
        assert first.event.status == EventStatus.new
        assert first.event.operator_note is None
        assert first.event.snapshot_path == "snapshots/gateway.jpg"
        assert first.event.clip_path is None
        gateway.update("event-1", status="forwarded", operator_note="Ответственный")
        stale = gateway.apply_remote_event(old)
        assert not stale.created and not stale.changed
        assert stale.event.ended_at == 120.
        assert stale.event.details == {"closed": True}
        assert stale.event.operator_note == "Ответственный"
    with Journal(path) as gateway:
        for replay in (closed, old, closed):
            result = gateway.apply_remote_event(decode(encode(replay)))
            assert not result.created and not result.changed
            assert result.event.status == EventStatus.forwarded
            assert result.event.operator_note == "Ответственный"
            assert result.event.snapshot_path == "snapshots/gateway.jpg"
        assert gateway.query()[1] == 1
        assert gateway.pending_messages() == []


def test_remote_paths_are_scrubbed_and_local_path_is_checked(tmp_path):
    remote = message(make_event(snapshot_path="../../evil", clip_path="file:///tmp/evil"))
    with Journal(tmp_path / "gateway.db") as gateway:
        result = gateway.apply_remote_event(remote)
        assert result.event.snapshot_path is None
        assert result.event.clip_path is None
        for invalid in ("../evil", "/tmp/evil", "a/../evil", "a\\evil", "a//b", "./x", "http://evil", "a/\x00b"):
            with pytest.raises(ValueError):
                gateway.apply_remote_event(remote, snapshot_path=invalid)
        assert gateway.query()[1] == 1


def test_gateway_checks_origin_identity_and_message_id_reuse(tmp_path):
    original = message()
    with Journal(tmp_path / "gateway.db") as gateway:
        gateway.apply_remote_event(original)
        with pytest.raises(ValueError, match="origin"):
            gateway.apply_remote_event(message(), expected_edge_id="other-edge")
        for changes in ({"edge_id": "other-edge", "producer_id": original.producer_id},
                        {"producer_id": str(uuid4())}):
            with pytest.raises(ValueError, match="origin"):
                gateway.apply_remote_event(message(**changes))
        for identity in ({"camera_id": "camera-2"}, {"started_at": 99.}, {"rule_id": "s2"},
                         {"site": "Другой цех"}, {"bbox": (0., 0., 30., 40.)}):
            with pytest.raises(ValueError, match="identity"):
                gateway.apply_remote_event(message(make_event(**identity), sequence=2,
                                                   producer_id=original.producer_id))
        reused = original.model_copy(update={"payload": event_payload(make_event(details={"changed": True}))})
        with pytest.raises(ValueError, match="identity was reused"):
            gateway.apply_remote_event(reused)
        other_event = original.model_copy(update={"payload": event_payload(make_event("event-2"))})
        with pytest.raises(ValueError, match="identity was reused"):
            gateway.apply_remote_event(other_event)
        assert gateway.query()[1] == 1


def test_remote_event_cannot_claim_preexisting_local_event(tmp_path):
    with Journal(tmp_path / "gateway.db") as gateway:
        gateway.add(make_event())
        with pytest.raises(ValueError, match="origin"):
            gateway.apply_remote_event(message())


@pytest.mark.parametrize("failure_table", ["transport_receipts", "transport_origins"])
@pytest.mark.parametrize("existing", [False, True])
def test_gateway_failure_rolls_back_event_receipt_and_revision(tmp_path, failure_table, existing):
    path = tmp_path / "gateway.db"
    producer = str(uuid4())
    with Journal(path) as gateway:
        if existing:
            gateway.apply_remote_event(message(producer_id=producer))
        update = message(make_event(ended_at=120.), sequence=2,
                         producer_id=producer, kind="event_update")
        with sqlite3.connect(path) as db:
            db.execute(f"CREATE TRIGGER fail_apply BEFORE INSERT ON {failure_table} "
                       "BEGIN SELECT RAISE(ABORT, 'injected gateway failure'); END")
        with pytest.raises(sqlite3.IntegrityError, match="injected"):
            gateway.apply_remote_event(update)
        assert gateway.get("event-1") == (make_event() if existing else None)
        with sqlite3.connect(path) as db:
            assert db.execute("SELECT COUNT(*) FROM transport_receipts").fetchone()[0] == int(existing)
            assert db.execute("SELECT COUNT(*) FROM transport_origins").fetchone()[0] == int(existing)
            db.execute("DROP TRIGGER fail_apply")
        applied = gateway.apply_remote_event(update)
        assert applied.event.ended_at == 120.
        assert applied.created is not existing


def test_gateway_status_edit_cannot_be_lost_during_remote_update(tmp_path):
    """Pause replication inside its write transaction; a second writer must wait."""
    path = tmp_path / "gateway.db"
    producer = str(uuid4())
    entered = threading.Event()
    release = threading.Event()
    with Journal(path) as gateway, Journal(path) as operator:
        gateway.apply_remote_event(message(producer_id=producer))
        def hold():
            entered.set()
            assert release.wait(5)
            return 1
        gateway._conn.create_function("hold_apply", 0, hold)
        with sqlite3.connect(path) as db:
            db.execute("CREATE TRIGGER hold_remote BEFORE UPDATE ON transport_origins "
                       "BEGIN SELECT hold_apply(); END")
        with ThreadPoolExecutor(max_workers=2) as executor:
            applying = executor.submit(gateway.apply_remote_event,
                                       message(make_event(ended_at=120.), sequence=2, producer_id=producer))
            assert entered.wait(5)
            editing = executor.submit(operator.update, "event-1", status="confirmed", operator_note="Оператор")
            release.set()
            applying.result(timeout=5)
            edited = editing.result(timeout=5)
        assert edited.status == EventStatus.confirmed
        assert edited.operator_note == "Оператор"
        assert edited.ended_at == 120.
        # Reverse ordering: a committed operator update precedes the next arrival.
        newest = gateway.apply_remote_event(message(make_event(ended_at=130.), sequence=3, producer_id=producer))
        assert newest.event.operator_note == "Оператор"
        assert newest.event.status == EventStatus.confirmed
        assert newest.event.ended_at == 130.


def test_two_edge_connections_allocate_unique_durable_sequences(tmp_path):
    path = tmp_path / "edge.db"
    with Journal(path, replication_edge_id="edge-1") as first, Journal(path, replication_edge_id="edge-1") as second:
        def create(args):
            index, journal = args
            journal.add(make_event(f"event-{index}"))
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(create, ((i, first if i % 2 else second) for i in range(40))))
        assert first.producer_id == second.producer_id
        pending = first.pending_messages()
        assert len({item.message_id for item in pending}) == 40
        assert [item.sequence for item in pending] == list(range(1, 41))
