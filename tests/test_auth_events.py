"""PATCH /api/events/{id}: кто и когда принял решение — handled_by и handled_at в самом событии."""

import sqlite3
import time
from contextlib import closing
from uuid import uuid4

import pytest

from guard.auth import passwords
from guard.auth.testing import auth_env
from guard.contracts import Event, Severity
from guard.events.journal import Journal
from guard.pipeline._fallbacks import MemoryJournal
from guard.transport.protocol import Envelope, event_payload

# Схема events до добавления handled_by и handled_at (user_version = 1): так выглядит data/guard.db демо.
OLD_EVENTS_TABLE = """
CREATE TABLE events (
    id TEXT PRIMARY KEY NOT NULL, camera_id TEXT NOT NULL, site TEXT NOT NULL, zone_id TEXT,
    rule_id TEXT NOT NULL, type TEXT NOT NULL, severity TEXT NOT NULL, track_id INTEGER,
    started_at REAL NOT NULL, confirmed_at REAL NOT NULL, ended_at REAL, bbox TEXT NOT NULL,
    snapshot_path TEXT, clip_path TEXT, status TEXT NOT NULL, operator_note TEXT,
    details TEXT NOT NULL, latency_ms TEXT NOT NULL
)
"""


@pytest.fixture(autouse=True)
def fast_scrypt(monkeypatch):
    monkeypatch.setattr(passwords, "SCRYPT_N", 2**10)


def make_event(event_id="ev1", **kw):
    base = dict(
        id=event_id, camera_id="cam1", site="Цех 1", zone_id="z1", rule_id="s1", type="zone_intrusion",
        severity=Severity.high, track_id=1, started_at=1000.0, confirmed_at=1002.0, bbox=(1, 2, 3, 4),
    )
    base.update(kw)
    return Event(**base)


def test_event_model_has_handled_fields():
    ev = make_event()
    assert ev.handled_by is None and ev.handled_at is None


def test_patch_records_who_and_when(tmp_path):
    with auth_env(tmp_path) as env:
        env.engine.journal.add(make_event())
        headers = env.login("operator")
        before = time.time()
        with env.client.websocket_connect("/ws/events", headers=headers) as ws:
            r = env.client.patch("/api/events/ev1", json={"status": "confirmed"}, headers=headers)
            assert r.status_code == 200
            body = r.json()
            assert body["handled_by"] == "operator"
            assert before - 1 <= body["handled_at"] <= time.time() + 1
            msg = ws.receive_json()
            while msg["kind"] != "event_update":
                msg = ws.receive_json()
            assert msg["event"]["handled_by"] == "operator"
            assert msg["event"]["handled_at"] == body["handled_at"]
        listed = env.client.get("/api/events", headers=headers).json()["items"][0]
        assert listed["handled_by"] == "operator" and listed["handled_at"] == body["handled_at"]
        one = env.client.get("/api/events/ev1", headers=env.login("worker")).json()
        assert one["handled_by"] == "operator"


def test_client_cannot_forge_handled_by(tmp_path):
    with auth_env(tmp_path) as env:
        env.engine.journal.add(make_event())
        r = env.client.patch("/api/events/ev1", json={"operator_note": "x", "handled_by": "admin", "handled_at": 1.0},
                             headers=env.login("operator"))
        assert r.status_code == 200
        assert r.json()["handled_by"] == "operator" and r.json()["handled_at"] != 1.0


def test_empty_patch_does_not_mark_handled(tmp_path):
    with auth_env(tmp_path) as env:
        env.engine.journal.add(make_event())
        r = env.client.patch("/api/events/ev1", json={}, headers=env.login("operator"))
        assert r.status_code == 200 and r.json()["handled_by"] is None


def test_auth_disabled_marks_admin(tmp_path):
    with auth_env(tmp_path, auth=False) as env:
        env.engine.journal.add(make_event())
        r = env.client.patch("/api/events/ev1", json={"status": "false"})
        assert r.status_code == 200 and r.json()["handled_by"] == "admin"


def test_journal_migrates_old_database_in_place(tmp_path):
    db = tmp_path / "guard.db"
    with closing(sqlite3.connect(db)) as conn:
        conn.execute(OLD_EVENTS_TABLE)
        conn.execute(
            "INSERT INTO events VALUES ('old', 'cam1', 'Цех 1', NULL, 's1', 'zone_intrusion', 'high', 1,"
            " 1000.0, 1002.0, NULL, '[1, 2, 3, 4]', NULL, NULL, 'confirmed', 'старая заметка', '{}', '{}')"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    with Journal(db) as journal:
        old = journal.get("old")
        assert old.operator_note == "старая заметка" and old.handled_by is None and old.handled_at is None
        updated = journal.update("old", status="forwarded", handled_by="safety", handled_at=2000.0)
        assert (updated.handled_by, updated.handled_at) == ("safety", 2000.0)
    with closing(sqlite3.connect(db)) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(events)")}
        # версия схемы не меняется: прежний код по-прежнему открывает эту базу
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 1
    assert {"handled_by", "handled_at"} <= columns
    with Journal(db) as journal:  # повторное открытие ничего не ломает
        assert journal.get("old").handled_by == "safety"


def _envelope(event, *, sequence, producer_id, kind):
    return Envelope(
        edge_id="edge-1", producer_id=producer_id, run_id=str(uuid4()), message_id=f"{producer_id}:{sequence}",
        kind=kind, sequence=sequence, sent_at=200.0, payload=event_payload(event),
    )


def test_edge_revision_keeps_gateway_decision(tmp_path):
    """Решение диспетчера принадлежит шлюзу, как status и operator_note: ревизия с edge его не стирает."""
    producer = str(uuid4())
    with Journal(tmp_path / "gw.db") as gateway:
        gateway.apply_remote_event(_envelope(make_event(), sequence=1, producer_id=producer, kind="event"))
        gateway.update("ev1", status="confirmed", handled_by="operator", handled_at=1500.0)
        closed = make_event(ended_at=1100.0)
        result = gateway.apply_remote_event(
            _envelope(closed, sequence=2, producer_id=producer, kind="event_update"), expected_edge_id="edge-1",
        )
        assert result.event.ended_at == 1100.0
        assert (result.event.status.value, result.event.handled_by, result.event.handled_at) == (
            "confirmed", "operator", 1500.0)


def test_wire_payload_omits_gateway_only_fields():
    """Шлюз без этих полей в модели Event не отвергнет сообщение нового edge."""
    payload = event_payload(make_event(handled_by="operator", handled_at=5.0))
    assert "handled_by" not in payload["event"] and "handled_at" not in payload["event"]


def test_memory_journal_accepts_handled_fields():
    journal = MemoryJournal()
    journal.add(make_event())
    updated = journal.update("ev1", handled_by="operator", handled_at=10.0)
    assert updated.handled_by == "operator" and updated.handled_at == 10.0


def test_csv_has_handled_columns(tmp_path):
    with Journal(tmp_path / "g.db") as journal:
        journal.add(make_event(handled_by="safety", handled_at=0.0))
        journal.export_csv(tmp_path / "out.csv")
    text = (tmp_path / "out.csv").read_text(encoding="utf-8-sig")
    header = text.splitlines()[0]
    assert "Кто обработал" in header and "Время обработки (UTC)" in header
    assert "safety" in text and "1970-01-01T00:00:00+00:00" in text
