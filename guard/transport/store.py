"""SQLite replication state; every method runs in its caller's transaction.

The Journal owns the connection and BEGIN IMMEDIATE boundary, so an event,
its outbox entry and its sequence can only commit together. Gateway receipts
and revisions share the transaction that inserts/updates the event.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any
from uuid import uuid4

from guard.contracts import Event
from guard.transport.protocol import (
    MAX_SEQUENCE,
    Envelope,
    MessageKind,
    decode,
    encode,
    event_payload,
    identifier,
)

RUN_ID = str(uuid4())

# An update can evolve evidence/timing/closure, but cannot move an event to a
# different track, camera, rule, place, start time or confirmation time.
IDENTITY_FIELDS = (
    "id", "camera_id", "site", "zone_id", "rule_id", "type", "severity",
    "track_id", "started_at", "confirmed_at", "bbox",
)

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS transport_producer (
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        edge_id TEXT NOT NULL,
        producer_id TEXT NOT NULL,
        next_sequence INTEGER NOT NULL CHECK(next_sequence > 0)
    )""",
    """CREATE TABLE IF NOT EXISTS transport_outbox (
        sequence INTEGER PRIMARY KEY,
        message_id TEXT UNIQUE NOT NULL,
        envelope BLOB NOT NULL,
        prepared INTEGER NOT NULL DEFAULT 0 CHECK(prepared IN (0, 1))
    )""",
    """CREATE TABLE IF NOT EXISTS transport_origins (
        event_id TEXT PRIMARY KEY,
        edge_id TEXT NOT NULL,
        producer_id TEXT NOT NULL,
        sequence INTEGER NOT NULL
    )""",
    """CREATE TABLE IF NOT EXISTS transport_receipts (
        message_id TEXT PRIMARY KEY,
        edge_id TEXT NOT NULL,
        producer_id TEXT NOT NULL,
        sequence INTEGER NOT NULL,
        event_id TEXT NOT NULL,
        fingerprint TEXT NOT NULL,
        UNIQUE(producer_id, sequence)
    )""",
)


@dataclass(frozen=True, slots=True)
class ApplyResult:
    event: Event
    created: bool
    changed: bool


def local_snapshot_path(value: str | None) -> str | None:
    """Only a gateway-owned relative path may reach the persisted event."""
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 512:
        raise ValueError("Invalid local snapshot path")
    if "\\" in value or ":" in value or "\x00" in value:
        raise ValueError("Invalid local snapshot path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("Snapshot path must be a safe relative POSIX path")
    return value


class ReplicationStore:
    """Internal Journal helper, deliberately independent of the broker."""

    def __init__(self, connection: sqlite3.Connection, edge_id: str | None) -> None:
        self.connection = connection
        for statement in _SCHEMA:
            connection.execute(statement)
        columns = {row[1] for row in connection.execute("PRAGMA table_info(transport_outbox)")}
        if "prepared" not in columns:
            connection.execute("ALTER TABLE transport_outbox ADD COLUMN prepared INTEGER NOT NULL DEFAULT 0")
        self.edge_id = None if edge_id is None else identifier(edge_id, "edge_id")
        self.producer_id: str | None = None
        if self.edge_id is not None:
            row = connection.execute(
                "SELECT edge_id, producer_id FROM transport_producer WHERE singleton = 1"
            ).fetchone()
            if row is None:
                self.producer_id = str(uuid4())
                connection.execute(
                    "INSERT INTO transport_producer VALUES (1, ?, ?, 1)",
                    (self.edge_id, self.producer_id),
                )
            else:
                if row["edge_id"] != self.edge_id:
                    raise ValueError("Journal already belongs to another replication edge")
                self.producer_id = row["producer_id"]

    def make_message(self, kind: MessageKind, payload: dict[str, Any]) -> Envelope:
        if self.edge_id is None or self.producer_id is None:
            raise ValueError("Journal replication is not enabled")
        sequence = self.connection.execute(
            "SELECT next_sequence FROM transport_producer WHERE singleton = 1"
        ).fetchone()[0]
        if sequence >= MAX_SEQUENCE:
            raise ValueError("Producer sequence exhausted")
        message = Envelope(
            edge_id=self.edge_id,
            producer_id=self.producer_id,
            run_id=RUN_ID,
            sequence=sequence,
            message_id=f"{self.producer_id}:{sequence}",
            kind=kind,
            sent_at=time.time(),
            payload=payload,
        )
        self.connection.execute(
            "UPDATE transport_producer SET next_sequence = ? WHERE singleton = 1",
            (sequence + 1,),
        )
        return message

    def enqueue(self, kind: MessageKind, event: Event) -> None:
        if self.edge_id is None:
            return
        message = self.make_message(kind, event_payload(event))
        self.connection.execute(
            "INSERT INTO transport_outbox(sequence, message_id, envelope) VALUES (?, ?, ?)",
            (message.sequence, message.message_id, encode(message)),
        )

    def pending(self, limit: int) -> list[Envelope]:
        if type(limit) is not int or limit < 0 or limit > 10_000:
            raise ValueError("limit must be an integer from 0 to 10000")
        rows = self.connection.execute(
            "SELECT envelope FROM transport_outbox ORDER BY sequence LIMIT ?", (limit,)
        ).fetchall()
        return [decode(row[0], expected_edge_id=self.edge_id) for row in rows]

    def mark_published(self, message_id: str) -> None:
        # A confirmed broker publish allows removal; a crash before this DELETE
        # deliberately causes redelivery, which the gateway handles durably.
        self.connection.execute(
            "DELETE FROM transport_outbox WHERE message_id = ?", (message_id,)
        )

    def prepare_message(self, message_id: str, snapshot_jpeg: bytes | None) -> Envelope:
        row = self.connection.execute(
            "SELECT envelope, prepared FROM transport_outbox WHERE message_id = ?", (message_id,)
        ).fetchone()
        if row is None:
            raise KeyError(message_id)
        message = decode(row["envelope"], expected_edge_id=self.edge_id)
        if row["prepared"]:
            return message
        event = Event.model_validate(message.payload["event"])
        prepared = Envelope.model_validate({
            **message.model_dump(),
            "payload": event_payload(event, snapshot_jpeg=snapshot_jpeg),
        })
        self.connection.execute(
            "UPDATE transport_outbox SET envelope = ?, prepared = 1 WHERE message_id = ?",
            (encode(prepared), message_id),
        )
        return prepared

    def stats(self) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT next_sequence FROM transport_producer WHERE singleton = 1"
        ).fetchone()
        return {
            "pending": self.connection.execute("SELECT COUNT(*) FROM transport_outbox").fetchone()[0],
            "producer_id": self.producer_id,
            "run_id": RUN_ID,
            "next_sequence": None if row is None else row[0],
        }

    def receipt(self, envelope: Envelope, event_id: str) -> bool:
        """Return True for an identical receipt, fail on ID reuse."""
        fingerprint = hashlib.sha256(encode(envelope)).hexdigest()
        existing = self.connection.execute(
            "SELECT * FROM transport_receipts WHERE message_id = ?",
            (envelope.message_id,),
        ).fetchone()
        if existing is not None:
            if (existing["edge_id"], existing["producer_id"], existing["sequence"],
                existing["event_id"], existing["fingerprint"]) != (
                envelope.edge_id, envelope.producer_id, envelope.sequence, event_id, fingerprint
            ):
                raise ValueError("Replication message identity was reused")
            return True
        self.connection.execute(
            "INSERT INTO transport_receipts VALUES (?, ?, ?, ?, ?, ?)",
            (envelope.message_id, envelope.edge_id, envelope.producer_id,
             envelope.sequence, event_id, fingerprint),
        )
        return False

    def origin(self, event_id: str) -> sqlite3.Row | None:
        return self.connection.execute(
            "SELECT * FROM transport_origins WHERE event_id = ?", (event_id,)
        ).fetchone()

    def set_origin(self, event_id: str, envelope: Envelope) -> None:
        self.connection.execute(
            "INSERT INTO transport_origins VALUES (?, ?, ?, ?) "
            "ON CONFLICT(event_id) DO UPDATE SET sequence = excluded.sequence",
            (event_id, envelope.edge_id, envelope.producer_id, envelope.sequence),
        )
