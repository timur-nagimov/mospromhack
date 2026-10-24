"""Потокобезопасный журнал событий SQLite.

Границы времени включительны и относятся к confirmed_at. Пустой список
значений фильтра не находит событий. При одинаковом времени порядок
определяется идентификатором события.

CSV содержит все совпавшие события, русские заголовки и даты в UTC.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import threading
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from guard.contracts import Event, EventStatus
from guard.transport.protocol import Envelope, MessageKind, decode, encode
from guard.transport.store import (
    IDENTITY_FIELDS, RUN_ID, ApplyResult, ReplicationStore, local_snapshot_path,
)

_Filter = str | list[str] | None

_SCHEMA_VERSION = 1
_COLUMNS = (
    "id",
    "camera_id",
    "site",
    "zone_id",
    "rule_id",
    "type",
    "severity",
    "track_id",
    "started_at",
    "confirmed_at",
    "ended_at",
    "bbox",
    "snapshot_path",
    "clip_path",
    "status",
    "operator_note",
    "details",
    "latency_ms",
    "handled_by",
    "handled_at",
)
# Колонки, добавленные после первой версии схемы: в старую базу они дописываются ALTER TABLE при открытии.
# user_version остаётся 1, чтобы прежний код по-прежнему открывал обновлённую базу.
_ADDED_COLUMNS = (("handled_by", "TEXT"), ("handled_at", "REAL"))
_JSON_COLUMNS = ("bbox", "details", "latency_ms")
_FILTER_COLUMNS = frozenset(
    {"type", "camera_id", "zone_id", "status", "severity"}
)
_UPDATE_COLUMNS = frozenset(
    {
        "status",
        "operator_note",
        "ended_at",
        "snapshot_path",
        "clip_path",
        "details",
        "handled_by",
        "handled_at",
    }
)
_GROUP_COLUMNS = {
    "type": "type",
    "camera": "camera_id",
    "zone": "zone_id",
    "severity": "severity",
    "status": "status",
    "site": "site",
}
_CSV_HEADERS = (
    "Идентификатор события",
    "Камера",
    "Участок",
    "Зона",
    "Правило",
    "Тип нарушения",
    "Критичность",
    "Идентификатор трека",
    "Начало (UTC)",
    "Подтверждение (UTC)",
    "Окончание (UTC)",
    "Рамка (JSON)",
    "Путь к кадру",
    "Путь к клипу",
    "Статус",
    "Комментарий ответственного",
    "Подробности",
    "Задержки обработки (мс, JSON)",
    "Кто обработал",
    "Время обработки (UTC)",
)
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY NOT NULL,
    camera_id TEXT NOT NULL,
    site TEXT NOT NULL,
    zone_id TEXT,
    rule_id TEXT NOT NULL,
    type TEXT NOT NULL,
    severity TEXT NOT NULL,
    track_id INTEGER,
    started_at REAL NOT NULL,
    confirmed_at REAL NOT NULL,
    ended_at REAL,
    bbox TEXT NOT NULL,
    snapshot_path TEXT,
    clip_path TEXT,
    status TEXT NOT NULL,
    operator_note TEXT,
    details TEXT NOT NULL,
    latency_ms TEXT NOT NULL,
    handled_by TEXT,
    handled_at REAL
)
"""
_CREATE_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_events_camera_confirmed "
    "ON events(camera_id, confirmed_at)",
    "CREATE INDEX IF NOT EXISTS idx_events_type ON events(type)",
    "CREATE INDEX IF NOT EXISTS idx_events_status ON events(status)",
    "CREATE INDEX IF NOT EXISTS idx_events_zone ON events(zone_id)",
    "CREATE INDEX IF NOT EXISTS idx_events_confirmed ON events(confirmed_at)",
)
_INSERT_SQL = (
    f"INSERT INTO events ({', '.join(_COLUMNS)}) "
    f"VALUES ({', '.join('?' for _ in _COLUMNS)})"
)


def _serialize(event: Event) -> dict[str, Any]:
    data = event.model_dump(mode="json")
    for column in _JSON_COLUMNS:
        data[column] = json.dumps(
            data[column], ensure_ascii=False, allow_nan=False
        )
    return data


def _deserialize(row: sqlite3.Row) -> Event:
    data = dict(row)
    for column in _JSON_COLUMNS:
        data[column] = json.loads(data[column])
    return Event.model_validate(data)


def _where(
    filters: dict[str, Any],
    ts_from: float | None,
    ts_to: float | None,
) -> tuple[str, list[Any]]:
    unknown = filters.keys() - _FILTER_COLUMNS
    if unknown:
        raise TypeError(f"Unknown filters: {', '.join(sorted(unknown))}")

    clauses: list[str] = []
    parameters: list[Any] = []
    for column, value in filters.items():
        if value is None:
            continue
        if isinstance(value, str):
            values = [value]
        elif isinstance(value, list):
            values = value
        else:
            raise TypeError(f"{column} must be a string or a list of strings")
        if not all(isinstance(item, str) for item in values):
            raise TypeError(f"{column} must contain only strings")
        if not values:
            clauses.append("0")
            continue
        placeholders = ", ".join("?" for _ in values)
        clauses.append(f"{column} IN ({placeholders})")
        parameters.extend(
            item.value if isinstance(item, Enum) else item for item in values
        )

    if ts_from is not None:
        clauses.append("confirmed_at >= ?")
        parameters.append(ts_from)
    if ts_to is not None:
        clauses.append("confirmed_at <= ?")
        parameters.append(ts_to)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), parameters


def _order_sql(order: str) -> str:
    if not isinstance(order, str) or order.lower() not in {"asc", "desc"}:
        raise ValueError("order must be 'asc' or 'desc'")
    direction = order.upper()
    return f" ORDER BY confirmed_at {direction}, id {direction}"


class Journal:
    """SQLite-хранилище с отдельным соединением и блокировкой на экземпляр."""

    def __init__(self, db_path: str | Path, *, replication_edge_id: str | None = None) -> None:
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._closed = False
        self._conn = sqlite3.connect(
            path, timeout=30.0, check_same_thread=False
        )
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                version = self._conn.execute(
                    "PRAGMA user_version"
                ).fetchone()[0]
                if version not in (0, _SCHEMA_VERSION):
                    raise ValueError(
                        f"Unsupported journal schema version: {version}"
                    )
                if version == 0:
                    self._conn.execute(_CREATE_TABLE)
                    for statement in _CREATE_INDEXES:
                        self._conn.execute(statement)
                    self._conn.execute(
                        f"PRAGMA user_version = {_SCHEMA_VERSION}"
                    )
                self._add_missing_columns()
                self._replication = ReplicationStore(self._conn, replication_edge_id)
        except BaseException:
            self._conn.close()
            raise

    def _add_missing_columns(self) -> None:
        """Дописать в существующую базу колонки, появившиеся позже (handled_by, handled_at).

        Только ADD COLUMN со значением NULL: старые строки и прежний код не затрагиваются.
        """
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(events)")}
        for name, sql_type in _ADDED_COLUMNS:
            if name not in existing:
                self._conn.execute(f"ALTER TABLE events ADD COLUMN {name} {sql_type}")

    def __enter__(self) -> Journal:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Освободить соединение; повторное закрытие безопасно."""
        with self._lock:
            if not self._closed:
                self._conn.close()
                self._closed = True

    def add(self, event: Event) -> None:
        """Добавить событие. Повторный id вызывает sqlite3.IntegrityError."""
        validated = Event.model_validate(event.model_dump())
        data = _serialize(validated)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._conn.execute(
                _INSERT_SQL, [data[column] for column in _COLUMNS]
            )
            self._replication.enqueue("event", validated)

    def update(self, event_id: str, *, details_patch: dict[str, Any] | None = None, **fields: Any) -> Event:
        """Атомарно обновить разрешённые поля; отсутствующий id — KeyError."""
        if details_patch is not None and details_patch.keys() - {"reopened_count", "active_duration_s", "clip_error"}:
            raise ValueError("Unsupported machine detail patch")
        unknown = fields.keys() - _UPDATE_COLUMNS
        if unknown:
            raise ValueError(
                f"Fields cannot be updated: {', '.join(sorted(unknown))}"
            )
        with self._lock, self._conn:
            # Защищает read-modify-write также от других экземпляров Journal.
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            if row is None:
                raise KeyError(event_id)
            current = _deserialize(row)
            if details_patch:
                fields["details"] = {**current.details, **details_patch}
            if not fields:
                return current
            data = current.model_dump()
            data.update(fields)
            updated = Event.model_validate(data)
            serialized = _serialize(updated)
            assignments = ", ".join(f"{column} = ?" for column in fields)
            self._conn.execute(
                f"UPDATE events SET {assignments} WHERE id = ?",
                [serialized[column] for column in fields] + [event_id],
            )
            stored = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
            result = _deserialize(stored)
            self._replication.enqueue("event_update", updated)
            return result

    @property
    def producer_id(self) -> str | None:
        """Persistent producer UUID, or None when outbox replication is off."""
        return self._replication.producer_id

    @property
    def run_id(self) -> str:
        """One UUID for this Python process, shared by its Journal instances."""
        return RUN_ID

    def pending_messages(self, limit: int = 100) -> list[Envelope]:
        """Unconfirmed durable messages, ordered by producer sequence."""
        with self._lock:
            return self._replication.pending(limit)

    def mark_published(self, message_id: str) -> None:
        """Remove one outbox item after a successful publisher confirmation."""
        with self._lock, self._conn:
            self._replication.mark_published(message_id)

    def prepare_message(self, message_id: str, *, snapshot_jpeg: bytes | None) -> Envelope:
        """Freeze an outbox wire body once, including an unavailable snapshot.

        Repeat calls return the durable original body even if a local media
        file has since changed or disappeared. Unknown IDs raise KeyError.
        """
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            return self._replication.prepare_message(message_id, snapshot_jpeg)

    def make_message(self, kind: MessageKind, payload: dict[str, Any]) -> Envelope:
        """Allocate a transient message sequence; frames/metrics are not queued."""
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            return self._replication.make_message(kind, payload)

    def replication_stats(self) -> dict[str, Any]:
        with self._lock, self._conn:
            self._conn.execute("BEGIN")
            return self._replication.stats()

    def apply_remote_event(
        self,
        envelope: Envelope,
        *,
        snapshot_path: str | None = None,
        expected_edge_id: str | None = None,
    ) -> ApplyResult:
        """Atomically apply a remote revision and its durable receipt.

        Gateway status/note always win over edge fields. A later revision can
        arrive before creation; obsolete revisions cannot reopen a closed event.
        A remote event cannot claim an existing local event or another origin.
        """
        message = decode(encode(envelope), expected_edge_id=expected_edge_id)
        if message.kind not in {"event", "event_update"}:
            raise ValueError("Only event messages can be applied to Journal")
        incoming = Event.model_validate(message.payload["event"])
        snapshot_path = local_snapshot_path(snapshot_path)
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (incoming.id,)
            ).fetchone()
            current = None if row is None else _deserialize(row)
            origin = self._replication.origin(incoming.id)
            if current is not None:
                if origin is None or (origin["edge_id"], origin["producer_id"]) != (
                    message.edge_id, message.producer_id
                ):
                    raise ValueError("Event belongs to another origin")
                if any(getattr(current, field) != getattr(incoming, field) for field in IDENTITY_FIELDS):
                    raise ValueError("Remote event identity cannot change")
            elif origin is not None:
                raise ValueError("Replication origin has no event")
            duplicate = self._replication.receipt(message, incoming.id)
            if duplicate or (origin is not None and message.sequence <= origin["sequence"]):
                if current is None:
                    raise ValueError("Replication receipt has no event")
                return ApplyResult(current, created=False, changed=False)

            data = incoming.model_dump()
            data.update(
                status=EventStatus.new if current is None else current.status,
                operator_note=None if current is None else current.operator_note,
                handled_by=None if current is None else current.handled_by,
                handled_at=None if current is None else current.handled_at,
                snapshot_path=snapshot_path if snapshot_path is not None else (
                    None if current is None else current.snapshot_path
                ),
                clip_path=None,
            )
            result = Event.model_validate(data)
            serialized = _serialize(result)
            if current is None:
                self._conn.execute(_INSERT_SQL, [serialized[column] for column in _COLUMNS])
            else:
                assignments = ", ".join(f"{column} = ?" for column in _COLUMNS if column != "id")
                self._conn.execute(
                    f"UPDATE events SET {assignments} WHERE id = ?",
                    [serialized[column] for column in _COLUMNS if column != "id"] + [result.id],
                )
            self._replication.set_origin(result.id, message)
            return ApplyResult(result, created=current is None, changed=current != result)

    def get(self, event_id: str) -> Event | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM events WHERE id = ?", (event_id,)
            ).fetchone()
        return None if row is None else _deserialize(row)

    def query(
        self,
        type: _Filter = None,
        camera_id: _Filter = None,
        zone_id: _Filter = None,
        status: _Filter = None,
        severity: _Filter = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
        limit: int = 50,
        offset: int = 0,
        order: str = "desc",
    ) -> tuple[list[Event], int]:
        for name, value in (("limit", limit), ("offset", offset)):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        ordering = _order_sql(order)
        where, parameters = _where(
            {
                "type": type,
                "camera_id": camera_id,
                "zone_id": zone_id,
                "status": status,
                "severity": severity,
            },
            ts_from,
            ts_to,
        )
        with self._lock, self._conn:
            # total и страница читаются из одного снимка базы.
            self._conn.execute("BEGIN")
            total = self._conn.execute(
                "SELECT COUNT(*) FROM events" + where, parameters
            ).fetchone()[0]
            rows = self._conn.execute(
                "SELECT * FROM events"
                + where
                + ordering
                + " LIMIT ? OFFSET ?",
                parameters + [limit, offset],
            ).fetchall()
        return [_deserialize(row) for row in rows], total

    def stats(
        self,
        group_by: str,
        ts_from: float | None = None,
        ts_to: float | None = None,
        tz: str = "Europe/Moscow",
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """Вернуть непустые группы по возрастанию key; зона без id — None."""
        with self._lock:
            return self._stats_unlocked(group_by, ts_from, ts_to, tz, **filters)

    def _stats_unlocked(
        self,
        group_by: str,
        ts_from: float | None,
        ts_to: float | None,
        tz: str,
        **filters: Any,
    ) -> list[dict[str, Any]]:
        """SQL-агрегат на текущем соединении; вызывается под self._lock."""
        if group_by not in _GROUP_COLUMNS and group_by not in {"hour", "day", "weekday_hour"}:
            raise ValueError(f"Unsupported group_by: {group_by!r}")
        where, parameters = _where(filters, ts_from, ts_to)
        temporal = group_by in {"hour", "day", "weekday_hour"}
        if temporal:
            local_tz = ZoneInfo(tz)

            def local_bucket(timestamp: float) -> int | str:
                local = datetime.fromtimestamp(timestamp, local_tz)
                if group_by == "hour":
                    return local.hour
                if group_by == "weekday_hour":
                    return f"{local.weekday()}:{local.hour}"
                return local.date().isoformat()

            expression = "_journal_local_bucket(confirmed_at)"
        else:
            expression = _GROUP_COLUMNS[group_by]

        if temporal:
            self._conn.create_function(
                "_journal_local_bucket",
                1,
                local_bucket,
                deterministic=True,
            )
        rows = self._conn.execute(
            f"SELECT {expression} AS key, COUNT(*) AS count FROM events"
            + where
            + " GROUP BY 1 ORDER BY 1",
            parameters,
        ).fetchall()
        return [dict(row) for row in rows]

    def analytics(
        self,
        ts_from: float | None = None,
        ts_to: float | None = None,
        tz: str = "Europe/Moscow",
    ) -> dict[str, Any]:
        """Все агрегаты страницы из одного снимка SQLite, без загрузки Event.

        Блокировка защищает соединение от потоков этого Journal, а явная
        транзакция сохраняет снимок также при записях через другие соединения.
        В WAL писатели могут завершить транзакцию, пока читается аналитика.
        """
        with self._lock, self._conn:
            self._conn.execute("BEGIN")

            def aggregate(group_by: str, **filters: Any) -> list[dict[str, Any]]:
                return self._stats_unlocked(group_by, ts_from, ts_to, tz, **filters)

            return {
                "types": aggregate("type"),
                "cameras": aggregate("camera"),
                "sites": aggregate("site"),
                "days": aggregate("day"),
                "weekdayHours": aggregate("weekday_hour"),
                "criticalTypes": aggregate("type", severity="critical"),
                "statusTypes": {
                    status.value: aggregate("type", status=status.value)
                    for status in EventStatus
                },
            }

    def export_csv(self, path: str | Path, **filters: Any) -> None:
        """Выгрузить все совпадения в CSV UTF-8 с BOM.

        Принимает фильтры query и order, но не limit/offset.
        Даты записываются в ISO 8601 с часовым поясом UTC.
        """
        ts_from = filters.pop("ts_from", None)
        ts_to = filters.pop("ts_to", None)
        ordering = _order_sql(filters.pop("order", "desc"))
        where, parameters = _where(filters, ts_from, ts_to)
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM events" + where + ordering, parameters
            ).fetchall()

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(_CSV_HEADERS)
            for row in rows:
                values = dict(row)
                for column in ("started_at", "confirmed_at", "ended_at", "handled_at"):
                    if values[column] is not None:
                        values[column] = datetime.fromtimestamp(
                            values[column], timezone.utc
                        ).isoformat()
                writer.writerow([values[column] for column in _COLUMNS])
