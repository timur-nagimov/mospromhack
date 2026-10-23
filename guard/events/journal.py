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

from guard.contracts import Event

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
)
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
    }
)
_GROUP_COLUMNS = {
    "type": "type",
    "camera": "camera_id",
    "zone": "zone_id",
    "severity": "severity",
    "status": "status",
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
    latency_ms TEXT NOT NULL
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

    def __init__(self, db_path: str | Path) -> None:
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
        except BaseException:
            self._conn.close()
            raise

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
            self._conn.execute(
                _INSERT_SQL, [data[column] for column in _COLUMNS]
            )

    def update(self, event_id: str, **fields: Any) -> Event:
        """Атомарно обновить разрешённые поля; отсутствующий id — KeyError."""
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
            return _deserialize(stored)

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
        if group_by not in _GROUP_COLUMNS and group_by not in {"hour", "day"}:
            raise ValueError(f"Unsupported group_by: {group_by!r}")
        where, parameters = _where(filters, ts_from, ts_to)
        temporal = group_by in {"hour", "day"}
        if temporal:
            local_tz = ZoneInfo(tz)

            def local_bucket(timestamp: float) -> int | str:
                local = datetime.fromtimestamp(timestamp, local_tz)
                return local.hour if group_by == "hour" else local.date().isoformat()

            expression = "_journal_local_bucket(confirmed_at)"
        else:
            expression = _GROUP_COLUMNS[group_by]

        with self._lock:
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
                for column in ("started_at", "confirmed_at", "ended_at"):
                    if values[column] is not None:
                        values[column] = datetime.fromtimestamp(
                            values[column], timezone.utc
                        ).isoformat()
                writer.writerow([values[column] for column in _COLUMNS])
