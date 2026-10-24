"""Хранилище маховика: SQLite flywheel.db и кадры frames/<camera_id>/<sample_id>.jpg (docs/FLYWHEEL.md).

Доступ к БД — только через FlywheelStore: им пользуются конвейер (фоновая запись), API и офлайн-скрипты
ml/flywheel/*. Одно соединение на объект, все операции под блокировкой; WAL и busy_timeout позволяют
читать и писать одновременно из сервиса и из CLI.

Сэмпл — словарь с полями таблицы samples. JSON-поля (detections, labels, meta) раскрыты в списки
и словари. Кадр пишется отдельно (write_frame) до вставки строки: строка без кадра не появляется.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from guard.rules.geometry import LOCAL_TZ

SAMPLE_SOURCES = ("event", "uncertain", "flicker", "disagreement", "operator", "import")
SAMPLE_STATUSES = ("new", "auto", "accepted", "rejected")
REVIEW_STATUSES = ("accepted", "rejected")
# Решение оператора по событию → метка сэмпла event этого события
OPERATOR_LABELS = {"confirmed": "positive", "false": "hard_negative", "forwarded": "forwarded"}
LABEL_ORIGINS = ("teacher", "operator", "model", "dataset")
# Источники вне суточного лимита: событие и ручной захват оператора
UNCAPPED_SOURCES = frozenset({"event", "operator"})
ORDERS = {
    "-created_at": "created_at DESC, id DESC",
    "created_at": "created_at ASC, id ASC",
    "-reviewed_at": "reviewed_at DESC, id DESC",
    "reviewed_at": "reviewed_at ASC, id ASC",
}

_COLUMNS = (
    "id",
    "created_at",
    "camera_id",
    "source",
    "event_id",
    "frame_path",
    "width",
    "height",
    "detections",
    "labels",
    "status",
    "operator_label",
    "reason",
    "holdout",
    "reviewed_at",
    "dataset_version",
    "meta",
)
_JSON_COLUMNS = ("detections", "labels", "meta")
_UPDATE_COLUMNS = frozenset(
    {"status", "labels", "operator_label", "reason", "reviewed_at", "dataset_version", "meta", "holdout"}
)
_DATASET_COLUMNS = ("version", "created_at", "path", "classes", "n_train", "n_val", "n_hard_negative", "manifest")
_DATASET_JSON = ("classes", "manifest")

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS samples (
        id TEXT PRIMARY KEY NOT NULL,
        created_at REAL NOT NULL,
        camera_id TEXT NOT NULL,
        source TEXT NOT NULL,
        event_id TEXT,
        frame_path TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        detections TEXT NOT NULL DEFAULT '[]',
        labels TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'new',
        operator_label TEXT,
        reason TEXT NOT NULL DEFAULT '',
        holdout INTEGER NOT NULL DEFAULT 0,
        reviewed_at REAL,
        dataset_version TEXT,
        meta TEXT NOT NULL DEFAULT '{}'
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS datasets (
        version TEXT PRIMARY KEY NOT NULL,
        created_at REAL NOT NULL,
        path TEXT NOT NULL,
        classes TEXT NOT NULL DEFAULT '[]',
        n_train INTEGER NOT NULL DEFAULT 0,
        n_val INTEGER NOT NULL DEFAULT 0,
        n_hard_negative INTEGER NOT NULL DEFAULT 0,
        manifest TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_samples_created ON samples(created_at)",
    "CREATE INDEX IF NOT EXISTS idx_samples_camera_created ON samples(camera_id, created_at)",
    "CREATE INDEX IF NOT EXISTS idx_samples_status ON samples(status)",
    "CREATE INDEX IF NOT EXISTS idx_samples_source ON samples(source)",
    "CREATE INDEX IF NOT EXISTS idx_samples_event ON samples(event_id)",
)

_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


class SampleError(ValueError):
    """Некорректный сэмпл или правка (→ HTTP 422)."""


def new_sample_id() -> str:
    return uuid.uuid4().hex


def safe_name(value: str) -> str:
    """Имя каталога камеры без разделителей пути и спецсимволов."""
    name = _SAFE_NAME.sub("_", str(value)).strip("._")
    return name or "camera"


def frame_rel_path(camera_id: str, sample_id: str) -> str:
    return f"frames/{safe_name(camera_id)}/{safe_name(sample_id)}.jpg"


def day_start(ts: float) -> float:
    """Начало местных (МСК) суток для момента ts."""
    local = datetime.fromtimestamp(ts, LOCAL_TZ)
    return local.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def _num(value: Any, digits: int) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return round(f, digits) if math.isfinite(f) else None


def normalize_box(box: Any) -> list[float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise SampleError("bbox must be [x1, y1, x2, y2]")
    out = [_num(v, 1) for v in box]
    if any(v is None for v in out):
        raise SampleError("bbox values must be finite numbers")
    x1, y1, x2, y2 = out  # type: ignore[misc]
    if x2 <= x1 or y2 <= y1:
        raise SampleError("bbox must have x2 > x1 and y2 > y1")
    return [x1, y1, x2, y2]  # type: ignore[list-item]


def normalize_detection(det: Mapping[str, Any]) -> dict[str, Any]:
    track = det.get("track_id")
    return {
        "model_id": str(det.get("model_id") or det.get("model") or ""),
        "cls": str(det.get("cls") or ""),
        "conf": _num(det.get("conf"), 4),
        "bbox": normalize_box(det.get("bbox")),
        "track_id": int(track) if track is not None else None,
    }


def normalize_label(label: Mapping[str, Any], default_origin: str = "operator") -> dict[str, Any]:
    cls = str(label.get("cls") or "").strip()
    if not cls:
        raise SampleError("label cls must not be empty")
    origin = str(label.get("origin") or default_origin)
    if origin not in LABEL_ORIGINS:
        raise SampleError(f"label origin must be one of {LABEL_ORIGINS}")
    return {"cls": cls, "bbox": normalize_box(label.get("bbox")), "origin": origin, "conf": _num(label.get("conf"), 4)}


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _row_to_sample(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for column in _JSON_COLUMNS:
        data[column] = json.loads(data[column]) if data[column] else ([] if column != "meta" else {})
    data["holdout"] = int(data["holdout"] or 0)
    return data


def _row_to_dataset(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for column in _DATASET_JSON:
        data[column] = json.loads(data[column]) if data[column] else ([] if column == "classes" else {})
    return data


def _as_list(value: str | Iterable[str] | None) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        return parts or None
    out = [str(v) for v in value]
    return out or None


class FlywheelStore:
    """SQLite-хранилище сэмплов и версий датасетов. Потокобезопасно."""

    def __init__(self, root: str | Path, db_name: str = "flywheel.db") -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / db_name
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=10.0, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            for stmt in _SCHEMA:
                self._conn.execute(stmt)

    # ---------- кадры ----------

    def resolve_frame(self, rel: str | None) -> Path | None:
        """Путь кадра внутри хранилища; чужие пути и отсутствующие файлы — None."""
        if not rel:
            return None
        root = self.root.resolve()
        path = (root / rel).resolve()
        if root not in path.parents or not path.is_file():
            return None
        return path

    def write_frame(self, rel: str, data: bytes) -> Path:
        """Атомарная запись JPEG: временный файл и переименование."""
        root = self.root.resolve()
        path = (root / rel).resolve()
        if root not in path.parents:
            raise SampleError(f"frame path {rel!r} escapes the store")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return path

    # ---------- сэмплы ----------

    def add(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        """Вставить сэмпл. Повтор с тем же id не меняет строку (доставка at-least-once)."""
        row = self._prepare(sample)
        values = [row[c] for c in _COLUMNS]
        with self._lock:
            self._conn.execute(
                f"INSERT OR IGNORE INTO samples ({', '.join(_COLUMNS)}) VALUES ({', '.join('?' for _ in _COLUMNS)})",
                values,
            )
            stored = self.get(row["id"])
        assert stored is not None
        return stored

    def _prepare(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        source = str(sample.get("source") or "")
        if source not in SAMPLE_SOURCES:
            raise SampleError(f"source must be one of {SAMPLE_SOURCES}")
        status = str(sample.get("status") or "new")
        if status not in SAMPLE_STATUSES:
            raise SampleError(f"status must be one of {SAMPLE_STATUSES}")
        camera_id = str(sample.get("camera_id") or "").strip()
        if not camera_id:
            raise SampleError("camera_id must not be empty")
        sample_id = str(sample.get("id") or new_sample_id())
        frame_path = str(sample.get("frame_path") or frame_rel_path(camera_id, sample_id))
        width, height = int(sample.get("width") or 0), int(sample.get("height") or 0)
        if width <= 0 or height <= 0:
            raise SampleError("width and height must be positive")
        operator_label = sample.get("operator_label")
        if operator_label is not None and operator_label not in OPERATOR_LABELS.values():
            raise SampleError(f"operator_label must be one of {sorted(OPERATOR_LABELS.values())}")
        detections = [normalize_detection(d) for d in sample.get("detections") or []]
        labels = [normalize_label(lb, default_origin="model") for lb in sample.get("labels") or []]
        return {
            "id": sample_id,
            "created_at": float(sample.get("created_at") or time.time()),
            "camera_id": camera_id,
            "source": source,
            "event_id": sample.get("event_id") or None,
            "frame_path": frame_path,
            "width": width,
            "height": height,
            "detections": _dumps(detections),
            "labels": _dumps(labels),
            "status": status,
            "operator_label": operator_label,
            "reason": str(sample.get("reason") or ""),
            "holdout": 1 if sample.get("holdout") else 0,
            "reviewed_at": sample.get("reviewed_at"),
            "dataset_version": sample.get("dataset_version"),
            "meta": _dumps(dict(sample.get("meta") or {})),
        }

    def get(self, sample_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
        return _row_to_sample(row) if row is not None else None

    def query(
        self,
        *,
        status: str | Iterable[str] | None = None,
        source: str | Iterable[str] | None = None,
        camera_id: str | Iterable[str] | None = None,
        cls: str | Iterable[str] | None = None,
        holdout: bool | int | None = None,
        event_id: str | Iterable[str] | None = None,
        ids: Iterable[str] | None = None,
        since: float | None = None,
        until: float | None = None,
        limit: int | None = 50,
        offset: int = 0,
        order: str = "-created_at",
    ) -> tuple[list[dict[str, Any]], int]:
        """Фильтры через запятую или списком; cls — класс в метках или детекциях сэмпла."""
        if order not in ORDERS:
            raise SampleError(f"order must be one of {sorted(ORDERS)}")
        where: list[str] = []
        params: list[Any] = []
        for column, value in (("status", status), ("source", source), ("camera_id", camera_id), ("event_id", event_id)):
            values = _as_list(value)
            if values is not None:
                where.append(f"{column} IN ({', '.join('?' for _ in values)})")
                params.extend(values)
        if ids is not None:
            id_list = list(ids)
            if not id_list:
                return [], 0
            where.append(f"id IN ({', '.join('?' for _ in id_list)})")
            params.extend(id_list)
        classes = _as_list(cls)
        if classes is not None:
            marks = ", ".join("?" for _ in classes)
            where.append(
                "(EXISTS (SELECT 1 FROM json_each(samples.labels) j WHERE json_extract(j.value, '$.cls') IN "
                f"({marks})) OR EXISTS (SELECT 1 FROM json_each(samples.detections) j "
                f"WHERE json_extract(j.value, '$.cls') IN ({marks})))"
            )
            params.extend(classes)
            params.extend(classes)
        if holdout is not None:
            where.append("holdout = ?")
            params.append(1 if holdout else 0)
        if since is not None:
            where.append("created_at >= ?")
            params.append(float(since))
        if until is not None:
            where.append("created_at <= ?")
            params.append(float(until))
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            total = int(self._conn.execute(f"SELECT COUNT(*) FROM samples{clause}", params).fetchone()[0])
            sql = f"SELECT * FROM samples{clause} ORDER BY {ORDERS[order]}"
            page_params = list(params)
            if limit is not None:
                sql += " LIMIT ? OFFSET ?"
                page_params += [int(limit), int(offset)]
            rows = self._conn.execute(sql, page_params).fetchall()
        return [_row_to_sample(r) for r in rows], total

    def update(self, sample_id: str, **fields: Any) -> dict[str, Any] | None:
        unknown = set(fields) - _UPDATE_COLUMNS
        if unknown:
            raise SampleError(f"cannot update fields {sorted(unknown)}")
        if not fields:
            return self.get(sample_id)
        values: dict[str, Any] = {}
        for key, value in fields.items():
            if key == "status" and value not in SAMPLE_STATUSES:
                raise SampleError(f"status must be one of {SAMPLE_STATUSES}")
            if key == "operator_label" and value is not None and value not in OPERATOR_LABELS.values():
                raise SampleError(f"operator_label must be one of {sorted(OPERATOR_LABELS.values())}")
            if key == "labels":
                value = _dumps([normalize_label(lb, default_origin="operator") for lb in value or []])
            elif key == "meta":
                value = _dumps(dict(value or {}))
            elif key == "holdout":
                value = 1 if value else 0
            values[key] = value
        assignments = ", ".join(f"{k} = ?" for k in values)
        with self._lock:
            cur = self._conn.execute(f"UPDATE samples SET {assignments} WHERE id = ?", [*values.values(), sample_id])
            if cur.rowcount == 0:
                return None
            return self.get(sample_id)

    def review(
        self,
        sample_id: str,
        *,
        status: str | None = None,
        labels: list[Mapping[str, Any]] | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Решение человека (PATCH): status accepted/rejected и/или итоговые метки.

        Метка, совпавшая с прежней по классу и рамке, сохраняет origin и conf; изменённые и новые
        получают origin = operator. Ставится reviewed_at."""
        if status is not None and status not in REVIEW_STATUSES:
            raise SampleError(f"status must be one of {REVIEW_STATUSES}")
        with self._lock:
            current = self.get(sample_id)
            if current is None:
                return None
            fields: dict[str, Any] = {"reviewed_at": float(now if now is not None else time.time())}
            if status is not None:
                fields["status"] = status
            if labels is not None:
                old = {(lb["cls"], tuple(lb["bbox"])): lb for lb in current["labels"]}
                merged = []
                for raw in labels:
                    probe = normalize_label({**raw, "origin": "operator"})
                    prev = old.get((probe["cls"], tuple(probe["bbox"])))
                    merged.append(prev if prev is not None else probe)
                fields["labels"] = merged
            return self.update(sample_id, **fields)

    def set_operator_label(self, event_id: str, label: str | None) -> int:
        """Решение оператора по событию → operator_label сэмплов event этого события."""
        if label is not None and label not in OPERATOR_LABELS.values():
            raise SampleError(f"operator_label must be one of {sorted(OPERATOR_LABELS.values())}")
        with self._lock:
            cur = self._conn.execute(
                "UPDATE samples SET operator_label = ? WHERE event_id = ? AND source = 'event'", (label, event_id)
            )
            return int(cur.rowcount)

    def count(
        self, *, camera_id: str | None = None, sources: Iterable[str] | None = None, since: float | None = None
    ) -> int:
        where, params = [], []
        if camera_id is not None:
            where.append("camera_id = ?")
            params.append(camera_id)
        if sources is not None:
            src = list(sources)
            where.append(f"source IN ({', '.join('?' for _ in src)})")
            params.extend(src)
        if since is not None:
            where.append("created_at >= ?")
            params.append(since)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self._lock:
            return int(self._conn.execute(f"SELECT COUNT(*) FROM samples{clause}", params).fetchone()[0])

    def capped_today(self, now: float | None = None) -> dict[str, int]:
        """Сколько сэмплов под суточным лимитом (всё, кроме event и operator) уже есть сегодня по камерам."""
        since = day_start(now if now is not None else time.time())
        marks = ", ".join("?" for _ in UNCAPPED_SOURCES)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT camera_id, COUNT(*) FROM samples WHERE created_at >= ? AND source NOT IN ({marks}) "
                "GROUP BY camera_id",
                (since, *sorted(UNCAPPED_SOURCES)),
            ).fetchall()
        return {r[0]: int(r[1]) for r in rows}

    def summary(self, now: float | None = None) -> dict[str, Any]:
        """Счётчики для /api/flywheel/summary (без полей сборщика и моделей — их добавляет API)."""
        since = day_start(now if now is not None else time.time())
        with self._lock:
            total = int(self._conn.execute("SELECT COUNT(*) FROM samples").fetchone()[0])
            today = int(self._conn.execute("SELECT COUNT(*) FROM samples WHERE created_at >= ?", (since,)).fetchone()[0])
            by_source = dict(self._conn.execute("SELECT source, COUNT(*) FROM samples GROUP BY source").fetchall())
            by_status = dict(self._conn.execute("SELECT status, COUNT(*) FROM samples GROUP BY status").fetchall())
            by_camera = dict(self._conn.execute("SELECT camera_id, COUNT(*) FROM samples GROUP BY camera_id").fetchall())
            by_class = dict(
                self._conn.execute(
                    "SELECT json_extract(j.value, '$.cls') AS c, COUNT(DISTINCT s.id) FROM samples s, "
                    "json_each(s.labels) j GROUP BY c"
                ).fetchall()
            )
            by_detected = dict(
                self._conn.execute(
                    "SELECT json_extract(j.value, '$.cls') AS c, COUNT(DISTINCT s.id) FROM samples s, "
                    "json_each(s.detections) j GROUP BY c"
                ).fetchall()
            )
            by_operator = dict(
                self._conn.execute(
                    "SELECT operator_label, COUNT(*) FROM samples WHERE operator_label IS NOT NULL GROUP BY operator_label"
                ).fetchall()
            )
            reviewed = int(
                self._conn.execute("SELECT COUNT(*) FROM samples WHERE status IN ('accepted', 'rejected')").fetchone()[0]
            )
            holdout = int(self._conn.execute("SELECT COUNT(*) FROM samples WHERE holdout = 1").fetchone()[0])
        return {
            "total": total,
            "today": today,
            "by_source": {s: int(by_source.get(s, 0)) for s in SAMPLE_SOURCES},
            "by_status": {s: int(by_status.get(s, 0)) for s in SAMPLE_STATUSES},
            "by_class": {str(k): int(v) for k, v in sorted(by_class.items(), key=lambda kv: -kv[1]) if k},
            "by_detected_class": {str(k): int(v) for k, v in sorted(by_detected.items(), key=lambda kv: -kv[1]) if k},
            "by_camera": {str(k): int(v) for k, v in by_camera.items()},
            "by_operator_label": {str(k): int(v) for k, v in by_operator.items()},
            "reviewed": reviewed,
            "holdout": holdout,
            "last_dataset": self.last_dataset(),
        }

    # ---------- версии датасетов ----------

    def add_dataset(self, dataset: Mapping[str, Any]) -> dict[str, Any]:
        row = {
            "version": str(dataset["version"]),
            "created_at": float(dataset.get("created_at") or time.time()),
            "path": str(dataset.get("path") or ""),
            "classes": _dumps(list(dataset.get("classes") or [])),
            "n_train": int(dataset.get("n_train") or 0),
            "n_val": int(dataset.get("n_val") or 0),
            "n_hard_negative": int(dataset.get("n_hard_negative") or 0),
            "manifest": _dumps(dict(dataset.get("manifest") or {})),
        }
        with self._lock:
            self._conn.execute(
                f"INSERT OR REPLACE INTO datasets ({', '.join(_DATASET_COLUMNS)}) "
                f"VALUES ({', '.join('?' for _ in _DATASET_COLUMNS)})",
                [row[c] for c in _DATASET_COLUMNS],
            )
            stored = self.get_dataset(row["version"])
        assert stored is not None
        return stored

    def get_dataset(self, version: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets WHERE version = ?", (version,)).fetchone()
        return _row_to_dataset(row) if row is not None else None

    def list_datasets(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC, version DESC").fetchall()
        return [_row_to_dataset(r) for r in rows]

    def last_dataset(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM datasets ORDER BY created_at DESC, version DESC LIMIT 1").fetchone()
        return _row_to_dataset(row) if row is not None else None

    def mark_dataset_version(self, sample_ids: Iterable[str], version: str) -> int:
        ids = list(sample_ids)
        if not ids:
            return 0
        changed = 0
        with self._lock:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                cur = self._conn.execute(
                    f"UPDATE samples SET dataset_version = ? WHERE id IN ({', '.join('?' for _ in chunk)})",
                    [version, *chunk],
                )
                changed += cur.rowcount
        return changed

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
