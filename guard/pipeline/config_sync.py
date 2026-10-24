"""Синхронизация конфигурации камер gateway → edge (docs/EDGE.md §8, docs/EDGE_PROTOCOL.md).

Шлюз хранит желаемую конфигурацию каждой камеры с монотонной версией и отправляет её на edge
сообщением camera_config в конверте v1. Edge применяет только версию новее уже обработанной —
через атомарную конфигурацию движка (add/update/delete_camera: сначала запись, затем runtime), —
записывает её в оверлей рядом с базовым файлом камер и отвечает camera_config_ack.

Модуль лёгкий (без inference, OpenCV и источников): его импортирует и шлюз.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

import yaml

from guard.contracts import CameraConfig
from guard.pipeline.config import _dump_camera
from guard.pipeline.errors import CameraError
from guard.rules.base import available_rules, get_rule, with_default_severity
from guard.rules.geometry import validate_active_hours
from guard.transport.protocol import MAX_CONFIG_ERROR, MAX_SEQUENCE, Envelope, encode, identifier
from guard.transport.store import RUN_ID

log = logging.getLogger(__name__)

CONFIG_KIND = "camera_config"
ACK_KIND = "camera_config_ack"
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://\S*")


def edge_source_ref(edge_id: str, camera_id: str) -> str:
    """URI источника, под которым шлюз видит камеру edge: настоящий URI наружу не уходит."""
    return f"edge://{edge_id}/{camera_id}"


def safe_error(exc: BaseException | str) -> str:
    """Текст ошибки для подтверждения: без URI (в них бывают пароли), не длиннее предела протокола."""
    text = exc if isinstance(exc, str) else (str(exc) or type(exc).__name__)
    if isinstance(exc, KeyError):
        text = f"{type(exc).__name__}: {text}"
    text = _URL.sub("<uri>", " ".join(text.split())) or "Configuration was not applied"
    return text[:MAX_CONFIG_ERROR]


def ack_payload(camera_id: str, version: int, ok: bool, error: str | None, current_version: int) -> dict[str, Any]:
    return {
        "camera_id": camera_id, "version": version, "ok": ok,
        "error": None if ok else safe_error(error or "Configuration was not applied"),
        "current_version": current_version,
    }


def _same(left: CameraConfig, right: CameraConfig) -> bool:
    return left.model_dump(mode="json") == right.model_dump(mode="json")


def validate_camera_light(cam: CameraConfig) -> None:
    """Проверка на шлюзе до отправки: то же, что PipelineEngine.validate_camera, кроме реестра моделей.

    Реестр моделей есть только на edge, поэтому окончательно конфигурацию проверяет edge и сообщает
    ошибку в camera_config_ack. ID камеры, зон и правил — идентификаторы протокола событий."""
    try:
        identifier(cam.camera_id, "camera_id")
    except ValueError as exc:
        raise CameraError("camera_id: latin letters, digits, '_', '-', '.' (up to 128)") from exc
    if not 0 < cam.sample_fps <= 30:
        raise CameraError("sample_fps must be in (0, 30]")
    zone_ids = [z.zone_id for z in cam.zones]
    if len(set(zone_ids)) != len(zone_ids):
        raise CameraError("zone_id values must be unique")
    for z in cam.zones:
        try:
            identifier(z.zone_id, "zone_id")
        except ValueError as exc:
            raise CameraError(f"zone {z.zone_id!r}: zone_id must be a latin identifier") from exc
        if len(z.polygon) < 3:
            raise CameraError(f"zone {z.zone_id}: polygon needs at least 3 points")
        if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in z.polygon):
            raise CameraError(f"zone {z.zone_id}: polygon coordinates must be normalized to 0..1")
        try:
            validate_active_hours(z.active_hours)
        except ValueError as exc:
            raise CameraError(f"zone {z.zone_id}: {exc}") from exc
    known = {r["type"] for r in available_rules()}
    rule_ids = [r.rule_id for r in cam.rules]
    if len(set(rule_ids)) != len(rule_ids):
        raise CameraError("rule_id values must be unique within a camera")
    for r in cam.rules:
        try:
            identifier(r.rule_id, "rule_id")
        except ValueError as exc:
            raise CameraError(f"rule {r.rule_id!r}: rule_id must be a latin identifier") from exc
        if r.type not in known:
            raise CameraError(f"rule {r.rule_id}: unknown type {r.type!r}; available: {sorted(known)}")
        missing = set(r.zones or []) - set(zone_ids)
        if missing:
            raise CameraError(f"rule {r.rule_id}: unknown zones {sorted(missing)}")
        if r.min_duration_s < 0 or not 0 <= r.min_ratio <= 1:
            raise CameraError(f"rule {r.rule_id}: min_duration_s >= 0 and 0 <= min_ratio <= 1 required")
        try:
            get_rule(r.type)(r, cam)
        except (ValueError, TypeError, OverflowError) as exc:
            raise CameraError(f"rule {r.rule_id} ({r.type}): invalid configuration: {exc}") from exc


# ================================================================ edge: оверлей и применение


def overlay_path(cameras_config: str | Path, explicit: str | Path | None = None) -> Path:
    """Оверлей лежит рядом с базовым файлом: configs/cameras.stand.yaml → configs/.cameras.stand.overlay.yaml.

    Точка в начале имени: tools/stand.py sync не выгружает скрытые файлы, поэтому оверлей ноутбука
    не перезапишет оверлей стенда, а синхронизация кода не удалит его."""
    if explicit:
        return Path(explicit).expanduser()
    base = Path(cameras_config)
    return base.with_name(f".{base.stem}.overlay.yaml")


_OVERLAY_HEADER = (
    "# Оверлей камер edge (docs/EDGE.md §8): правки, принятые от шлюза по каналу camera_config.\n"
    "# Базовый файл камер не меняется. При старте edge читает базовый файл и применяет этот оверлей.\n"
    "# Чтобы вернуться к базовой конфигурации, остановите edge, удалите файл и запустите edge снова.\n"
)


class CameraOverlay:
    """Изменения камер поверх базового файла; запись атомарная (временный файл, fsync, os.replace).

    Запись камеры: version — последняя обработанная команда, ok/error — её результат,
    applied_version — версия действующей правки, op/config — сама правка (upsert | delete)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self.base: dict[str, CameraConfig] = {}
        self.entries: dict[str, dict[str, Any]] = self._read()

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.is_file():
            return {}
        try:
            raw = yaml.safe_load(self.path.read_text(encoding="utf-8")) or {}
            cameras = raw.get("cameras") or {}
            if not isinstance(cameras, dict):
                raise ValueError("cameras must be a mapping")
        except (OSError, ValueError, yaml.YAMLError) as exc:
            # Не роняем edge: файл откладываем для разбора, шлюз повторит свои правки (docs/EDGE.md §8).
            aside = self.path.with_name(f"{self.path.name}.corrupt-{int(time.time())}")
            log.error("cameras overlay %s unreadable (%s); moved to %s", self.path, type(exc).__name__, aside.name)
            try:
                os.replace(self.path, aside)
            except OSError:
                log.exception("cannot move unreadable overlay %s", self.path)
            return {}
        entries: dict[str, dict[str, Any]] = {}
        for camera_id, entry in cameras.items():
            try:
                identifier(camera_id, "camera_id")
                if not isinstance(entry, dict) or entry.get("op") not in ("upsert", "delete", None):
                    raise ValueError("invalid entry")
                for key in ("version", "applied_version"):
                    value = entry.get(key, 0)
                    if type(value) is not int or not 0 <= value <= MAX_SEQUENCE:
                        raise ValueError(f"invalid {key}")
                if entry.get("op") == "upsert":
                    CameraConfig.model_validate(entry.get("config"))
            except (ValueError, TypeError) as exc:
                log.error("cameras overlay %s: entry %r skipped: %s", self.path, camera_id, str(exc).splitlines()[0])
                continue
            entries[camera_id] = dict(entry)
        return entries

    # ---- чтение ----

    def merge(self, base: list[CameraConfig]) -> list[CameraConfig]:
        """Камеры при старте edge: базовый файл, затем правки оверлея (удаление, замена, новые камеры)."""
        with self._lock:
            self.base = {cam.camera_id: cam for cam in base}
            result = dict(self.base)
            for camera_id, entry in self.entries.items():
                if entry.get("op") == "delete":
                    result.pop(camera_id, None)
                elif entry.get("op") == "upsert":
                    result[camera_id] = with_default_severity(CameraConfig.model_validate(entry["config"]))
            if self.entries:
                log.info("cameras overlay %s: %s", self.path, ", ".join(
                    f"{cid}={entry.get('op') or 'base'}@{entry.get('applied_version', 0)}"
                    for cid, entry in self.entries.items()))
            return list(result.values())

    def record(self, camera_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self.entries.get(camera_id)
            return dict(entry) if entry is not None else None

    def applied_version(self, camera_id: str) -> int:
        with self._lock:
            return int((self.entries.get(camera_id) or {}).get("applied_version", 0))

    # ---- запись ----

    def commit(self, cameras: dict[str, CameraConfig], change: tuple[str, int] | None) -> None:
        """Новый набор камер → оверлей (разница с базовым файлом). change — (camera_id, версия) команды шлюза.

        Вызывается из PipelineEngine._persist: если запись не удалась, runtime не меняется."""
        with self._lock:
            target = change[0] if change else None
            entries = copy.deepcopy(self.entries)
            for camera_id in dict.fromkeys([*self.base, *cameras, *entries]):
                effective, base, entry = cameras.get(camera_id), self.base.get(camera_id), entries.get(camera_id)
                if effective is None:
                    if base is None and entry is None and camera_id != target:
                        continue
                    entry = {**(entry or {}), "op": "delete", "config": None}
                elif entry is None and camera_id != target and base is not None and _same(effective, base):
                    continue  # базовая конфигурация, правок нет
                else:
                    entry = {**(entry or {}), "op": "upsert", "config": _dump_camera(effective)}
                entries[camera_id] = entry
            if change is not None:
                camera_id, version = change
                entries[camera_id].update(version=version, applied_version=version, ok=True, error=None,
                                          processed_at=round(time.time(), 3))
            self._write(entries)
            self.entries = entries

    def record_result(self, camera_id: str, version: int, *, ok: bool, error: str | None) -> None:
        """Результат команды, которая не изменила действующую конфигурацию (или изменила, но с ошибкой запуска)."""
        with self._lock:
            entries = copy.deepcopy(self.entries)
            entry = entries.setdefault(camera_id, {"op": None, "config": None, "applied_version": 0})
            entry.update(version=version, ok=ok, error=error, processed_at=round(time.time(), 3))
            self._write(entries)
            self.entries = entries

    def _write(self, entries: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = _OVERLAY_HEADER + yaml.safe_dump(
            {"format": 1, "cameras": entries}, allow_unicode=True, sort_keys=False, default_flow_style=None, width=120,
        )
        fd, tmp = tempfile.mkstemp(prefix=f"{self.path.name}.", suffix=".tmp", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class EdgeConfigApplier:
    """Edge: camera_config → атомарная конфигурация движка, строго по версиям камеры.

    Версия меньше обработанной — устаревшая команда, не применяется. Равная — повтор доставки:
    edge отвечает прежним результатом и ничего не перезапускает."""

    def __init__(self, engine: Any, overlay: CameraOverlay, edge_id: str) -> None:
        self.engine = engine
        self.overlay = overlay
        self.edge_id = identifier(edge_id, "edge_id")
        self._lock = threading.Lock()
        self.counters = {"applied": 0, "failed": 0, "duplicate": 0, "stale": 0}
        self.last_error: str | None = None
        self.last_applied_at: float | None = None

    def handle(self, envelope: Envelope) -> dict[str, Any]:
        """Ответ camera_config_ack (payload). Ошибки конфигурации не бросает — они уходят в ответ."""
        if envelope.kind != CONFIG_KIND or envelope.edge_id != self.edge_id:
            raise ValueError("Unexpected control message")
        payload = envelope.payload
        camera_id, version = payload["camera_id"], payload["version"]
        with self._lock:
            record = self.overlay.record(camera_id) or {}
            seen, current = int(record.get("version", 0)), int(record.get("applied_version", 0))
            if version < seen:
                self.counters["stale"] += 1
                return ack_payload(camera_id, version, False, f"stale version {version}: edge already processed {seen}", current)
            if version == seen:
                self.counters["duplicate"] += 1
                ok = bool(record.get("ok"))
                return ack_payload(camera_id, version, ok, record.get("error"), current)
            ok, error = self._apply(payload)
            self.counters["applied" if ok else "failed"] += 1
            self.last_error = error
            if ok:
                self.last_applied_at = time.time()
            return ack_payload(camera_id, version, ok, error, self.overlay.applied_version(camera_id))

    def _apply(self, payload: dict[str, Any]) -> tuple[bool, str | None]:
        camera_id, version, op = payload["camera_id"], payload["version"], payload["op"]
        engine = self.engine
        with engine._lock:  # RLock движка: _persist видит версию этой команды
            engine._config_change = (camera_id, version)
            try:
                current = engine.get_camera(camera_id)
                if op == "delete":
                    if current is None:
                        self.overlay.commit(dict(engine.cameras), (camera_id, version))
                    else:
                        engine.delete_camera(camera_id)
                else:
                    camera = self._resolve_source(CameraConfig.model_validate(payload["config"]), current)
                    if current is None:
                        engine.add_camera(camera)
                    elif _same(with_default_severity(camera), current):
                        self.overlay.commit(dict(engine.cameras), (camera_id, version))  # без перезапуска камеры
                    else:
                        engine.update_camera(camera_id, camera)
            except Exception as exc:  # noqa: BLE001 — любая ошибка уходит шлюзу в подтверждении
                error = safe_error(exc)
                committed = self.overlay.applied_version(camera_id) == version
                log.warning("camera_config %s v%s %s: %s", camera_id, version,
                            "committed, runtime failed" if committed else "rejected", error)
                self.overlay.record_result(camera_id, version, ok=False, error=error)
                return False, error
            finally:
                engine._config_change = None
        log.info("camera_config %s v%s applied: %s", camera_id, version, op)
        return True, None

    def _resolve_source(self, camera: CameraConfig, current: CameraConfig | None) -> CameraConfig:
        """edge://<edge>/<camera> — «источник не меняется»: подставляем свой URI. Иначе — новый источник."""
        ref = edge_source_ref(self.edge_id, camera.camera_id)
        if camera.source.uri != ref:
            if camera.source.uri.startswith("edge://"):
                raise CameraError("source.uri refers to another edge camera")
            return camera
        if current is None:
            raise CameraError("source.uri is required for a new camera")
        if camera.source.type != current.source.type:
            raise CameraError("changing source.type requires a new source.uri")
        return camera.model_copy(update={"source": camera.source.model_copy(update={"uri": current.source.uri})})

    def status(self) -> dict[str, Any]:
        return {**self.counters, "last_error": self.last_error, "last_applied_at": self.last_applied_at,
                "overlay": self.overlay.path.name}


# ================================================================ шлюз: желаемая конфигурация


_GATEWAY_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS control_producer (
        singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
        edge_id TEXT NOT NULL,
        producer_id TEXT NOT NULL,
        next_sequence INTEGER NOT NULL CHECK(next_sequence > 0),
        version_floor INTEGER NOT NULL DEFAULT 0
    )""",
    """CREATE TABLE IF NOT EXISTS camera_config_desired (
        camera_id TEXT PRIMARY KEY,
        version INTEGER NOT NULL,
        op TEXT NOT NULL CHECK(op IN ('upsert', 'delete')),
        config TEXT,
        state TEXT NOT NULL CHECK(state IN ('pending', 'synced', 'error')),
        error TEXT,
        message_id TEXT NOT NULL,
        wire BLOB NOT NULL,
        updated_at REAL NOT NULL,
        acked_at REAL
    )""",
)


class GatewayConfigSync:
    """Шлюз: желаемая конфигурация камер, её версии и состояние синхронизации с edge (SQLite).

    Команда camera_config «замораживается» при создании (как outbox событий edge): повторная
    отправка после переподключения шлёт те же байты с тем же message_id. Версия — max(предыдущая + 1,
    время в мс): монотонна и после потери БД шлюза. Состояния: pending → synced | error."""

    def __init__(self, db_path: str | Path, edge_id: str, *, notify: Callable[[dict[str, Any]], None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.edge_id = identifier(edge_id, "edge_id")
        self._notify = notify
        self._clock = clock
        self._lock = threading.Lock()
        self._send: Callable[[str, bytes], None] | None = None
        self._edge_versions: dict[str, int] = {}
        self._edge_run: tuple[str, str] | None = None
        path = str(db_path)
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = FULL")
            with self._conn:
                self._conn.execute("BEGIN IMMEDIATE")
                for statement in _GATEWAY_SCHEMA:
                    self._conn.execute(statement)
                row = self._conn.execute("SELECT edge_id, producer_id FROM control_producer WHERE singleton = 1").fetchone()
                if row is None:
                    self.producer_id = str(uuid4())
                    self._conn.execute("INSERT INTO control_producer VALUES (1, ?, ?, 1, 0)", (self.edge_id, self.producer_id))
                elif row["edge_id"] != self.edge_id:
                    raise ValueError("Config sync store already belongs to another edge")
                else:
                    self.producer_id = row["producer_id"]
        except BaseException:
            self._conn.close()
            raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def attach(self, send: Callable[[str, bytes], None] | None) -> None:
        """send(message_id, wire) — очередь отправки канала; без канала команды ждут переподключения."""
        self._send = send

    # ---- команды ----

    def submit(self, op: str, camera_id: str, config: CameraConfig | None = None) -> dict[str, Any]:
        if op not in ("upsert", "delete") or (op == "upsert") != (config is not None):
            raise ValueError("Invalid camera_config operation")
        identifier(camera_id, "camera_id")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            producer = self._conn.execute("SELECT * FROM control_producer WHERE singleton = 1").fetchone()
            previous = self._row(camera_id)
            version = max(producer["version_floor"] + 1, int(self._clock() * 1000),
                          (previous["version"] + 1) if previous is not None else 1)
            if version > MAX_SEQUENCE or producer["next_sequence"] >= MAX_SEQUENCE:
                raise ValueError("Config version exhausted")
            if config is not None:
                config = self._keep_source(config, previous)
            payload = {"camera_id": camera_id, "version": version, "op": op,
                       "config": None if config is None else config.model_dump(mode="json")}
            sequence = producer["next_sequence"]
            envelope = Envelope(
                edge_id=self.edge_id, producer_id=self.producer_id, run_id=RUN_ID, sequence=sequence,
                message_id=f"{self.producer_id}:{sequence}", kind=CONFIG_KIND, sent_at=self._clock(), payload=payload,
            )
            wire = encode(envelope)
            self._conn.execute("UPDATE control_producer SET next_sequence = ?, version_floor = ? WHERE singleton = 1",
                               (sequence + 1, version))
            self._conn.execute(
                "INSERT OR REPLACE INTO camera_config_desired VALUES (?, ?, ?, ?, 'pending', NULL, ?, ?, ?, NULL)",
                (camera_id, version, op, None if config is None else json.dumps(payload["config"], ensure_ascii=False),
                 envelope.message_id, wire, self._clock()),
            )
            state = self._state(self._row(camera_id))
        self._emit([state])
        self._dispatch([(envelope.message_id, wire)])
        return state

    @staticmethod
    def _keep_source(config: CameraConfig, previous: sqlite3.Row | None) -> CameraConfig:
        """Ещё не применённая смена источника не теряется, если следующая правка пришла с edge-ссылкой."""
        if previous is None or previous["op"] != "upsert" or not config.source.uri.startswith("edge://"):
            return config
        known = CameraConfig.model_validate(json.loads(previous["config"])).source
        if known.uri.startswith("edge://") or known.type != config.source.type:
            return config
        return config.model_copy(update={"source": config.source.model_copy(update={"uri": known.uri})})

    def on_ack(self, envelope: Envelope) -> dict[str, Any] | None:
        if envelope.kind != ACK_KIND or envelope.edge_id != self.edge_id:
            raise ValueError("Unexpected control message")
        payload = envelope.payload
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._raise_floor(max(payload["version"], payload["current_version"]))
            row = self._row(payload["camera_id"])
            if row is None or row["version"] != payload["version"]:
                return None  # подтверждение старой или чужой версии
            state, error = ("synced", None) if payload["ok"] else ("error", payload["error"])
            if row["state"] == state and row["error"] == error:
                return None  # повтор подтверждения
            self._conn.execute("UPDATE camera_config_desired SET state = ?, error = ?, acked_at = ? WHERE camera_id = ?",
                               (state, error, self._clock(), row["camera_id"]))
            result = self._state(self._row(row["camera_id"]))
        self._emit([result])
        return result

    def observe_catalogue(self, producer_id: str, run_id: str, cameras: dict[str, CameraConfig],
                          statuses: dict[str, Any]) -> None:
        """Каталог edge: версии конфигурации камер (status.config_version). Применённая версия подтверждает
        правку, даже если ack потерялся. Новый процесс edge получает неподтверждённые правки повторно, а
        правки, которых у него больше нет (например, удалён оверлей), — снова в pending."""
        versions = {cid: status["config_version"] for cid, status in statuses.items()
                    if isinstance(status, dict) and type(status.get("config_version")) is int}
        new_run = (producer_id, run_id) != self._edge_run
        changed, resend = [], []
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            self._raise_floor(max(versions.values(), default=0))
            for row in self._conn.execute("SELECT * FROM camera_config_desired ORDER BY version").fetchall():
                camera_id, edge_version = row["camera_id"], versions.get(row["camera_id"])
                if row["state"] == "pending":
                    if edge_version is not None and edge_version >= row["version"]:
                        self._set_state(camera_id, "synced", None)
                        changed.append(camera_id)
                    elif new_run:
                        resend.append((row["message_id"], row["wire"]))
                elif row["state"] == "synced" and new_run and versions:
                    lost = ((row["op"] == "upsert" and (edge_version or 0) < row["version"])
                            or (row["op"] == "delete" and camera_id in cameras))
                    if lost:
                        self._set_state(camera_id, "pending", None)
                        changed.append(camera_id)
                        resend.append((row["message_id"], row["wire"]))
            states = [self._state(self._row(camera_id)) for camera_id in changed]
            self._edge_versions = versions
            self._edge_run = (producer_id, run_id)
        self._emit(states)
        self._dispatch(resend)

    def pending_wires(self) -> list[tuple[str, bytes]]:
        """Неподтверждённые команды — для повторной отправки при (пере)подключении канала."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT message_id, wire FROM camera_config_desired WHERE state = 'pending' ORDER BY version"
            ).fetchall()
        return [(row["message_id"], bytes(row["wire"])) for row in rows]

    # ---- представление для API ----

    def view(self, catalogue: dict[str, CameraConfig]) -> dict[str, CameraConfig]:
        """Камеры для /api/cameras: каталог edge, поверх — желаемые правки, которые edge ещё не показал.

        Отклонённая правка существующей камеры не показывается как действующая: виден каталог edge,
        а ошибка — в status.config_sync."""
        result = dict(catalogue)
        with self._lock:
            rows = self._conn.execute("SELECT camera_id, version, op, config, state FROM camera_config_desired").fetchall()
            versions = dict(self._edge_versions)
        for row in rows:
            camera_id = row["camera_id"]
            shown = versions.get(camera_id, 0) if camera_id in catalogue else 0
            if row["version"] <= shown or (row["state"] == "error" and camera_id in catalogue):
                continue
            if row["op"] == "delete":
                result.pop(camera_id, None)
                continue
            camera = CameraConfig.model_validate(json.loads(row["config"]))
            result[camera_id] = camera.model_copy(update={
                "source": camera.source.model_copy(update={"uri": edge_source_ref(self.edge_id, camera_id)})})
        return result

    def status(self, camera_id: str) -> dict[str, Any]:
        with self._lock:
            row = self._row(camera_id)
            edge_version = self._edge_versions.get(camera_id)
        if row is None:  # шлюз камеру не менял: действует конфигурация edge
            return {"camera_id": camera_id, "state": "synced", "version": edge_version, "op": None, "error": None,
                    "updated_at": None, "acked_at": None, "edge_version": edge_version}
        return {**self._state(row), "edge_version": edge_version}  # edge_version — из каталога edge

    def summary(self) -> dict[str, Any]:
        with self._lock:
            counts = dict(self._conn.execute(
                "SELECT state, COUNT(*) FROM camera_config_desired GROUP BY state").fetchall())
        return {state: counts.get(state, 0) for state in ("pending", "synced", "error")}

    # ---- внутреннее ----

    def _row(self, camera_id: str) -> sqlite3.Row | None:
        return self._conn.execute("SELECT * FROM camera_config_desired WHERE camera_id = ?", (camera_id,)).fetchone()

    def _set_state(self, camera_id: str, state: str, error: str | None) -> None:
        self._conn.execute("UPDATE camera_config_desired SET state = ?, error = ?, acked_at = ? WHERE camera_id = ?",
                           (state, error, self._clock() if state != "pending" else None, camera_id))

    def _raise_floor(self, version: int) -> None:
        self._conn.execute("UPDATE control_producer SET version_floor = MAX(version_floor, ?) WHERE singleton = 1",
                           (int(version),))

    @staticmethod
    def _state(row: sqlite3.Row) -> dict[str, Any]:
        return {"camera_id": row["camera_id"], "state": row["state"], "version": row["version"], "op": row["op"],
                "error": row["error"], "updated_at": row["updated_at"], "acked_at": row["acked_at"]}

    def _emit(self, states: list[dict[str, Any] | None]) -> None:
        if self._notify is None:
            return
        for state in states:
            if state is not None:
                try:
                    self._notify({"kind": "camera_sync", **state})
                except Exception:  # noqa: BLE001 — уведомление UI не влияет на состояние
                    log.exception("camera_sync notification failed")

    def _dispatch(self, items: list[tuple[str, bytes]]) -> None:
        send = self._send
        if send is None:
            return
        for message_id, wire in items:
            try:
                send(message_id, bytes(wire))
            except Exception as exc:  # noqa: BLE001 — команда остаётся pending и уйдёт при переподключении
                log.warning("camera_config %s not sent now (%s); will resend on reconnect", message_id, type(exc).__name__)
