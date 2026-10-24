"""Durable edge CLI and compatibility helpers for upstream integrations.

The production build/run path uses EdgeEngine: SQLite outbox, versioned JSON
Envelope messages, separate event and bounded telemetry.v2 channels. EdgeSink,
GatewayQueueGuard and EdgeStats below remain compatibility helpers only.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import signal
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "2")  # контейнер стенда: 1.5 CPU, 4 ГиБ

import yaml  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field  # noqa: E402

from guard.pipeline.engine import PipelineEngine  # noqa: E402
from guard.pipeline.settings import CODE_ROOT, Settings, load_dotenv_guard, project_root  # noqa: E402
from guard.pipeline.transport import make_sink, metrics_msg  # noqa: E402
from guard.sysinfo import DEFAULT_NTP_SERVERS, CpuSampler, NtpClock, memory_snapshot  # noqa: E402
from guard.transport import HeartbeatMsg  # noqa: E402

log = logging.getLogger("guard.edge")

STAND_BROKER_HOST = "10.0.114.30"
GATEWAY_QUEUE = "zazerkalye.gateway"
HEARTBEAT_IDLE_S = 30.0


def disable_ort_spinning() -> bool:
    """Все сессии onnxruntime процесса — без активного ожидания потоков (allow_spinning=0).

    Хвосты бэкенда linq-tpu (DFL-декод, ~1 мс) создаются с 2 потоками и spinning по умолчанию: на стенде
    это 91 мс CPU на вызов вместо 21.6 мс (замер 24.10, docs/EDGE.md), и три хвоста съедают квоту 1.5 CPU.
    Подмена класса на уровне модуля действует и на сессии, которые бэкенды создадут позже.
    Отключение: GUARD_ORT_SPINNING=1."""
    if os.environ.get("GUARD_ORT_SPINNING", "0") == "1":
        return False
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    base = ort.InferenceSession
    if getattr(base, "_guard_no_spin", False):
        return True

    class NoSpinSession(base):  # type: ignore[misc, valid-type]
        _guard_no_spin = True

        def __init__(self, path_or_bytes: Any, sess_options: Any = None, *args: Any, **kwargs: Any) -> None:
            so = sess_options if sess_options is not None else ort.SessionOptions()
            for key in ("session.intra_op.allow_spinning", "session.inter_op.allow_spinning"):
                try:  # onnx_cpu уже выключает spinning сам: повторная запись даёт предупреждение ORT
                    already = so.get_session_config_entry(key) == "0"
                except RuntimeError:
                    already = False
                if not already:
                    so.add_session_config_entry(key, "0")
            super().__init__(path_or_bytes, so, *args, **kwargs)

    ort.InferenceSession = NoSpinSession
    return True


# ---------------------------------------------------------------- секция edge в файле камер


class LinqSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    raw: bool = True  # raw-упаковка на клиенте: трафик в 3–4 раза меньше, к/с ×2 (LINQ_NOTES §13)
    window: int | None = Field(default=None, ge=1, le=32)  # задач в полёте для submit()
    timeout_s: float | None = Field(default=None, gt=0)


class GuardSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool = True
    backlog_limit: int = Field(default=300, ge=1)
    period_s: float = Field(default=2.0, gt=0)


class EdgeSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: str | None = None
    models_config: str | None = None
    models_dir: str | None = None
    data_dir: str | None = None
    rabbit_url: str | None = None
    node_id: str | None = None
    publish_frame_meta: bool | None = None
    ntp_servers: list[str] = Field(default_factory=lambda: list(DEFAULT_NTP_SERVERS))
    linq: LinqSection = Field(default_factory=LinqSection)
    gateway_guard: GuardSection = Field(default_factory=GuardSection)
    settings: dict[str, Any] = Field(default_factory=dict)  # поля guard.pipeline.settings.Settings


def load_edge_section(config: str | Path | None) -> EdgeSection:
    if not config:
        return EdgeSection()
    path = Path(config).expanduser()
    if not path.is_file():
        return EdgeSection()
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    section = raw.get("edge") if isinstance(raw, dict) else None
    return EdgeSection.model_validate(section or {})


def _project_path(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (project_root() / p)


def on_stand() -> bool:
    """Процесс работает в Jupyter-контейнере стенда (как linq_tpu._default_host)."""
    return bool(os.environ.get("JUPYTERHUB_USER")) or Path("/auto/cache").is_dir()


def default_rabbit_url() -> str | None:
    """На стенде брокер — 10.0.114.30 (RTT 1 мс); иначе None: RabbitSink возьмёт GUARD_TRANSPORT_URL или 127.0.0.1."""
    return f"amqp://guest:guest@{STAND_BROKER_HOST}:5672/" if on_stand() else None


def apply_edge_section(settings: Settings, section: EdgeSection) -> list[str]:
    """Секция edge → настройки (если соответствующая GUARD_* не задана). Возвращает применённые ключи."""
    applied: list[str] = []
    env = os.environ
    if section.backend and not env.get("GUARD_BACKEND"):
        settings.backend = section.backend
        applied.append("backend")
    if section.models_config and not env.get("GUARD_MODELS_CONFIG"):
        settings.models_config = _project_path(section.models_config)
        applied.append("models_config")
    if section.models_dir and not env.get("GUARD_MODELS_DIR"):
        settings.models_dir = _project_path(section.models_dir)
        applied.append("models_dir")
    if section.data_dir and not env.get("GUARD_DATA_DIR"):
        settings.data_dir = _project_path(section.data_dir)
        if not env.get("GUARD_MEDIA_DIR"):
            settings.media_dir = settings.data_dir / "media"
        if not env.get("GUARD_DB_PATH"):
            settings.db_path = settings.data_dir / "guard.db"
        applied.append("data_dir")
    if section.rabbit_url and not env.get("GUARD_RABBIT_URL"):
        settings.rabbit_url = section.rabbit_url
        applied.append("rabbit_url")
    if section.publish_frame_meta is not None and not env.get("GUARD_FRAME_META"):
        settings.publish_frame_meta = section.publish_frame_meta
        applied.append("publish_frame_meta")
    known = {f.name: f for f in dataclasses.fields(Settings)}
    for key, value in section.settings.items():
        if key not in known or key in {"data_dir", "models_dir", "cameras_config", "models_config", "db_path",
                                       "media_dir", "frontend_dist", "mode"}:
            raise ValueError(f"edge.settings: unsupported key {key!r}")
        current = getattr(settings, key)
        if isinstance(current, bool):
            value = bool(value)
        elif isinstance(current, int):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        setattr(settings, key, value)
        applied.append(f"settings.{key}")
    return applied


def _linq_env(section: LinqSection) -> None:
    """LinqTpuBackend читает режим из окружения при создании (движок создаёт бэкенд сам)."""
    os.environ.setdefault("LINQ_RAW", "1" if section.raw else "0")
    if section.window:
        os.environ.setdefault("LINQ_WINDOW", str(section.window))
    if section.timeout_s:
        os.environ.setdefault("LINQ_TIMEOUT_S", str(section.timeout_s))


# ---------------------------------------------------------------- транспорт: учёт и защита канала


class GatewayQueueGuard:
    """Состояние очереди шлюза по passive declare раз в period_s (отдельное соединение, свой поток)."""

    def __init__(self, url: str | None, backlog_limit: int = 300, period_s: float = 2.0,
                 queue: str = GATEWAY_QUEUE) -> None:
        self.url = url or os.environ.get("GUARD_TRANSPORT_URL") or "amqp://guest:guest@127.0.0.1:5672/"
        self.queue = queue
        self.backlog_limit = backlog_limit
        self.period_s = period_s
        self.consumers: int | None = None
        self.messages: int | None = None
        self.checked_at: float | None = None
        self.error: str | None = None
        self.backlog_paused = False
        self._last_idle_heartbeat = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="edge-gateway-guard", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def update(self, consumers: int, messages: int) -> None:
        self.consumers, self.messages = int(consumers), int(messages)
        self.checked_at = time.time()
        self.error = None
        if self.messages > self.backlog_limit:
            self.backlog_paused = True
        elif self.messages <= self.backlog_limit // 2:
            self.backlog_paused = False

    def _run(self) -> None:
        import pika

        connection = None
        channel = None
        while not self._stop.is_set():
            try:
                if connection is None or not connection.is_open:
                    params = pika.URLParameters(self.url)
                    params.connection_attempts = 1
                    params.socket_timeout = 3.0
                    params.blocked_connection_timeout = 3.0
                    params.heartbeat = 30
                    connection = pika.BlockingConnection(params)
                    channel = None
                if channel is None or not channel.is_open:  # канал один на всё время: без шума в логе
                    channel = connection.channel()
                res = channel.queue_declare(queue=self.queue, passive=True)
                self.update(res.method.consumer_count, res.method.message_count)
            except Exception as exc:  # noqa: BLE001 — без сведений об очереди ничего не блокируем
                channel = None
                if "NOT_FOUND" in str(exc) or "404" in str(exc):
                    self.update(0, 0)  # очередь ещё не объявлена: шлюза точно нет (брокер закрыл канал)
                else:
                    self.error = type(exc).__name__
                    self.consumers = self.messages = None
                    try:
                        if connection is not None:
                            connection.close()
                    except Exception:  # noqa: BLE001
                        pass
                    connection = None
            self._stop.wait(self.period_s)
        try:
            if connection is not None and connection.is_open:
                connection.close()
        except Exception:  # noqa: BLE001
            pass

    def allows(self, kind: str) -> bool:
        if kind in ("event", "event_update") or self.consumers is None:
            return True
        if self.consumers == 0:
            if kind == "heartbeat" and time.monotonic() - self._last_idle_heartbeat >= HEARTBEAT_IDLE_S:
                self._last_idle_heartbeat = time.monotonic()
                return True
            return False
        if kind == "frame_meta":
            return not self.backlog_paused
        return True

    def state(self) -> dict[str, Any]:
        return {
            "queue": self.queue,
            "consumers": self.consumers,
            "messages": self.messages,
            "backlog_limit": self.backlog_limit,
            "frame_meta_paused": bool(self.consumers == 0 or self.backlog_paused) if self.consumers is not None else False,
            "error": self.error,
        }


class EdgeSink:
    """Обёртка Sink движка: счётчики по видам и камерам, details.edge.sent_at в событиях, защита канала.

    Исключения внутреннего sink пробрасываются: их считает PipelineEngine._sink_publish (sink_errors)."""

    def __init__(self, inner: Any, node_id: str = "edge", guard: GatewayQueueGuard | None = None) -> None:
        self.inner = inner
        self.node_id = node_id
        self.guard = guard
        self._lock = threading.Lock()
        self.sent: dict[str, int] = defaultdict(int)
        self.held: dict[str, int] = defaultdict(int)  # не отправлено защитой канала (шлюз не слушает/backlog)
        self.rejected: dict[str, int] = defaultdict(int)  # sink вернул False (переполнение очереди отправки)
        self.frame_meta_by_camera: dict[str, int] = defaultdict(int)
        self.snapshot_bytes = 0
        self.snapshots = 0
        self.last_event_sent_at: float | None = None

    @staticmethod
    def _kind(message: Any) -> str:
        kind = getattr(message, "kind", None)
        if kind is None and isinstance(message, dict):
            kind = message.get("kind")
        return str(kind or "?")

    def _stamp(self, message: Any) -> Any:
        event = getattr(message, "event", None)
        if event is None:
            return message
        sent_at = time.time()
        details = {**event.details, "edge": {"node": self.node_id, "sent_at": round(sent_at, 3)}}
        self.last_event_sent_at = sent_at
        return message.model_copy(update={"event": event.model_copy(update={"details": details})})

    def publish(self, message: Any, **kwargs: Any) -> bool:
        kind = self._kind(message)
        if self.guard is not None and not self.guard.allows(kind):
            with self._lock:
                self.held[kind] += 1
            return False
        if kind in ("event", "event_update"):
            message = self._stamp(message)
        accepted = self.inner.publish(message, **kwargs)
        with self._lock:
            if accepted is False:
                self.rejected[kind] += 1
            else:
                self.sent[kind] += 1
                if kind == "frame_meta":
                    self.frame_meta_by_camera[getattr(message, "camera_id", "?")] += 1
                snap = getattr(message, "snapshot_jpeg", None)
                if snap:
                    self.snapshots += 1
                    self.snapshot_bytes += len(snap)
        return accepted

    def close(self, timeout: float | None = 5.0) -> None:
        if self.guard is not None:
            self.guard.stop()
        close = getattr(self.inner, "close", None)
        if callable(close):
            close(timeout=timeout)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            out: dict[str, Any] = {
                "sent": dict(self.sent),
                "held_by_guard": dict(self.held),
                "rejected": dict(self.rejected),
                "frame_meta_by_camera": dict(self.frame_meta_by_camera),
                "snapshots": self.snapshots,
                "snapshot_kb_avg": round(self.snapshot_bytes / self.snapshots / 1024, 1) if self.snapshots else None,
            }
        pending = getattr(self.inner, "pending", None)
        dropped = getattr(self.inner, "dropped", None)
        if isinstance(pending, int):
            out["outbox_pending"] = pending
        if isinstance(dropped, dict):
            out["outbox_dropped"] = dict(dropped)  # вытеснено из очереди отправки или истёк TTL
        if self.guard is not None:
            out["gateway_queue"] = self.guard.state()
        return out


# ---------------------------------------------------------------- сборка и запуск


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m guard.edge", description="Конвейер «Зазеркалья» без UI (режим edge)")
    p.add_argument("--config", help="cameras.yaml (по умолчанию GUARD_CAMERAS_CONFIG или configs/cameras.yaml)")
    p.add_argument("--models-config", help="реестр моделей (по умолчанию edge.models_config или configs/models.yaml)")
    p.add_argument("--backend", help="onnx-cpu | linq-tpu (по умолчанию GUARD_BACKEND или edge.backend)")
    p.add_argument("--transport", choices=["rabbit", "memory", "log"], help="по умолчанию GUARD_TRANSPORT или rabbit")
    p.add_argument("--url", help="event-bus AMQP URL (GUARD_RABBIT_URL / edge.rabbit_url), отдельно от LINQ_RABBIT_*")
    p.add_argument("--no-frame-meta", action="store_true", help="не слать frame_meta (только события и метрики)")
    p.add_argument("--no-gateway-guard", action="store_true", help="совместимый флаг; durable transport использует отдельную bounded telemetry.v2")
    p.add_argument("--metrics-period", type=float, default=2.0)
    p.add_argument("--stats-period", type=float, default=10.0, help="строка статистики в лог раз в N секунд")
    p.add_argument("--duration", type=float, default=0.0, help="остановиться через N секунд (смоук-тест)")
    p.add_argument("--node-id", help="durable edge ID (GUARD_EDGE_ID / GUARD_NODE_ID / edge.node_id)")
    args = p.parse_args(argv)
    if not math.isfinite(args.duration) or args.duration < 0:
        p.error("--duration must be finite and nonnegative")
    if any(not math.isfinite(value) or value <= 0 for value in (args.metrics_period, args.stats_period)):
        p.error("metric and stats periods must be finite and positive")
    return args


class EnvelopeMemoryPublisher:
    """Explicit in-process/debug transport; it never changes the durable wire."""

    def __init__(self, *, log_messages: bool = False):
        from collections import deque
        self.events = deque()
        self.telemetry = deque(maxlen=32)
        self.log_messages = log_messages
        self.published = 0

    def publish(self, envelope, *, durable=True):
        from guard.transport.protocol import encode
        if durable != (envelope.kind in {"event", "event_update"}):
            raise ValueError("Only events belong in the durable channel")
        wire = encode(envelope)
        (self.events if durable else self.telemetry).append(wire)
        self.published += 1
        if self.log_messages:
            log.info("edge envelope kind=%s sequence=%s", envelope.kind, envelope.sequence)

    def close(self):
        pass

    def status(self):
        return {"connected": True, "published": self.published, "events_pending": len(self.events)}


def build(args: argparse.Namespace, *, backend=None, publisher=None, telemetry_publisher=None):
    from guard.edge.runner import EdgeEngine
    from guard.inference import normalize_backend_name

    # Identity and camera selection must see dotenv values before the temporary
    # edge-mode overrides used to satisfy Settings validation are installed.
    load_dotenv_guard()
    config = _project_path(str(args.config or os.environ.get("GUARD_CAMERAS_CONFIG") or CODE_ROOT / "configs/cameras.yaml"))
    section = load_edge_section(config)
    node_id = args.node_id or os.environ.get("GUARD_EDGE_ID") or os.environ.get("GUARD_NODE_ID") or section.node_id or "edge"
    # Settings validates durable identity at construction; CLI identity must be
    # available before from_env(), without leaving test/global mode overrides.
    overrides = {"GUARD_MODE": "edge", "GUARD_EDGE_ID": node_id}
    previous = {key: os.environ.get(key) for key in overrides}
    try:
        os.environ.update(overrides)
        settings = Settings.from_env()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    applied = apply_edge_section(settings, section)
    settings.mode = "edge"
    settings.edge_id = node_id
    settings.clips_enabled = False
    settings.autostart = True
    settings.telemetry_interval_s = args.metrics_period
    settings.cameras_config = _project_path(str(config)).resolve()
    if args.models_config:
        settings.models_config = _project_path(args.models_config).resolve()
    if args.backend:
        settings.backend = args.backend
    settings.backend = normalize_backend_name(settings.backend)
    if args.transport:
        settings.transport = args.transport
    if args.url:
        settings.rabbit_url = args.url
    if args.no_frame_meta:
        settings.publish_frame_meta = False
    if settings.transport == "rabbit" and not settings.rabbit_url:
        settings.rabbit_url = os.environ.get("GUARD_TRANSPORT_URL") or default_rabbit_url() or "amqp://guest:guest@127.0.0.1:5672/"
    if "tpu" in settings.backend:
        # External inference is fail-closed even if a shell inherited CPU fallback.
        os.environ["GUARD_BACKEND_FALLBACK"] = "none"
        _linq_env(section.linq)
        os.environ.setdefault("LINQ_TIMEOUT_S", "5")
    if disable_ort_spinning():
        applied.append("ort_no_spinning")
    settings.__post_init__()
    if settings.transport in {"memory", "log"}:
        publisher = publisher if publisher is not None else EnvelopeMemoryPublisher(log_messages=settings.transport == "log")
        telemetry_publisher = telemetry_publisher if telemetry_publisher is not None else EnvelopeMemoryPublisher(log_messages=settings.transport == "log")
    engine_cls = EdgeEngine
    if settings.config_sync:  # команды конфигурации камер от шлюза, оверлей рядом с файлом камер (docs/EDGE.md §8)
        from guard.edge.config_sync import ConfigSyncEdgeEngine
        engine_cls = ConfigSyncEdgeEngine
    engine = engine_cls(settings, backend=backend, publisher=publisher, telemetry_publisher=telemetry_publisher)
    engine.edge_node_id = node_id
    engine.edge_ntp_servers = list(section.ntp_servers)
    if applied:
        log.info("edge section applied: %s", ", ".join(applied))
    return settings, engine


def _json_safe(value: Any) -> Any:
    """Метрики → JSON без NaN/inf и numpy-скаляров (codec пишет JSON с allow_nan=False)."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


class EdgeStats:
    """Сводка узла для metrics.edge и строки лога: CPU/память контейнера, NTP, отправка."""

    def __init__(self, engine: PipelineEngine, node_id: str, ntp_servers: list[str]) -> None:
        self.engine = engine
        self.node_id = node_id
        self.cpu = CpuSampler()
        self.ntp = NtpClock(ntp_servers)
        self.started = time.time()
        self.last: dict[str, Any] = {}
        self.cpu_samples: list[float] = []
        self.proc_samples: list[float] = []

    def start(self) -> None:
        self.ntp.start()

    def collect(self) -> dict[str, Any]:
        sink = self.engine.sink
        cpu = self.cpu.sample()
        if cpu.get("container_cores") is not None:
            self.cpu_samples.append(cpu["container_cores"])
        if cpu.get("process_cores") is not None:
            self.proc_samples.append(cpu["process_cores"])
        snap = self.engine.metrics.snapshot()
        self.last = {
            "node_id": self.node_id,
            "ts": round(time.time(), 3),
            "uptime_s": round(time.time() - self.started, 1),
            "backend": self.engine.backend.name,
            "cpu": cpu,
            "memory": memory_snapshot(),
            "clock": self.ntp.state,
            "transport": sink.stats() if hasattr(sink, "stats") else {},
            "sink_errors": self.engine.sink_errors,
            "events_created": self.engine.events_created,
            "frames_processed": {cid: s.get("frames_processed", 0) for cid, s in snap.items()},
        }
        return self.last

    def line(self) -> str:
        e = self.last
        snap = self.engine.metrics.snapshot()
        fps = " ".join(f"{cid}={s['fps_processed']:.1f}" for cid, s in sorted(snap.items()))
        cpu = e.get("cpu") or {}
        tr = e.get("transport") or {}
        gq = tr.get("gateway_queue") or {}
        clock = e.get("clock") or {}
        return (
            f"stats: fps {fps} | cpu ctr {cpu.get('container_cores')}/{cpu.get('limit_cores')} "
            f"({cpu.get('container_pct_of_limit')}%) proc {cpu.get('process_cores')} | "
            f"rss {(e.get('memory') or {}).get('process_rss_mb')} MB | events {e.get('events_created')} | "
            f"sent {tr.get('sent')} held {tr.get('held_by_guard')} dropped {tr.get('outbox_dropped')} "
            f"pending {tr.get('outbox_pending')} | gw consumers {gq.get('consumers')} backlog {gq.get('messages')} | "
            f"ntp {clock.get('offset_ms')} ms"
        )


def publish_edge_metrics(engine: PipelineEngine, stats: EdgeStats) -> None:
    if engine.sink is None:
        return
    data = engine.metrics_snapshot()
    data["edge"] = stats.collect()
    engine._sink_publish(metrics_msg(_json_safe(data)))


def run(args: argparse.Namespace) -> int:
    settings, engine = build(args)
    stop = threading.Event()
    old_handlers = {}
    if threading.current_thread() is threading.main_thread():
        for sig in (signal.SIGINT, signal.SIGTERM):
            old_handlers[sig] = signal.getsignal(sig)
            signal.signal(sig, lambda *_: stop.set())
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        engine.start()
        log.info("edge process started: node=%s backend=%s transport=%s durable_outbox=true",
                 settings.edge_id, settings.backend, settings.transport)
        while not stop.is_set():
            remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
            if remaining == 0:
                break
            if stop.wait(min(args.stats_period, remaining) if remaining is not None else args.stats_period):
                break
            health = engine.health()
            log.info("edge status=%s; external inference acceptance requires active per-model counters", health.get("status"))
    finally:
        try:
            engine.stop()
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pika").setLevel(logging.WARNING)  # INFO pika — по строке на каждое соединение и канал
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
