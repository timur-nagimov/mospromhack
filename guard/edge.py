"""Режим edge (D-005): конвейер без HTTP рядом с TPU, результаты — в транспорт.

    python -m guard.edge --config configs/cameras.stand.yaml [--transport rabbit] [--url amqp://...]

Публикуются:
- event / event_update — Event + снапшот JPEG ≤ 60 КБ (q≈70, ≤ 640 px). В details.edge — node и sent_at
  (epoch по часам edge в момент передачи в транспорт): по нему считается задержка канала;
- frame_meta — на каждый обработанный кадр: camera_id, frame_idx (позиция в файле), ts,
  рамки людей и объектов (int) с ID, нарушители;
- metrics — каждые --metrics-period секунд (как /api/metrics) + ключ edge: загрузка CPU контейнера
  (cgroup), память, сдвиг часов по NTP, счётчики отправки по видам и камерам, состояние очереди шлюза;
  затем heartbeat.
Журнал на edge — в памяти (SQLite и клипы ведёт gateway), кольцевой буфер клипов выключен.

Секция `edge:` в файле камер (необязательна; load_cameras читает только `cameras`) описывает
развёртывание, чтобы `python -m guard.edge --config <файл>` работал без переменных окружения:

    edge:
      backend: linq-tpu
      models_config: configs/models.stand.yaml   # относительно корня проекта
      models_dir: ~/work/models
      data_dir: ~/work/mosprom/data
      rabbit_url: null          # null: на стенде брокер 10.0.114.30, иначе 127.0.0.1 (VPN-контейнер)
      node_id: stand-linq
      linq: {raw: true, window: 4}
      gateway_guard: {backlog_limit: 300}
      settings: {max_batch: 8, batch_timeout_ms: 4.0}

Приоритет: флаги CLI > переменные окружения GUARD_* > секция edge > значения по умолчанию.

Защита канала (gateway_guard). Очередь zazerkalye.gateway общая для событий и телеметрии, у событий
и метрик нет TTL, а TTL frame_meta RabbitMQ применяет только в голове очереди. Без потребителя
очередь растёт до x-max-length (2000), после чего брокер отклоняет публикации, и канал встаёт.
Поэтому edge раз в 2 с смотрит очередь (passive declare): без потребителя не шлёт frame_meta и
metrics (heartbeat — раз в 30 с), при очереди > backlog_limit приостанавливает frame_meta.
События отправляются всегда.
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

from guard.pipeline._fallbacks import MemoryJournal  # noqa: E402
from guard.pipeline.engine import PipelineEngine  # noqa: E402
from guard.pipeline.settings import CODE_ROOT, Settings  # noqa: E402
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
    return p if p.is_absolute() else (CODE_ROOT / p)


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
        if kind in ("frame_meta", "flywheel_sample"):  # сэмплы маховика ждут вместе с телеметрией
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
    p.add_argument("--url", help="AMQP URL (по умолчанию GUARD_RABBIT_URL, edge.rabbit_url, на стенде 10.0.114.30)")
    p.add_argument("--no-frame-meta", action="store_true", help="не слать frame_meta (только события и метрики)")
    p.add_argument("--no-gateway-guard", action="store_true", help="слать телеметрию, даже если шлюз не слушает")
    p.add_argument("--metrics-period", type=float, default=2.0)
    p.add_argument("--stats-period", type=float, default=10.0, help="строка статистики в лог раз в N секунд")
    p.add_argument("--duration", type=float, default=0.0, help="остановиться через N секунд (смоук-тест)")
    p.add_argument("--node-id", default=os.environ.get("GUARD_NODE_ID"))
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> tuple[Settings, PipelineEngine]:
    section = load_edge_section(args.config)
    settings = Settings.from_env()
    applied = apply_edge_section(settings, section)
    if disable_ort_spinning():
        applied.append("ort_no_spinning")
    settings.mode = "edge"
    settings.clips_enabled = False
    settings.autostart = True
    if args.config:
        settings.cameras_config = Path(args.config).expanduser().resolve()
    if args.models_config:
        settings.models_config = Path(args.models_config).expanduser().resolve()
    if args.backend:
        settings.backend = args.backend
    if args.transport:
        settings.transport = args.transport
    if args.url:
        settings.rabbit_url = args.url
    if args.no_frame_meta:
        settings.publish_frame_meta = False
    if settings.transport == "rabbit" and not settings.rabbit_url and not os.environ.get("GUARD_TRANSPORT_URL"):
        settings.rabbit_url = default_rabbit_url()
    if "tpu" in settings.backend:
        _linq_env(section.linq)
    node_id = args.node_id or section.node_id or "edge"
    guard = None
    if settings.transport == "rabbit" and section.gateway_guard.enabled and not args.no_gateway_guard:
        guard = GatewayQueueGuard(settings.rabbit_url, section.gateway_guard.backlog_limit, section.gateway_guard.period_s)
    sink = EdgeSink(make_sink(settings.transport, settings.rabbit_url), node_id=node_id, guard=guard)
    engine = PipelineEngine(settings, journal=MemoryJournal(), sink=sink)
    engine.edge_node_id = node_id  # для метрик и логов
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
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    node_id = getattr(engine, "edge_node_id", args.node_id or "edge")
    stats = EdgeStats(engine, node_id, getattr(engine, "edge_ntp_servers", []))
    stats.start()
    guard = getattr(engine.sink, "guard", None)
    if guard is not None:
        guard.start()
    engine.start()
    log.info(
        "edge started: node=%s, %d cameras, backend=%s, transport=%s (%s), frame_meta=%s, guard=%s",
        node_id, len(engine.cameras), engine.backend.name, settings.transport,
        (settings.rabbit_url or "default").split("@")[-1], settings.publish_frame_meta, guard is not None,
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    next_line = time.monotonic() + args.stats_period
    try:
        while not stop.wait(args.metrics_period):
            publish_edge_metrics(engine, stats)
            engine._sink_publish(HeartbeatMsg(node_id=node_id))
            if time.monotonic() >= next_line:
                next_line += args.stats_period
                log.info(stats.line())
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        try:  # снимок до остановки: после неё темп кадров в окне 5 с начинает падать
            final = engine.metrics_snapshot()
        except Exception:  # noqa: BLE001
            final = {}
        engine.stop()
        stats.ntp.stop()
        close = getattr(engine.sink, "close", None)
        if callable(close):
            try:
                close(timeout=5.0)
            except TimeoutError as exc:  # медленный канал: недоотправленная телеметрия не повод падать
                log.warning("sink close: %s", exc)
        summary = {
            "node_id": node_id,
            "uptime_s": round(time.time() - stats.started, 1),
            "events": engine.events_created,
            "sink_errors": engine.sink_errors,
            "transport": engine.sink.stats() if hasattr(engine.sink, "stats") else {},
            "frames_processed": (stats.last or {}).get("frames_processed"),
            "container_cores_avg": round(sum(stats.cpu_samples) / len(stats.cpu_samples), 3) if stats.cpu_samples else None,
            "process_cores_avg": round(sum(stats.proc_samples) / len(stats.proc_samples), 3) if stats.proc_samples else None,
            "cameras": {
                cid: {
                    "fps_processed": round(c.get("fps_processed", 0.0), 2),
                    "lag_s": c.get("lag_s"),
                    "timings_ms": {k: v.get("p50") for k, v in (c.get("timings_ms") or {}).items()},
                    "timings_p95_ms": {k: v.get("p95") for k, v in (c.get("timings_ms") or {}).items()},
                }
                for cid, c in (final.get("cameras") or {}).items()
            },
            "inference": final.get("inference"),
        }
        log.info("edge stopped: %s", json.dumps(_json_safe(summary), ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("pika").setLevel(logging.WARNING)  # INFO pika — по строке на каждое соединение и канал
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
