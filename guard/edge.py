"""Режим edge (D-005): конвейер без HTTP рядом с TPU, результаты — в транспорт.

    python -m guard.edge --config configs/cameras.stand.yaml [--transport rabbit] [--url amqp://...]

Публикуются:
- event / event_update — Event + снапшот JPEG ≤ 60 КБ (q≈70, ≤ 640 px);
- frame_meta — на каждый обработанный кадр: camera_id, frame_idx (позиция в файле), ts,
  рамки людей и объектов (int) с ID, нарушители;
- metrics — каждые --metrics-period секунд (как /api/metrics) + heartbeat.
Журнал на edge — в памяти (SQLite и клипы ведёт gateway), кольцевой буфер клипов выключен.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")  # контейнер стенда: 1.5 CPU, 4 ГиБ

from guard.pipeline._fallbacks import MemoryJournal  # noqa: E402
from guard.pipeline.engine import PipelineEngine  # noqa: E402
from guard.pipeline.settings import Settings  # noqa: E402
from guard.pipeline.transport import make_sink  # noqa: E402
from guard.transport import HeartbeatMsg  # noqa: E402

log = logging.getLogger("guard.edge")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m guard.edge", description="Конвейер «Зазеркалья» без UI (режим edge)")
    p.add_argument("--config", help="cameras.yaml (по умолчанию GUARD_CAMERAS_CONFIG или configs/cameras.yaml)")
    p.add_argument("--models-config", help="реестр моделей (по умолчанию configs/models.yaml)")
    p.add_argument("--backend", help="onnx-cpu | linq-tpu (по умолчанию GUARD_BACKEND)")
    p.add_argument("--transport", choices=["rabbit", "memory", "log"], help="по умолчанию GUARD_TRANSPORT или rabbit")
    p.add_argument("--url", help="AMQP URL (по умолчанию GUARD_RABBIT_URL или amqp://guest:guest@127.0.0.1:5672/)")
    p.add_argument("--no-frame-meta", action="store_true", help="не слать frame_meta (только события и метрики)")
    p.add_argument("--metrics-period", type=float, default=2.0)
    p.add_argument("--duration", type=float, default=0.0, help="остановиться через N секунд (смоук-тест)")
    p.add_argument("--node-id", default=os.environ.get("GUARD_NODE_ID", "edge"))
    return p.parse_args(argv)


def build(args: argparse.Namespace) -> tuple[Settings, PipelineEngine]:
    settings = Settings.from_env()
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
    sink = make_sink(settings.transport, settings.rabbit_url)
    engine = PipelineEngine(settings, journal=MemoryJournal(), sink=sink)
    return settings, engine


def run(args: argparse.Namespace) -> int:
    settings, engine = build(args)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    engine.start()
    log.info(
        "edge started: %d cameras, backend=%s, transport=%s, frame_meta=%s",
        len(engine.cameras), engine.backend.name, settings.transport, settings.publish_frame_meta,
    )
    deadline = time.monotonic() + args.duration if args.duration > 0 else None
    try:
        while not stop.wait(args.metrics_period):
            engine.publish_metrics()
            engine._sink_publish(HeartbeatMsg(node_id=args.node_id))
            if deadline is not None and time.monotonic() >= deadline:
                break
    finally:
        engine.stop()
        close = getattr(engine.sink, "close", None)
        if callable(close):
            close(timeout=5.0)
        log.info("edge stopped: events=%d sink_errors=%d", engine.events_created, engine.sink_errors)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
