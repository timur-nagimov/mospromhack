"""Edge-режим маховика (docs/FLYWHEEL.md, «Edge-режим»): сэмплы edge уходят на шлюз частями.

EdgeSender (edge, фоновый поток): сырой кадр → JPEG ≤ 960 px (≤ 60 КБ) → части по 8 КБ → конверты
flywheel_sample протокола edge → шлюз (guard/transport/protocol.py) в свою очередь AMQP (guard/flywheel/amqp.py).
- Конверт выдаёт журнал edge (Journal.make_message): тот же producer и сквозная sequence, что у событий и
  телеметрии. В outbox сэмплы не попадают: доставка без гарантии, как у телеметрии.
- Трафик ограничен ведром токенов по байтам конверта: GUARD_FLYWHEEL_EDGE_KBPS (4 КБ/с) на камеру,
  контракт — не больше 10.
- Части по 8 КБ: на шлюзе в полёте не больше одной части (~11 КБ), и событие, идущее по тому же туннелю
  стенд → ноутбук (0.1–0.4 МБ/с), делит канал с ней 30–110 мс, а не с целым кадром 0.2–0.8 с.
- Сэмплы событий и ручного захвата идут вперёд остальных; при полном буфере вытесняется самый старый
  обычный сэмпл. Ошибка публикации бросает сэмпл целиком (dropped_channel), следующая попытка — после паузы.

SampleAssembler (шлюз): части по sample_id → проверка JPEG и полей → фоновый писатель (LocalWriter.put).
Незавершённые сэмплы забываются через ttl_s. Повторная доставка частей и сэмплов не дублирует строки.
Edge не задаёт ни путь кадра, ни статус сэмпла, ни метку оператора: их ставит шлюз.
"""

from __future__ import annotations

import base64
import io
import logging
import math
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

from guard.flywheel.collector import SampleJob, _scale_meta
from guard.flywheel.frames import detections_to_dicts, encode_jpeg
from guard.flywheel.store import SAMPLE_SOURCES, UNCAPPED_SOURCES, new_sample_id
from guard.transport.protocol import FLYWHEEL_SAMPLE_FIELDS, MAX_FLYWHEEL_PARTS, Envelope, encode, identifier

log = logging.getLogger(__name__)

KIND = "flywheel_sample"
CHUNK_BYTES = 8 * 1024
EDGE_MAX_WIDTH = 960
EDGE_QUALITY = 80
EDGE_MAX_KB = 60
QUEUE_SIZE = 8  # сырые кадры ждут кодирования: 8 × 1080p ≈ 50 МБ
RETRY_PAUSE_S = (1.0, 15.0)
STOP_TIMEOUT_S = 12.0


class _JobBuffer:
    """Ограниченный буфер с приоритетом: события и ручной захват — вперёд, вытесняется старый обычный сэмпл."""

    def __init__(self, maxsize: int) -> None:
        self.maxsize = max(1, maxsize)
        self._high: deque[SampleJob] = deque()
        self._low: deque[SampleJob] = deque()
        self._cv = threading.Condition()
        self.evicted = 0

    def __len__(self) -> int:
        return len(self._high) + len(self._low)

    def put(self, job: SampleJob) -> bool:
        high = job.sample.get("source") in UNCAPPED_SOURCES
        with self._cv:
            if len(self) >= self.maxsize:
                if not (high and self._low):
                    return False
                self._low.popleft()
                self.evicted += 1
            (self._high if high else self._low).append(job)
            self._cv.notify()
            return True

    def get(self, timeout: float) -> SampleJob | None:
        with self._cv:
            if not len(self):
                self._cv.wait(timeout)
            if self._high:
                return self._high.popleft()
            if self._low:
                return self._low.popleft()
            return None

    def wake(self) -> None:
        with self._cv:
            self._cv.notify_all()


class TokenBucket:
    """Ведро токенов в байтах. rate — байт/с; burst — сколько можно отправить подряд."""

    def __init__(self, rate: Callable[[], float], burst: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.rate = rate
        self.burst = burst
        self.clock = clock
        self.tokens = burst
        self.updated = clock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.burst, self.tokens + (now - self.updated) * max(1.0, self.rate()))
        self.updated = now

    def wait_time(self, size: float) -> float:
        self._refill()
        need = min(size, self.burst)  # часть больше ведра ждёт полного ведра, а не вечность
        if self.tokens >= need:
            return 0.0
        return (need - self.tokens) / max(1.0, self.rate())

    def take(self, size: float) -> None:
        self._refill()
        self.tokens -= size


def sample_fields(sample: dict[str, Any]) -> dict[str, Any]:
    """Поля сэмпла, которые edge передаёт шлюзу (протокол отклоняет остальные)."""
    return {key: value for key, value in sample.items() if key in FLYWHEEL_SAMPLE_FIELDS and value is not None}


class EdgeSender:
    """Отправитель сэмплов edge → шлюз. Интерфейс как у LocalWriter: start, stop, put, stats."""

    def __init__(
        self,
        publisher: Any,
        make_message: Callable[[str, dict[str, Any]], Envelope],
        *,
        kbps_per_camera: float = 4.0,
        cameras: Callable[[], int] | int = 1,
        chunk_bytes: int = CHUNK_BYTES,
        max_width: int = EDGE_MAX_WIDTH,
        quality: int = EDGE_QUALITY,
        max_kb: int = EDGE_MAX_KB,
        queue_size: int = QUEUE_SIZE,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        # publisher — объект с publish(envelope, durable=False) или фабрика без аргументов (вызов в потоке отправки)
        self._publisher = publisher
        self.make_message = make_message
        self.kbps_per_camera = float(kbps_per_camera)
        self._cameras = cameras if callable(cameras) else (lambda n=int(cameras): n)
        self.chunk_bytes = max(1024, int(chunk_bytes))
        self.max_width = max_width
        self.quality = quality
        self.max_kb = max_kb
        self._buffer = _JobBuffer(queue_size)
        # ведро вмещает часть с метаданными части 0 и base64 (+ ~40 %)
        self.bucket = TokenBucket(self.rate_bps, burst=3 * self.chunk_bytes, clock=clock)
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait
        self._thread: threading.Thread | None = None
        self.sent_samples = 0
        self.sent_parts = 0
        self.sent_bytes = 0
        self.dropped_channel = 0
        self.dropped_queue = 0
        self.errors = 0
        self.last_error: str | None = None
        self.started_mono = clock()
        self.clock = clock

    def rate_bps(self) -> float:
        return self.kbps_per_camera * 1024.0 * max(1, int(self._cameras()))

    @property
    def publisher(self) -> Any:
        if callable(self._publisher) and not hasattr(self._publisher, "publish"):
            self._publisher = self._publisher()
        return self._publisher

    # ---- жизненный цикл ----

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="flywheel-edge-sender", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._buffer.wake()
        thread = self._thread
        if thread is not None:
            # публикация в полёте ждёт подтверждения брокера: таймауты pika — 5 с на сокет и на блокировку
            thread.join(max(timeout, STOP_TIMEOUT_S))
            if thread.is_alive():
                raise RuntimeError("Flywheel edge sender did not stop")
            self._thread = None
        # своё соединение закрываем; общий объект телеметрии (тесты, отладка) закрывает EdgeRunner
        if getattr(self._publisher, "dedicated", False):
            try:
                self._publisher.close()
            except Exception:  # noqa: BLE001
                pass

    def put(self, job: SampleJob) -> bool:
        ok = self._buffer.put(job)
        if not ok:
            self.dropped_queue += 1
        return ok

    def _run(self) -> None:
        pause = RETRY_PAUSE_S[0]
        while not self._stop.is_set():
            job = self._buffer.get(timeout=0.5)
            if job is None:
                continue
            try:
                self.send(job)
                pause = RETRY_PAUSE_S[0]
            except Exception as exc:  # noqa: BLE001 — сбой отправки сэмпла не трогает конвейер
                self.errors += 1
                self.dropped_channel += 1
                self.last_error = type(exc).__name__
                if self.errors in (1, 10, 100) or self.errors % 1000 == 0:
                    log.warning("flywheel: edge sample send failed (%d): %s", self.errors, self.last_error)
                self._stop.wait(pause)  # брокер недоступен: не долбим его каждым сэмплом
                pause = min(RETRY_PAUSE_S[1], pause * 2)

    # ---- отправка ----

    def encode(self, job: SampleJob) -> tuple[dict[str, Any], bytes]:
        if job.frame is None:
            raise ValueError("edge sample job has no frame")
        jpeg, width, height, scale = encode_jpeg(job.frame, self.max_width, self.quality, self.max_kb)
        sample = sample_fields(dict(job.sample))
        sample.setdefault("id", new_sample_id())
        sample.update(
            width=width,
            height=height,
            detections=detections_to_dicts(job.detections or {}, job.tracks, scale),
            meta=_scale_meta(sample.get("meta") or {}, scale),
        )
        return sample, jpeg

    def _wait_tokens(self, size: int) -> bool:
        while not self._stop.is_set():
            delay = self.bucket.wait_time(size)
            if delay <= 0:
                self.bucket.take(size)
                return True
            self._sleep(min(delay, 1.0))
        return False

    def send(self, job: SampleJob) -> bool:
        """Все части сэмпла. False — остановка; исключение публикации бросает сэмпл целиком."""
        sample, jpeg = self.encode(job)
        chunk = max(self.chunk_bytes, math.ceil(len(jpeg) / MAX_FLYWHEEL_PARTS))
        chunks = [jpeg[i : i + chunk] for i in range(0, len(jpeg), chunk)]
        publisher = self.publisher
        for index, part in enumerate(chunks):
            payload = {
                "sample_id": sample["id"],
                "part": index,
                "parts": len(chunks),
                "sample": sample if index == 0 else None,
                "jpeg_b64": base64.b64encode(part).decode("ascii"),
            }
            envelope = self.make_message(KIND, payload)
            size = len(encode(envelope))
            if not self._wait_tokens(size):
                return False
            publisher.publish(envelope, durable=False)
            self.sent_parts += 1
            self.sent_bytes += size
        self.sent_samples += 1
        self.last_error = None
        return True

    def stats(self) -> dict[str, Any]:
        elapsed = max(1e-6, self.clock() - self.started_mono)
        status = getattr(self._publisher, "status", None)
        return {
            "mode": "edge",
            "sent_samples": self.sent_samples,
            "sent_parts": self.sent_parts,
            "sent_kb": round(self.sent_bytes / 1024, 1),
            "avg_kbps": round(self.sent_bytes / 1024 / elapsed, 2),
            "rate_limit_kbps": round(self.rate_bps() / 1024, 1),
            "chunk_kb": round(self.chunk_bytes / 1024, 1),
            "queue": len(self._buffer),
            "evicted": self._buffer.evicted,
            "dropped_queue": self.dropped_queue,
            "dropped_channel": self.dropped_channel,
            "errors": self.errors,
            "transport": status() if callable(status) and getattr(self._publisher, "dedicated", False) else None,
        }


class SampleAssembler:
    """Шлюз: части flywheel_sample → готовый сэмпл → фоновый писатель (LocalWriter.put)."""

    def __init__(self, writer: Any, *, ttl_s: float = 120.0, max_pending: int = 32,
                 max_pending_bytes: int = 8 * 1024 * 1024, clock: Callable[[], float] = time.monotonic) -> None:
        self.writer = writer
        self.ttl_s = ttl_s
        self.max_pending = max_pending
        self.max_pending_bytes = max_pending_bytes
        self.clock = clock
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._pending_bytes = 0
        self._done: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self.completed = 0
        self.parts = 0
        self.dropped = {"incomplete": 0, "overflow": 0, "invalid": 0, "writer": 0}

    def add_envelope(self, envelope: Envelope) -> bool:
        """True — сэмпл собран и передан писателю. Конверт уже проверен протоколом (decode)."""
        if envelope.kind != KIND:
            raise ValueError("Not a flywheel sample")
        payload = envelope.payload
        return self.add(payload["sample_id"], payload["part"], payload["parts"], payload["sample"],
                        base64.b64decode(payload["jpeg_b64"]))

    def add(self, sample_id: str, part: int, parts: int, sample: dict[str, Any] | None, chunk: bytes) -> bool:
        now = self.clock()
        with self._lock:
            self.parts += 1
            self._expire(now)
            if sample_id in self._done:
                return False  # повторная доставка уже собранного сэмпла
            entry = self._pending.get(sample_id)
            if entry is not None and entry["n"] != parts:
                self._forget(sample_id)
                self.dropped["invalid"] += 1
                return False
            if entry is None:
                entry = {"parts": {}, "n": parts, "sample": None, "t": now, "bytes": 0}
                self._pending[sample_id] = entry
            if part not in entry["parts"]:
                entry["parts"][part] = chunk
                entry["bytes"] += len(chunk)
                self._pending_bytes += len(chunk)
            if part == 0 and sample is not None:
                entry["sample"] = dict(sample)
            self._trim(keep=sample_id)
            if sample_id not in self._pending or len(entry["parts"]) < entry["n"] or entry["sample"] is None:
                return False
            self._forget(sample_id)
            self._done[sample_id] = None
            while len(self._done) > 1024:
                self._done.popitem(last=False)
        jpeg = b"".join(entry["parts"][i] for i in range(entry["n"]))
        try:
            job = self._job(sample_id, entry["sample"], jpeg)
        except ValueError:
            self.dropped["invalid"] += 1
            return False
        if not self.writer.put(job):
            self.dropped["writer"] += 1
            return False
        self.completed += 1
        return True

    @staticmethod
    def _job(sample_id: str, meta: dict[str, Any], jpeg: bytes) -> SampleJob:
        """Сэмпл из полей edge: путь, статус и метку оператора ставит шлюз; размеры — из самого JPEG."""
        from PIL import Image

        try:
            with Image.open(io.BytesIO(jpeg)) as image:
                width, height = image.size
                if image.format != "JPEG" or not 1 <= width <= 4096 or not 1 <= height <= 4096:
                    raise ValueError("Invalid sample JPEG")
                image.verify()
        except (OSError, Image.DecompressionBombError) as exc:
            raise ValueError("Invalid sample JPEG") from exc
        source = meta.get("source")
        if source not in SAMPLE_SOURCES:
            raise ValueError("Invalid sample source")
        identifier(meta.get("camera_id"), "camera_id")
        created = meta.get("created_at")
        sample = {
            "id": sample_id,
            "created_at": float(created) if isinstance(created, (int, float)) and created > 0 else time.time(),
            "camera_id": meta["camera_id"],
            "source": source,
            "event_id": meta.get("event_id"),
            "reason": str(meta.get("reason") or ""),
            "holdout": 1 if meta.get("holdout") else 0,
            "width": width,
            "height": height,
            "meta": dict(meta.get("meta") or {}),
        }
        detections = [d for d in meta.get("detections") or [] if isinstance(d, dict)]
        return SampleJob(sample=sample, jpeg=jpeg, det_dicts=detections)

    def _forget(self, sample_id: str) -> None:
        entry = self._pending.pop(sample_id, None)
        if entry is not None:
            self._pending_bytes -= entry["bytes"]

    def _trim(self, keep: str) -> None:
        """Сверх лимита — вытесняется самый старый незавершённый сэмпл; текущий — только если он один не влез."""
        while len(self._pending) > self.max_pending or self._pending_bytes > self.max_pending_bytes:
            victim = next((sid for sid in self._pending if sid != keep), keep)
            self._forget(victim)
            self.dropped["overflow"] += 1
            if victim == keep:
                break

    def _expire(self, now: float) -> None:
        stale = [sid for sid, e in self._pending.items() if now - e["t"] > self.ttl_s]
        for sid in stale:
            self._forget(sid)
            self.dropped["incomplete"] += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            pending = len(self._pending)
            pending_kb = round(self._pending_bytes / 1024, 1)
        return {"completed": self.completed, "parts": self.parts, "pending": pending, "pending_kb": pending_kb,
                "dropped": dict(self.dropped)}


def receive_sample(engine: Any, envelope: Envelope) -> bool:
    """GatewayEngine.receive: часть сэмпла → сборщик шлюза. Без маховика часть молча пропускается."""
    assembler = getattr(engine, "flywheel_assembler", None)
    if assembler is None:
        return False
    return assembler.add_envelope(envelope)
