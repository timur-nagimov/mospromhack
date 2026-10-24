"""Edge-режим маховика (docs/FLYWHEEL.md, «Edge-режим»): на edge нет журнала, сэмплы уходят на шлюз.

EdgeSender (edge, фоновый поток): сырой кадр → JPEG ≤ 960 px (≤ 60 КБ) → части по 8 КБ → сообщения
flywheel_sample в тот же транспорт, что события. Части 0 несёт метаданные сэмпла.
- Трафик ограничен ведром токенов: GUARD_FLYWHEEL_EDGE_KBPS (4 КБ/с) на камеру, контракт — не больше 10.
- Части по 8 КБ ограничивают «пробку» в канале стенд → ноутбук. Событие, отправленное во время сэмпла,
  ждёт не больше одной части: 8 КБ при 0.1–0.4 МБ/с — 20–80 мс. Целый кадр (40–60 КБ) держал бы его 0.1–0.6 с.
- Сэмплы событий и ручного захвата идут вперёд остальных; при переполнении буфера вытесняется самый старый
  обычный сэмпл.
- Защита канала edge (GatewayQueueGuard) придерживает flywheel_sample вместе с frame_meta. Если шлюз не слушает,
  сэмпл бросается целиком (dropped_channel), без отправки оставшихся частей.

SampleAssembler (шлюз, поток транспорта): собирает части по sample_id и отдаёт готовый сэмпл фоновому писателю.
Незавершённые сэмплы забываются через ttl_s. Повторная доставка частей и сэмплов не дублирует строки.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable
from typing import Any

from guard.flywheel.collector import SampleJob, _scale_meta
from guard.flywheel.frames import detections_to_dicts, encode_jpeg
from guard.flywheel.store import UNCAPPED_SOURCES, new_sample_id
from guard.transport import FlywheelSampleMsg

log = logging.getLogger(__name__)

CHUNK_BYTES = 8 * 1024
EDGE_MAX_WIDTH = 960
EDGE_QUALITY = 80
EDGE_MAX_KB = 60
QUEUE_SIZE = 16


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
        if self.tokens >= size:
            return 0.0
        return (size - self.tokens) / max(1.0, self.rate())

    def take(self, size: float) -> None:
        self._refill()
        self.tokens -= size


class EdgeSender:
    """Отправитель сэмплов edge → шлюз. Интерфейс как у LocalWriter: start, stop, put, stats."""

    def __init__(
        self,
        sink: Any,
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
        self.sink = sink
        self.kbps_per_camera = float(kbps_per_camera)
        self._cameras = cameras if callable(cameras) else (lambda n=int(cameras): n)
        self.chunk_bytes = max(1024, int(chunk_bytes))
        self.max_width = max_width
        self.quality = quality
        self.max_kb = max_kb
        self._buffer = _JobBuffer(queue_size)
        self.bucket = TokenBucket(self.rate_bps, burst=2 * self.chunk_bytes, clock=clock)
        self._stop = threading.Event()
        self._sleep = sleep or self._stop.wait
        self._thread: threading.Thread | None = None
        self.sent_samples = 0
        self.sent_parts = 0
        self.sent_bytes = 0
        self.dropped_channel = 0
        self.dropped_queue = 0
        self.errors = 0
        self.started_mono = clock()
        self.clock = clock

    def rate_bps(self) -> float:
        return self.kbps_per_camera * 1024.0 * max(1, int(self._cameras()))

    # ---- жизненный цикл ----

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="flywheel-edge-sender", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._buffer.wake()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def put(self, job: SampleJob) -> bool:
        ok = self._buffer.put(job)
        if not ok:
            self.dropped_queue += 1
        return ok

    def _run(self) -> None:
        while not self._stop.is_set():
            job = self._buffer.get(timeout=0.5)
            if job is None:
                continue
            try:
                self.send(job)
            except Exception:  # noqa: BLE001 — сбой отправки сэмпла не трогает конвейер
                self.errors += 1
                if self.errors in (1, 10, 100):
                    log.exception("flywheel: edge sample send failed")

    # ---- отправка ----

    def encode(self, job: SampleJob) -> tuple[dict[str, Any], bytes]:
        if job.frame is None:
            raise ValueError("edge sample job has no frame")
        jpeg, width, height, scale = encode_jpeg(job.frame, self.max_width, self.quality, self.max_kb)
        sample = dict(job.sample)
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
        sample, jpeg = self.encode(job)
        chunks = [jpeg[i : i + self.chunk_bytes] for i in range(0, len(jpeg), self.chunk_bytes)] or [b""]
        for index, chunk in enumerate(chunks):
            if not self._wait_tokens(len(chunk) + 256):
                return False
            msg = FlywheelSampleMsg(
                sample_id=sample["id"], part=index, parts=len(chunks), sample=sample if index == 0 else None, jpeg=chunk
            )
            if self.sink.publish(msg) is False:  # защита канала придержала или очередь отправки полна
                self.dropped_channel += 1
                return False
            self.sent_parts += 1
            self.sent_bytes += len(chunk)
        self.sent_samples += 1
        return True

    def stats(self) -> dict[str, Any]:
        elapsed = max(1e-6, self.clock() - self.started_mono)
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
        }


class SampleAssembler:
    """Шлюз: части flywheel_sample → готовый сэмпл → фоновый писатель (LocalWriter.put)."""

    def __init__(self, writer: Any, *, ttl_s: float = 120.0, max_pending: int = 32,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.writer = writer
        self.ttl_s = ttl_s
        self.max_pending = max_pending
        self.clock = clock
        self._pending: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._done: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()
        self.completed = 0
        self.dropped = {"incomplete": 0, "overflow": 0, "writer": 0}

    def add(self, msg: FlywheelSampleMsg) -> bool:
        """True — сэмпл собран и передан писателю."""
        now = self.clock()
        with self._lock:
            self._expire(now)
            if msg.sample_id in self._done:
                return False  # повторная доставка уже собранного сэмпла
            entry = self._pending.get(msg.sample_id)
            if entry is None:
                entry = {"parts": {}, "n": msg.parts, "sample": None, "t": now}
                self._pending[msg.sample_id] = entry
                while len(self._pending) > self.max_pending:
                    self._pending.popitem(last=False)
                    self.dropped["overflow"] += 1
            entry["parts"][msg.part] = msg.jpeg or b""
            if msg.part == 0 and msg.sample is not None:
                entry["sample"] = dict(msg.sample)
            if len(entry["parts"]) < entry["n"] or entry["sample"] is None:
                return False
            del self._pending[msg.sample_id]
            self._done[msg.sample_id] = None
            while len(self._done) > 512:
                self._done.popitem(last=False)
        jpeg = b"".join(entry["parts"][i] for i in range(entry["n"]))
        sample = entry["sample"]
        sample["id"] = msg.sample_id
        job = SampleJob(sample=sample, jpeg=jpeg, det_dicts=list(sample.get("detections") or []))
        if not self.writer.put(job):
            self.dropped["writer"] += 1
            return False
        self.completed += 1
        return True

    def _expire(self, now: float) -> None:
        stale = [sid for sid, e in self._pending.items() if now - e["t"] > self.ttl_s]
        for sid in stale:
            del self._pending[sid]
            self.dropped["incomplete"] += 1

    def stats(self) -> dict[str, Any]:
        with self._lock:
            pending = len(self._pending)
        return {"completed": self.completed, "pending": pending, "dropped": dict(self.dropped)}
