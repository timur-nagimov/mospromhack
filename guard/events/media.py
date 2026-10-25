"""Медиа событий: кольцевой буфер кадров камеры, снапшот JPEG и клип H.264 ±N секунд.

Память ограничена жёстко (ноутбук падал по OOM):
- FrameRing хранит не больше max_frames (150) кадров, не чаще clip_fps, уменьшенными до
  clip_max_width (640 px); при превышении бюджета байт уменьшает кадры дальше и сокращает окно;
- клип пишется потоково: ffmpeg открывается сразу при подтверждении события, кадры идут из
  буфера по мере появления (пересчёт на постоянный fps), всё видео в памяти не копится;
- одновременно пишется не больше max_writers клипов, очередь заданий ограничена.
Кодирование: ffmpeg из imageio-ffmpeg, libx264, yuv420p, -movflags +faststart (играет в браузере).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import weakref
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from guard.contracts import Event

log = logging.getLogger(__name__)

MIN_RING_WIDTH = 320
# resolve отдаёт только кадры и ролики: в data_dir лежат и базы (guard.db, auth.db), их по ссылке события не отдать
MEDIA_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp", ".mp4", ".webm", ".mkv", ".avi", ".mov"})


class FrameRing:
    """Кольцевой буфер (ts, кадр) одной камеры с ограничением по времени, числу кадров, fps и байтам."""

    def __init__(
        self,
        seconds: float = 12.0,
        fps: float = 10.0,
        max_width: int = 640,
        budget_bytes: int = 128 << 20,
        max_frames: int = 150,
    ) -> None:
        self.seconds = seconds
        self.fps = max(1.0, fps)
        self.max_width = max_width
        self._base_max_width = max_width
        self.budget_bytes = budget_bytes
        self._frames: deque[tuple[float, np.ndarray]] = deque(maxlen=max(2, max_frames))
        self._bytes = 0
        self._next_due: float | None = None
        self._last_ts = -1e18
        self._lock = threading.Lock()
        self.downscaled = False
        self._sealed = False

    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            if budget_bytes > self.budget_bytes and self.max_width < self._base_max_width:
                # бюджет вырос (камеру удалили) — возвращаем исходное разрешение, иначе клипы остаются мелкими навсегда
                self.max_width = self._base_max_width
                self.downscaled = False
            self.budget_bytes = budget_bytes
            while len(self._frames) > 2 and self._bytes > self.budget_bytes:
                self._bytes -= self._frames.popleft()[1].nbytes

    def seal(self) -> None:
        """A capture ended: writers may drain, but no later scene may append."""
        with self._lock:
            self._sealed = True

    @property
    def sealed(self) -> bool:
        return self._sealed

    def _due(self, ts: float) -> bool:
        """Прореживание до fps по накопителю: 60 к/с → 10, 12 к/с → ~10."""
        step = 1.0 / self.fps
        if self._next_due is None or ts - self._next_due > step:
            self._next_due = ts + step
            return True
        if ts >= self._next_due - 0.5 * step:
            self._next_due += step
            return True
        return False

    def push(self, frame: np.ndarray, ts: float) -> bool:
        with self._lock:
            if self._sealed or not self._due(ts):
                return False
        h, w = frame.shape[:2]
        width = min(self.max_width, w)
        if width < w:
            small = cv2.resize(frame, (width, max(2, int(h * width / w))), interpolation=cv2.INTER_AREA)
        else:
            small = frame.copy()  # не держим ссылку на полноразмерный кадр источника
        with self._lock:
            if self._sealed:
                return False
            if len(self._frames) == self._frames.maxlen:
                self._bytes -= self._frames[0][1].nbytes  # deque(maxlen) выбросит самый старый
            self._last_ts = ts
            self._frames.append((ts, small))
            self._bytes += small.nbytes
            edge = ts - self.seconds
            while self._frames and self._frames[0][0] < edge:
                self._bytes -= self._frames.popleft()[1].nbytes
            if self._bytes > self.budget_bytes:
                # нехватка памяти: дальше храним кадры меньше, лишнее старое выбрасываем
                if self.max_width > MIN_RING_WIDTH:
                    self.max_width = max(MIN_RING_WIDTH, int(self.max_width * 0.8))
                    self.downscaled = True
                while len(self._frames) > 2 and self._bytes > self.budget_bytes:
                    self._bytes -= self._frames.popleft()[1].nbytes
        return True

    def snapshot(self, since: float | None = None, until: float | None = None) -> list[tuple[float, np.ndarray]]:
        """Ссылки на кадры (без копирования) в интервале [since, until]."""
        with self._lock:
            items = list(self._frames)
        return [(t, f) for t, f in items if (since is None or t >= since) and (until is None or t <= until)]

    @property
    def last_ts(self) -> float:
        return self._last_ts

    def __len__(self) -> int:
        return len(self._frames)

    def stats(self) -> dict:
        with self._lock:
            n = len(self._frames)
            span = self._frames[-1][0] - self._frames[0][0] if n > 1 else 0.0
            return {"frames": n, "span_s": round(span, 2), "mb": round(self._bytes / 2**20, 1), "max_width": self.max_width}


def _even(x: int) -> int:
    return max(2, x - (x % 2))


class ClipEncoder:
    """Потоковый энкодер H.264: кадры BGR любого размера приводятся к размеру первого (чётному)."""

    def __init__(self, path: Path, fps: float, crf: int = 26) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self.crf = crf
        self._writer = None
        self._size: tuple[int, int] | None = None
        self._tmp = self.path.with_suffix(".part.mp4")
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        import imageio_ffmpeg

        if self._writer is None:
            h0, w0 = frame.shape[:2]
            self._size = (_even(w0), _even(h0))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = imageio_ffmpeg.write_frames(
                str(self._tmp),
                self._size,
                pix_fmt_in="bgr24",
                pix_fmt_out="yuv420p",
                fps=self.fps,
                codec="libx264",
                quality=None,
                macro_block_size=2,
                ffmpeg_log_level="error",
                output_params=["-preset", "veryfast", "-crf", str(self.crf), "-movflags", "+faststart"],
            )
            self._writer.send(None)
        w, h = self._size
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        self._writer.send(np.ascontiguousarray(frame))
        self.frames += 1

    def close(self) -> bool:
        """Закрыть и атомарно переименовать; False — не было ни одного кадра."""
        if self._writer is None:
            return False
        self._writer.close()
        self._writer = None
        self._tmp.replace(self.path)
        return True

    def abort(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                pass
            self._writer = None
        self._tmp.unlink(missing_ok=True)


def write_clip(frames: list[np.ndarray], path: Path, fps: float, crf: int = 26) -> None:
    """Готовые кадры → mp4 (для тестов и офлайн-сборки)."""
    if not frames:
        raise ValueError("no frames for clip")
    enc = ClipEncoder(path, fps, crf)
    try:
        for f in frames:
            enc.write(f)
    except BaseException:
        enc.abort()
        raise
    enc.close()


@dataclass
class _ClipJob:
    event: Event
    ring: FrameRing
    start_ts: float
    end_ts: float
    on_done: Callable[[Event, str | None], None] | None


class MediaStore:
    """Снапшоты и клипы событий в data/media/<camera_id>/<YYYYMMDD>/<event_id>.{jpg,mp4}."""

    def __init__(
        self,
        data_dir: Path,
        media_dir: Path | None = None,
        pre_s: float = 5.0,
        post_s: float = 5.0,
        clip_fps: float = 10.0,
        max_width: int = 640,
        budget_mb: int = 256,
        jpeg_quality: int = 90,
        max_frames: int = 150,
        max_writers: int = 3,
        max_pending: int = 16,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.media_dir = Path(media_dir) if media_dir else self.data_dir / "media"
        self.pre_s, self.post_s = pre_s, post_s
        self.clip_fps = clip_fps
        self.max_width = max_width
        self.max_frames = max_frames
        self.budget_bytes = int(budget_mb) << 20
        self.jpeg_quality = jpeg_quality
        self.max_writers = max(1, max_writers)
        self._rings: dict[str, FrameRing] = {}
        self._capture_ids: dict[str, str] = {}
        self._previous: dict[str, tuple[str, FrameRing]] = {}
        self._all_rings: weakref.WeakSet[FrameRing] = weakref.WeakSet()
        self._lock = threading.Lock()
        self._jobs: queue.Queue[_ClipJob | None] = queue.Queue(maxsize=max_pending)
        self._threads: list[threading.Thread] = []
        self.clips_written = 0
        self.clip_errors = 0
        self.clips_dropped = 0

    # ---- кольцевые буферы ----

    def ring(self, camera_id: str) -> FrameRing:
        with self._lock:
            ring = self._rings.get(camera_id)
            if ring is None:
                ring = self._new_ring(camera_id, "")
            return ring

    def _new_ring(self, camera_id: str, capture_id: str) -> FrameRing:
        # Called under MediaStore._lock; no writer/user callback is invoked.
        ring = FrameRing(seconds=self.pre_s + 7.0, fps=self.clip_fps, max_width=self.max_width, max_frames=self.max_frames)
        self._rings[camera_id], self._capture_ids[camera_id] = ring, capture_id
        self._all_rings.add(ring)
        self._rebalance()
        return ring

    def _rebalance(self) -> None:
        # Include sealed rings held by queued/active writers, not only current
        # cameras. The previous-ring lookup retains at most one per camera.
        rings = list(self._all_rings)
        share = self.budget_bytes // max(1, len(rings))
        for r in rings:
            r.set_budget(share)

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            ring = self._rings.pop(camera_id, None)
            if ring is not None:
                ring.seal()
            self._capture_ids.pop(camera_id, None)
            self._previous.pop(camera_id, None)
            self._rebalance()

    def push_frame(self, camera_id: str, frame: np.ndarray, ts: float, capture_id: str = "") -> None:
        with self._lock:
            ring = self._rings.get(camera_id)
            current = self._capture_ids.get(camera_id, "")
            if ring is None:
                ring = self._new_ring(camera_id, capture_id)
            elif capture_id and current and capture_id != current:
                ring.seal()
                self._previous[camera_id] = (current, ring)
                ring = self._new_ring(camera_id, capture_id)
            elif capture_id and not current:
                self._capture_ids[camera_id] = capture_id  # Legacy → first known identity is bootstrap.
            ring.push(frame, ts)

    # ---- пути ----

    def rel_path(self, event: Event, suffix: str) -> str:
        day = datetime.fromtimestamp(event.confirmed_at).strftime("%Y%m%d")
        abs_path = self.media_dir / event.camera_id / day / f"{event.id}{suffix}"
        try:
            return str(abs_path.relative_to(self.data_dir))
        except ValueError:
            return str(abs_path)

    def abs_path(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.data_dir / p

    def resolve(self, rel: str | None) -> Path | None:
        """Путь из журнала → существующий файл внутри data_dir/media_dir (защита от выхода за пределы)."""
        if not rel:
            return None
        p = self.abs_path(rel).resolve()
        roots = {self.data_dir.resolve(), self.media_dir.resolve()}
        if not any(p == r or r in p.parents for r in roots):
            return None
        if p.suffix.lower() not in MEDIA_SUFFIXES:
            return None
        return p if p.is_file() else None

    # ---- снапшот ----

    def save_snapshot(self, event: Event, image: np.ndarray) -> str:
        rel = self.rel_path(event, ".jpg")
        self.write_jpeg(self.abs_path(rel), image, self.jpeg_quality)
        return rel

    def save_snapshot_bytes(self, event: Event, data: bytes) -> str:
        rel = self.rel_path(event, ".jpg")
        path = self.abs_path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(path)
        return rel

    @staticmethod
    def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(path)

    # ---- клипы ----

    def start(self) -> None:
        with self._lock:
            if self._threads:
                return
            for i in range(self.max_writers):
                t = threading.Thread(target=self._run, name=f"clip-writer-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    def stop(self, timeout: float = 15.0) -> None:
        with self._lock:
            threads, self._threads = self._threads, []
        for _ in threads:
            try:
                self._jobs.put(None, timeout=1.0)
            except queue.Full:
                break
        deadline = time.monotonic() + timeout
        for t in threads:
            t.join(timeout=max(0.1, deadline - time.monotonic()))

    def schedule_clip(self, event: Event, on_done: Callable[[Event, str | None], None] | None = None) -> bool:
        """Поставить клип [started_at − pre_s, confirmed_at + post_s] в запись. False — очередь полна."""
        self.start()
        origin = event.details.get("origin", {})
        capture_id = origin.get("capture_id", "") if isinstance(origin, dict) else ""
        with self._lock:
            ring = self._rings.get(event.camera_id)
            current = self._capture_ids.get(event.camera_id, "")
            if capture_id and current and capture_id != current:
                previous = self._previous.get(event.camera_id)
                ring = previous[1] if previous is not None and previous[0] == capture_id else None
                if ring is None:
                    self.clips_dropped += 1
                    return False  # Never substitute another capture's frames.
            elif ring is None:
                ring = self._new_ring(event.camera_id, capture_id)
            elif capture_id and not current:
                self._capture_ids[event.camera_id] = capture_id
        job = _ClipJob(
            event=event,
            ring=ring,
            start_ts=min(event.started_at, event.confirmed_at) - self.pre_s,
            end_ts=event.confirmed_at + self.post_s,
            on_done=on_done,
        )
        try:
            self._jobs.put_nowait(job)
            return True
        except queue.Full:
            self.clips_dropped += 1
            log.warning("clip queue full, clip for event %s dropped", event.id)
            return False

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._write_streaming(job)
            job = None  # Release a completed capture's ring before waiting.
            with self._lock:
                self._rebalance()

    def _write_streaming(self, job: _ClipJob) -> None:
        """Кадры буфера → ffmpeg по мере появления; постоянный fps (повтор/пропуск кадров по ts)."""
        rel = self.rel_path(job.event, ".mp4")
        enc = ClipEncoder(self.abs_path(rel), self.clip_fps)
        step = 1.0 / self.clip_fps
        last_ts = job.start_ts - 1e-6
        prev: np.ndarray | None = None
        out_ts: float | None = None  # время следующего выходного кадра
        deadline = job.end_ts + 3.0  # источник встал — дописываем то, что есть
        t0 = time.perf_counter()
        try:
            while True:
                items = job.ring.snapshot(since=last_ts + 1e-6, until=job.end_ts)
                for ts, frame in items:
                    if out_ts is None:
                        out_ts = ts
                    # слот выходного кадра out_ts; кадр занимает слот, если |ts - out_ts| < step/2
                    while prev is not None and ts >= out_ts + 0.5 * step:
                        enc.write(prev)  # повтор, если в буфере пропуск
                        out_ts += step
                    if prev is None or ts >= out_ts - 0.5 * step:
                        enc.write(frame)
                        out_ts += step
                    prev, last_ts = frame, ts
                if job.ring.sealed or job.ring.last_ts >= job.end_ts or time.time() >= deadline:
                    # Encoding a snapshot can take long enough for the source
                    # to publish the remaining post frames. Read again until
                    # empty; source readiness does not mean this batch included
                    # every frame already present in the ring.
                    if not items:
                        break
                    continue
                time.sleep(0.2)
            if enc.frames < 2:
                raise RuntimeError(f"not enough frames for clip ({enc.frames})")
            enc.close()
            self.clips_written += 1
            log.info("clip %s: %d frames @ %.0f fps, %.0f ms incl. waiting", job.event.id, enc.frames, self.clip_fps, (time.perf_counter() - t0) * 1000)
        except Exception:  # noqa: BLE001 — клип не должен ронять сервис
            enc.abort()
            self.clip_errors += 1
            rel = None
            log.exception("clip for event %s failed", job.event.id)
        if job.on_done is not None:
            try:
                job.on_done(job.event, rel)
            except Exception:  # noqa: BLE001
                log.exception("clip callback failed for %s", job.event.id)

    def stats(self) -> dict:
        with self._lock:
            rings = {cid: r.stats() for cid, r in self._rings.items()}
            retained_rings = len(self._all_rings)
        return {
            "rings": rings,
            "retained_rings": retained_rings,
            "clips_written": self.clips_written,
            "clip_errors": self.clip_errors,
            "clips_dropped": self.clips_dropped,
            "pending": self._jobs.qsize(),
        }
