"""Медиа событий: кольцевой буфер кадров камеры, снапшот JPEG и клип H.264 ±N секунд.

- FrameRing получает каждый кадр источника, хранит не чаще clip_fps и уменьшенным до
  clip_max_width; при превышении бюджета памяти уменьшает кадры дальше и сокращает окно.
- Снапшот пишется сразу при подтверждении события (синхронно, ~10 мс).
- Клип собирается в фоновом потоке: кадры до события снимаются из буфера в момент
  подтверждения, кадры после — когда истечёт post_s; кодирование ffmpeg из imageio-ffmpeg:
  libx264, yuv420p, -movflags +faststart (играет в браузере).
"""

from __future__ import annotations

import logging
import queue
import threading
import time
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


class FrameRing:
    """Кольцевой буфер (ts, кадр) одной камеры с ограничением по времени, fps и памяти."""

    def __init__(self, seconds: float = 12.0, fps: float = 10.0, max_width: int = 960, budget_bytes: int = 256 << 20) -> None:
        self.seconds = seconds
        self.fps = max(1.0, fps)
        self.max_width = max_width
        self.budget_bytes = budget_bytes
        self._frames: deque[tuple[float, np.ndarray]] = deque()
        self._bytes = 0
        self._last_ts = -1e18
        self._lock = threading.Lock()
        self.downscaled = False

    def set_budget(self, budget_bytes: int) -> None:
        with self._lock:
            self.budget_bytes = budget_bytes

    def push(self, frame: np.ndarray, ts: float) -> bool:
        if ts - self._last_ts < 0.9 / self.fps:
            return False
        h, w = frame.shape[:2]
        width = min(self.max_width, w)
        small = frame if width >= w else cv2.resize(frame, (width, max(2, int(h * width / w))), interpolation=cv2.INTER_AREA)
        if small is frame:
            small = frame.copy()  # кадр источника может переиспользоваться декодером
        with self._lock:
            self._last_ts = ts
            self._frames.append((ts, small))
            self._bytes += small.nbytes
            edge = ts - self.seconds
            while self._frames and (self._frames[0][0] < edge):
                self._bytes -= self._frames.popleft()[1].nbytes
            if self._bytes > self.budget_bytes:
                # нехватка памяти: дальше храним кадры меньше, лишнее старое выбрасываем
                if self.max_width > MIN_RING_WIDTH:
                    self.max_width = max(MIN_RING_WIDTH, int(self.max_width * 0.8))
                    self.downscaled = True
                while self._frames and self._bytes > self.budget_bytes:
                    self._bytes -= self._frames.popleft()[1].nbytes
        return True

    def snapshot(self, since: float | None = None, until: float | None = None) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            items = list(self._frames)
        return [(t, f) for t, f in items if (since is None or t >= since) and (until is None or t <= until)]

    @property
    def last_ts(self) -> float:
        return self._last_ts

    def stats(self) -> dict:
        with self._lock:
            n = len(self._frames)
            span = self._frames[-1][0] - self._frames[0][0] if n > 1 else 0.0
            return {"frames": n, "span_s": round(span, 2), "mb": round(self._bytes / 2**20, 1), "max_width": self.max_width}


def _even(x: int) -> int:
    return max(2, x - (x % 2))


def write_clip(frames: list[np.ndarray], path: Path, fps: float, crf: int = 26) -> None:
    """BGR-кадры → mp4 H.264 yuv420p с faststart. Кадры приводятся к размеру первого (чётному)."""
    import imageio_ffmpeg

    if not frames:
        raise ValueError("no frames for clip")
    h0, w0 = frames[0].shape[:2]
    w, h = _even(w0), _even(h0)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(tmp),
        (w, h),
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        fps=float(fps),
        codec="libx264",
        quality=None,
        macro_block_size=2,
        ffmpeg_log_level="error",
        output_params=["-preset", "veryfast", "-crf", str(crf), "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for f in frames:
            if f.shape[1] != w or f.shape[0] != h:
                f = cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA)
            writer.send(np.ascontiguousarray(f))
    finally:
        writer.close()
    tmp.replace(path)


@dataclass
class _ClipJob:
    event: Event
    camera_id: str
    start_ts: float
    end_ts: float
    pre_frames: list[tuple[float, np.ndarray]]
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
        max_width: int = 960,
        budget_mb: int = 768,
        jpeg_quality: int = 90,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.media_dir = Path(media_dir) if media_dir else self.data_dir / "media"
        self.pre_s, self.post_s = pre_s, post_s
        self.clip_fps = clip_fps
        self.max_width = max_width
        self.budget_bytes = int(budget_mb) << 20
        self.jpeg_quality = jpeg_quality
        self._rings: dict[str, FrameRing] = {}
        self._lock = threading.Lock()
        self._jobs: queue.Queue[_ClipJob | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self.clips_written = 0
        self.clip_errors = 0

    # ---- кольцевые буферы ----

    def ring(self, camera_id: str) -> FrameRing:
        with self._lock:
            ring = self._rings.get(camera_id)
            if ring is None:
                # окно: pre_s до начала нарушения + запас на окно подтверждения
                ring = self._rings[camera_id] = FrameRing(
                    seconds=self.pre_s + 7.0, fps=self.clip_fps, max_width=self.max_width
                )
                self._rebalance()
            return ring

    def _rebalance(self) -> None:
        share = self.budget_bytes // max(1, len(self._rings))
        for r in self._rings.values():
            r.set_budget(share)

    def remove_camera(self, camera_id: str) -> None:
        with self._lock:
            self._rings.pop(camera_id, None)
            self._rebalance()

    def push_frame(self, camera_id: str, frame: np.ndarray, ts: float) -> None:
        self.ring(camera_id).push(frame, ts)

    # ---- пути ----

    def rel_path(self, event: Event, suffix: str) -> str:
        day = datetime.fromtimestamp(event.confirmed_at).strftime("%Y%m%d")
        abs_path = self.media_dir / event.camera_id / day / f"{event.id}{suffix}"
        try:
            return str(abs_path.relative_to(self.data_dir))
        except ValueError:
            return str(abs_path)

    def resolve(self, rel: str | None) -> Path | None:
        """Путь из журнала → файл внутри data_dir/media_dir (защита от выхода за пределы)."""
        if not rel:
            return None
        p = Path(rel)
        p = p if p.is_absolute() else self.data_dir / p
        p = p.resolve()
        roots = {self.data_dir.resolve(), self.media_dir.resolve()}
        if not any(p == r or r in p.parents for r in roots):
            return None
        return p if p.is_file() else None

    # ---- снапшот ----

    def save_snapshot(self, event: Event, image: np.ndarray) -> str:
        rel = self.rel_path(event, ".jpg")
        path = self.data_dir / rel if not Path(rel).is_absolute() else Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            raise RuntimeError("JPEG encode failed")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(buf.tobytes())
        tmp.replace(path)
        return rel

    # ---- клипы ----

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, name="clip-writer", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        if self._thread is not None:
            self._jobs.put(None)
            self._thread.join(timeout=timeout)
            self._thread = None

    def schedule_clip(self, event: Event, on_done: Callable[[Event, str | None], None] | None = None) -> None:
        start_ts = min(event.started_at, event.confirmed_at) - self.pre_s
        end_ts = event.confirmed_at + self.post_s
        pre = self.ring(event.camera_id).snapshot(since=start_ts)
        self.start()
        self._jobs.put(_ClipJob(event, event.camera_id, start_ts, end_ts, pre, on_done))

    def _run(self) -> None:
        pending: list[_ClipJob] = []
        while True:
            timeout = 0.25 if pending else None
            try:
                job = self._jobs.get(timeout=timeout)
                if job is None:
                    break
                pending.append(job)
            except queue.Empty:
                pass
            now = time.time()
            ready = [j for j in pending if now >= j.end_ts + 0.3]
            for job in ready:
                pending.remove(job)
                self._finish(job)
        for job in pending:  # при остановке пишем то, что есть
            self._finish(job)

    def _finish(self, job: _ClipJob) -> None:
        rel: str | None = None
        try:
            last_pre = job.pre_frames[-1][0] if job.pre_frames else job.start_ts - 1
            ring = self._rings.get(job.camera_id)
            post = ring.snapshot(since=last_pre + 1e-6, until=job.end_ts) if ring else []
            items = job.pre_frames + post
            if len(items) < 2:
                raise RuntimeError(f"not enough frames for clip ({len(items)})")
            span = items[-1][0] - items[0][0]
            fps = min(30.0, max(1.0, (len(items) - 1) / span)) if span > 0 else self.clip_fps
            rel = self.rel_path(job.event, ".mp4")
            t0 = time.perf_counter()
            write_clip([f for _, f in items], self.data_dir / rel, fps)
            self.clips_written += 1
            log.info(
                "clip %s: %d frames, %.1f s, %.1f fps, encoded in %.0f ms",
                job.event.id, len(items), span, fps, (time.perf_counter() - t0) * 1000,
            )
        except Exception:  # noqa: BLE001 — клип не должен ронять сервис
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
        return {"rings": rings, "clips_written": self.clips_written, "clip_errors": self.clip_errors, "pending": self._jobs.qsize()}
