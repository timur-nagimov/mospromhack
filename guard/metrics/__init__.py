"""Метрики конвейера: скользящие перцентили по этапам, частоты кадров и лаг.

Всё потокобезопасно: пишут потоки источников и камер, читают API и рассылка WS.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

import numpy as np

from guard.contracts import TIMING_KEYS


class RollingStats:
    """Окно последних значений (не больше maxlen и не старше window_s) с перцентилями."""

    def __init__(self, maxlen: int = 300, window_s: float | None = 60.0) -> None:
        self._values: deque[tuple[float, float]] = deque(maxlen=maxlen)
        self._window_s = window_s
        self._lock = threading.Lock()

    def add(self, value: float, ts: float | None = None) -> None:
        with self._lock:
            self._values.append((time.monotonic() if ts is None else ts, float(value)))

    def _recent(self) -> np.ndarray:
        with self._lock:
            items = list(self._values)
        if self._window_s is not None and items:
            edge = time.monotonic() - self._window_s
            items = [it for it in items if it[0] >= edge]
        return np.fromiter((v for _, v in items), dtype=np.float64, count=len(items))

    def summary(self) -> dict[str, float]:
        vals = self._recent()
        if vals.size == 0:
            return {"p50": 0.0, "p95": 0.0, "n": 0}
        p50, p95 = np.percentile(vals, (50, 95))
        return {"p50": round(float(p50), 2), "p95": round(float(p95), 2), "n": int(vals.size)}


class RateMeter:
    """Частота событий (кадров) за последние window_s секунд."""

    def __init__(self, window_s: float = 5.0) -> None:
        self._ticks: deque[float] = deque()
        self._window_s = window_s
        self._lock = threading.Lock()
        self.total = 0

    def tick(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._ticks.append(now)
            self.total += 1
            self._prune(now)

    def _prune(self, now: float) -> None:
        edge = now - self._window_s
        while self._ticks and self._ticks[0] < edge:
            self._ticks.popleft()

    def rate(self, now: float | None = None) -> float:
        now = time.monotonic() if now is None else now
        with self._lock:
            self._prune(now)
            n = len(self._ticks)
            if n < 2:
                return 0.0
            # (n-1) интервалов от первого тика окна до «сейчас»: при остановке потока частота спадает
            return round((n - 1) / max(now - self._ticks[0], 1e-3), 2)


class CameraMetrics:
    """Метрики одной камеры: fps на входе и в обработке, лаг, перцентили по TIMING_KEYS."""

    def __init__(self, camera_id: str, window: int = 300) -> None:
        self.camera_id = camera_id
        self.fps_in = RateMeter()
        self.fps_processed = RateMeter()
        self.timings = {k: RollingStats(maxlen=window) for k in TIMING_KEYS}
        self._lag_ema: float | None = None
        self._last_done_mono: float | None = None
        self._last_capture_ts: float | None = None
        self._period_s = 0.2
        self._lock = threading.Lock()
        self.frames_processed = 0
        self.errors = 0

    def set_period(self, sample_fps: float) -> None:
        self._period_s = 1.0 / max(sample_fps, 0.1)

    def on_source_frame(self) -> None:
        self.fps_in.tick()

    def record_frame(self, timings: dict[str, float], capture_ts: float, done_ts: float | None = None) -> None:
        done_ts = time.time() if done_ts is None else done_ts
        for key, value in timings.items():
            stats = self.timings.get(key)
            if stats is not None:
                stats.add(value)
        self.fps_processed.tick()
        lag = max(0.0, done_ts - capture_ts)
        with self._lock:
            self._lag_ema = lag if self._lag_ema is None else 0.8 * self._lag_ema + 0.2 * lag
            self._last_done_mono = time.monotonic()
            self._last_capture_ts = capture_ts
            self.frames_processed += 1

    def lag_s(self) -> float:
        with self._lock:
            if self._lag_ema is None or self._last_done_mono is None or self._last_capture_ts is None:
                return 0.0
            stalled = time.monotonic() - self._last_done_mono > 2 * self._period_s + 1.0
            lag = time.time() - self._last_capture_ts if stalled else self._lag_ema
        return round(lag, 3)

    def snapshot(self) -> dict[str, Any]:
        return {
            "fps_in": self.fps_in.rate(),
            "fps_processed": self.fps_processed.rate(),
            "lag_s": self.lag_s(),
            "timings_ms": {k: self.timings[k].summary() for k in TIMING_KEYS},
            "frames_processed": self.frames_processed,
            "errors": self.errors,
        }


class MetricsRegistry:
    """Метрики всех камер."""

    def __init__(self) -> None:
        self._cameras: dict[str, CameraMetrics] = {}
        self._lock = threading.Lock()

    def camera(self, camera_id: str) -> CameraMetrics:
        with self._lock:
            cm = self._cameras.get(camera_id)
            if cm is None:
                cm = self._cameras[camera_id] = CameraMetrics(camera_id)
            return cm

    def reset(self, camera_id: str) -> CameraMetrics:
        with self._lock:
            cm = self._cameras[camera_id] = CameraMetrics(camera_id)
            return cm

    def remove(self, camera_id: str) -> None:
        with self._lock:
            self._cameras.pop(camera_id, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            cams = list(self._cameras.items())
        return {cid: cm.snapshot() for cid, cm in cams}
