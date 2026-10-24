"""Детектор: preprocess (letterbox) → backend.infer → postprocess (декодер YOLOv8).

Батчинг между камерами: BatchScheduler на модель собирает кадры разных камер в один
вызов infer, если бэкенд принимает batch > 1 (max_batch_size). Иначе кадры идут по одному.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field

import numpy as np

from guard.contracts import Detection, ModelSpec
from guard.inference.base import InferenceBackend
from guard.metrics import RollingStats
from guard.pipeline import compat

log = logging.getLogger(__name__)


@dataclass(slots=True)
class _Request:
    tensor: np.ndarray
    enqueued: float
    future: Future = field(default_factory=Future)


class BatchScheduler:
    """Очередь запросов одной модели: до max_batch кадров или batch_timeout с первого кадра."""

    def __init__(self, backend: InferenceBackend, model_id: str, max_batch: int, timeout_s: float) -> None:
        self.backend = backend
        self.model_id = model_id
        self.max_batch = max(1, max_batch)
        self.timeout_s = max(0.0, timeout_s)
        self._q: queue.Queue[_Request | None] = queue.Queue(maxsize=self.max_batch * 8)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"batch-{model_id}", daemon=True)
        self._thread.start()

    def submit(self, tensor: np.ndarray) -> Future:
        req = _Request(tensor=tensor, enqueued=time.perf_counter())
        if self._stop.is_set():
            req.future.set_exception(RuntimeError("scheduler stopped"))
            return req.future
        try:
            self._q.put(req, timeout=1.0)
        except queue.Full:
            req.future.set_exception(RuntimeError(f"batch queue for {self.model_id} is full"))
        return req.future

    def _collect(self, first: _Request) -> list[_Request]:
        batch = [first]
        deadline = first.enqueued + self.timeout_s
        while len(batch) < self.max_batch:
            remaining = deadline - time.perf_counter()
            try:
                item = self._q.get(timeout=remaining) if remaining > 0 else self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                self._stop.set()
                break
            batch.append(item)
        return batch

    def _run(self) -> None:
        while not self._stop.is_set():
            first = self._q.get()
            if first is None:
                break
            batch = [r for r in self._collect(first) if r.future.set_running_or_notify_cancel()]
            if not batch:  # все ждавшие уже отказались (таймаут) — не гоняем инференс впустую
                continue
            started = time.perf_counter()
            try:
                stacked = np.concatenate([r.tensor for r in batch], axis=0)
                outputs, t = self.backend.infer_timed(self.model_id, stacked)
            except Exception as exc:  # noqa: BLE001 — ошибку получает каждый ожидающий
                for r in batch:
                    r.future.set_exception(exc)
                continue
            for i, r in enumerate(batch):
                timings = {
                    "queue": (started - r.enqueued) * 1000.0,
                    "transfer": t.get("transfer", 0.0),
                    "infer": t.get("infer", 0.0),
                    "batch": float(len(batch)),
                }
                r.future.set_result(([o[i : i + 1] for o in outputs], timings))
        # отклоняем оставшихся
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is not None:
                item.future.set_exception(RuntimeError("scheduler stopped"))

    def stop(self) -> None:
        self._stop.set()
        self._q.put(None)
        self._thread.join(timeout=2)


class Detector:
    """Одна модель реестра поверх бэкенда; потокобезопасен (состояния нет)."""

    def __init__(self, spec: ModelSpec, backend: InferenceBackend, scheduler: BatchScheduler | None = None) -> None:
        self.spec = spec
        self.backend = backend
        self.scheduler = scheduler
        self.queue_stats = RollingStats()  # ожидание в батчере, мс
        if spec.output_format == "yolov8-pose":
            self._decode = compat.decode_yolov8_pose
        elif spec.output_format == "yolov8":
            self._decode = compat.decode_yolov8
        else:
            self._decode = None

    @property
    def model_id(self) -> str:
        return self.spec.model_id

    def preprocess(self, frame: np.ndarray):
        return compat.letterbox(frame, tuple(self.spec.input_size), self.spec.input_dtype)

    def postprocess(self, outputs: list[np.ndarray], meta) -> list[Detection]:
        if self._decode is None:
            return []
        return self._decode(outputs, meta, self.spec)

    def detect(self, frame: np.ndarray, timings: dict[str, float] | None = None) -> list[Detection]:
        """Детекции на кадре; времена этапов (мс) прибавляются к timings по ключам TIMING_KEYS."""
        t0 = time.perf_counter()
        tensor, meta = self.preprocess(frame)
        t1 = time.perf_counter()
        if self.scheduler is not None:
            future = self.scheduler.submit(tensor)
            try:
                outputs, t = future.result(timeout=30)
            except TimeoutError:
                future.cancel()  # после простоя бэкенда (TPU) не обрабатываем накопившиеся устаревшие кадры
                raise
        else:
            outputs, t = self.backend.infer_timed(self.model_id, tensor)
        t2 = time.perf_counter()
        dets = self.postprocess(outputs, meta)
        t3 = time.perf_counter()
        if timings is not None:
            infer_ms = t.get("infer", (t2 - t1) * 1000.0)
            queue_ms = t.get("queue", 0.0)
            self.queue_stats.add(queue_ms)
            transfer_ms = t.get("transfer", 0.0)
            timings["preprocess"] = timings.get("preprocess", 0.0) + (t1 - t0) * 1000.0
            timings["queue"] = timings.get("queue", 0.0) + queue_ms
            timings["transfer"] = timings.get("transfer", 0.0) + transfer_ms
            timings["infer"] = timings.get("infer", 0.0) + infer_ms
            timings["postprocess"] = timings.get("postprocess", 0.0) + (t3 - t2) * 1000.0
        return dets
