"""Сверка TPU с CPU на выборочных кадрах (source = disagreement, docs/FLYWHEEL.md).

Работает только при бэкенде linq-tpu и GUARD_FLYWHEEL_SHADOW=1. Раз в GUARD_FLYWHEEL_SHADOW_PERIOD_S (30 с)
на камеру кадр и детекции TPU уходят в отдельный поток. Там та же модель считается на CPU (onnxruntime fp32,
1 поток), и рамки сравниваются (select.compare_detections):
- сравниваются только целевые классы правил камеры;
- рамка «уверенная», если conf ≥ conf_threshold модели + 0.10;
- пара — рамка того же класса с IoU ≥ 0.5 среди всех рамок другой стороны.
Уверенная рамка без пары даёт сэмпл disagreement с детекциями обеих сторон (модель CPU — <model_id>_cpu).
Очередь сверки — 2 кадра: если CPU не успевает, кадр пропускается (skipped). Горячий путь не ждёт.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from typing import Any

from guard.flywheel.select import SHADOW_MARGIN, camera_targets, compare_detections

log = logging.getLogger(__name__)

SHADOW_FORMATS = ("yolov8", "yolov8-pose")


class ShadowChecker:
    def __init__(self, detector_factory: Callable[[str], Any], *, queue_size: int = 2) -> None:
        """detector_factory(model_id) → объект с detect(frame) и spec (CPU-детектор той же модели) или None."""
        self.detector_factory = detector_factory
        self._detectors: dict[str, Any] = {}
        self._q: queue.Queue[tuple[Any, Any, Any, dict[str, list]] | None] = queue.Queue(maxsize=max(1, queue_size))
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self.checks = 0
        self.mismatches = 0
        self.skipped = 0
        self.errors = 0
        self.ms: list[float] = []

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="flywheel-shadow", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def submit(self, collector: Any, worker: Any, ctx: Any) -> bool:
        """Из горячего пути: только постановка в очередь (ссылки на кадр и копия списков детекций)."""
        try:
            self._q.put_nowait((collector, worker, ctx, {m: list(d) for m, d in ctx.detections.items()}))
            return True
        except queue.Full:
            self.skipped += 1
            return False

    def _detector(self, model_id: str) -> Any:
        if model_id not in self._detectors:
            try:
                self._detectors[model_id] = self.detector_factory(model_id)
            except Exception:  # noqa: BLE001 — нет ONNX: эту модель не сверяем
                log.exception("flywheel shadow: CPU detector for %s unavailable", model_id)
                self._detectors[model_id] = None
        return self._detectors[model_id]

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._q.get()
            if item is None:
                return
            try:
                self.check(*item)
            except Exception:  # noqa: BLE001
                self.errors += 1
                if self.errors in (1, 10, 100):
                    log.exception("flywheel shadow check failed")

    def check(self, collector: Any, worker: Any, ctx: Any, tpu_dets: dict[str, list]) -> bool:
        """Сверить кадр. True — найдено расхождение и сэмпл поставлен в очередь."""
        t0 = time.perf_counter()
        classes = camera_targets(getattr(worker, "rules", []))
        found = []
        both: dict[str, list] = {}
        for model_id, dets in tpu_dets.items():
            det = self._detector(model_id)
            if det is None:
                continue
            spec = det.spec
            if getattr(spec, "output_format", "yolov8") not in SHADOW_FORMATS:
                continue
            cpu = det.detect(ctx.frame)
            both[model_id] = dets
            both[f"{model_id}_cpu"] = cpu
            found += compare_detections(model_id, dets, cpu, float(spec.conf_threshold) + SHADOW_MARGIN, classes)
        self.checks += 1
        self.ms.append((time.perf_counter() - t0) * 1000.0)
        if len(self.ms) > 200:
            del self.ms[:100]
        if not found:
            return False
        self.mismatches += 1
        found.sort(key=lambda m: -m.conf)
        reason = found[0].reason + (f" (+{len(found) - 1})" if len(found) > 1 else "")
        meta = {
            "mismatches": [
                {"model_id": m.model_id, "cls": m.cls, "conf": round(m.conf, 4), "bbox": list(m.bbox), "only_on": m.only_on}
                for m in found[:20]
            ],
            "focus_bbox": list(found[0].bbox),
        }
        return bool(collector.submit_disagreement(worker, ctx, reason, meta, both))

    def stats(self) -> dict[str, Any]:
        ms = sorted(self.ms)
        return {
            "checks": self.checks,
            "mismatches": self.mismatches,
            "skipped": self.skipped,
            "errors": self.errors,
            "cpu_ms_p50": round(ms[len(ms) // 2], 1) if ms else None,
        }


def make_shadow(engine: Any) -> ShadowChecker | None:
    """Сверка нужна только на TPU: на onnx-cpu сравнивать не с чем."""
    backend_name = getattr(getattr(engine, "backend", None), "name", "") or ""
    if "linq-tpu" not in backend_name:
        log.info("flywheel shadow: backend %s is not linq-tpu, check disabled", backend_name or "?")
        return None
    from guard.inference import get_backend
    from guard.pipeline.detector import Detector

    settings = engine.settings
    cpu_backend = get_backend("onnx-cpu", models_dir=settings.models_dir, intra_op_threads=1, max_batch=1)

    def factory(model_id: str) -> Any:
        spec = engine.models.get(model_id)
        if spec is None:
            return None
        cpu_backend.load(spec)
        return Detector(spec, cpu_backend)

    return ShadowChecker(factory)
