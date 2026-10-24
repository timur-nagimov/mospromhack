"""Помощники для тестов конвейера и API: фейковый бэкенд, настройки во временном каталоге."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from guard.contracts import Detection, ModelSpec
from guard.inference.base import BackendStats, InferenceBackend
from guard.pipeline.settings import Settings

COCO_LIKE = ["person"] + [f"cls{i}" for i in range(1, 80)]


class FakeBackend(InferenceBackend):
    """Бэкенд без моделей: пустой выход YOLOv8 (1, 84, 8400) — детекций нет."""

    name = "fake"

    def __init__(self, max_batch: int = 1) -> None:
        self.specs: dict[str, ModelSpec] = {}
        self._max_batch = max_batch
        self._stats = BackendStats()
        self.calls = 0

    def load(self, spec: ModelSpec) -> None:
        self.specs[spec.model_id] = spec

    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        self.calls += 1
        nc = len(self.specs[model_id].classes) or 80
        self._stats.record(model_id, items=int(batch.shape[0]), infer=0.1, transfer=0.0)
        return [np.zeros((batch.shape[0], 4 + nc, 8400), dtype=np.float32)]

    def max_batch_size(self, model_id: str) -> int:
        return self._max_batch

    def stats(self) -> dict[str, Any]:
        return {"backend": self.name, "device": "cpu", "utilization": self._stats.utilization(), "models": self._stats.snapshot()}


class ScriptedDetector:
    """Детектор-заглушка: детекции задаёт функция от номера вызова."""

    def __init__(self, script, model_id: str = "det") -> None:
        self.script = script
        self.model_id = model_id
        self.n = 0
        self.scheduler = None
        from guard.metrics import RollingStats

        self.queue_stats = RollingStats()

    def detect(self, frame: np.ndarray, timings: dict[str, float] | None = None) -> list[Detection]:
        dets = self.script(self.n, frame)
        self.n += 1
        if timings is not None:
            timings["infer"] = timings.get("infer", 0.0) + 1.0
        return dets


def make_settings(tmp: Path, **overrides: Any) -> Settings:
    """Изолированные настройки: свои data/, models.yaml, cameras.yaml, guard.db."""
    data = tmp / "data"
    (data / "media").mkdir(parents=True, exist_ok=True)
    models_dir = tmp / "models"
    models_dir.mkdir(exist_ok=True)
    models_cfg = tmp / "models.yaml"
    models_cfg.write_text(
        yaml.safe_dump({"models": [{"model_id": "det", "onnx_path": "fake.onnx", "classes": COCO_LIKE}]}),
        encoding="utf-8",
    )
    params: dict[str, Any] = dict(
        data_dir=data,
        models_dir=models_dir,
        cameras_config=tmp / "cameras.yaml",
        models_config=models_cfg,
        db_path=data / "guard.db",
        media_dir=data / "media",
        frontend_dist=tmp / "no-frontend",
        backend="fake",
        autostart=True,
        clip_pre_s=1.0,
        clip_post_s=0.5,
    )
    params.update(overrides)
    if params.get("mode") in {"edge", "gateway"} and "edge_id" not in overrides:
        params["edge_id"] = "test-edge"
    return Settings(**params)


def write_images(folder: Path, n: int = 5, size: tuple[int, int] = (320, 240)) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    w, h = size
    for i in range(n):
        img = np.full((h, w, 3), 40 + 30 * i % 200, dtype=np.uint8)
        cv2.putText(img, str(i), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.imwrite(str(folder / f"frame_{i:03d}.jpg"), img)
    return folder


def person(bbox: tuple[float, float, float, float], conf: float = 0.9) -> Detection:
    return Detection(model="det", cls="person", cls_id=0, conf=conf, bbox=bbox)
