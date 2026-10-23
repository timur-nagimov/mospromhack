"""Абстрактный бэкенд инференса: onnxruntime-CPU сейчас, LinQ TPU позже.

Контракт (docs/CONTRACTS.md, «Модели и инференс»):
- name: "onnx-cpu" | "linq-tpu";
- load(spec) — загрузить модель из реестра configs/models.yaml;
- infer(model_id, batch) — синхронно и потокобезопасно; NCHW в spec.input_dtype на входе,
  сырые выходы в формате ONNX на выходе;
- stats() — p50/p95 по этапам, fps, ошибки.

Необязательные расширения с реализацией по умолчанию (бэкенду переопределять не обязательно):
- max_batch_size(model_id) — сколько кадров принимает один вызов infer (батчинг между камерами);
- infer_timed(model_id, batch) — выходы плюс разбивка времени {"transfer", "infer"} в мс;
- class_names(model_id) — классы из метаданных модели, если реестр их не задал;
- close() — освободить ресурсы.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from guard.contracts import ModelSpec
from guard.metrics import RateMeter, RollingStats


class InferenceBackend(ABC):
    name: ClassVar[str] = "abstract"

    @abstractmethod
    def load(self, spec: ModelSpec) -> None:
        """Загрузить модель; повторный вызов с той же моделью ничего не делает."""

    @abstractmethod
    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        """Сырые выходы модели для батча NCHW. Синхронно и потокобезопасно."""

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Сводка: p50/p95 по этапам, fps, ошибки — по моделям."""

    # ---- необязательные расширения ----

    def max_batch_size(self, model_id: str) -> int:
        return 1

    def infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        t0 = time.perf_counter()
        outputs = self.infer(model_id, batch)
        return outputs, {"transfer": 0.0, "infer": (time.perf_counter() - t0) * 1000.0}

    def class_names(self, model_id: str) -> list[str] | None:
        return None

    def loaded_models(self) -> list[str]:
        return []

    def close(self) -> None:
        return None


def models_dir_default() -> Path:
    return Path(os.environ.get("GUARD_MODELS_DIR", "models")).expanduser()


def resolve_model_path(path: str, models_dir: str | Path | None = None) -> Path:
    """Путь модели: абсолютный как есть, иначе относительно GUARD_MODELS_DIR."""
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    base = Path(models_dir).expanduser() if models_dir else models_dir_default()
    return base / p


class BackendStats:
    """Учёт вызовов бэкенда по моделям: перцентили этапов, fps (кадров/с), ошибки.

    Общий помощник для реализаций (onnx_cpu, linq_tpu)."""

    STAGES = ("transfer", "infer")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._models: dict[str, dict[str, Any]] = {}

    def _model(self, model_id: str) -> dict[str, Any]:
        with self._lock:
            m = self._models.get(model_id)
            if m is None:
                m = self._models[model_id] = {
                    "stages": {s: RollingStats() for s in self.STAGES},
                    "fps": RateMeter(),
                    "calls": 0,
                    "items": 0,
                    "errors": 0,
                    "last_error": None,
                    "max_batch_seen": 0,
                }
            return m

    def record(self, model_id: str, items: int, **stage_ms: float) -> None:
        m = self._model(model_id)
        for stage, ms in stage_ms.items():
            st = m["stages"].get(stage)
            if st is None:
                with self._lock:
                    st = m["stages"].setdefault(stage, RollingStats())
            st.add(ms)
        now = time.monotonic()
        for _ in range(items):
            m["fps"].tick(now)
        with self._lock:
            m["calls"] += 1
            m["items"] += items
            m["max_batch_seen"] = max(m["max_batch_seen"], items)

    def error(self, model_id: str, exc: BaseException) -> None:
        m = self._model(model_id)
        with self._lock:
            m["errors"] += 1
            m["last_error"] = f"{type(exc).__name__}: {exc}"[:300]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            models = list(self._models.items())
        out: dict[str, Any] = {}
        for model_id, m in models:
            out[model_id] = {
                "timings_ms": {s: st.summary() for s, st in m["stages"].items()},
                "fps": m["fps"].rate(),
                "calls": m["calls"],
                "items": m["items"],
                "max_batch_seen": m["max_batch_seen"],
                "errors": m["errors"],
                "last_error": m["last_error"],
            }
        return out
