"""Бэкенд onnxruntime на CPU: одна сессия на модель, потокобезопасный infer."""

from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort

from guard.contracts import ModelSpec
from guard.inference.base import BackendStats, InferenceBackend, resolve_model_path

log = logging.getLogger(__name__)

_ORT_DTYPES = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(uint8)": np.uint8,
    "tensor(int8)": np.int8,
}
# Метаданные ultralytics: names = "{0: 'person', 1: 'bicycle', ...}"
_NAMES_RE = re.compile(r"(\d+)\s*:\s*(['\"])(.*?)\2")


@dataclass(slots=True)
class _LoadedModel:
    spec: ModelSpec
    path: Path
    session: ort.InferenceSession
    input_name: str
    output_names: list[str]
    input_dtype: Any
    fixed_batch: int | None  # None — батч динамический


class OnnxCpuBackend(InferenceBackend):
    name = "onnx-cpu"

    def __init__(
        self,
        models_dir: str | Path | None = None,
        intra_op_threads: int | None = None,
        max_batch: int = 8,
    ) -> None:
        self.models_dir = Path(models_dir).expanduser() if models_dir else None
        env_threads = os.environ.get("GUARD_ORT_THREADS")
        self.intra_op_threads = intra_op_threads or (int(env_threads) if env_threads else 4)
        self.max_batch = max(1, int(max_batch))
        self._models: dict[str, _LoadedModel] = {}
        self._lock = threading.Lock()
        self._stats = BackendStats()

    # ---- загрузка ----

    def load(self, spec: ModelSpec) -> None:
        path = resolve_model_path(spec.onnx_path, self.models_dir)
        with self._lock:
            cur = self._models.get(spec.model_id)
            if cur is not None and cur.path == path:
                cur.spec = spec  # пороги могли поменяться — сессию не пересоздаём
                return
        if not path.exists():
            raise FileNotFoundError(f"ONNX model for {spec.model_id!r} not found: {path}")
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = self.intra_op_threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        t0 = time.perf_counter()
        session = ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])
        inp = session.get_inputs()[0]
        batch_dim = inp.shape[0] if inp.shape else None
        fixed = batch_dim if isinstance(batch_dim, int) and batch_dim > 0 else None
        loaded = _LoadedModel(
            spec=spec,
            path=path,
            session=session,
            input_name=inp.name,
            output_names=[o.name for o in session.get_outputs()],
            input_dtype=_ORT_DTYPES.get(inp.type, np.float32),
            fixed_batch=fixed,
        )
        with self._lock:
            self._models[spec.model_id] = loaded
        log.info(
            "onnx-cpu: loaded %s from %s in %.0f ms (input %s %s, batch=%s, threads=%d)",
            spec.model_id, path, (time.perf_counter() - t0) * 1000, inp.name, inp.shape,
            fixed or "dynamic", self.intra_op_threads,
        )

    def _get(self, model_id: str) -> _LoadedModel:
        m = self._models.get(model_id)
        if m is None:
            raise KeyError(f"model {model_id!r} is not loaded")
        return m

    # ---- инференс ----

    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        outputs, _ = self.infer_timed(model_id, batch)
        return outputs

    def infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        m = self._get(model_id)
        if batch.dtype != m.input_dtype or not batch.flags["C_CONTIGUOUS"]:
            batch = np.ascontiguousarray(batch, dtype=m.input_dtype)
        n = int(batch.shape[0])
        t0 = time.perf_counter()
        try:
            if m.fixed_batch is None or n == m.fixed_batch:
                outputs = m.session.run(m.output_names, {m.input_name: batch})
            else:
                # Статический батч модели не совпадает с пришедшим — гоняем частями и склеиваем по оси 0
                step = m.fixed_batch
                parts = [m.session.run(m.output_names, {m.input_name: batch[i : i + step]}) for i in range(0, n, step)]
                outputs = [np.concatenate([p[k] for p in parts], axis=0) for k in range(len(m.output_names))]
        except Exception as exc:  # noqa: BLE001 — учитываем и пробрасываем
            self._stats.error(model_id, exc)
            raise
        infer_ms = (time.perf_counter() - t0) * 1000.0
        self._stats.record(model_id, items=n, infer=infer_ms, transfer=0.0)
        return outputs, {"transfer": 0.0, "infer": infer_ms}

    # ---- сведения ----

    def max_batch_size(self, model_id: str) -> int:
        m = self._models.get(model_id)
        if m is None:
            return 1
        return self.max_batch if m.fixed_batch is None else m.fixed_batch

    def class_names(self, model_id: str) -> list[str] | None:
        """Классы из метаданных ONNX (ultralytics кладёт их в custom_metadata_map["names"])."""
        m = self._models.get(model_id)
        if m is None:
            return None
        raw = m.session.get_modelmeta().custom_metadata_map.get("names")
        if not raw:
            return None
        pairs = sorted((int(idx), name) for idx, _, name in _NAMES_RE.findall(raw))
        return [name for _, name in pairs] or None

    def loaded_models(self) -> list[str]:
        return sorted(self._models)

    def stats(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "device": "cpu",
            "utilization": self._stats.utilization(),
            "providers": ["CPUExecutionProvider"],
            "intra_op_threads": self.intra_op_threads,
            "models": self._stats.snapshot(),
        }

    def close(self) -> None:
        with self._lock:
            self._models.clear()


def create_backend(**kwargs: Any) -> OnnxCpuBackend:
    return OnnxCpuBackend(**kwargs)
