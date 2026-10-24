"""Автоматическое переключение TPU → CPU с периодической проверкой TPU."""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from guard.contracts import ModelSpec
from guard.inference.base import InferenceBackend
from guard.inference.linq_tpu import BackendUnavailable


class FallbackBackend(InferenceBackend):
    """Один интерфейс инференса поверх двух бэкендов.

    name сохраняет имя основного бэкенда; фактический бэкенд указан в stats.active.
    CPU загружается лениво, с исходным ModelSpec и его onnx_path.
    Только один параллельный вызов проверяет primary после интервала восстановления.
    """

    def __init__(
        self,
        primary: InferenceBackend,
        secondary: InferenceBackend,
        retry_interval_s: float = 15.0,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.retry_interval_s = max(0.0, float(retry_interval_s))
        self._condition = threading.Condition(threading.RLock())
        self._close_lock = threading.Lock()
        self._load_locks = (threading.Lock(), threading.Lock())
        self._loaded: tuple[set[str], set[str]] = (set(), set())
        self._specs: dict[str, ModelSpec] = {}
        self._availability: dict[str, bool] = {}
        self._secondary_errors: dict[str, str] = {}
        self._primary_error: str | None = None
        self._fallback = False
        self._fallback_since: float | None = None
        self._fallback_seconds = 0.0
        self._next_probe = 0.0
        self._probing = False
        self._generation = 0
        self._switches = 0
        self._closed = False
        self._inflight = 0
        self._pool: ThreadPoolExecutor | None = None

    @property
    def name(self) -> str:
        return self.primary.name

    @property
    def _client(self) -> Any:
        """Совместимость с подменой транспорта в существующих офлайн-тестах."""
        return getattr(self.primary, "_client")

    @_client.setter
    def _client(self, value: Any) -> None:
        setattr(self.primary, "_client", value)

    @contextmanager
    def _operation(self) -> Iterator[None]:
        with self._condition:
            if self._closed:
                raise BackendUnavailable("fallback: бэкенд закрыт")
            self._inflight += 1
        try:
            yield
        finally:
            with self._condition:
                self._inflight -= 1
                self._condition.notify_all()

    def _ensure_loaded(self, model_id: str, primary: bool) -> None:
        index = 0 if primary else 1
        backend = self.primary if primary else self.secondary
        with self._load_locks[index]:
            with self._condition:
                if self._closed:
                    raise BackendUnavailable("fallback: бэкенд закрыт")
                if model_id in self._loaded[index]:
                    return
                spec = self._specs[model_id]
            backend.load(spec)
            with self._condition:
                self._loaded[index].add(model_id)

    @staticmethod
    def _error(exc: Exception) -> str:
        return f"{type(exc).__name__}: {exc}"[:300]

    def _activate(self, exc: Exception) -> None:
        with self._condition:
            now = time.monotonic()
            if not self._fallback:
                self._fallback = True
                self._fallback_since = now
                self._switches += 1
                self._generation += 1
            self._primary_error = self._error(exc)
            self._next_probe = now + self.retry_interval_s
            self._probing = False

    def _recover(self, generation: int, probe: bool) -> None:
        with self._condition:
            # Успех старого параллельного запроса не отменяет более новый отказ.
            if not probe and generation != self._generation:
                return
            if self._fallback:
                now = time.monotonic()
                if self._fallback_since is not None:
                    self._fallback_seconds += max(0.0, now - self._fallback_since)
                self._fallback_since = None
                self._fallback = False
                self._switches += 1
                self._generation += 1
            self._probing = False
            self._primary_error = None

    def _mark_available(self, model_id: str) -> None:
        with self._condition:
            self._availability[model_id] = True
            self._secondary_errors.pop(model_id, None)

    def _secondary_failed(self, model_id: str, exc: Exception) -> BackendUnavailable:
        with self._condition:
            self._availability[model_id] = False
            self._secondary_errors[model_id] = self._error(exc)
            primary_error = self._primary_error
        return BackendUnavailable(
            f"{model_id}: primary unavailable ({primary_error}); "
            f"{self.secondary.name}: {self._error(exc)}"
        )

    def load(self, spec: ModelSpec) -> None:
        with self._operation():
            with self._condition:
                self._specs.setdefault(spec.model_id, spec)
                self._availability.setdefault(spec.model_id, True)
                use_primary = not self._fallback
            if use_primary:
                try:
                    self._ensure_loaded(spec.model_id, True)
                except Exception as exc:
                    self._activate(exc)
                else:
                    self._mark_available(spec.model_id)
                    return
            try:
                self._ensure_loaded(spec.model_id, False)
            except Exception as exc:
                raise self._secondary_failed(spec.model_id, exc) from exc
            self._mark_available(spec.model_id)

    def _run(
        self, model_id: str, batch: np.ndarray, primary: bool,
    ) -> tuple[list[np.ndarray], dict[str, float]]:
        self._ensure_loaded(model_id, primary)
        if primary:
            return self.primary.infer_timed(model_id, batch)

        # Уже созданный BatchScheduler сохраняет размер TPU-батча при переключении.
        # Делим его по возможности CPU; последний неполный батч дополняем нулями,
        # чтобы поддержать и статические ONNX с batch > 1.
        x = np.asarray(batch)
        if x.ndim == 3:
            x = x[None]
        if x.shape[0] == 0:
            raise ValueError("fallback: пустой батч")
        step = max(1, int(self.secondary.max_batch_size(model_id)))
        parts: list[list[np.ndarray]] = []
        timings: dict[str, float] = {}
        for start in range(0, x.shape[0], step):
            piece = x[start:start + step]
            count = piece.shape[0]
            if count < step:
                padding = np.zeros((step - count,) + x.shape[1:], dtype=x.dtype)
                piece = np.concatenate([piece, padding], axis=0)
            outputs, measured = self.secondary.infer_timed(model_id, piece)
            parts.append([np.asarray(output)[:count] for output in outputs])
            for key, value in measured.items():
                timings[key] = timings.get(key, 0.0) + float(value)
        outputs = [
            np.concatenate([part[index] for part in parts], axis=0)
            for index in range(len(parts[0]))
        ]
        return outputs, timings

    def infer_timed(self, model_id: str, batch: np.ndarray) -> tuple[list[np.ndarray], dict[str, float]]:
        with self._operation():
            with self._condition:
                if model_id not in self._specs:
                    raise KeyError(f"модель {model_id} не загружена")
                probe = (
                    self._fallback
                    and not self._probing
                    and time.monotonic() >= self._next_probe
                )
                use_primary = not self._fallback or probe
                generation = self._generation
                if probe:
                    self._probing = True
                    self._next_probe = time.monotonic() + self.retry_interval_s
            if use_primary:
                try:
                    result = self._run(model_id, batch, True)
                except Exception as exc:
                    self._activate(exc)
                else:
                    self._recover(generation, probe)
                    self._mark_available(model_id)
                    return result
            try:
                result = self._run(model_id, batch, False)
            except Exception as exc:
                raise self._secondary_failed(model_id, exc) from exc
            self._mark_available(model_id)
            return result

    def infer(self, model_id: str, batch: np.ndarray) -> list[np.ndarray]:
        return self.infer_timed(model_id, batch)[0]

    def submit(self, model_id: str, batch: np.ndarray) -> Future:
        with self._condition:
            if self._closed:
                future: Future = Future()
                future.set_exception(BackendUnavailable("fallback: бэкенд закрыт"))
                return future
            if self._pool is None:
                workers = max(1, int(getattr(self.primary, "window", 4)))
                self._pool = ThreadPoolExecutor(workers, thread_name_prefix="inference-fallback")
            return self._pool.submit(self.infer_timed, model_id, batch)

    def loaded_models(self) -> list[str]:
        with self._condition:
            return sorted(self._loaded[0] | self._loaded[1])

    def max_batch_size(self, model_id: str) -> int:
        with self._condition:
            active = self.secondary if self._fallback else self.primary
        return max(1, int(active.max_batch_size(model_id)))

    def class_names(self, model_id: str) -> list[str] | None:
        with self._condition:
            spec = self._specs.get(model_id)
            if spec is not None and spec.classes:
                return list(spec.classes)
            backends = (
                (self.secondary, self.primary) if self._fallback
                else (self.primary, self.secondary)
            )
        for backend in backends:
            names = backend.class_names(model_id)
            if names:
                return names
        return None

    def stats(self) -> dict[str, Any]:
        primary_stats = dict(self.primary.stats())
        secondary_stats = dict(self.secondary.stats())
        with self._condition:
            elapsed = self._fallback_seconds
            if self._fallback_since is not None:
                elapsed += max(0.0, time.monotonic() - self._fallback_since)
            active = self.secondary if self._fallback else self.primary
            out = dict(secondary_stats if self._fallback else primary_stats)
            out.update({
                "backend": self.name,
                "active": active.name,
                "fallback": self._fallback,
                "switches": self._switches,
                "fallback_seconds": elapsed,
                "circuit_open": bool(primary_stats.get("circuit_open", False)),
                "last_error": (
                    next(iter(self._secondary_errors.values()), None)
                    or self._primary_error
                    or primary_stats.get("last_error")
                ),
                "available": not self._closed and (
                    not self._availability or any(self._availability.values())
                ),
                "closed": self._closed,
                "primary": primary_stats,
                "secondary": secondary_stats,
            })
            return out

    def close(self) -> None:
        with self._close_lock:
            with self._condition:
                if self._closed:
                    return
                self._closed = True
                pool = self._pool
                if self._fallback_since is not None:
                    self._fallback_seconds += max(
                        0.0, time.monotonic() - self._fallback_since,
                    )
                    self._fallback_since = None
            try:
                # Сначала прерываем ожидающие TPU-вызовы.
                self.primary.close()
            finally:
                try:
                    if pool is not None:
                        pool.shutdown(wait=True, cancel_futures=True)
                    with self._condition:
                        while self._inflight:
                            self._condition.wait()
                        self._pool = None
                        self._loaded[0].clear()
                        self._loaded[1].clear()
                finally:
                    self.secondary.close()


__all__ = ["BackendUnavailable", "FallbackBackend"]
