"""Выбор бэкенда инференса по имени; модули реализаций импортируются лениво.

    backend = get_backend()            # имя из GUARD_BACKEND, по умолчанию "onnx-cpu"
    backend = get_backend("linq-tpu")  # guard.inference.linq_tpu (A1)

Модуль реализации может предоставить фабрику create_backend(**kwargs); иначе берётся
подкласс InferenceBackend с подходящим name. Лишние kwargs, которых конструктор
не принимает, отбрасываются.
"""

from __future__ import annotations

import importlib
import inspect
import os
from typing import Any

from guard.inference.base import InferenceBackend

DEFAULT_BACKEND = "onnx-cpu"

_MODULES = {
    "onnx-cpu": "guard.inference.onnx_cpu",
    "linq-tpu": "guard.inference.linq_tpu",
}
_ALIASES = {"onnx": "onnx-cpu", "cpu": "onnx-cpu", "ort": "onnx-cpu", "tpu": "linq-tpu", "linq": "linq-tpu"}


def backend_name_from_env() -> str:
    return normalize_backend_name(os.environ.get("GUARD_BACKEND") or DEFAULT_BACKEND)


def normalize_backend_name(name: str) -> str:
    key = name.strip().lower().replace("_", "-")
    return _ALIASES.get(key, key)


def available_backends() -> list[str]:
    return sorted(_MODULES)


def _filter_kwargs(fn: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kwargs
    return {k: v for k, v in kwargs.items() if k in params}


def get_backend(name: str | None = None, **kwargs: Any) -> InferenceBackend:
    """Создать бэкенд по имени (или из GUARD_BACKEND)."""
    resolved = normalize_backend_name(name) if name else backend_name_from_env()
    module_name = _MODULES.get(resolved)
    if module_name is None:
        raise ValueError(f"unknown inference backend {resolved!r}; available: {available_backends()}")
    module = importlib.import_module(module_name)  # ImportError пробрасывается: бэкенда нет в сборке
    candidates = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj)
        and issubclass(obj, InferenceBackend)
        and not inspect.isabstract(obj)
        and getattr(obj, "name", None) == resolved
    ]
    cls = candidates[0] if candidates else None
    factory = getattr(module, "create_backend", None)
    # общие kwargs (models_dir, intra_op_threads, max_batch) — только те, что понимает реализация
    accepted = _filter_kwargs(cls, kwargs) if cls is not None else kwargs
    if callable(factory):
        backend = factory(**_filter_kwargs(factory, accepted))
    elif cls is not None:
        backend = cls(**accepted)
    else:
        raise ImportError(f"{module_name} has no InferenceBackend with name {resolved!r}")
    if not isinstance(backend, InferenceBackend):
        raise TypeError(f"{module_name} returned {type(backend).__name__}, expected InferenceBackend")
    return backend


__all__ = [
    "DEFAULT_BACKEND",
    "InferenceBackend",
    "available_backends",
    "backend_name_from_env",
    "get_backend",
    "normalize_backend_name",
]
