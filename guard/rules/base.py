"""API правил-плагинов. Новый тип нарушения = новый модуль в guard/rules/plugins/ с @register_rule.

Конвейер не меняется: он вызывает evaluate() у всех включённых правил камеры
и передаёт нарушения в EventManager, который отвечает за подтверждение и дедупликацию.
"""

from __future__ import annotations

import importlib
import pkgutil
from abc import ABC, abstractmethod
from typing import Any, ClassVar

from guard.contracts import CameraConfig, FrameContext, RuleConfig, Severity, Violation


class RulePlugin(ABC):
    type: ClassVar[str]  # уникальное имя, на него ссылается RuleConfig.type
    title: ClassVar[str]  # название для UI и журнала
    required_models: ClassVar[frozenset[str]] = frozenset({"det"})
    default_severity: ClassVar[Severity] = Severity.high
    params_schema: ClassVar[dict[str, Any]] = {}  # JSON Schema параметров — по ней UI строит форму

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        self.cfg = cfg
        self.camera = camera

    @abstractmethod
    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        """Нарушения на этом кадре. Без памяти о подтверждении: окно и дубли ведёт EventManager."""

    def violation(self, ctx: FrameContext, **kwargs: Any) -> Violation:
        kwargs.setdefault("severity", self.cfg.severity)
        return Violation(rule_id=self.cfg.rule_id, type=self.type, **kwargs)


_REGISTRY: dict[str, type[RulePlugin]] = {}
_PLUGINS_LOADED = False


def register_rule(cls: type[RulePlugin]) -> type[RulePlugin]:
    if cls.type in _REGISTRY and _REGISTRY[cls.type] is not cls:
        raise ValueError(f"rule type {cls.type!r} already registered")
    _REGISTRY[cls.type] = cls
    return cls


def load_plugins() -> None:
    """Импортирует все модули плагинов один раз. Флаг, а не пустота реестра:
    если какой-то плагин уже импортирован напрямую, остальные всё равно должны загрузиться."""
    global _PLUGINS_LOADED
    if _PLUGINS_LOADED:
        return
    from guard.rules import plugins

    for mod in pkgutil.iter_modules(plugins.__path__):
        importlib.import_module(f"{plugins.__name__}.{mod.name}")
    _PLUGINS_LOADED = True


def get_rule(rule_type: str) -> type[RulePlugin]:
    load_plugins()
    return _REGISTRY[rule_type]


def build_rules(camera: CameraConfig) -> list[RulePlugin]:
    return [get_rule(r.type)(r, camera) for r in camera.rules if r.enabled]


def with_default_severity(camera: CameraConfig) -> CameraConfig:
    """Правило без явной severity получает default_severity своего плагина явно.

    Иначе API отдаёт общий default RuleConfig (high), UI сохраняет его обратно через PUT,
    и fall_immobile, vehicle_proximity, smoking_fire теряют critical. Неизвестные типы не трогаем:
    их отвергнет валидация камеры.
    """
    load_plugins()
    rules = [
        r.model_copy(update={"severity": _REGISTRY[r.type].default_severity})
        if "severity" not in r.model_fields_set and r.type in _REGISTRY
        else r
        for r in camera.rules
    ]
    if all(new is old for new, old in zip(rules, camera.rules)):
        return camera
    return camera.model_copy(update={"rules": rules})


def available_rules() -> list[dict[str, Any]]:
    load_plugins()
    return [
        {
            "type": cls.type,
            "title": cls.title,
            "required_models": sorted(cls.required_models),
            "default_severity": cls.default_severity.value,
            "params_schema": cls.params_schema,
        }
        for cls in _REGISTRY.values()
    ]
