"""Отчётность по событиям охраны труда.

Экспорты загружаются лениво, чтобы модуль shift_report можно было запускать
через ``python -m guard.reports.shift_report`` без повторного импорта.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "Hotspot",
    "QualityStats",
    "ReportData",
    "Shift",
    "SummaryClient",
    "Trend",
    "ZoneCount",
    "build_report",
    "render_html",
    "render_markdown",
    "summarize_with_llm",
]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".shift_report", __name__), name)
    globals()[name] = value
    return value
