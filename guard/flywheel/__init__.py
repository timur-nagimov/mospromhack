"""Маховик данных: система сама собирает обучающую выборку. Контракт — docs/FLYWHEEL.md.

- store.py — FlywheelStore: SQLite data/flywheel/flywheel.db и кадры frames/<camera_id>/<id>.jpg;
- select.py — отбор кадров: полоса уверенности, мигание трека, расхождение TPU и CPU;
- collector.py — сборщик в конвейере: горячий путь только ставит задачу в очередь, запись в фоне;
- edge.py — edge-режим: сэмплы уходят на шлюз частями с ограничением трафика;
- holdout.py — кадры роликов офлайн-оценки помечаются holdout = 1;
- jobs.py — задачи ml/flywheel/* подпроцессом с логом.
"""

from __future__ import annotations

from guard.flywheel.store import (
    OPERATOR_LABELS,
    SAMPLE_SOURCES,
    SAMPLE_STATUSES,
    FlywheelStore,
)

__all__ = ["OPERATOR_LABELS", "SAMPLE_SOURCES", "SAMPLE_STATUSES", "FlywheelStore"]
