"""Выбор основных реализаций модулей или резервных вариантов из _fallbacks.

После слияния guard/vision, guard/tracking, guard/events/{manager,journal}.py
замены отключаются сами; что используется сейчас — в FALLBACKS (видно в /health).
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

FALLBACKS: dict[str, bool] = {}

try:
    from guard.vision.preprocess import letterbox
    from guard.vision.yolo import decode_yolov8, decode_yolov8_pose

    FALLBACKS["vision"] = False
except ImportError:
    from guard.pipeline._fallbacks import decode_yolov8, decode_yolov8_pose, letterbox

    FALLBACKS["vision"] = True

try:
    from guard.vision.draw import draw_event_snapshot, draw_overlay

    FALLBACKS["draw"] = False
except ImportError:
    from guard.pipeline._fallbacks import draw_event_snapshot, draw_overlay

    FALLBACKS["draw"] = True

try:
    from guard.tracking import Tracker, assign_attributes

    FALLBACKS["tracking"] = False
except ImportError:
    from guard.pipeline._fallbacks import IouTracker as Tracker
    from guard.pipeline._fallbacks import assign_attributes

    FALLBACKS["tracking"] = True

try:
    from guard.events.manager import EventManager

    FALLBACKS["event_manager"] = False
except ImportError:
    from guard.pipeline._fallbacks import SimpleEventManager as EventManager

    FALLBACKS["event_manager"] = True

try:
    from guard.events.journal import Journal

    FALLBACKS["journal"] = False
except ImportError:
    from guard.pipeline._fallbacks import MemoryJournal as Journal

    FALLBACKS["journal"] = True

# Отрисовку зон делаем сами всегда: она нужна и снапшоту, и потоку с оверлеями
from guard.pipeline._fallbacks import draw_overlay as _fallback_draw_overlay  # noqa: E402
from guard.pipeline._fallbacks import draw_zones  # noqa: E402

def safe_draw_overlay(frame: Any, people: Any, objects: Any, zones: Any, highlight: Any) -> Any:
    """Оверлей MJPEG. По уточнению API (24.10) он компактный: «#ID» и короткая метка нарушителя.
    Подписи guard.vision.draw_overlay длиннее, поэтому для потока используем свой рендерер."""
    return _fallback_draw_overlay(frame, people, objects, zones, events_highlight=highlight)


for _name, _fb in FALLBACKS.items():
    if _fb:
        log.info("compat: %s — временная замена из guard/pipeline/_fallbacks.py", _name)

__all__ = [
    "FALLBACKS",
    "EventManager",
    "Journal",
    "Tracker",
    "assign_attributes",
    "decode_yolov8",
    "decode_yolov8_pose",
    "draw_event_snapshot",
    "draw_overlay",
    "draw_zones",
    "letterbox",
    "safe_draw_overlay",
]
