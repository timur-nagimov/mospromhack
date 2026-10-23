"""Общие контракты системы — единственный источник типов для всех модулей.

Описание и правила изменения: docs/CONTRACTS.md. Меняет файл только оркестратор.
Рантайм-объекты конвейера — dataclass (быстро, допускают numpy),
конфигурация и API — pydantic (валидация и сериализация).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

BBox = tuple[float, float, float, float]  # x1, y1, x2, y2 в пикселях исходного кадра

# Ключи замеров по этапам обработки кадра, миллисекунды
TIMING_KEYS = (
    "decode",
    "preprocess",
    "transfer",
    "queue",
    "infer",
    "postprocess",
    "tracking",
    "rules",
    "alert",
    "total",
)


class Severity(str, Enum):
    critical = "critical"
    high = "high"
    medium = "medium"
    low = "low"


class EventStatus(str, Enum):
    new = "new"
    confirmed = "confirmed"
    false = "false"
    forwarded = "forwarded"


# ---------- Конфигурация (pydantic) ----------


class ModelSpec(BaseModel):
    """Описание модели; реестр — configs/models.yaml."""

    model_id: str  # det | ppe | pose | fire | ...
    task: Literal["detect", "pose", "classify"] = "detect"
    onnx_path: str  # путь относительно GUARD_MODELS_DIR или абсолютный
    tpu_program: str | None = None  # uri/путь программы TPU, если скомпилирована
    input_size: tuple[int, int] = (640, 640)  # (h, w)
    input_dtype: Literal["float32", "uint8"] = "float32"
    classes: list[str] = Field(default_factory=list)
    output_format: Literal["yolov8", "yolov8-pose", "raw"] = "yolov8"
    conf_threshold: float = 0.35
    iou_threshold: float = 0.5


class Zone(BaseModel):
    zone_id: str
    name: str
    kind: Literal[
        "danger", "ppe_required", "no_smoking", "smoking_area", "vehicle_path", "height", "fence"
    ] = "danger"
    polygon: list[tuple[float, float]]  # нормированные координаты 0..1 относительно кадра
    active_hours: list[str] | None = None  # ["07:55-12:00"] по местному времени; None = всегда


class RuleConfig(BaseModel):
    rule_id: str
    type: str  # имя плагина из реестра guard.rules
    enabled: bool = True
    zones: list[str] | None = None  # None = весь кадр
    params: dict[str, Any] = Field(default_factory=dict)
    min_duration_s: float = 2.0  # нарушение должно длиться не меньше
    min_ratio: float = 0.7  # доля нарушающих кадров в окне min_duration_s
    close_after_s: float = 3.0  # событие закрывается после стольких секунд без нарушения
    cooldown_s: float = 30.0  # окно дедупликации для того же места/правила
    severity: Severity = Severity.high


class SourceConfig(BaseModel):
    type: Literal["file", "rtsp", "images", "webcam"]
    uri: str
    loop: bool = True
    realtime: bool = True  # файлы и папки изображений отдаются в реальном темпе


class CameraConfig(BaseModel):
    camera_id: str
    name: str
    site: str  # участок
    source: SourceConfig
    sample_fps: float = 5.0  # сколько кадров в секунду анализировать
    enabled: bool = True
    zones: list[Zone] = Field(default_factory=list)
    rules: list[RuleConfig] = Field(default_factory=list)


class Event(BaseModel):
    id: str
    camera_id: str
    site: str
    zone_id: str | None = None
    rule_id: str
    type: str
    severity: Severity
    track_id: int | None = None
    started_at: float  # epoch ts первого нарушающего кадра
    confirmed_at: float  # epoch ts подтверждения = создания оповещения
    ended_at: float | None = None
    bbox: BBox
    snapshot_path: str | None = None
    clip_path: str | None = None
    status: EventStatus = EventStatus.new
    operator_note: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    latency_ms: dict[str, float] = Field(default_factory=dict)  # замеры обработки, ключи из TIMING_KEYS


# ---------- Рантайм конвейера (dataclass) ----------


@dataclass(slots=True)
class Detection:
    model: str  # model_id
    cls: str
    cls_id: int
    conf: float
    bbox: BBox
    keypoints: np.ndarray | None = None  # (17, 3): x, y, conf — для pose


@dataclass(slots=True)
class Track:
    track_id: int
    cls: str
    bbox: BBox
    conf: float
    first_seen: float
    last_seen: float
    hits: int = 1
    keypoints: np.ndarray | None = None
    velocity: tuple[float, float] = (0.0, 0.0)  # px/с, точка опоры (низ-центр рамки)
    speed_bh: float = 0.0  # скорость в высотах тела в секунду
    attributes: dict[str, Any] = field(default_factory=dict)  # helmet, vest, phone, pose_state, ...
    history: deque = field(default_factory=lambda: deque(maxlen=64))  # (ts, foot_x, foot_y, h)


@dataclass(slots=True)
class Violation:
    rule_id: str
    type: str
    severity: Severity
    track_id: int | None
    zone_id: str | None
    bbox: BBox
    score: float = 1.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FrameContext:
    camera: CameraConfig
    ts: float  # epoch времени захвата кадра
    frame_idx: int
    frame: np.ndarray  # BGR, исходное разрешение
    detections: dict[str, list[Detection]] = field(default_factory=dict)  # по model_id
    people: list[Track] = field(default_factory=list)  # треки людей
    objects: list[Track] = field(default_factory=list)  # треки техники и прочих объектов
    timings: dict[str, float] = field(default_factory=dict)  # мс, ключи из TIMING_KEYS

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.frame.shape[:2]
        return w, h
