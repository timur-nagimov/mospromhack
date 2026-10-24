"""S7 «Курение и открытый огонь».

Огонь (и опционально дым) — по центру рамки модели fire, как раньше.

Курение включается параметром cigarette_model — model_id детектора сигареты из реестра.
Тогда правило требует ещё det и pose. Сигарета засчитывается человеку, если её рамка рядом
с уверенной ключевой точкой головы (нос, глаза, уши) или кисти: ближе near_factor × высота
рамки человека. Нарушение — пока у трека за последние window_s секунд набралось min_hits
кадров с сигаретой. Сигарета мелкая и часто закрыта рукой: окно сглаживает пропуски детектора,
а подтверждение и дубли по-прежнему ведёт EventManager. Без cigarette_model правило не меняется.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Any, ClassVar

import numpy as np

from guard.contracts import BBox, CameraConfig, FrameContext, RuleConfig, Severity, Track, Violation
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_center,
    bbox_height,
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)

# COCO-17: 0 нос, 1–2 глаза, 3–4 уши; 9–10 кисти.
_HEAD_POINTS = (0, 1, 2, 3, 4)
_HAND_POINTS = (9, 10)
# Мягче, чем 0.5 у атрибутов: курящего часто видно вполоборота, а точность держат
# детектор сигареты и окно из нескольких кадров.
_POINT_CONFIDENCE = 0.3


def _valid_box(box: BBox) -> bool:
    return all(math.isfinite(value) for value in box) and box[2] > box[0] and box[3] > box[1]


def _distance_to_box(point: tuple[float, float], box: BBox) -> float:
    """Расстояние от точки до прямоугольника; внутри — ноль."""
    x, y = point
    dx = max(box[0] - x, 0.0, x - box[2])
    dy = max(box[1] - y, 0.0, y - box[3])
    return math.hypot(dx, dy)


def _targets(person: Track) -> list[tuple[str, tuple[float, float]]]:
    points = person.keypoints
    if points is None:
        return []
    points = np.asarray(points, dtype=float)
    if points.shape != (17, 3):
        return []
    out = []
    for where, indices in (("head", _HEAD_POINTS), ("hand", _HAND_POINTS)):
        for index in indices:
            x, y, conf = points[index]
            if math.isfinite(x) and math.isfinite(y) and math.isfinite(conf) and conf >= _POINT_CONFIDENCE:
                out.append((where, (float(x), float(y))))
    return out


@register_rule
class SmokingFireRule(RulePlugin):
    type: ClassVar[str] = "smoking_fire"
    title: ClassVar[str] = "Курение и открытый огонь"
    required_models: ClassVar[frozenset[str]] = frozenset({"fire"})
    default_severity: ClassVar[Severity] = Severity.critical
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Курение и открытый огонь",
        "properties": {
            "include_smoke": {
                "type": "boolean",
                "title": "Учитывать детекции дыма",
                "default": False,
            },
            "min_conf": {
                "type": "number",
                "title": "Мин. уверенность детекции",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
            },
            "min_area_px": {
                "type": "number",
                "title": "Мин. площадь рамки, квадратных пикселей",
                "minimum": 0.0,
                "default": 400,
            },
            "detect_fire": {
                "type": "boolean",
                "title": "Искать открытый огонь (модель fire)",
                "default": True,
            },
            "cigarette_model": {
                "type": "string",
                "title": "Модель сигареты (model_id из реестра); пусто — курение не ищем",
                "description": "Например, cigarette. Нужны ещё det и pose: сигарета засчитывается у головы или кисти",
                "default": "",
            },
            "cigarette_min_conf": {
                "type": "number",
                "title": "Мин. уверенность сигареты",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.4,
            },
            "near_factor": {
                "type": "number",
                "title": "Радиус «у лица или в руке», долей высоты человека",
                "minimum": 0.0,
                "default": 0.15,
            },
            "window_s": {
                "type": "number",
                "title": "Окно накопления сигареты у человека, с",
                "minimum": 0.0,
                "default": 3.0,
            },
            "min_hits": {
                "type": "integer",
                "title": "Мин. кадров с сигаретой в окне",
                "minimum": 1,
                "default": 2,
            },
            "min_box_h_px": {
                "type": "integer",
                "title": "Мин. высота рамки человека для курения, px",
                "minimum": 0,
                "default": 60,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        self.include_smoke = bool(params.get("include_smoke", False))
        self.min_conf = float(params.get("min_conf", 0.5))
        self.min_area_px = float(params.get("min_area_px", 400))
        self.detect_fire = bool(params.get("detect_fire", True))
        self.cigarette_model = str(params.get("cigarette_model") or "").strip()
        self.cigarette_min_conf = float(params.get("cigarette_min_conf", 0.4))
        self.near_factor = float(params.get("near_factor", 0.15))
        self.window_s = float(params.get("window_s", 3.0))
        self.min_hits = int(params.get("min_hits", 2))
        self.min_box_h_px = float(params.get("min_box_h_px", 60))
        for name in ("cigarette_min_conf", "near_factor", "window_s", "min_box_h_px"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.min_hits < 1:
            raise ValueError("min_hits must be >= 1")
        # Модели — на экземпляре: CameraWorker, run_offline и проверка камеры берут rule.required_models.
        # Без cigarette_model набор прежний — только fire.
        models: set[str] = {"fire"} if self.detect_fire else set()
        if self.cigarette_model:
            models |= {"det", "pose", self.cigarette_model}
        if not models:
            raise ValueError("nothing to detect: detect_fire is false and cigarette_model is empty")
        self.required_models = frozenset(models)
        self.severity = (
            cfg.severity
            if "severity" in cfg.model_fields_set
            else self.default_severity
        )
        if cfg.zones:
            wanted = set(cfg.zones)
            self.zones = [
                zone for zone in camera.zones if zone.zone_id in wanted
            ]
        else:
            self.zones = [
                zone for zone in camera.zones if zone.kind == "no_smoking"
            ]
        # Не расширяем область при неизвестных ID или неактивных зонах.
        self._zone_limited = bool(cfg.zones) or bool(self.zones)
        self.smoking_areas = [
            zone for zone in camera.zones if zone.kind == "smoking_area"
        ]
        # track_id → (ts, conf, где) кадров, на которых у человека была сигарета.
        self._hits: dict[int, deque[tuple[float, float, str]]] = {}

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        w, h = ctx.size
        active = [
            (zone, zone_polygon_px(zone, w, h))
            for zone in self.zones
            if is_zone_active(zone, ctx.ts)
        ]
        if self._zone_limited and not active:
            return []

        # Исключения smoking_area относятся к режиму проверки всего кадра.
        excluded = [
            zone_polygon_px(zone, w, h)
            for zone in self.smoking_areas
            if not self._zone_limited and is_zone_active(zone, ctx.ts)
        ]
        out: list[Violation] = []
        if self.detect_fire:
            out += self._fire(ctx, active, excluded)
        if self.cigarette_model:
            out += self._smoking(ctx, active, excluded)
        return out

    def _zone_of(self, point: tuple[float, float], active, excluded) -> tuple[bool, str | None]:
        zone_id = next(
            (
                zone.zone_id
                for zone, polygon in active
                if point_in_polygon(point, polygon)
            ),
            None,
        )
        if self._zone_limited and zone_id is None:
            return False, None
        if any(point_in_polygon(point, polygon) for polygon in excluded):
            return False, None
        return True, zone_id

    def _fire(self, ctx: FrameContext, active, excluded) -> list[Violation]:
        classes = {"fire", "smoke"} if self.include_smoke else {"fire"}
        out: list[Violation] = []
        for detection in ctx.detections.get("fire", []):
            if (
                detection.cls not in classes
                or not math.isfinite(detection.conf)
                or detection.conf < self.min_conf
            ):
                continue

            box = detection.bbox
            if not _valid_box(box):
                continue
            area = (box[2] - box[0]) * (box[3] - box[1])
            if not math.isfinite(area) or area < self.min_area_px:
                continue

            inside, zone_id = self._zone_of(bbox_center(box), active, excluded)
            if not inside:
                continue

            out.append(self.violation(
                ctx,
                severity=self.severity,
                track_id=None,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in box),
                score=float(detection.conf),
                details={
                    "cls": detection.cls,
                    "conf": float(detection.conf),
                },
            ))
        return out

    def _smoking(self, ctx: FrameContext, active, excluded) -> list[Violation]:
        """Сигарета у головы или кисти: копим кадры по треку, нарушение — при min_hits в окне."""
        people = [
            person for person in ctx.people
            if _valid_box(person.bbox) and bbox_height(person.bbox) >= self.min_box_h_px
        ]
        cigarettes = [
            detection for detection in ctx.detections.get(self.cigarette_model, [])
            if math.isfinite(detection.conf)
            and detection.conf >= self.cigarette_min_conf
            and _valid_box(detection.bbox)
        ]
        # Каждую сигарету отдаём ближайшему человеку, у которого она в радиусе.
        # Кадр — одно попадание на трек, даже если сигарет у человека несколько.
        frame_hits: dict[int, tuple[float, str]] = {}
        targets = {person.track_id: _targets(person) for person in people}
        for cigarette in cigarettes:
            best: tuple[float, Track, str] | None = None
            for person in people:
                radius = self.near_factor * bbox_height(person.bbox)
                for where, point in targets[person.track_id]:
                    distance = _distance_to_box(point, cigarette.bbox)
                    if distance <= radius and (best is None or distance < best[0]):
                        best = (distance, person, where)
            if best is not None:
                _, person, where = best
                conf = float(cigarette.conf)
                if conf > frame_hits.get(person.track_id, (-1.0, ""))[0]:
                    frame_hits[person.track_id] = (conf, where)
        for track_id, (conf, where) in frame_hits.items():
            hits = self._hits.setdefault(track_id, deque())
            if hits and hits[-1][0] == ctx.ts:  # повторный вызов на том же кадре
                hits.pop()
            hits.append((ctx.ts, conf, where))

        for track_id in list(self._hits):
            hits = self._hits[track_id]
            while hits and hits[0][0] < ctx.ts - self.window_s:
                hits.popleft()
            if not hits:
                del self._hits[track_id]

        out: list[Violation] = []
        for person in people:
            hits = self._hits.get(person.track_id)
            if not hits or len(hits) < self.min_hits:
                continue
            inside, zone_id = self._zone_of(foot_point(person.bbox), active, excluded)
            if not inside:
                continue
            _, conf, where = max(hits, key=lambda hit: hit[1])
            out.append(self.violation(
                ctx,
                severity=self.severity,
                track_id=person.track_id,
                zone_id=zone_id,
                bbox=tuple(float(value) for value in person.bbox),
                score=conf,
                details={
                    "cls": "cigarette",
                    "conf": conf,
                    "near": where,
                    "hits": len(hits),
                },
            ))
        return out
