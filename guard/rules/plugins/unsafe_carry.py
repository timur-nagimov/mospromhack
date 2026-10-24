"""S10: осторожные двумерные эвристики небезопасной переноски.

Груз должен пересекать рамку человека и находиться рядом хотя бы с одним
надёжным запястьем COCO-17 (9, 10). Это не доказывает захват: рюкзак на спине,
перспектива и перекрытия людей остаются источниками ошибок. Неоднозначное
соответствие груза нескольким людям отбрасывается.

Закрытый обзор оценивается только когда рамка груза перекрывает нос по X/Y,
а её верх строго выше носа. Направление взгляда и реальная видимость неизвестны.
Наклон — угол линии между серединами плеч и бёдер, а не изгиб позвоночника.

Track.history содержит только (ts, foot_x, foot_y, h), без прошлых поз и грузов.
Поэтому для наклона сохраняются до 64 временных меток реально наблюдавшегося
сочетания «груз у руки + наклон». Длительность считается по непрерывному суффиксу
Track.history, для каждой записи которого есть такое наблюдение. История,
предшествующая первому evaluate(), не доказывает переноску. Пропуск наблюдения,
потеря груза/позы, большой интервал или смена трека прерывают длительность.
Это история признака, а не подтверждение события; EventManager подтверждает
и дедуплицирует события. Треки и их history не изменяются.

Без cfg.zones проверки обзора/наклона действуют во всём кадре. Явный список
ограничивает все проверки активными выбранными зонами. Зональное нарушение
возможно только в danger/vehicle_path и проверяется по точке опоры.
"""

from __future__ import annotations

from collections import deque
from math import atan2, degrees, hypot, isfinite
from typing import Any, ClassVar

import numpy as np

from guard.contracts import (
    BBox,
    CameraConfig,
    FrameContext,
    RuleConfig,
    Severity,
    Track,
    Violation,
)
from guard.rules.base import RulePlugin, register_rule
from guard.rules.geometry import (
    bbox_iou,
    foot_point,
    is_zone_active,
    point_in_polygon,
    zone_polygon_px,
)

_DEFAULT_LOAD_CLASSES = ["suitcase", "backpack", "handbag", "box", "load"]
_EvidenceKey = tuple[str, int, float]
_Load = tuple[str, BBox, float]


def _valid_box(box: BBox) -> bool:
    return (
        all(isfinite(value) for value in box)
        and box[2] > box[0]
        and box[3] > box[1]
    )


@register_rule
class UnsafeCarryRule(RulePlugin):
    type: ClassVar[str] = "unsafe_carry"
    title: ClassVar[str] = "Переноска груза с нарушением"
    required_models: ClassVar[frozenset[str]] = frozenset({"det", "pose"})
    default_severity: ClassVar[Severity] = Severity.medium
    params_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "title": "Переноска груза с нарушением",
        "description": (
            "Эвристики по двумерным рамкам и позе; не определяют вес, захват "
            "или направление взгляда. Наклон требует наблюдений груза и позы "
            "на всём проверяемом суффиксе history; возраст трека недостаточен."
        ),
        "properties": {
            "load_classes": {
                "type": "array",
                "title": "Классы грузов",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
                "default": _DEFAULT_LOAD_CLASSES,
                "description": "Классы box и load учитываются всегда.",
            },
            "check_view": {
                "type": "boolean",
                "title": "Проверять перекрытие обзора",
                "default": True,
                "description": (
                    "Верх груза выше носа, рамка груза перекрывает нос по X/Y."
                ),
            },
            "check_bend": {
                "type": "boolean",
                "title": "Проверять длительный наклон торса",
                "default": True,
            },
            "check_zone": {
                "type": "boolean",
                "title": "Проверять переноску в danger/vehicle_path",
                "default": True,
            },
            "kp_conf": {
                "type": "number",
                "title": "Мин. уверенность ключевой точки",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
            },
            "min_load_conf": {
                "type": "number",
                "title": "Мин. уверенность обнаружения груза",
                "minimum": 0.0,
                "maximum": 1.0,
                "default": 0.5,
            },
            "hand_distance_bh": {
                "type": "number",
                "title": "Макс. расстояние запястья до груза, высот тела",
                "minimum": 0.0,
                "default": 0.05,
            },
            "torso_angle_deg": {
                "type": "number",
                "title": "Порог наклона торса от вертикали, градусы",
                "minimum": 0.0,
                "maximum": 90.0,
                "default": 45.0,
            },
            "bend_duration_s": {
                "type": "number",
                "title": "Порог длительности переноски с наклоном, с",
                "minimum": 0.0,
                "default": 1.0,
                "description": "Длительность должна быть строго больше порога.",
            },
            "max_history_gap_s": {
                "type": "number",
                "title": "Макс. интервал между наблюдениями history, с",
                "exclusiveMinimum": 0.0,
                "default": 1.0,
            },
        },
        "additionalProperties": False,
    }

    def __init__(self, cfg: RuleConfig, camera: CameraConfig) -> None:
        super().__init__(cfg, camera)
        params = cfg.params
        classes = params.get("load_classes", _DEFAULT_LOAD_CLASSES)
        if (
            not isinstance(classes, (list, tuple))
            or any(not isinstance(name, str) or not name for name in classes)
        ):
            raise ValueError("load_classes must be a list of nonempty strings")
        self.load_classes = frozenset(classes) | {"box", "load"}

        for name in ("check_view", "check_bend", "check_zone"):
            value = params.get(name, True)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be boolean")
            setattr(self, name, value)

        for name, default, maximum in (
            ("kp_conf", 0.5, 1.0),
            ("min_load_conf", 0.5, 1.0),
            ("hand_distance_bh", 0.05, None),
            ("torso_angle_deg", 45.0, 90.0),
            ("bend_duration_s", 1.0, None),
            ("max_history_gap_s", 1.0, None),
        ):
            value = float(params.get(name, default))
            if (
                not isfinite(value)
                or value < 0
                or (maximum is not None and value > maximum)
                or (name == "max_history_gap_s" and value == 0)
            ):
                raise ValueError(f"invalid {name}")
            setattr(self, name, value)

        wanted = None if cfg.zones is None else set(cfg.zones)
        self.zones = [
            zone for zone in camera.zones
            if wanted is None or zone.zone_id in wanted
        ]
        self._zone_limited = wanted is not None
        self._bend_observations: dict[_EvidenceKey, deque[float]] = {}

    def _points(self, track: Track) -> dict[int, tuple[float, float]]:
        if track.keypoints is None:
            return {}
        try:
            points = np.asarray(track.keypoints, dtype=float)
        except (TypeError, ValueError):
            return {}
        if points.shape != (17, 3):
            return {}
        x1, y1, x2, y2 = track.bbox
        return {
            index: (float(row[0]), float(row[1]))
            for index, row in enumerate(points)
            if np.all(np.isfinite(row))
            and self.kp_conf <= row[2] <= 1.0
            and row[2] > 0
            and x1 <= row[0] <= x2
            and y1 <= row[1] <= y2
        }

    def _loads(self, ctx: FrameContext) -> list[_Load]:
        # Детекции нужны в том числе для классов, которые трекер не отслеживает.
        candidates = list(ctx.detections.get("det", []))
        candidates.extend(
            obj for obj in ctx.objects if obj.last_seen == ctx.ts
        )
        unique: dict[tuple[str, BBox], float] = {}
        for obj in candidates:
            if (
                obj.cls not in self.load_classes
                or not _valid_box(obj.bbox)
                or not isfinite(obj.conf)
                or not self.min_load_conf <= obj.conf <= 1.0
                or obj.conf <= 0
            ):
                continue
            box = tuple(float(value) for value in obj.bbox)
            key = (obj.cls, box)
            unique[key] = max(unique.get(key, 0.0), float(obj.conf))
        return [(name, box, conf) for (name, box), conf in unique.items()]

    def _in_hand(
        self,
        track: Track,
        points: dict[int, tuple[float, float]],
        box: BBox,
    ) -> bool:
        if bbox_iou(track.bbox, box) <= 0:
            return False
        limit = self.hand_distance_bh * (track.bbox[3] - track.bbox[1])
        for index in (9, 10):
            if index not in points:
                continue
            x, y = points[index]
            dx = max(box[0] - x, 0.0, x - box[2])
            dy = max(box[1] - y, 0.0, y - box[3])
            if hypot(dx, dy) <= limit:
                return True
        return False

    @staticmethod
    def _torso_angle(points: dict[int, tuple[float, float]]) -> float | None:
        if not all(index in points for index in (5, 6, 11, 12)):
            return None
        sx = (points[5][0] + points[6][0]) / 2.0
        sy = (points[5][1] + points[6][1]) / 2.0
        hx = (points[11][0] + points[12][0]) / 2.0
        hy = (points[11][1] + points[12][1]) / 2.0
        if hypot(hx - sx, hy - sy) < 1e-6:
            return None
        return degrees(atan2(abs(hx - sx), abs(hy - sy)))

    def _bent_for_s(
        self,
        key: _EvidenceKey,
        track: Track,
        ts: float,
        qualifies: bool,
    ) -> float:
        if not qualifies:
            self._bend_observations.pop(key, None)
            return 0.0

        observed = self._bend_observations.setdefault(key, deque(maxlen=64))
        if observed and ts < observed[-1]:
            observed.clear()
        if not observed or observed[-1] != ts:
            observed.append(ts)
        timestamps = set(observed)

        since = ts
        previous = ts
        found_current = False
        for row in reversed(track.history):
            try:
                sample_ts, x, y, body_height = (float(value) for value in row)
            except (TypeError, ValueError, OverflowError):
                break
            if not isfinite(sample_ts):
                break
            if sample_ts > ts:
                continue
            if (
                not all(isfinite(value) for value in (x, y, body_height))
                or body_height <= 0
            ):
                break
            if not found_current:
                if sample_ts != ts:
                    return 0.0
                found_current = True
            if (
                sample_ts > previous
                or previous - sample_ts > self.max_history_gap_s
                or sample_ts not in timestamps
            ):
                break
            since = sample_ts
            previous = sample_ts
        return max(0.0, ts - since) if found_current else 0.0

    def evaluate(self, ctx: FrameContext) -> list[Violation]:
        if not isfinite(ctx.ts):
            self._bend_observations.clear()
            return []

        people = [
            track for track in ctx.people
            if track.cls == "person"
            and _valid_box(track.bbox)
            and isfinite(track.conf)
            and isfinite(track.first_seen)
            and track.first_seen <= ctx.ts
            and track.last_seen == ctx.ts
        ]
        keys = {
            (ctx.camera.camera_id, track.track_id, track.first_seen)
            for track in people
        }
        for key in list(self._bend_observations):
            if key not in keys:
                del self._bend_observations[key]

        width, height = ctx.size
        active = []
        for zone in self.zones:
            if not is_zone_active(zone, ctx.ts):
                continue
            polygon = zone_polygon_px(zone, width, height)
            if len(polygon) >= 3 and all(
                isfinite(x) and isfinite(y) for x, y in polygon
            ):
                active.append((zone, polygon))

        points = [self._points(track) for track in people]
        associated: list[list[_Load]] = [[] for _ in people]
        for load in self._loads(ctx):
            owners = [
                index for index, track in enumerate(people)
                if self._in_hand(track, points[index], load[1])
            ]
            # В толпе пересечение не позволяет надёжно выбрать владельца.
            if len(owners) == 1:
                associated[owners[0]].append(load)

        out: list[Violation] = []
        for track, joints, loads in zip(people, points, associated):
            key = (ctx.camera.camera_id, track.track_id, track.first_seen)
            inside = [
                zone for zone, polygon in active
                if point_in_polygon(foot_point(track.bbox), polygon)
            ]
            in_scope = not self._zone_limited or bool(inside)
            angle = self._torso_angle(joints)
            bent_s = self._bent_for_s(
                key,
                track,
                ctx.ts,
                qualifies=(
                    self.check_bend
                    and in_scope
                    and bool(loads)
                    and angle is not None
                    and angle > self.torso_angle_deg
                ),
            )
            if not in_scope or not loads:
                continue

            conditions: list[str] = []
            nose = joints.get(0)
            if self.check_view and nose is not None and any(
                box[1] < nose[1] <= box[3]
                and box[0] <= nose[0] <= box[2]
                for _, box, _ in loads
            ):
                conditions.append("view_obstructed")
            if self.check_bend and bent_s > self.bend_duration_s:
                conditions.append("bent_back")

            danger = next(
                (zone for zone in inside if zone.kind in {"danger", "vehicle_path"}),
                None,
            )
            if self.check_zone and danger is not None:
                conditions.append("danger_zone")
            if not conditions:
                continue

            zone = (
                danger if "danger_zone" in conditions
                else inside[0] if inside else None
            )
            details: dict[str, Any] = {
                "conditions": conditions,
                "loads": [
                    {"cls": name, "bbox": list(box), "conf": conf}
                    for name, box, conf in loads
                ],
                "evidence": "geometry_heuristic",
            }
            if "bent_back" in conditions:
                details["torso_angle_deg"] = angle
                details["bent_duration_s"] = bent_s
            if zone is not None:
                details["zone_name"] = zone.name
                details["zone_kind"] = zone.kind

            out.append(self.violation(
                ctx,
                severity=Severity.medium,
                track_id=track.track_id,
                zone_id=zone.zone_id if zone is not None else None,
                bbox=tuple(float(value) for value in track.bbox),
                score=min(float(track.conf), max(load[2] for load in loads)),
                details=details,
            ))
        return out
