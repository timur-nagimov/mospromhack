"""Отбор кадров в выборку (docs/FLYWHEEL.md, «Когда сохраняется сэмпл»). Чистые функции и состояние камеры.

- uncertain: у целевого класса правила камеры уверенность в полосе [порог − 0.10, порог + 0.05];
- flicker: трек человека пропал, и через ≤ 1 с рядом родился новый ID;
- disagreement: детекции TPU сверены с CPU fp32 той же модели, есть рамка без пары с уверенностью выше порога.

Порог «решения» для полосы. Декодер отбрасывает детекции ниже conf_threshold модели, поэтому полоса
считается от порога, который реально решает судьбу рамки:
- min_conf правила, если он есть (zone_intrusion 0.5, smoking_fire 0.5);
- для классов, которые ведёт трекер (person и техника из OBJECT_CLASSES), — порог трекера tracker_high_thresh
  (0.4): ниже него трек не продолжается первым проходом;
- иначе conf_threshold модели (ppe 0.4, fire 0.4, cell phone 0.25).
Нижняя часть полосы ниже conf_threshold модели не видна — её детекций нет на выходе декодера.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from guard.contracts import Detection
from guard.pipeline.worker import OBJECT_CLASSES, PERSON_CLASSES

TRACKED_CLASSES = PERSON_CLASSES | OBJECT_CLASSES

BAND_BELOW = 0.10
BAND_ABOVE = 0.05
FLICKER_GAP_S = 1.0
FLICKER_CENTER_K = 0.75
SHADOW_IOU = 0.5
SHADOW_MARGIN = 0.10  # рамка «уверенная», если conf ≥ conf_threshold модели + 0.10

# Целевые классы правил по умолчанию. Плагин может задать свои атрибутом класса flywheel_classes.
RULE_TARGETS: dict[str, frozenset[str]] = {
    "zone_intrusion": frozenset({"person"}),
    "fall_immobile": frozenset({"person"}),
    "running": frozenset({"person"}),
    "fence_climb": frozenset({"person"}),
    "height_no_harness": frozenset({"person"}),
    "phone_usage": frozenset({"cell phone"}),
    "smoking_fire": frozenset({"fire"}),
    "vehicle_proximity": frozenset(
        {"car", "truck", "bus", "forklift", "motorcycle", "excavator", "wheel_loader", "dump_truck"}
    ),
    "unsafe_carry": frozenset({"suitcase", "backpack", "handbag", "box", "load"}),
    "ppe_missing": frozenset({"helmet", "no_helmet", "vest", "no_vest"}),
}
HEAVY_VEHICLES = frozenset({"excavator", "wheel_loader", "dump_truck"})


def rule_type_of(rule: Any) -> str:
    return getattr(getattr(rule, "cfg", None), "type", None) or getattr(rule, "type", "") or ""


def rule_id_of(rule: Any) -> str:
    return getattr(getattr(rule, "cfg", None), "rule_id", None) or getattr(rule, "rule_id", "") or ""


def rule_targets(rule: Any) -> frozenset[str]:
    """Целевые классы правила-плагина (экземпляр RulePlugin или объект с cfg.type)."""
    custom = getattr(rule, "flywheel_classes", None)
    if custom:
        return frozenset(custom)
    kind = rule_type_of(rule)
    if kind == "ppe_missing":
        required = getattr(rule, "required", None) or ("helmet", "vest")
        return frozenset({c for r in required for c in (r, f"no_{r}")})
    if kind == "vehicle_proximity" and getattr(rule, "vehicle_classes", None):
        return frozenset(rule.vehicle_classes) | HEAVY_VEHICLES
    if kind == "unsafe_carry" and getattr(rule, "load_classes", None):
        return frozenset(rule.load_classes)
    if kind == "smoking_fire" and getattr(rule, "include_smoke", False):
        return frozenset({"fire", "smoke"})
    return RULE_TARGETS.get(kind, frozenset({"person"}))


def camera_targets(rules: Iterable[Any]) -> frozenset[str]:
    out: set[str] = set()
    for rule in rules:
        out |= rule_targets(rule)
    return frozenset(out)


def decision_threshold(rule: Any, cls: str, model_threshold: float, tracker_high: float) -> float:
    """Порог, который решает судьбу рамки класса cls для этого правила (см. docstring модуля)."""
    min_conf = getattr(rule, "min_conf", None)
    if isinstance(min_conf, (int, float)) and not isinstance(min_conf, bool):
        return max(float(model_threshold), float(min_conf))
    if cls in TRACKED_CLASSES:
        return max(float(model_threshold), float(tracker_high))
    return float(model_threshold)


@dataclass(frozen=True, slots=True)
class UncertainHit:
    model_id: str
    cls: str
    conf: float
    lo: float
    hi: float
    bbox: tuple[float, float, float, float]
    rule_id: str

    @property
    def reason(self) -> str:
        return f"{self.cls} conf {self.conf:.2f} ∈ [{self.lo:.2f}, {self.hi:.2f}], правило {self.rule_id}"


def find_uncertain(
    detections: Mapping[str, Iterable[Detection]],
    rules: Iterable[Any],
    model_thresholds: Mapping[str, float],
    tracker_high: float,
) -> UncertainHit | None:
    """Самая неуверенная рамка целевого класса: ближе всех к порогу решения внутри полосы."""
    best: UncertainHit | None = None
    best_gap = 1e9
    for rule in rules:
        targets = rule_targets(rule)
        rule_id = rule_id_of(rule)
        for model_id, dets in detections.items():
            model_thr = model_thresholds.get(model_id)
            if model_thr is None:
                continue
            for det in dets:
                if det.cls not in targets:
                    continue
                thr = decision_threshold(rule, det.cls, model_thr, tracker_high)
                lo, hi = round(thr - BAND_BELOW, 4), round(thr + BAND_ABOVE, 4)
                if not lo <= det.conf <= hi:
                    continue
                gap = abs(det.conf - thr)
                if gap < best_gap:
                    best_gap = gap
                    best = UncertainHit(model_id, det.cls, float(det.conf), lo, hi, tuple(det.bbox), rule_id)
    return best


# ---------------------------------------------------------------- мигание трека


def box_iou(a: Iterable[float], b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def boxes_near(old: Iterable[float], new: Iterable[float], k: float = FLICKER_CENTER_K) -> bool:
    """Рамки рядом: пересекаются или центры не дальше k · max(h, w) старой рамки."""
    a, b = tuple(old), tuple(new)
    if box_iou(a, b) > 0:
        return True
    acx, acy = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bcx, bcy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    size = max(a[2] - a[0], a[3] - a[1])
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5 <= k * size


@dataclass(frozen=True, slots=True)
class FlickerHit:
    old_id: int
    new_id: int
    gap_s: float
    old_bbox: tuple[float, float, float, float]
    new_bbox: tuple[float, float, float, float]

    @property
    def reason(self) -> str:
        return f"трек #{self.old_id} пропал, через {self.gap_s:.1f} с рядом новый #{self.new_id}"


@dataclass
class FlickerState:
    """Жизненный цикл треков людей одной камеры: кто пропал, кто родился."""

    gap_s: float = FLICKER_GAP_S
    present: dict[int, tuple[tuple[float, ...], float]] = field(default_factory=dict)
    lost: dict[int, tuple[tuple[float, ...], float]] = field(default_factory=dict)
    seen: dict[int, float] = field(default_factory=dict)

    def update(self, tracks: Iterable[tuple[int, Iterable[float]]], ts: float) -> FlickerHit | None:
        """tracks — (track_id, bbox) подтверждённых треков кадра. Возвращает первое мигание кадра."""
        current = {int(tid): (tuple(float(v) for v in bbox), ts) for tid, bbox in tracks}
        for tid, (bbox, last_ts) in self.present.items():
            if tid not in current:
                self.lost[tid] = (bbox, last_ts)  # пропал: последний кадр, где его видели
        hit: FlickerHit | None = None
        for tid, (bbox, _) in current.items():
            self.lost.pop(tid, None)  # вернулся тот же ID — не мигание
            if tid in self.seen:
                continue
            for old_id, (old_bbox, lost_ts) in sorted(self.lost.items(), key=lambda kv: -kv[1][1]):
                gap = ts - lost_ts
                if 0 <= gap <= self.gap_s and boxes_near(old_bbox, bbox):
                    if hit is None:
                        hit = FlickerHit(old_id, tid, gap, old_bbox, bbox)  # type: ignore[arg-type]
                    self.lost.pop(old_id, None)
                    break
        for tid in current:
            self.seen[tid] = ts
        self.present = current
        # старое не копим: пропавшие дольше 2·gap и давно не виденные ID забываем
        self.lost = {k: v for k, v in self.lost.items() if ts - v[1] <= 2 * self.gap_s}
        if len(self.seen) > 512:
            self.seen = {k: v for k, v in self.seen.items() if ts - v <= 120.0}
        return hit


# ---------------------------------------------------------------- расхождение TPU и CPU


@dataclass(frozen=True, slots=True)
class Mismatch:
    model_id: str
    cls: str
    conf: float
    bbox: tuple[float, float, float, float]
    only_on: str  # tpu | cpu

    @property
    def reason(self) -> str:
        other = "CPU" if self.only_on == "tpu" else "TPU"
        return f"{self.model_id}: {self.cls} {self.conf:.2f} только на {self.only_on.upper()}, на {other} пары нет"


def compare_detections(
    model_id: str,
    tpu: Iterable[Detection],
    cpu: Iterable[Detection],
    threshold: float,
    classes: frozenset[str] | None = None,
    iou: float = SHADOW_IOU,
) -> list[Mismatch]:
    """Уверенные рамки (conf ≥ threshold) без пары того же класса с IoU ≥ iou на другой стороне.

    Пара ищется среди всех рамок другой стороны: TPU в int8 занижает conf на ~0.05, и рамка CPU 0.45
    с парой TPU 0.38 расхождением не считается."""
    tpu_list = [d for d in tpu if classes is None or d.cls in classes]
    cpu_list = [d for d in cpu if classes is None or d.cls in classes]
    out: list[Mismatch] = []
    for side, mine, other in (("tpu", tpu_list, cpu_list), ("cpu", cpu_list, tpu_list)):
        for det in mine:
            if det.conf < threshold:
                continue
            paired = any(o.cls == det.cls and box_iou(det.bbox, o.bbox) >= iou for o in other)
            if not paired:
                out.append(Mismatch(model_id, det.cls, float(det.conf), tuple(det.bbox), side))  # type: ignore[arg-type]
    out.sort(key=lambda m: -m.conf)
    return out
