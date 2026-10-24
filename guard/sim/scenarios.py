"""Синтетические детекции для полного конвейера при 5 кадрах в секунду.

Каждая рамка получает воспроизводимое смещение по обеим осям:
случайный знак и величину от 3 до 8 px. Геометрия позы и СИЗ следует
за рамкой человека.

Один из каждых 13 кадров теряет все детекции. Среди кадров, где акторы
присутствуют по сценарию, это составляет 5–10%. Намеренный выход из
кадра и завершающие пустые кадры не считаются потерями детектора.
Отдельно моделируются полностью отсутствующие и частично ненадёжные
keypoints. Расписание пропусков фиксировано, шум зависит от seed.

Последние пять или более секунд каждого потока позволяют проверить
закрытие событий. Время виртуальное: ожиданий, инференса и I/O нет.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

import numpy as np

from guard.contracts import (
    BBox,
    CameraConfig,
    Detection,
    RuleConfig,
    Severity,
    SourceConfig,
    Zone,
)

FPS = 5
FRAME_SIZE = (1920, 720)  # width, height
START_TS = 1_700_000_000.0

FrameSample = tuple[float, dict[str, list[Detection]]]

_DURATIONS = {
    "zone_intrusion": 21.0,
    "ppe_missing": 13.0,
    "fall_immobile": 18.0,
    "vehicle_proximity": 19.0,
    "running": 10.0,
    "noisy_frame": 15.0,
    "track_return": 19.0,
}
SCENARIO_NAMES = tuple(_DURATIONS)

_RULE_TYPES = {
    "zone_intrusion": "zone_intrusion",
    "ppe_missing": "ppe_missing",
    "fall_immobile": "fall_immobile",
    "vehicle_proximity": "vehicle_proximity",
    "running": "running",
    "noisy_frame": "ppe_missing",
    "track_return": "zone_intrusion",
}

_SEVERITIES = {
    "zone_intrusion": Severity.high,
    "ppe_missing": Severity.high,
    "fall_immobile": Severity.critical,
    "vehicle_proximity": Severity.critical,
    "running": Severity.medium,
}


@dataclass(slots=True)
class Scenario:
    name: str
    camera: CameraConfig
    frames: list[FrameSample]
    track_buffer_s: float = 3.0


@dataclass(frozen=True, slots=True)
class _Actor:
    cls: str
    bbox: BBox
    helmet: str | None = None
    fall_progress: float = 0.0


def _person(
    center_x: float,
    *,
    helmet: str | None = None,
) -> _Actor:
    return _Actor(
        cls="person",
        bbox=(center_x - 50.0, 320.0, center_x + 50.0, 500.0),
        helmet=helmet,
    )


def _actors(name: str, t: float, frame_idx: int) -> list[_Actor]:
    if name == "zone_intrusion":
        if t >= 16.0:
            return []
        if t < 3.0:
            center_x = 260.0 + 80.0 * t
        elif t < 13.0:
            center_x = 500.0
        else:
            center_x = 500.0 - 80.0 * (t - 13.0)
        return [_person(center_x)]

    if name == "ppe_missing":
        if t >= 8.0:
            return []
        return [
            _person(560.0, helmet="no_helmet"),
            _person(760.0, helmet="helmet"),
        ]

    if name == "fall_immobile":
        if t >= 12.5:
            return []
        # Стоит до t=2; переход занимает 0.5 с; затем лежит 10 с.
        progress = min(1.0, max(0.0, (t - 2.0) / 0.5))
        width = 90.0 + 150.0 * progress
        height = 180.0 - 100.0 * progress
        return [
            _Actor(
                cls="person",
                bbox=(
                    600.0 - width / 2.0,
                    500.0 - height,
                    600.0 + width / 2.0,
                    500.0,
                ),
                fall_progress=progress,
            )
        ]

    if name == "vehicle_proximity":
        if t >= 14.0:
            return []
        left = 80.0 + 90.0 * t
        # Проезд рядом: между точкой опоры человека и техникой 40 px.
        return [
            _person(850.0),
            _Actor("truck", (left, 540.0, left + 220.0, 680.0)),
        ]

    if name == "running":
        if t >= 5.0:
            return []
        left = 100.0 + 320.0 * t
        # 320 / 140 > 1.8 высот тела/с. Ширина позволяет начальному
        # IoU-сопоставлению пережить движение и максимальный шум.
        return [_Actor("person", (left, 360.0, left + 170.0, 500.0))]

    if name == "noisy_frame":
        if t >= 10.0:
            return []
        # В остальных кадрах PPE неизвестны, положительных голосов нет.
        helmet = "no_helmet" if frame_idx == 4 * FPS else None
        return [_person(560.0, helmet=helmet)]

    if name == "track_return":
        if t < 6.0 or 8.0 <= t < 14.0:
            return [_person(560.0)]
        return []

    raise ValueError(f"Unknown scenario: {name!r}")


def _jitter(box: BBox, rng: np.random.Generator) -> BBox:
    delta = rng.uniform(3.0, 8.0, size=2)
    delta *= rng.choice((-1.0, 1.0), size=2)
    dx, dy = (float(value) for value in delta)
    x1, y1, x2, y2 = box
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def _pose_points(box: BBox, progress: float) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    points = np.empty((17, 3), dtype=float)
    points[:, 0] = x1 + 0.5 * width
    points[:, 1] = y1 + np.linspace(0.08, 0.92, 17) * height
    points[:, 2] = 0.95

    standing = np.array([
        [0.35, 0.25],
        [0.65, 0.25],
        [0.40, 0.60],
        [0.60, 0.60],
    ])
    lying = np.array([
        [0.30, 0.35],
        [0.30, 0.65],
        [0.65, 0.40],
        [0.65, 0.60],
    ])
    torso = (1.0 - progress) * standing + progress * lying
    points[[5, 6, 11, 12], :2] = (
        torso * np.array([width, height]) + np.array([x1, y1])
    )
    return points


def _detections(
    actors: list[_Actor],
    frame_idx: int,
    rng: np.random.Generator,
) -> dict[str, list[Detection]]:
    result: dict[str, list[Detection]] = {
        "det": [],
        "ppe": [],
        "pose": [],
    }
    if frame_idx % 13 == 8:
        return result

    for actor in actors:
        box = _jitter(actor.bbox, rng)
        result["det"].append(Detection(
            model="det",
            cls=actor.cls,
            cls_id=0 if actor.cls == "person" else 7,
            conf=0.96,
            bbox=box,
        ))
        if actor.cls != "person":
            continue

        points = _pose_points(box, actor.fall_progress)
        if frame_idx % 9 == 4:
            points = None
        elif frame_idx % 11 == 5:
            points[[5, 11], 2] = 0.0

        result["pose"].append(Detection(
            model="pose",
            cls="person",
            cls_id=0,
            conf=0.97,
            bbox=box,
            keypoints=points,
        ))

        if actor.helmet is not None:
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            # Центр PPE — на 12% высоты человека, в области головы.
            head_box = (
                x1 + 0.30 * width,
                y1 + 0.04 * height,
                x1 + 0.70 * width,
                y1 + 0.20 * height,
            )
            result["ppe"].append(Detection(
                model="ppe",
                cls=actor.helmet,
                cls_id=0 if actor.helmet == "helmet" else 1,
                conf=0.98,
                bbox=head_box,
            ))

    return result


def generate_frames(
    name: str,
    *,
    seed: int = 0,
    start_ts: float = START_TS,
) -> list[FrameSample]:
    """Вернуть независимый список кадров (ts, detections) при 5 к/с."""
    if name not in _DURATIONS:
        raise ValueError(f"Unknown scenario: {name!r}")
    if not isfinite(start_ts):
        raise ValueError("start_ts must be finite")

    rng = np.random.default_rng(seed)
    frames: list[FrameSample] = []
    for frame_idx in range(round(_DURATIONS[name] * FPS)):
        elapsed = frame_idx / FPS
        actors = _actors(name, elapsed, frame_idx)
        frames.append((
            float(start_ts + elapsed),
            _detections(actors, frame_idx, rng),
        ))
    return frames


def _zone(kind: str) -> Zone:
    width, height = FRAME_SIZE
    return Zone(
        zone_id=f"{kind}-1",
        name=f"Synthetic {kind}",
        kind=kind,
        polygon=[
            (400.0 / width, 350.0 / height),
            (1000.0 / width, 350.0 / height),
            (1000.0 / width, 650.0 / height),
            (400.0 / width, 650.0 / height),
        ],
    )


def make_scenario(
    name: str,
    *,
    seed: int = 0,
    start_ts: float = START_TS,
) -> Scenario:
    """Создать поток и конфигурацию камеры с соответствующим правилом."""
    frames = generate_frames(name, seed=seed, start_ts=start_ts)
    rule_type = _RULE_TYPES[name]
    zone_kind = {
        "zone_intrusion": "danger",
        "ppe_missing": "ppe_required",
        "vehicle_proximity": "vehicle_path",
    }.get(rule_type)
    zones = [_zone(zone_kind)] if zone_kind is not None else []

    rule = RuleConfig(
        rule_id=f"{name}-rule",
        type=rule_type,
        zones=[zone.zone_id for zone in zones] if zones else None,
        params={"required": ["helmet"]} if rule_type == "ppe_missing" else {},
        min_duration_s=2.0,
        min_ratio=0.7,
        close_after_s=3.0,
        cooldown_s=30.0,
        severity=_SEVERITIES[rule_type],
    )
    camera = CameraConfig(
        camera_id=f"sim-{name}",
        name=f"Synthetic {name}",
        site="synthetic-workshop",
        source=SourceConfig(
            type="images",
            uri=f"synthetic://{name}",
            loop=False,
            realtime=False,
        ),
        sample_fps=FPS,
        zones=zones,
        rules=[rule],
    )
    return Scenario(
        name=name,
        camera=camera,
        frames=frames,
        # Стандартный буфер 3 с сохранил бы ID после отсутствия 2 с.
        track_buffer_s=1.0 if name == "track_return" else 3.0,
    )
