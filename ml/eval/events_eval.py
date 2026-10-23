"""Оценка событий относительно ручной разметки видео.

Порядок жадного сопоставления:
* отдельно для каждой пары (video, type);
* предсказания — по (confirmed, start, id);
* достоверные метки — по (t_start, t_end);
* сначала свободная достоверная метка, затем дубль, затем uncertain.

Uncertain-интервалы исключают совпавшие предсказания из оценки, не дают
TP/FN и могут исключать несколько предсказаний. Достоверные метки имеют
приоритет. Границы интервалов включительны.

Precision/recall/F1 при нулевом знаменателе равны 0. Задержки могут быть
отрицательными из-за допуска; при отсутствии TP они равны null.
P95 вычисляется линейной интерполяцией между соседними значениями.
Общие метрики считаются по суммарным счётчикам и объединённым задержкам.

durations — словарь {"имя видео": длительность_в_секундах}, включая чистые
видео. Каждое видео учитывается один раз, независимо от количества камер.
Без durations частота ложных тревог равна null.

CLI:
    python -m ml.eval.events_eval --labels data/labels --preds preds.json \
        --mapping mapping.json --durations durations.json --out metrics.json

Вместо --preds можно передать --db. SQLite открывается только для чтения.
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field
from numbers import Real
from pathlib import Path
from typing import Any

from guard.contracts import Event


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


@dataclass(frozen=True, slots=True)
class LabelEvent:
    """Ручная метка; время в секундах от начала видео."""

    video: str
    type: str
    t_start: float
    t_end: float
    uncertain: bool = False
    note: str = ""

    def __post_init__(self) -> None:
        _text(self.video, "video")
        _text(self.type, "type")
        start = _number(self.t_start, "t_start")
        end = _number(self.t_end, "t_end")
        if start < 0 or end < start:
            raise ValueError("Labels require 0 <= t_start <= t_end")
        if not isinstance(self.uncertain, bool):
            raise ValueError("uncertain must be a boolean")
        if not isinstance(self.note, str):
            raise ValueError("note must be a string")
        object.__setattr__(self, "t_start", start)
        object.__setattr__(self, "t_end", end)


@dataclass(frozen=True, slots=True)
class PredEvent:
    """Предсказание после перевода epoch-времени во время видео."""

    video: str
    type: str
    start: float
    confirmed: float
    id: str = ""

    def __post_init__(self) -> None:
        _text(self.video, "video")
        _text(self.type, "type")
        start = _number(self.start, "start")
        confirmed = _number(self.confirmed, "confirmed")
        if confirmed < start:
            raise ValueError("Predictions require start <= confirmed")
        if not isinstance(self.id, str):
            raise ValueError("id must be a string")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "confirmed", confirmed)


@dataclass(slots=True)
class _Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    duplicates: int = 0
    ignored_uncertain: int = 0
    delays: list[float] = field(default_factory=list)


def _overlaps(pred: PredEvent, label: LabelEvent, tol: float) -> bool:
    return (
        pred.start - tol <= label.t_end
        and pred.confirmed + tol >= label.t_start
    )


def _percentile(sorted_values: list[float], quantile: float) -> float | None:
    if not sorted_values:
        return None
    position = (len(sorted_values) - 1) * quantile
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    return (
        sorted_values[lower]
        + (sorted_values[upper] - sorted_values[lower]) * fraction
    )


def _metrics(counts: _Counts, duration_s: float | None) -> dict[str, Any]:
    precision_denominator = counts.tp + counts.fp
    recall_denominator = counts.tp + counts.fn
    f1_denominator = 2 * counts.tp + counts.fp + counts.fn
    delays = sorted(counts.delays)
    return {
        "tp": counts.tp,
        "fp": counts.fp,
        "fn": counts.fn,
        "duplicates": counts.duplicates,
        "ignored_uncertain": counts.ignored_uncertain,
        "precision": (
            counts.tp / precision_denominator if precision_denominator else 0.0
        ),
        "recall": counts.tp / recall_denominator if recall_denominator else 0.0,
        "f1": 2 * counts.tp / f1_denominator if f1_denominator else 0.0,
        "latency_median_s": _percentile(delays, 0.5),
        "latency_p95_s": _percentile(delays, 0.95),
        "false_alarms_per_hour": (
            counts.fp / (duration_s / 3600.0)
            if duration_s is not None and duration_s > 0
            else None
        ),
    }


def _validate_durations(durations: Any) -> dict[str, float]:
    if not isinstance(durations, Mapping):
        raise ValueError("durations must map video filenames to seconds")
    result = {}
    for video, value in durations.items():
        _text(video, "duration video")
        duration = _number(value, f"duration for {video!r}")
        if duration <= 0:
            raise ValueError(f"Duration for {video!r} must be positive")
        result[video] = duration
    return result


def evaluate(
    labels: list[LabelEvent],
    preds: list[PredEvent],
    tol_s: float = 2.0,
    *,
    durations: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Вернуть by_type, overall и параметры оценки.

    durations добавлен как необязательный keyword-параметр. При его
    передаче обязательна длительность каждого видео из labels и preds;
    дополнительные ключи представляют чистые видео.
    """
    tol = _number(tol_s, "tol_s")
    if tol < 0:
        raise ValueError("tol_s must be non-negative")
    if not all(isinstance(label, LabelEvent) for label in labels):
        raise TypeError("labels must contain LabelEvent instances")
    if not all(isinstance(pred, PredEvent) for pred in preds):
        raise TypeError("preds must contain PredEvent instances")

    videos = {event.video for event in [*labels, *preds]}
    duration_s = None
    if durations is not None:
        validated_durations = _validate_durations(durations)
        missing = videos - validated_durations.keys()
        if missing:
            raise ValueError(f"Missing durations for: {', '.join(sorted(missing))}")
        videos.update(validated_durations)
        duration_s = math.fsum(validated_durations.values())

    label_groups: dict[tuple[str, str], list[LabelEvent]] = defaultdict(list)
    pred_groups: dict[tuple[str, str], list[PredEvent]] = defaultdict(list)
    for label in labels:
        label_groups[label.video, label.type].append(label)
    for pred in preds:
        pred_groups[pred.video, pred.type].append(pred)

    by_type_counts: dict[str, _Counts] = {}
    for key in sorted(label_groups.keys() | pred_groups.keys()):
        counts = by_type_counts.setdefault(key[1], _Counts())
        certain = sorted(
            (label for label in label_groups[key] if not label.uncertain),
            key=lambda label: (label.t_start, label.t_end),
        )
        uncertain = [
            label for label in label_groups[key] if label.uncertain
        ]
        matched: set[int] = set()
        ordered_preds = sorted(
            pred_groups[key],
            key=lambda pred: (pred.confirmed, pred.start, pred.id),
        )
        for pred in ordered_preds:
            candidates = [
                index
                for index, label in enumerate(certain)
                if _overlaps(pred, label, tol)
            ]
            available = next(
                (index for index in candidates if index not in matched), None
            )
            if available is not None:
                matched.add(available)
                counts.tp += 1
                counts.delays.append(pred.confirmed - certain[available].t_start)
            elif candidates:
                counts.duplicates += 1
            elif any(_overlaps(pred, label, tol) for label in uncertain):
                counts.ignored_uncertain += 1
            else:
                counts.fp += 1
        counts.fn += len(certain) - len(matched)

    overall = _Counts()
    for counts in by_type_counts.values():
        overall.tp += counts.tp
        overall.fp += counts.fp
        overall.fn += counts.fn
        overall.duplicates += counts.duplicates
        overall.ignored_uncertain += counts.ignored_uncertain
        overall.delays.extend(counts.delays)

    return {
        "tol_s": tol,
        "duration_s": duration_s,
        "videos": sorted(videos),
        "by_type": {
            event_type: _metrics(by_type_counts[event_type], duration_s)
            for event_type in sorted(by_type_counts)
        },
        "overall": _metrics(overall, duration_s),
    }


def _read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_labels(directory: str | Path) -> tuple[list[LabelEvent], set[str]]:
    """Загрузить метки и сохранить имена видео с пустой разметкой."""
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Labels directory does not exist: {directory}")
    paths = sorted(directory.glob("*.json"))
    if not paths:
        raise ValueError(f"No label JSON files found in {directory}")

    labels: list[LabelEvent] = []
    videos: set[str] = set()
    for path in paths:
        data = _read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a JSON object")
        video = _text(data.get("video"), f"{path}: video")
        if video in videos:
            raise ValueError(f"{path}: duplicate annotation for {video!r}")
        videos.add(video)
        events = data.get("events")
        if not isinstance(events, list):
            raise ValueError(f"{path}: events must be a list")
        for index, raw in enumerate(events):
            if not isinstance(raw, dict):
                raise ValueError(f"{path}: event {index} must be an object")
            try:
                labels.append(LabelEvent(video=video, **raw))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: event {index}: {exc}") from exc
    return labels, videos


def _validate_mapping(mapping: Any) -> dict[str, tuple[str, float]]:
    if not isinstance(mapping, Mapping):
        raise ValueError("mapping must be an object keyed by camera_id")
    result = {}
    for camera_id, entry in mapping.items():
        _text(camera_id, "camera_id")
        if not isinstance(entry, Mapping):
            raise ValueError(f"Mapping for {camera_id!r} must be an object")
        video = _text(entry.get("video"), f"{camera_id}: video")
        t0 = _number(entry.get("t0"), f"{camera_id}: t0")
        result[camera_id] = (video, t0)
    return result


def predictions_from_events(
    events: Iterable[Event], mapping: Mapping[str, Any]
) -> list[PredEvent]:
    """Перевести started_at и confirmed_at в секунды видео.

    ended_at, операторский status и latency_ms не меняют оценку:
    задержка по заданному определению равна confirmed - t_start.
    """
    camera_mapping = _validate_mapping(mapping)
    result = []
    for event in events:
        if event.camera_id not in camera_mapping:
            raise ValueError(
                f"No mapping for camera {event.camera_id!r} "
                f"(event {event.id!r})"
            )
        video, t0 = camera_mapping[event.camera_id]
        result.append(
            PredEvent(
                video=video,
                type=event.type,
                start=event.started_at - t0,
                confirmed=event.confirmed_at - t0,
                id=event.id,
            )
        )
    return result


def load_predictions(
    path: str | Path, mapping: Mapping[str, Any]
) -> list[PredEvent]:
    """Прочитать JSON-список Event или полный API-ответ с items."""
    data = _read_json(path)
    if isinstance(data, dict):
        items = data.get("items")
        if not isinstance(items, list):
            raise ValueError("Predictions object must contain an items list")
        if "total" in data and data["total"] != len(items):
            raise ValueError(
                "Predictions contain an incomplete API page; export all events"
            )
        data = items
    if not isinstance(data, list):
        raise ValueError("Predictions must be a JSON list of Event objects")
    return predictions_from_events(
        (Event.model_validate(item) for item in data), mapping
    )


def load_database(
    path: str | Path, mapping: Mapping[str, Any]
) -> list[PredEvent]:
    """Прочитать все события журнала в одном SQLite-снимке.

    Используется схема Journal. Прямое read-only соединение не создаёт
    базу, не запускает миграции и не ограничивает выборку страницей query.
    """
    uri = Path(path).resolve().as_uri() + "?mode=ro"

    def deserialize(row: sqlite3.Row) -> Event:
        data = dict(row)
        for column in ("bbox", "details", "latency_ms"):
            data[column] = json.loads(data[column])
        return Event.model_validate(data)

    with closing(sqlite3.connect(uri, uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            "SELECT * FROM events ORDER BY confirmed_at ASC, id ASC"
        )
        return predictions_from_events(
            (deserialize(row) for row in rows), mapping
        )


def format_table(result: Mapping[str, Any]) -> str:
    """Сформировать текстовую таблицу с общей строкой."""
    rows = [[
        "Type", "TP", "FP", "FN", "Duplicates", "Ignored",
        "Precision", "Recall", "F1", "Median(s)", "P95(s)", "FP/h",
    ]]
    metrics = list(result["by_type"].items())
    metrics.append(("TOTAL", result["overall"]))
    for event_type, values in metrics:
        row = [event_type]
        row.extend(
            str(values[key])
            for key in ("tp", "fp", "fn", "duplicates", "ignored_uncertain")
        )
        row.extend(
            "-" if values[key] is None else f"{values[key]:.4f}"
            for key in (
                "precision", "recall", "f1", "latency_median_s",
                "latency_p95_s", "false_alarms_per_hour",
            )
        )
        rows.append(row)
    widths = [max(len(value) for value in column) for column in zip(*rows)]
    return "\n".join(
        "  ".join(value.ljust(width) for value, width in zip(row, widths)).rstrip()
        for row in rows
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preds", type=Path)
    source.add_argument("--db", type=Path)
    parser.add_argument("--mapping", required=True, type=Path)
    parser.add_argument(
        "--durations", required=True, type=Path,
        help='JSON object: {"video.mp4": duration_in_seconds}',
    )
    parser.add_argument("--tol-s", type=float, default=2.0)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        labels, annotated_videos = load_labels(args.labels)
        durations = _validate_durations(_read_json(args.durations))
        missing = annotated_videos - durations.keys()
        extra = durations.keys() - annotated_videos
        if missing or extra:
            raise ValueError(
                "Durations must cover exactly the annotated videos; "
                f"missing={sorted(missing)}, unannotated={sorted(extra)}"
            )
        mapping = _read_json(args.mapping)
        if args.preds is not None:
            preds = load_predictions(args.preds, mapping)
        else:
            preds = load_database(args.db, mapping)
        unknown = {pred.video for pred in preds} - annotated_videos
        if unknown:
            raise ValueError(
                f"Predictions reference unannotated videos: {sorted(unknown)}"
            )
        result = evaluate(labels, preds, args.tol_s, durations=durations)
        serialized = json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, sqlite3.Error) as exc:
        parser.error(str(exc))

    print(format_table(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
