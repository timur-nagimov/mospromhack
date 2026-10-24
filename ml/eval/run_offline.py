"""Последовательная офлайн-оценка событий без ожидания реального времени.

Приоритет путей: флаги CLI, явная конфигурация YAML, GUARD_*, значения
по умолчанию. root разрешается относительно scenarios.yaml, остальные
пути сценария — относительно root. Необязательный models_dir из реестра
моделей разрешается относительно models.yaml; scenarios.yaml имеет
приоритет над этим значением. Относительные пути CLI разрешаются от cwd.

camera.source формируется из videos_dir и имени video. Время кадра:
t0 + frame_idx / native_fps, где frame_idx начинается с нуля.

Выбирается первый кадр не раньше очередного момента выборки. Частота
выборки ограничена FPS исходника. Пропуски из-за скорости инференса,
потоки источников, медиа и искусственные кадры после EOF отсутствуют.

Трекеры, атрибуты и менеджер событий берутся из pipeline.compat.
Параметры трекеров совпадают с CameraWorker: переданный Settings
или Settings.from_env(). Пути из Settings не переопределяют Suite.

JSON содержит метрики events_eval, items/total с полными Event, mapping,
durations и статистику обработки. Его можно читать через
events_eval.load_predictions(path, report["mapping"]).

Отсутствующая модель, неизвестное правило или ошибка обработки прерывают
прогон. Метки не фильтруются по включённым правилам: отключённый детектор
не должен незаметно улучшать recall.
"""

from __future__ import annotations

import os

# До импорта numpy, OpenCV и модулей инференса.
os.environ["OMP_NUM_THREADS"] = "2"

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml

from guard.contracts import TIMING_KEYS, CameraConfig, Event, FrameContext
from guard.inference import get_backend
from guard.pipeline import compat
from guard.pipeline import settings as pipeline_settings
from guard.pipeline.config import load_models
from guard.pipeline.detector import Detector
from guard.pipeline.settings import Settings
from guard.pipeline.worker import OBJECT_CLASSES, PERSON_CLASSES, object_detections
from guard.rules.base import RulePlugin, build_rules, load_plugins
from guard.rules.geometry import validate_active_hours
from ml.eval import events_eval


DEFAULT_T0 = 1_700_000_000.0


@dataclass(frozen=True)
class Scenario:
    id: str
    video: str
    camera: CameraConfig
    t0: float = DEFAULT_T0


@dataclass(frozen=True)
class Suite:
    labels_dir: Path
    models_config: Path
    models_dir: Path
    scenarios: tuple[Scenario, ...]


@dataclass
class VideoResult:
    events: list[Event]
    duration_s: float
    processing: dict[str, Any]


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _path(base: Path, value: Any, name: str) -> Path:
    if isinstance(value, Path):
        path = value.expanduser()
    else:
        path = Path(_text(value, name)).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def _config_or_env(
    raw: Mapping[str, Any], key: str, env_name: str, default: Any
) -> Any:
    # Явное, но некорректное значение YAML должно пройти в валидацию,
    # а не незаметно замениться окружением.
    if key in raw:
        return raw[key]
    return os.environ.get(env_name) or default


def _resolve_models_dir(
    root: Path,
    raw: Mapping[str, Any],
    models_config: Path,
    override: str | Path | None,
) -> Path:
    if override is not None:
        return _path(Path.cwd(), override, "--models-dir")
    if "models_dir" in raw:
        return _path(root, raw["models_dir"], "models_dir")

    # Допускается models_dir рядом с models в реестре. Отсутствие такого
    # поля сохраняет поддержку всех форматов guard.pipeline.config.
    if models_config.is_file():
        registry = yaml.safe_load(models_config.read_text(encoding="utf-8"))
        if isinstance(registry, dict) and "models_dir" in registry:
            return _path(
                models_config.parent,
                registry["models_dir"],
                f"{models_config}: models_dir",
            )
    return _path(
        root, os.environ.get("GUARD_MODELS_DIR") or "models", "models_dir"
    )


def _validate_camera(camera: CameraConfig) -> None:
    _text(camera.camera_id, "camera_id")
    if not camera.enabled:
        raise ValueError(f"{camera.camera_id}: scenario camera must be enabled")
    if _finite(camera.sample_fps, "sample_fps") <= 0:
        raise ValueError("sample_fps must be positive")

    zone_ids = [zone.zone_id for zone in camera.zones]
    if len(zone_ids) != len(set(zone_ids)):
        raise ValueError(f"{camera.camera_id}: duplicate zone_id")
    for zone in camera.zones:
        if len(zone.polygon) < 3 or any(
            not (0 <= x <= 1 and 0 <= y <= 1) for x, y in zone.polygon
        ):
            raise ValueError(f"{zone.zone_id}: expected a normalized polygon")
        validate_active_hours(zone.active_hours)

    rule_ids = [rule.rule_id for rule in camera.rules]
    if len(rule_ids) != len(set(rule_ids)):
        raise ValueError(f"{camera.camera_id}: duplicate rule_id")
    for rule in camera.rules:
        if set(rule.zones or []) - set(zone_ids):
            raise ValueError(f"{rule.rule_id}: unknown zones")
        for name in ("min_duration_s", "close_after_s", "cooldown_s"):
            if _finite(getattr(rule, name), name) < 0:
                raise ValueError(f"{rule.rule_id}: {name} must be non-negative")
        if not 0 <= _finite(rule.min_ratio, "min_ratio") <= 1:
            raise ValueError(f"{rule.rule_id}: min_ratio must be in [0, 1]")


def load_scenarios(
    path: str | Path,
    *,
    models_config: str | Path | None = None,
    models_dir: str | Path | None = None,
) -> Suite:
    path = Path(path).expanduser().resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("version") != 1:
        raise ValueError("Scenarios must be an object with version: 1")

    # .env заполняет только резервные значения. Вызов через модуль позволяет
    # тестам изолировать загрузчик вместе со всеми GUARD_*.
    pipeline_settings.load_dotenv_guard()
    root = _path(
        path.parent, _config_or_env(raw, "root", "GUARD_ROOT", "."), "root"
    )
    data_dir = _path(
        root,
        _config_or_env(raw, "data_dir", "GUARD_DATA_DIR", "data"),
        "data_dir",
    )
    videos_dir = _path(
        root, raw.get("videos_dir", data_dir / "videos"), "videos_dir"
    )
    labels_dir = _path(
        root, raw.get("labels_dir", data_dir / "labels"), "labels_dir"
    )
    if models_config is not None:
        registry_path = _path(Path.cwd(), models_config, "--models-config")
    else:
        registry_path = _path(
            root,
            _config_or_env(
                raw, "models_config", "GUARD_MODELS_CONFIG", "configs/models.yaml"
            ),
            "models_config",
        )
    resolved_models_dir = _resolve_models_dir(
        root, raw, registry_path, models_dir
    )

    sample_fps = _finite(raw.get("sample_fps", 5.0), "sample_fps")
    if sample_fps <= 0:
        raise ValueError("sample_fps must be positive")
    default_t0 = _finite(raw.get("t0", DEFAULT_T0), "t0")

    entries = raw.get("scenarios")
    if not isinstance(entries, list) or not entries:
        raise ValueError("scenarios must be a non-empty list")

    scenarios = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each scenario must be an object")
        scenario_id = _text(entry.get("id"), "scenario id")
        video = _text(entry.get("video"), f"{scenario_id}: video")
        if Path(video).is_absolute() or ".." in Path(video).parts:
            raise ValueError(f"{scenario_id}: video must be relative to videos_dir")
        camera_raw = entry.get("camera")
        if not isinstance(camera_raw, dict):
            raise ValueError(f"{scenario_id}: camera must be an object")
        camera_raw = dict(camera_raw)
        camera_raw.setdefault("sample_fps", sample_fps)
        camera_raw["source"] = {
            "type": "file",
            "uri": str(videos_dir / video),
            "loop": False,
            "realtime": False,
        }
        camera = CameraConfig.model_validate(camera_raw)
        _validate_camera(camera)
        scenarios.append(
            Scenario(
                id=scenario_id,
                video=video,
                camera=camera,
                t0=_finite(entry.get("t0", default_t0), f"{scenario_id}: t0"),
            )
        )

    for name, values in (
        ("scenario id", [scenario.id for scenario in scenarios]),
        ("video", [scenario.video for scenario in scenarios]),
        ("camera_id", [scenario.camera.camera_id for scenario in scenarios]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate {name} in scenarios")

    return Suite(
        labels_dir, registry_path, resolved_models_dir, tuple(scenarios)
    )


def _select(suite: Suite, only: Sequence[str] | None) -> list[Scenario]:
    if only is None:
        return list(suite.scenarios)
    requested = {value.strip() for value in only}
    if not requested or "" in requested:
        raise ValueError("--only must contain non-empty scenario IDs")
    unknown = requested - {scenario.id for scenario in suite.scenarios}
    if unknown:
        raise ValueError(f"Unknown scenario IDs: {', '.join(sorted(unknown))}")
    return [scenario for scenario in suite.scenarios if scenario.id in requested]


def _load_detectors(
    model_ids: set[str], backend_name: str, suite: Suite
) -> tuple[Any, dict[str, Detector]]:
    if not model_ids:
        return None, {}

    specs = load_models(suite.models_config, suite.models_dir)
    missing = model_ids - specs.keys()
    if missing:
        raise ValueError(
            f"Missing required models: {', '.join(sorted(missing))}; "
            f"registry: {suite.models_config}. Rules were not silently disabled."
        )
    if compat.FALLBACKS.get("vision", False):
        raise RuntimeError("Offline evaluation requires guard.vision, not fallbacks")

    backend = get_backend(
        backend_name,
        models_dir=suite.models_dir,
        intra_op_threads=2,
        max_batch=1,
    )
    try:
        detectors = {}
        for model_id in sorted(model_ids):
            spec = specs[model_id]
            backend.load(spec)
            if not spec.classes:
                names = backend.class_names(model_id)
                if names:
                    spec = spec.model_copy(update={"classes": names})
            detectors[model_id] = Detector(spec, backend)
        return backend, detectors
    except BaseException:
        backend.close()
        raise


def _tracker_kwargs(settings: Settings) -> dict[str, Any]:
    """Те же параметры, которые CameraWorker передаёт обоим трекерам."""
    return {
        "high_thresh": settings.tracker_high_thresh,
        "low_thresh": settings.tracker_low_thresh,
        "match_iou": settings.tracker_match_iou,
        "new_track_thresh": settings.tracker_new_track_thresh,
        "track_buffer_s": settings.tracker_buffer_s,
        "min_hits": settings.tracker_min_hits,
    }


def _remember(events: dict[str, Event], event: Event) -> None:
    # Закрытие/повторное открытие обновляет тот же Event, а не создаёт дубль.
    previous = events.get(event.id)
    if previous is not None:
        event.latency_ms = dict(previous.latency_ms)
    events[event.id] = event


def run_video(
    scenario: Scenario,
    detectors: Mapping[str, Any],
    *,
    rules: Sequence[RulePlugin] | None = None,
    settings: Settings | None = None,
) -> VideoResult:
    """Прочитать один файл потоково; трекеры и EventManager изолированы."""
    camera = scenario.camera
    _validate_camera(camera)
    _finite(scenario.t0, "t0")
    if settings is None:
        settings = Settings.from_env()
    if rules is None:
        load_plugins()
        rules = build_rules(camera)
    needed = sorted({model_id for rule in rules for model_id in rule.required_models})
    missing = set(needed) - detectors.keys()
    if missing:
        raise ValueError(f"{scenario.id}: missing detectors: {sorted(missing)}")

    path = Path(camera.source.uri)
    if not path.is_file():
        raise ValueError(f"{scenario.id}: video does not exist: {path}")

    started = time.perf_counter()
    capture = cv2.VideoCapture(str(path))
    events: dict[str, Event] = {}
    decoded = analyzed = 0
    try:
        if not capture.isOpened():
            raise ValueError(f"{scenario.id}: cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError(f"{scenario.id}: invalid source FPS: {fps}")
        reported_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        expected_count = (
            round(reported_count)
            if math.isfinite(reported_count) and reported_count > 0
            else None
        )
        effective_fps = min(camera.sample_fps, fps)

        tracker_kwargs = _tracker_kwargs(settings)
        person_tracker = compat.Tracker(
            classes=set(PERSON_CLASSES), **tracker_kwargs
        )
        object_tracker = compat.Tracker(
            classes=set(OBJECT_CLASSES), **tracker_kwargs
        )
        manager = compat.EventManager()

        while True:
            read_started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter() - read_started) * 1000.0
            if not ok or frame is None:
                break
            frame_idx = decoded
            decoded += 1

            # Индекс цели вместо накопления периода: нет дрейфа на дробных FPS.
            if frame_idx + 1e-9 < analyzed * fps / effective_fps:
                continue
            ts = scenario.t0 + frame_idx / fps
            frame_started = time.perf_counter()
            timings = {key: 0.0 for key in TIMING_KEYS}
            timings["decode"] = decode_ms
            detections = {
                model_id: detectors[model_id].detect(frame, timings)
                for model_id in needed
            }

            # Та же последовательность и фильтры, что в CameraWorker.process.
            tracking_started = time.perf_counter()
            if "det" in detections:
                person_detections = [
                    det for det in detections["det"] if det.cls in PERSON_CLASSES
                ]
            else:
                person_detections = list(detections.get("pose", []))
            people = person_tracker.update(person_detections, ts)
            # Техника из всех моделей кадра, как в CameraWorker.process.
            objects = object_tracker.update(object_detections(detections), ts)
            compat.assign_attributes(people, detections, ts)
            rules_started = time.perf_counter()
            timings["tracking"] = (rules_started - tracking_started) * 1000.0

            context = FrameContext(
                camera=camera,
                ts=ts,
                frame_idx=frame_idx,
                frame=frame,
                detections=detections,
                people=people,
                objects=objects,
                timings=timings,
            )
            violations = [
                violation for rule in rules for violation in rule.evaluate(context)
            ]
            alert_started = time.perf_counter()
            timings["rules"] = (alert_started - rules_started) * 1000.0
            new, closed = manager.process(camera, ts, violations, dict(timings))
            finished = time.perf_counter()
            timings["alert"] = (finished - alert_started) * 1000.0
            timings["total"] = decode_ms + (finished - frame_started) * 1000.0
            for event in new:
                event.latency_ms = dict(timings)
                _remember(events, event)
            for event in closed:
                _remember(events, event)
            analyzed += 1

        if decoded == 0:
            raise ValueError(f"{scenario.id}: video has no readable frames")
        # Допуск в один кадр учитывает округление метаданных контейнера.
        if expected_count is not None and decoded + 1 < expected_count:
            raise ValueError(
                f"{scenario.id}: incomplete decode: "
                f"{decoded} frames, container reports {expected_count}"
            )

        # EOF не означает исчезновение человека. Не добавляем пустые кадры
        # и не подтверждаем кандидатов вне наблюдаемого интервала.
        for event in manager.open_events(camera.camera_id):
            _remember(events, event)
    finally:
        capture.release()

    elapsed = time.perf_counter() - started
    duration_s = decoded / fps
    return VideoResult(
        events=sorted(events.values(), key=lambda event: (event.confirmed_at, event.id)),
        duration_s=duration_s,
        processing={
            "native_fps": fps,
            "sample_fps": camera.sample_fps,
            "effective_sample_fps": effective_fps,
            "frames_decoded": decoded,
            "frames_analyzed": analyzed,
            "elapsed_s": elapsed,
            "fps": analyzed / elapsed if elapsed > 0 else None,
            "decoded_fps": decoded / elapsed if elapsed > 0 else None,
        },
    )


def run_evaluation(
    suite: Suite,
    *,
    backend: str = "onnx-cpu",
    only: Sequence[str] | None = None,
    tol_s: float = 2.0,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Оценить выбранные видео, включая пустую разметку и все её типы."""
    if backend not in {"onnx-cpu", "linq-tpu"}:
        raise ValueError(f"Unknown backend: {backend}")
    if _finite(tol_s, "tol_s") < 0:
        raise ValueError("tol_s must be non-negative")

    wall_started = time.perf_counter()
    scenarios = _select(suite, only)
    if settings is None:
        settings = Settings.from_env()
    labels, annotated_videos = events_eval.load_labels(suite.labels_dir)
    selected_videos = {scenario.video for scenario in scenarios}
    unknown = selected_videos - annotated_videos
    missing = annotated_videos - selected_videos if only is None else set()
    if unknown or missing:
        raise ValueError(
            "Scenarios must cover annotated videos; "
            f"unannotated={sorted(unknown)}, missing={sorted(missing)}"
        )
    labels = [label for label in labels if label.video in selected_videos]

    # Явная загрузка нужна и при уже частично заполненном реестре плагинов.
    load_plugins()
    prepared = []
    for scenario in scenarios:
        try:
            rules = build_rules(scenario.camera)
        except KeyError as exc:
            raise ValueError(f"{scenario.id}: unknown rule type: {exc}") from exc
        prepared.append((scenario, rules))
    model_ids = {
        model_id
        for _, rules in prepared
        for rule in rules
        for model_id in rule.required_models
    }

    inference, detectors = _load_detectors(model_ids, backend, suite)
    events = []
    mapping = {}
    durations = {}
    runs = []
    try:
        for scenario, rules in prepared:
            result = run_video(
                scenario, detectors, rules=rules, settings=settings
            )
            events.extend(result.events)
            mapping[scenario.camera.camera_id] = {
                "video": scenario.video,
                "t0": scenario.t0,
            }
            durations[scenario.video] = result.duration_s
            runs.append(
                {
                    "id": scenario.id,
                    "video": scenario.video,
                    "t0": scenario.t0,
                    "camera": scenario.camera.model_dump(mode="json"),
                    "duration_s": result.duration_s,
                    **result.processing,
                }
            )
    finally:
        if inference is not None:
            inference.close()

    predictions = events_eval.predictions_from_events(events, mapping)
    metrics = events_eval.evaluate(labels, predictions, tol_s, durations=durations)
    elapsed = math.fsum(run["elapsed_s"] for run in runs)
    frames_decoded = sum(run["frames_decoded"] for run in runs)
    frames_analyzed = sum(run["frames_analyzed"] for run in runs)
    return {
        **metrics,
        "backend": backend,
        "items": [event.model_dump(mode="json") for event in events],
        "total": len(events),
        "mapping": mapping,
        "durations": durations,
        "runs": runs,
        "models_config": str(suite.models_config),
        "models_dir": str(suite.models_dir),
        "required_models": sorted(model_ids),
        "tracker": _tracker_kwargs(settings),
        "processing": {
            "frames_decoded": frames_decoded,
            "frames_analyzed": frames_analyzed,
            # Последовательное декодирование и анализ; без загрузки моделей.
            "elapsed_s": elapsed,
            "fps": frames_analyzed / elapsed if elapsed > 0 else None,
            "decoded_fps": frames_decoded / elapsed if elapsed > 0 else None,
            # Включает конфигурацию, загрузку/закрытие моделей и оценку.
            "wall_elapsed_s": time.perf_counter() - wall_started,
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", required=True, type=Path)
    parser.add_argument("--backend", choices=("onnx-cpu", "linq-tpu"), default="onnx-cpu")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--only", help="Comma-separated scenario IDs, e.g. s1_01,s3_02")
    parser.add_argument("--tol-s", type=float, default=2.0)
    parser.add_argument("--models-config", type=Path, help="Override model registry")
    parser.add_argument("--models-dir", type=Path, help="Override model directory")
    args = parser.parse_args(argv)

    try:
        suite = load_scenarios(
            args.scenarios,
            models_config=args.models_config,
            models_dir=args.models_dir,
        )
        result = run_evaluation(
            suite,
            backend=args.backend,
            only=args.only.split(",") if args.only is not None else None,
            tol_s=args.tol_s,
        )
        serialized = json.dumps(
            result, ensure_ascii=False, indent=2, allow_nan=False
        )
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, RuntimeError, KeyError, ImportError, cv2.error) as exc:
        parser.error(str(exc))

    print(events_eval.format_table(result))
    processing = result["processing"]
    fps = processing["fps"]
    fps_text = "-" if fps is None else f"{fps:.2f}"
    print(
        f"\nBackend: {result['backend']}; "
        f"decoded: {processing['frames_decoded']}; "
        f"analyzed: {processing['frames_analyzed']}; "
        f"processing: {processing['elapsed_s']:.3f}s; "
        f"analyzed FPS: {fps_text}; "
        f"wall: {processing['wall_elapsed_s']:.3f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
