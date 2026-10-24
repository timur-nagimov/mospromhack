"""Read-only offline preflight: configs and local filesystem, never inference or I/O sources."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Sequence

import yaml
from pydantic import ValidationError

from guard.contracts import CameraConfig, ModelSpec
from guard.inference import available_backends, normalize_backend_name
from guard.inference.base import resolve_model_path
from guard.pipeline.config import resolve_source_uri
from guard.pipeline.settings import Settings
from guard.rules.base import available_rules, get_rule, load_plugins
from guard.rules.geometry import validate_active_hours

Status = Literal["ok", "error", "warning", "skipped"]
# Do not render raw YAML, Pydantic inputs, or exception messages. Even a malformed
# camera can contain credentials. URI values in identifiers/paths are hidden too.
# pathlib normalizes "rtsp://..." to "rtsp:/..." when a URI is mistakenly
# configured as a local path. Hide that form as well, including malformed URIs.
_URI = re.compile(r"[a-zA-Z][a-zA-Z0-9+.-]*:/+[^\s\"'<>]*")
_IMAGE_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def _safe(value: Any) -> str:
    text = str(value)
    match = _URI.search(text)
    # Hide the entire suffix, since a malformed URI can contain whitespace in
    # its password; stopping at that whitespace would expose the remainder.
    return text[:match.start()] + "<URI скрыт>" if match else text


class _Checks:
    def __init__(self) -> None:
        self.items: list[dict[str, str]] = []

    def add(self, check: str, status: Status, message: str) -> None:
        self.items.append({"check": _safe(check), "status": status, "message": _safe(message)})

    def report(self, required: set[str]) -> dict[str, Any]:
        counts = Counter(item["status"] for item in self.items)
        return {
            "scope": "offline",
            "ok": counts["error"] == 0,
            "summary": {status: counts[status] for status in ("ok", "error", "warning", "skipped")},
            "required_models": sorted(_safe(model) for model in required),
            "checks": self.items,
        }


def _read_yaml(path: Path, check: str, out: _Checks, *, required: bool = True) -> tuple[bool, Any]:
    try:
        if not path.is_file():
            out.add(check, "error" if required else "warning", f"Файл конфигурации отсутствует или не является файлом: {path}")
            return False, None
        if not os.access(path, os.R_OK):
            out.add(check, "error", f"Нет разрешения на чтение конфигурации: {path}")
            return False, None
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError, ValueError):
        out.add(check, "error", "Не удалось прочитать YAML конфигурации; содержимое скрыто")
        return False, None
    return True, raw


def _load_cameras(path: Path, out: _Checks) -> list[CameraConfig]:
    read, raw = _read_yaml(path, "config:cameras", out)
    if not read:
        return []
    if raw is None or raw == {}:
        raw = []
    if isinstance(raw, dict):
        raw = raw.get("cameras")
    if not isinstance(raw, list):
        out.add("config:cameras", "error", "Ожидается список камер или объект cameras со списком")
        return []
    cameras = []
    for index, item in enumerate(raw):
        try:
            cameras.append(CameraConfig.model_validate(item))
        except ValidationError:
            out.add(f"config:cameras:{index}", "error", "Камера не соответствует CameraConfig; содержимое скрыто")
    ids = [cam.camera_id for cam in cameras]
    if len(set(ids)) != len(ids):
        out.add("config:cameras", "error", "camera_id должны быть уникальными")
    out.add("config:cameras:loaded", "ok" if cameras else "warning", f"Прочитано камер: {len(cameras)}; включено: {sum(cam.enabled for cam in cameras)}")
    return cameras


def _load_models(path: Path, out: _Checks, required: bool) -> dict[str, ModelSpec]:
    read, raw = _read_yaml(path, "config:models", out, required=required)
    if not read:
        return {}
    if raw is None:
        raw = []
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if isinstance(raw, dict):
        if any(not isinstance(item, dict) for item in raw.values()):
            out.add("config:models", "error", "Записи реестра моделей должны быть объектами")
            return {}
        raw = [{"model_id": key, **item} for key, item in raw.items()]
    if not isinstance(raw, list):
        out.add("config:models", "error", "Ожидается список моделей или объект с моделями")
        return {}
    models = {}
    for index, item in enumerate(raw):
        try:
            if isinstance(item, dict):
                item = dict(item)
                if "onnx_path" not in item and "onnx" in item:
                    item["onnx_path"] = item.pop("onnx")
            spec = ModelSpec.model_validate(item)
        except ValidationError:
            out.add(f"config:models:{index}", "error", "Модель не соответствует ModelSpec; содержимое скрыто")
            continue
        if spec.model_id in models:
            out.add("config:models", "error", "model_id должны быть уникальными")
        if (not spec.model_id.strip() or not spec.onnx_path.strip()
                or any(size <= 0 for size in spec.input_size)
                or not math.isfinite(spec.conf_threshold) or not 0 <= spec.conf_threshold <= 1
                or not math.isfinite(spec.iou_threshold) or not 0 <= spec.iou_threshold <= 1):
            out.add(f"model:{spec.model_id}:config", "error", "Нужны непустые ID/путь, положительный input_size и конечные пороги в [0, 1]")
        models[spec.model_id] = spec
    out.add("config:models:loaded", "ok" if models else "warning", f"Прочитано моделей: {len(models)}")
    return models


def _camera_semantics(cam: CameraConfig, known: set[str], out: _Checks) -> set[str]:
    check = f"camera:{cam.camera_id}:config"
    errors = []
    if not cam.camera_id.strip() or not cam.source.uri.strip():
        errors.append("camera_id и source.uri должны быть непустыми")
    if not math.isfinite(cam.sample_fps) or not 0 < cam.sample_fps <= 30:
        errors.append("sample_fps должен быть в (0, 30]")
    zone_ids = [zone.zone_id for zone in cam.zones]
    if len(set(zone_ids)) != len(zone_ids):
        errors.append("zone_id должны быть уникальными")
    for zone in cam.zones:
        if (not zone.zone_id.strip() or len(zone.polygon) < 3
                or any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in zone.polygon)):
            errors.append("зоне нужны непустой ID и полигон минимум из трёх точек в [0, 1]")
        try:
            validate_active_hours(zone.active_hours)
        except ValueError:
            errors.append("некорректное расписание active_hours")
    rule_ids = [rule.rule_id for rule in cam.rules]
    if len(set(rule_ids)) != len(rule_ids):
        errors.append("rule_id должны быть уникальными в камере")
    required: set[str] = set()
    for rule in cam.rules:
        if not rule.rule_id.strip():
            errors.append("rule_id должен быть непустым")
        if rule.type not in known:
            errors.append("неизвестный тип правила")
            continue
        if set(rule.zones or []) - set(zone_ids):
            errors.append("правило ссылается на неизвестные зоны")
        if (not all(math.isfinite(value) and value >= 0 for value in
                    (rule.min_duration_s, rule.close_after_s, rule.cooldown_s))
                or not math.isfinite(rule.min_ratio) or not 0 <= rule.min_ratio <= 1):
            errors.append("длительности должны быть конечными и неотрицательными, min_ratio в [0, 1]")
        if cam.enabled and rule.enabled:
            cls = get_rule(rule.type)
            required.update(cls.required_models)
            try:
                cls(rule, cam)  # Pure rule constructors; no engine, backend or source.
            except (ValueError, TypeError, KeyError, OverflowError):
                errors.append("параметры включённого правила не позволяют создать плагин")
    if errors:
        for message in dict.fromkeys(errors):
            out.add(check, "error", message)
    else:
        out.add(check, "ok", "Структура камеры, зоны и правила корректны")
    return required


def _input_file(path: Path, check: str, out: _Checks) -> None:
    if not path.is_file():
        out.add(check, "error", f"Локальный файл отсутствует или имеет неверный тип: {path}")
    elif not os.access(path, os.R_OK):
        out.add(check, "error", f"Нет разрешения на чтение: {path}")
    else:
        out.add(check, "ok", f"Локальный файл доступен для чтения: {path}; содержимое не проверялось")


def _source(cam: CameraConfig, settings: Settings, out: _Checks) -> None:
    check = f"camera:{cam.camera_id}:source"
    if not cam.enabled:
        out.add(check, "skipped", "Камера выключена; наличие источника не требуется")
        return
    if cam.source.type not in {"file", "images"}:
        out.add(check, "skipped", "RTSP/webcam не открывались; доступность и учётные данные не проверены")
        return
    path = Path(resolve_source_uri(cam.source.uri, settings.data_dir))
    if cam.source.type == "file":
        _input_file(path, check, out)
        return
    if path.is_dir():
        if not os.access(path, os.R_OK | os.X_OK):
            out.add(check, "error", "Нет разрешения на чтение каталога изображений")
            return
        files = list(path.iterdir())
    else:
        files = [Path(item) for item in glob.glob(str(path))]
    images = [item for item in files if item.is_file() and item.suffix.lower() in _IMAGE_EXTS]
    if not images:
        out.add(check, "error", f"Нет локальных файлов изображений поддерживаемых типов: {path}")
    elif any(not os.access(item, os.R_OK) for item in images):
        out.add(check, "error", "Не все файлы изображений доступны для чтения")
    else:
        out.add(check, "ok", f"Найдено изображений: {len(images)}; декодирование не выполнялось")


def _output(path: Path, check: str, kind: Literal["file", "dir"], out: _Checks) -> None:
    if path.exists():
        if not (path.is_dir() if kind == "dir" else path.is_file()):
            out.add(check, "error", f"Неверный тип выходного пути ({kind}): {path}")
            return
        access = os.W_OK | (os.X_OK if kind == "dir" else os.R_OK)
        if not os.access(path, access):
            out.add(check, "error", f"Нет разрешений для выходного пути: {path}")
            return
    elif path.is_symlink():
        out.add(check, "error", f"Выходной путь — неработающая символическая ссылка: {path}")
        return
    # SQLite creates sibling files; config save uses an atomic replacement.
    parent = path if path.exists() and kind == "dir" else path.parent
    while not parent.exists() and not parent.is_symlink() and parent != parent.parent:
        parent = parent.parent
    if not parent.is_dir() or not os.access(parent, os.W_OK | os.X_OK):
        out.add(check, "error", f"Ближайший существующий родитель не является доступным для записи каталогом: {parent}")
    else:
        out.add(check, "ok", f"Тип и разрешения пути/существующего родителя подходят: {path}; запись не выполнялась")


def run_checks(settings: Settings | None = None) -> dict[str, Any]:
    """Return a JSON-compatible report without creating files or runtime objects."""
    out = _Checks()
    required: set[str] = set()
    if settings is None:
        try:
            settings = Settings.from_env()
        except (ValueError, OSError, UnicodeError):
            out.add("settings", "error", "Не удалось загрузить GUARD_* настройки; значения скрыты")
            return out.report(required)
    if settings.mode in {"edge", "gateway"}:
        from guard.transport.rabbit import RabbitConfig
        try:
            RabbitConfig(settings.rabbit_url or "", settings.edge_id)
        except (ValueError, TypeError):
            out.add("transport:config", "error", "Некорректная AMQP-конфигурация; значения скрыты")
        else:
            out.add("transport:config", "ok", "Локальная AMQP-конфигурация корректна; соединение не проверялось")
    if settings.mode == "gateway":
        out.add("mode:gateway", "ok", "Gateway принимает события и превью без локальных моделей и источников")
        for check, path, kind in (
            ("output:data", settings.data_dir, "dir"),
            ("output:media", settings.media_dir, "dir"),
            ("output:db", settings.db_path, "file"),
        ):
            _output(path, check, kind, out)
        out.add("runtime", "skipped", "Сеть и фактическая запись не проверялись")
        return out.report(required)
    backend = normalize_backend_name(settings.backend)
    out.add("backend:config", "ok" if backend in available_backends() else "error",
            "Имя бэкенда поддерживается" if backend in available_backends() else "Неизвестное имя бэкенда")
    try:
        cameras = _load_cameras(settings.cameras_config, out)
        # A prior import may have registered only one plugin. Populate the full
        # registry explicitly; available_rules() loads only an empty registry.
        load_plugins()
        known = {rule["type"] for rule in available_rules()}
        for cam in cameras:
            required.update(_camera_semantics(cam, known, out))
            _source(cam, settings, out)
        models = _load_models(settings.models_config, out, bool(required))
        for model_id in sorted(required):
            spec = models.get(model_id)
            check = f"model:{model_id}"
            if spec is None:
                out.add(check, "error", "Обязательная модель включённого правила отсутствует в реестре")
            elif backend == "onnx-cpu":
                _input_file(resolve_model_path(spec.onnx_path, settings.models_dir), check, out)
            elif backend == "linq-tpu":
                if not spec.tpu_program:
                    out.add(check, "error", "У обязательной модели не задан tpu_program")
                elif spec.tpu_program.startswith("file://"):
                    out.add(check, "skipped", "URI программы относится к TPU-воркеру; удалённый файл не проверен")
                else:
                    _input_file(resolve_model_path(spec.tpu_program, settings.models_dir), check, out)
                out.add(f"{check}:onnx-fallback", "skipped", "Потребность в ONNX/manifest/tail определяется загрузчиком TPU и здесь не проверяется")
        if not required:
            out.add("models:required", "warning", "Нет моделей, требуемых включёнными камерами и правилами")
        for check, path, kind in (
            ("output:data", settings.data_dir, "dir"),
            ("output:media", settings.media_dir, "dir"),
            ("output:db", settings.db_path, "file"),
            ("output:cameras", settings.cameras_config, "file"),
        ):
            _output(path, check, kind, out)
    except (OSError, ValueError, TypeError, OverflowError):
        out.add("offline:inspection", "error", "Локальная проверка не завершилась; значения и текст исключения скрыты")
    out.add("runtime", "skipped", "Сеть, источники кадров, инференс, ORT, TPU, содержимое моделей/видео и качество детекции не проверялись")
    return out.report(required)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read-only offline-проверка конфигурации и локальных путей GUARD_*")
    parser.add_argument("--json", action="store_true", help="вывести один JSON-объект в stdout")
    args = parser.parse_args(argv)
    report = run_checks()
    if args.json:
        print(json.dumps(report, ensure_ascii=False))
    else:
        print("guard doctor: scope=offline")
        for item in report["checks"]:
            print(f"[{item['status']}] {item['check']}: {item['message']}")
        counts = report["summary"]
        print("Итог: " + ", ".join(f"{key}={value}" for key, value in counts.items()))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
