"""Загрузка и сохранение конфигов: configs/cameras.yaml и реестр моделей configs/models.yaml."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from guard.contracts import CameraConfig, ModelSpec

log = logging.getLogger(__name__)

_CAMERAS_HEADER = (
    "# Камеры сервиса «Зазеркалье». Файл меняет API (/api/cameras), правки руками тоже допустимы.\n"
    "# source.uri для file/images: абсолютный путь или относительно GUARD_DATA_DIR.\n"
    "# Полигоны зон — нормированные координаты 0..1; active_hours — по Europe/Moscow.\n"
)


# ---------- камеры ----------


def load_cameras(path: str | Path) -> list[CameraConfig]:
    path = Path(path)
    if not path.exists():
        return []
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    items = raw.get("cameras", []) if isinstance(raw, dict) else raw
    cams = [CameraConfig.model_validate(item) for item in items or []]
    ids = [c.camera_id for c in cams]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"duplicate camera_id in {path}: {sorted(dup)}")
    return cams


def save_cameras(path: str | Path, cameras: Iterable[CameraConfig]) -> None:
    """Атомарная запись: временный файл рядом + os.replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"cameras": [c.model_dump(mode="json") for c in cameras]}
    text = _CAMERAS_HEADER + yaml.safe_dump(data, allow_unicode=True, sort_keys=False, default_flow_style=None, width=120)
    fd, tmp = tempfile.mkstemp(prefix=".cameras.", suffix=".yaml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def resolve_source_uri(uri: str, data_dir: Path) -> str:
    """Путь к файлу/папке источника: абсолютный, относительно GUARD_DATA_DIR или корня проекта."""
    p = Path(uri).expanduser()
    if p.is_absolute():
        return str(p)
    for base in (data_dir, data_dir.parent, Path.cwd()):
        cand = base / p
        if cand.exists():
            return str(cand)
    return str(data_dir / p)


# ---------- модели ----------


def _iter_model_items(raw: Any) -> Iterable[dict[str, Any]]:
    """Допустимые формы: {models: [..]}, {models: {id: {..}}}, [..], {id: {..}}."""
    if isinstance(raw, dict) and "models" in raw:
        raw = raw["models"]
    if isinstance(raw, list):
        yield from raw
    elif isinstance(raw, dict):
        for model_id, item in raw.items():
            if isinstance(item, dict):
                yield {"model_id": model_id, **item}


def load_models(path: str | Path, models_dir: str | Path | None = None) -> dict[str, ModelSpec]:
    """Реестр моделей; пустые classes дополняются из <models_dir>/<model_id>.json."""
    path = Path(path)
    if not path.exists():
        log.warning("models registry %s not found", path)
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    specs: dict[str, ModelSpec] = {}
    for item in _iter_model_items(raw):
        item = dict(item)
        if "onnx_path" not in item and "onnx" in item:
            item["onnx_path"] = item.pop("onnx")
        known = set(ModelSpec.model_fields)
        spec = ModelSpec.model_validate({k: v for k, v in item.items() if k in known})
        if not spec.classes and models_dir:
            meta = Path(models_dir) / f"{spec.model_id}.json"
            if meta.is_file():
                try:
                    classes = json.loads(meta.read_text(encoding="utf-8")).get("classes") or []
                    spec = spec.model_copy(update={"classes": [str(c) for c in classes]})
                except (OSError, ValueError) as exc:
                    log.warning("cannot read classes from %s: %s", meta, exc)
        specs[spec.model_id] = spec
    return specs
