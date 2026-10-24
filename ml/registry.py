"""Чтение реестра моделей configs/models.yaml → ModelSpec.

Формат файла: ключ `models` — список записей ModelSpec (поля строго из guard.contracts.ModelSpec);
ключ `meta` — справочные данные по model_id (источник, лицензия, исходные классы), в ModelSpec не входит.
onnx_path относительный — считается от GUARD_MODELS_DIR (по умолчанию <корень>/models).
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml

from guard.contracts import ModelSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "models.yaml"


def models_dir() -> Path:
    return Path(os.environ.get("GUARD_MODELS_DIR") or REPO_ROOT / "models").expanduser()


def load_raw(path: str | Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_specs(path: str | Path = DEFAULT_CONFIG) -> dict[str, ModelSpec]:
    raw = load_raw(path)
    items = raw["models"] if isinstance(raw, dict) else raw
    specs = [ModelSpec.model_validate(item) for item in items]
    ids = [s.model_id for s in specs]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        raise ValueError(f"повторяются model_id: {sorted(dup)}")
    return {s.model_id: s for s in specs}


def onnx_file(spec: ModelSpec) -> Path:
    p = Path(spec.onnx_path)
    return p if p.is_absolute() else models_dir() / p
