"""Отложенная выборка: кадры роликов офлайн-оценки в обучение не попадают (docs/FLYWHEEL.md, «Принципы»).

Ролики берутся из ml/eval/scenarios.yaml и из наборов ml/eval/scenarios_<имя>.yaml: это тоже оценочные
ролики. Камера считается отложенной, если в пути её источника встречается имя ролика без расширения
(s1_01, clean_02) или id сценария. Так ловятся и data/videos/s1_01.mp4, и субпоток стенда
data_360/videos/s1_01.mp4, и каталоги кадров images/s2 (кадры s2_01 и s2_02). Ошибка в сторону
holdout = 1 безопасна: такой кадр просто не уйдёт в обучение.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import yaml

from guard.pipeline.settings import CODE_ROOT

log = logging.getLogger(__name__)

EVAL_DIR = CODE_ROOT / "ml" / "eval"


def scenario_files(eval_dir: Path = EVAL_DIR) -> list[Path]:
    return sorted(p for p in eval_dir.glob("scenarios*.yaml") if p.is_file())


def load_holdout_names(eval_dir: Path = EVAL_DIR) -> frozenset[str]:
    """Имена роликов (без расширения) и id сценариев из всех scenarios*.yaml."""
    names: set[str] = set()
    for path in scenario_files(eval_dir):
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError):
            log.exception("holdout: cannot read %s", path)
            continue
        for scenario in raw.get("scenarios") or []:
            if not isinstance(scenario, dict):
                continue
            video = scenario.get("video")
            if video:
                names.add(Path(str(video)).stem)
            if scenario.get("id"):
                names.add(str(scenario["id"]))
    return frozenset(n for n in names if n)


@lru_cache(maxsize=4)
def _cached_names(eval_dir: str) -> frozenset[str]:
    return load_holdout_names(Path(eval_dir))


def holdout_names(eval_dir: Path = EVAL_DIR) -> frozenset[str]:
    return _cached_names(str(eval_dir))


def is_holdout_uri(uri: str | None, names: frozenset[str] | None = None) -> bool:
    """Источник камеры — отложенный ролик (по имени файла или каталога в пути)."""
    if not uri:
        return False
    names = holdout_names() if names is None else names
    if not names:
        return False
    parts = [p for p in Path(str(uri)).parts if p not in ("/", "")]
    stems = {Path(p).stem for p in parts} | set(parts)
    if stems & names:
        return True
    # каталоги кадров вида s2_01_frames, s2_01-360p: имя ролика — префикс части пути;
    # каталог серии вида images/s2 (кадры s2_01 и s2_02): часть пути — префикс имени ролика
    return any(
        part.startswith(f"{name}_") or part.startswith(f"{name}-") or name.startswith(f"{part}_")
        for part in stems
        for name in names
    )
