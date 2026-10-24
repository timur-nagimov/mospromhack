"""Карточки версий моделей: ml/model_cards/<version>.json (схема — docs/FLYWHEEL.md, «Карточки моделей»).

Общий модуль для train, gate и deploy: чтение, запись, поиск по версии.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CARDS_DIR = ROOT / "ml" / "model_cards"
STATUSES = ("candidate", "deployed", "rejected", "archived")
REQUIRED = ("model_id", "version", "parent", "created_at", "task", "classes", "status")


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def card_path(version_or_path: str | Path, cards_dir: Path = CARDS_DIR) -> Path:
    """Версия (forklift-v1) или путь к JSON → путь карточки."""
    p = Path(version_or_path)
    if p.suffix == ".json" or p.exists():
        return p
    return cards_dir / f"{version_or_path}.json"


def load_card(version_or_path: str | Path, cards_dir: Path = CARDS_DIR) -> dict[str, Any]:
    path = card_path(version_or_path, cards_dir)
    card = json.loads(path.read_text(encoding="utf-8"))
    validate_card(card)
    return card


def validate_card(card: dict[str, Any]) -> None:
    missing = [k for k in REQUIRED if k not in card]
    if missing:
        raise ValueError(f"карточка {card.get('version')}: нет полей {missing}")
    if card["status"] not in STATUSES:
        raise ValueError(f"карточка {card['version']}: status {card['status']!r} не из {STATUSES}")


def save_card(card: dict[str, Any], cards_dir: Path = CARDS_DIR) -> Path:
    validate_card(card)
    cards_dir.mkdir(parents=True, exist_ok=True)
    path = cards_dir / f"{card['version']}.json"
    path.write_text(json.dumps(card, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def list_cards(cards_dir: Path = CARDS_DIR) -> list[dict[str, Any]]:
    """Все карточки, новые сверху (как отдаёт /api/flywheel/models)."""
    cards = []
    for path in cards_dir.glob("*.json"):
        try:
            cards.append(load_card(path))
        except (ValueError, json.JSONDecodeError):
            continue
    return sorted(cards, key=lambda c: c.get("created_at", ""), reverse=True)
