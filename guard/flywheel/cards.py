"""Карточки моделей ml/model_cards/<version>.json (docs/FLYWHEEL.md, «Карточки моделей»)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from guard.pipeline.settings import CODE_ROOT

log = logging.getLogger(__name__)

CARDS_DIR = CODE_ROOT / "ml" / "model_cards"
CARD_STATUSES = ("candidate", "deployed", "rejected", "archived")


def load_model_cards(cards_dir: Path | None = None) -> list[dict[str, Any]]:
    """Все карточки, новые сверху. Битые файлы пропускаются с записью в лог."""
    cards_dir = cards_dir or CARDS_DIR
    cards: list[dict[str, Any]] = []
    if not cards_dir.is_dir():
        return cards
    for path in sorted(cards_dir.glob("*.json")):
        try:
            card = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            log.warning("model card %s is not valid JSON, skipped", path.name)
            continue
        if not isinstance(card, dict):
            continue
        card.setdefault("version", path.stem)
        card["card_file"] = f"ml/model_cards/{path.name}"
        cards.append(card)
    cards.sort(key=lambda c: (str(c.get("created_at") or ""), str(c.get("version"))), reverse=True)
    return cards


def models_summary(cards: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Для сводки: выкаченные и кандидаты, коротко."""
    out: dict[str, list[dict[str, Any]]] = {"deployed": [], "candidate": []}
    for card in cards:
        status = card.get("status")
        if status in out:
            out[status].append(
                {
                    "model_id": card.get("model_id"),
                    "version": card.get("version"),
                    "created_at": card.get("created_at"),
                    "status": status,
                    "promote": (card.get("gate") or {}).get("promote"),
                }
            )
    return out
