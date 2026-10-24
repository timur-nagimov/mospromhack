"""Журнал действий: GET /api/audit (право audit.view, docs/ROLES.md)."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from guard.auth.deps import require
from guard.auth.roles import AUDIT_VIEW
from guard.rules.geometry import LOCAL_TZ

router = APIRouter(prefix="/api/audit", tags=["audit"])


def _parse_time(value: str | None, name: str) -> float | None:
    """from/to: epoch-секунды или ISO 8601 (без зоны — по Москве), как в /api/events."""
    if value is None or value == "":
        return None
    try:
        ts = float(value)
    except ValueError:
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise HTTPException(422, f"{name}: expected epoch seconds or ISO 8601") from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=LOCAL_TZ)
        return dt.timestamp()
    if not math.isfinite(ts):
        raise HTTPException(422, f"{name}: expected finite epoch seconds")
    return ts


@router.get("", dependencies=[require(AUDIT_VIEW)])
def list_audit(
    request: Request,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    username: str | None = None,
    action: str | None = Query(None, description="через запятую; «auth.*» — все действия группы"),
    from_: str | None = Query(None, alias="from"),
    to: str | None = None,
) -> dict[str, Any]:
    """Записи от новых к старым: {items: [{id, ts, username, role, action, target, details, ip}], total}."""
    items, total = request.app.state.auth.store.query_audit(
        limit=limit,
        offset=offset,
        username=username,
        action=action,
        ts_from=_parse_time(from_, "from"),
        ts_to=_parse_time(to, "to"),
    )
    return {"items": items, "total": total}
