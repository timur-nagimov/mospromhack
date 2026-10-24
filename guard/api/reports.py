"""HTTP-отчёты по журналу событий.

Периоды полуоткрытые: [from, to), отбор по confirmed_at.
Границы принимаются в epoch-секундах или ISO 8601; даты без зоны — по МСК.
Отсутствующая или пустая граница заменяется соответствующей границей
текущей полной смены: 08–20 либо 20–08 по Москве.

Пресеты «сутки» и «7 дней» заканчиваются в момент запроса.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Literal

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from guard.auth.deps import require
from guard.auth.roles import ANALYTICS_VIEW
from guard.reports.shift_report import (
    ReportData,
    build_report,
    render_html,
    render_markdown,
)
from guard.rules.geometry import LOCAL_TZ

# отчёты — часть аналитики (docs/ROLES.md): право на весь роутер
router = APIRouter(prefix="/api/reports", tags=["reports"], dependencies=[require(ANALYTICS_VIEW)])


class ReportPreset(BaseModel):
    id: Literal["current_shift", "previous_shift", "day", "week"]
    label: str
    ts_from: float
    ts_to: float
    tz: str


def _now() -> datetime:
    return datetime.now(LOCAL_TZ)


def _shift_bounds(now: datetime) -> tuple[datetime, datetime]:
    local = now.astimezone(LOCAL_TZ)
    start = local.replace(hour=8, minute=0, second=0, microsecond=0)
    if local.hour < 8:
        start -= timedelta(hours=12)
    elif local.hour >= 20:
        start += timedelta(hours=12)
    return start, start + timedelta(hours=12)


def _parse_bound(value: str | None, default: float, name: str) -> float:
    if value is None or not value.strip():
        return default
    value = value.strip()
    try:
        try:
            result = float(value)
        except ValueError:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=LOCAL_TZ)
            result = parsed.timestamp()
        if not math.isfinite(result):
            raise ValueError("non-finite timestamp")
        datetime.fromtimestamp(result, LOCAL_TZ)
    except (ValueError, OverflowError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"{name}: expected finite epoch seconds or valid ISO 8601",
        ) from exc
    return result


def _report_bounds(from_: str | None, to: str | None) -> tuple[float, float]:
    shift_start, shift_end = _shift_bounds(_now())
    start = _parse_bound(from_, shift_start.timestamp(), "from")
    end = _parse_bound(to, shift_end.timestamp(), "to")
    if end <= start:
        raise HTTPException(
            status_code=422,
            detail="to must be later than from",
        )

    previous_start = start - (end - start)
    try:
        if not math.isfinite(previous_start):
            raise ValueError("non-finite previous period")
        datetime.fromtimestamp(previous_start, LOCAL_TZ)
    except (ValueError, OverflowError, OSError) as exc:
        raise HTTPException(
            status_code=422,
            detail="previous comparison period is outside the supported date range",
        ) from exc
    return start, end


@router.get(
    "/shift",
    response_model=ReportData,
    responses={
        200: {
            "description": "Отчёт в JSON, печатном HTML или Markdown.",
            "content": {
                "text/html": {"schema": {"type": "string"}},
                "text/markdown": {"schema": {"type": "string"}},
            },
        },
    },
)
def shift_report(
    request: Request,
    from_: str | None = Query(
        None,
        alias="from",
        description="Начало включительно: epoch или ISO 8601; по умолчанию начало текущей смены.",
    ),
    to: str | None = Query(
        None,
        description="Конец исключительно: epoch или ISO 8601; по умолчанию конец текущей смены.",
    ),
    format: Literal["json", "html", "md"] = Query("json"),
) -> ReportData | Response:
    """Построить отчёт по тому же журналу, который обслуживает /api/events."""
    start, end = _report_bounds(from_, to)
    report = build_report(
        request.app.state.engine.journal,
        start,
        end,
        tz=str(LOCAL_TZ),
    )
    if format == "html":
        return HTMLResponse(render_html(report))
    if format == "md":
        return Response(render_markdown(report), media_type="text/markdown")
    return report


@router.get("/presets", response_model=list[ReportPreset])
def report_presets() -> list[ReportPreset]:
    """Готовые периоды; ts_from/ts_to можно передать как from/to в /shift."""
    now = _now().astimezone(LOCAL_TZ)
    start, end = _shift_bounds(now)
    periods = [
        ("current_shift", "Текущая смена", start, end),
        ("previous_shift", "Прошлая смена", start - timedelta(hours=12), start),
        ("day", "Сутки", now - timedelta(days=1), now),
        ("week", "7 дней", now - timedelta(days=7), now),
    ]
    return [
        ReportPreset(
            id=key,
            label=label,
            ts_from=period_start.timestamp(),
            ts_to=period_end.timestamp(),
            tz=str(LOCAL_TZ),
        )
        for key, label, period_start, period_end in periods
    ]
