"""Справочник и разбор инцидентов в режиме локального переключателя ролей.

X-Guard-Role — выбранная демо-роль, не удостоверенная личность пользователя.
Проверки новых операций выполняются и в API; полноценный вход — отдельная задача.
"""
from __future__ import annotations

import math
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict

from guard.employees import Employee, EmployeeExists, EmployeeInput, EmployeeStore, UnknownEmployee, report_csv
from guard.rules.base import available_rules, load_plugins

router = APIRouter(tags=["employees"])
ROLES = {"worker", "operator", "safety", "admin"}


def selected_role(request: Request) -> str:
    # При включённом входе роль — из сессии (guard/auth), заголовок не доверяем
    auth = getattr(request.app.state, "auth", None)
    user = request.scope.get("guard.user")
    if auth is not None and getattr(auth, "enabled", False) and user is not None:
        return user.role
    role = request.headers.get("X-Guard-Role", "")
    if role not in ROLES:
        raise HTTPException(403, "Для работы с сотрудниками выберите роль в интерфейсе")
    return role


def _store(request: Request) -> EmployeeStore:
    return request.app.state.engine.journal.employees


class Assignment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    employee_ids: list[str]


@router.get("/api/employees", response_model=list[Employee])
def employees(request: Request, role: str = Depends(selected_role)) -> list[dict[str, Any]]:
    return _store(request).list()


@router.post("/api/employees", status_code=201, response_model=Employee)
def create_employee(body: EmployeeInput, request: Request, role: str = Depends(selected_role)) -> dict[str, Any]:
    if role != "admin":
        raise HTTPException(403, "Добавлять сотрудников может только администратор")
    try:
        return _store(request).create(body)
    except EmployeeExists as exc:
        raise HTTPException(409, str(exc)) from exc


@router.get("/api/events/{event_id}/employees", response_model=list[Employee])
def event_employees(event_id: str, request: Request, role: str = Depends(selected_role)) -> list[dict[str, Any]]:
    try:
        return _store(request).for_event(event_id)
    except KeyError as exc:
        raise HTTPException(404, "Событие не найдено") from exc


@router.put("/api/events/{event_id}/employees", response_model=list[Employee])
def assign_employees(event_id: str, body: Assignment, request: Request, role: str = Depends(selected_role)) -> list[dict[str, Any]]:
    if role not in {"operator", "admin"}:
        raise HTTPException(403, "Указывать виновников может только диспетчер смены или администратор")
    try:
        return _store(request).assign(event_id, body.employee_ids)
    except KeyError as exc:
        raise HTTPException(404, "Событие не найдено") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.get("/api/employees/{employee_id}/report")
def employee_report(
    employee_id: str, request: Request, role: str = Depends(selected_role),
    from_: float | None = Query(None, alias="from"), to: float | None = None,
    format: Literal["json", "csv"] = "json",
) -> Any:
    if any(value is not None and not math.isfinite(value) for value in (from_, to)):
        raise HTTPException(422, "Границы периода должны быть конечными epoch-секундами")
    if from_ is not None and to is not None and to < from_:
        raise HTTPException(422, "Конец периода должен быть не раньше начала")
    try:
        report = _store(request).report(employee_id, from_, to)
    except UnknownEmployee as exc:
        raise HTTPException(404, str(exc)) from exc
    if format == "csv":
        load_plugins()
        titles = {rule["type"]: rule["title"] for rule in available_rules()}
        return Response(report_csv(report, titles), media_type="text/csv; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="employee-{employee_id}.csv"'})
    return report
