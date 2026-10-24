"""Справочник сотрудников и ручные назначения виновников в том же SQLite, что и события.

Связи хранятся отдельно от машинного Event: обновление детекции или replay с edge
не меняет результат разбора оператором. Все операции используют блокировку Journal.
"""
from __future__ import annotations

from collections import Counter
import csv
from datetime import datetime
from io import StringIO
import sqlite3
import threading
import time
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator

from guard.rules.geometry import LOCAL_TZ


class EmployeeInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    full_name: str = Field(min_length=1, max_length=200)
    personnel_number: str | None = Field(default=None, max_length=64)
    department: str = Field(default="", max_length=200)
    position: str = Field(default="", max_length=200)

    @field_validator("personnel_number")
    @classmethod
    def empty_number(cls, value: str | None) -> str | None:
        return value or None


class Employee(EmployeeInput):
    id: str
    created_at: float


class UnknownEmployee(ValueError):
    pass


class EmployeeExists(ValueError):
    pass


class EmployeeStore:
    def __init__(self, connection: sqlite3.Connection, lock: threading.Lock) -> None:
        self._conn = connection
        self._lock = lock
        # Вызывается внутри транзакции инициализации Journal, в том числе для старых БД.
        connection.execute("""CREATE TABLE IF NOT EXISTS employees (
            id TEXT PRIMARY KEY NOT NULL, full_name TEXT NOT NULL,
            personnel_number TEXT UNIQUE, department TEXT NOT NULL,
            position TEXT NOT NULL, created_at REAL NOT NULL
        )""")
        connection.execute("""CREATE TABLE IF NOT EXISTS event_employees (
            event_id TEXT NOT NULL REFERENCES events(id),
            employee_id TEXT NOT NULL REFERENCES employees(id),
            assigned_at REAL NOT NULL,
            PRIMARY KEY (event_id, employee_id)
        )""")
        connection.execute("CREATE INDEX IF NOT EXISTS idx_event_employees_person ON event_employees(employee_id, event_id)")

    def list(self) -> list[dict[str, Any]]:
        """Весь справочник, без скрытого лимита или отбора по наличию инцидентов."""
        with self._lock:
            rows = self._conn.execute("SELECT * FROM employees ORDER BY full_name COLLATE NOCASE, id").fetchall()
        return [dict(row) for row in rows]

    def create(self, value: EmployeeInput) -> dict[str, Any]:
        employee = Employee(**value.model_dump(), id=uuid4().hex, created_at=time.time()).model_dump()
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "INSERT INTO employees (id, full_name, personnel_number, department, position, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    tuple(employee[key] for key in ("id", "full_name", "personnel_number", "department", "position", "created_at")),
                )
        except sqlite3.IntegrityError as exc:
            raise EmployeeExists("Сотрудник с таким табельным номером уже существует") from exc
        return employee

    def for_event(self, event_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if not self._conn.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone():
                raise KeyError(event_id)
            return self._for_event_unlocked(event_id)

    def _for_event_unlocked(self, event_id: str) -> list[dict[str, Any]]:
        rows = self._conn.execute(
            "SELECT p.* FROM employees p JOIN event_employees a ON a.employee_id = p.id "
            "WHERE a.event_id = ? ORDER BY p.full_name COLLATE NOCASE, p.id", (event_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def assign(self, event_id: str, employee_ids: list[str]) -> list[dict[str, Any]]:
        if len(employee_ids) != len(set(employee_ids)):
            raise ValueError("Список сотрудников содержит повторяющиеся идентификаторы")
        with self._lock, self._conn:
            self._conn.execute("BEGIN IMMEDIATE")
            if not self._conn.execute("SELECT 1 FROM events WHERE id = ?", (event_id,)).fetchone():
                raise KeyError(event_id)
            # Сначала проверяем весь список; ошибочный запрос не стирает прежнее назначение.
            for employee_id in employee_ids:
                if not self._conn.execute("SELECT 1 FROM employees WHERE id = ?", (employee_id,)).fetchone():
                    raise UnknownEmployee(f"Сотрудник не найден: {employee_id}")
            self._conn.execute("DELETE FROM event_employees WHERE event_id = ?", (event_id,))
            self._conn.executemany(
                "INSERT INTO event_employees (event_id, employee_id, assigned_at) VALUES (?, ?, ?)",
                [(event_id, employee_id, time.time()) for employee_id in employee_ids],
            )
            return self._for_event_unlocked(event_id)

    def report(self, employee_id: str, start: float | None = None, end: float | None = None) -> dict[str, Any]:
        clauses = ["a.employee_id = ?", "e.status = 'confirmed'"]
        parameters: list[Any] = [employee_id]
        if start is not None:
            clauses.append("e.confirmed_at >= ?")
            parameters.append(start)
        if end is not None:
            clauses.append("e.confirmed_at <= ?")
            parameters.append(end)
        with self._lock, self._conn:
            self._conn.execute("BEGIN")
            employee = self._conn.execute("SELECT * FROM employees WHERE id = ?", (employee_id,)).fetchone()
            if employee is None:
                raise UnknownEmployee("Сотрудник не найден")
            rows = self._conn.execute(
                "SELECT e.id, e.confirmed_at, e.started_at, e.ended_at, e.type, e.severity, e.status, "
                "e.camera_id, e.site, e.zone_id, e.operator_note FROM events e "
                "JOIN event_employees a ON a.event_id = e.id WHERE " + " AND ".join(clauses) +
                " ORDER BY e.confirmed_at DESC, e.id DESC", parameters,
            ).fetchall()
        items = [dict(row) for row in rows]
        statuses = Counter(item["status"] for item in items)
        types = Counter(item["type"] for item in items)
        return {"employee": dict(employee), "from": start, "to": end, "total": len(items),
                "statuses": {status: statuses[status] for status in ("new", "confirmed", "forwarded", "false")},
                "types": [{"key": key, "count": count} for key, count in sorted(types.items())], "items": items}


def report_csv(report: dict[str, Any], titles: dict[str, str]) -> str:
    """Полная выгрузка периода, UTF-8 BOM, Excel «;», даты по Москве."""
    def safe(value: Any) -> str:
        text = "" if value is None else str(value)
        return "'" + text if text.lstrip().startswith(("=", "+", "-", "@")) or text.startswith(("\t", "\r", "\n")) else text

    stream = StringIO(newline="")
    writer = csv.writer(stream, delimiter=";", lineterminator="\r\n")
    writer.writerow(["ID сотрудника", "ФИО", "Табельный номер", "Подразделение", "Должность", "ID события",
                     "Оповещение (МСК)", "Тип (код)", "Тип нарушения", "Статус", "Важность", "Камера (ID)",
                     "Участок", "Зона", "Комментарий оператора"])
    employee = report["employee"]
    statuses = {"new": "Новое", "confirmed": "Подтверждено", "forwarded": "Передано мастеру", "false": "Ложное"}
    severity = {"critical": "Критическая", "high": "Высокая", "medium": "Средняя", "low": "Низкая"}
    for event in report["items"]:
        writer.writerow([safe(value) for value in [employee["id"], employee["full_name"], employee["personnel_number"],
            employee["department"], employee["position"], event["id"],
            datetime.fromtimestamp(event["confirmed_at"], LOCAL_TZ).strftime("%d.%m.%Y %H:%M:%S"),
            event["type"], titles.get(event["type"], event["type"]), statuses.get(event["status"], event["status"]),
            severity.get(event["severity"], event["severity"]), event["camera_id"], event["site"], event["zone_id"], event["operator_note"]]])
    return "\ufeff" + stream.getvalue()
