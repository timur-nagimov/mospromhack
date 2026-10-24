"""Явно добавить десять вымышленных сотрудников в выбранный локальный журнал.

python -m tools.seed_demo_employees --db-path data/local-ui/guard.db
Набор общий с автономным мок-интерфейсом. Повторный запуск сохраняет существующие записи.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guard.employees import EmployeeExists, EmployeeInput
from guard.events.journal import Journal

CATALOG = Path(__file__).resolve().parents[1] / "frontend/src/domain/demoEmployees.json"


def seed_demo_employees(journal: Journal) -> dict[str, int]:
    catalog = [EmployeeInput.model_validate(item) for item in json.loads(CATALOG.read_text(encoding="utf-8"))]
    numbers = {employee.personnel_number for employee in catalog}
    if None in numbers or len(numbers) != len(catalog):
        raise ValueError("Демонстрационные табельные номера должны быть непустыми и уникальными")
    existing = {employee["personnel_number"] for employee in journal.employees.list()}
    added = 0
    for employee in catalog:
        if employee.personnel_number in existing:
            continue
        try:
            journal.employees.create(employee)
            added += 1
        except EmployeeExists:
            # Другой процесс мог добавить этот номер после чтения справочника.
            continue
    return {"catalog": len(catalog), "added": added, "skipped": len(catalog) - added,
            "total": len(journal.employees.list())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, required=True, help="SQLite выбранного локального запуска")
    args = parser.parse_args()
    with Journal(args.db_path) as journal:
        result = seed_demo_employees(journal)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
