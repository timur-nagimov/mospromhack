"""Сотрудники, права выбранной демо-роли, ручной разбор и полные персональные отчёты."""
import csv
from io import StringIO
import sqlite3

from fastapi.testclient import TestClient
import pytest

from guard.api.app import create_app
from guard.contracts import Event
from guard.employees import EmployeeInput
from guard.events.journal import Journal
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings


def headers(role="admin"):
    return {"X-Guard-Role": role}


@pytest.fixture
def env(tmp_path):
    settings = make_settings(tmp_path)
    engine = PipelineEngine(settings, backend=FakeBackend())
    with TestClient(create_app(settings, engine)) as client:
        yield client, engine.journal, settings


def add_event(journal, event_id="ev", ts=1000.0, **extra):
    event = Event(id=event_id, camera_id="cam", site="Цех", rule_id="s1", type="zone_intrusion",
                  severity="high", started_at=ts - 2, confirmed_at=ts, bbox=(1, 2, 3, 4), **extra)
    journal.add(event)
    return event


def employee(client, name="Иванов Иван", number=None):
    response = client.post("/api/employees", headers=headers(), json={
        "full_name": name, "personnel_number": number, "department": "Цех 1", "position": "Слесарь"})
    assert response.status_code == 201, response.text
    return response.json()


def assign(client, event_id, ids, role="operator"):
    return client.put(f"/api/events/{event_id}/employees", headers=headers(role), json={"employee_ids": ids})


@pytest.mark.parametrize("role", ["worker", "operator", "safety", "", "administrator"])
def test_only_admin_can_add_and_denial_changes_nothing(env, role):
    client, journal, _ = env
    response = client.post("/api/employees", headers=headers(role), json={"full_name": "Петров Пётр"})
    assert response.status_code == 403
    assert journal.employees.list() == []


def test_validation_optional_numbers_duplicates_and_all_roles_read(env):
    client, _, _ = env
    first = employee(client, "  Иванов Иван  ", "  001  ")
    assert first["full_name"] == "Иванов Иван" and first["personnel_number"] == "001"
    assert client.post("/api/employees", headers=headers(), json={"full_name": "   "}).status_code == 422
    assert client.post("/api/employees", headers=headers(), json={"full_name": "Пётр", "role": "admin"}).status_code == 422
    assert client.post("/api/employees", headers=headers(), json={"full_name": "Другой", "personnel_number": "001"}).status_code == 409
    second = employee(client, "Иванов Иван")
    third = employee(client, "Без инцидентов", "   ")
    assert second["id"] != first["id"] and third["personnel_number"] is None
    for role in ("worker", "operator", "safety", "admin"):
        response = client.get("/api/employees", headers=headers(role))
        assert response.status_code == 200 and len(response.json()) == 3
    assert client.get("/api/employees").status_code == 403


def test_multiple_culprits_replace_clear_atomic_invalid_and_missing_event(env):
    client, journal, _ = env
    add_event(journal)
    first, second = employee(client), employee(client, "Петров Пётр")
    ids = [first["id"], second["id"]]
    response = assign(client, "ev", ids)
    assert response.status_code == 200
    assert {item["id"] for item in response.json()} == set(ids)
    assert assign(client, "ev", [first["id"], "unknown"]).status_code == 422
    assert assign(client, "ev", [first["id"], first["id"]]).status_code == 422
    assert {item["id"] for item in client.get("/api/events/ev/employees", headers=headers("worker")).json()} == set(ids)
    assert assign(client, "missing", ids).status_code == 404
    assert client.get("/api/events/missing/employees", headers=headers()).status_code == 404
    assert assign(client, "ev", [second["id"]], "admin").json() == [second]
    assert assign(client, "ev", [], "admin").json() == []


@pytest.mark.parametrize("role", ["worker", "safety", "", "operator-admin"])
def test_assignment_denied_for_other_roles(env, role):
    client, journal, _ = env
    add_event(journal)
    person = employee(client)
    assert assign(client, "ev", [person["id"]], role).status_code == 403
    assert journal.employees.for_event("ev") == []


def test_confirmed_period_summary_multiple_people_and_complete_excel_csv(env):
    client, journal, _ = env
    first = employee(client, '=HYPERLINK("x")', "001")
    second = employee(client, "Без событий", "002")
    for index, status in enumerate(("new", "confirmed", "forwarded", "false")):
        add_event(journal, f"e{index}", 1000.0 + index, status=status, operator_note="@formula;\nкомментарий")
        assert assign(client, f"e{index}", [first["id"]]).status_code == 200
    add_event(journal, "unassigned", 1001.0, status="confirmed")
    path = f'/api/employees/{first["id"]}/report'
    response = client.get(path, headers=headers("worker"), params={"from": 1001, "to": 1003})
    report = response.json()
    assert response.status_code == 200 and report["total"] == 1
    assert report["statuses"] == {"new": 0, "confirmed": 1, "forwarded": 0, "false": 0}
    assert [item["id"] for item in report["items"]] == ["e1"]
    assert report["types"] == [{"key": "zone_intrusion", "count": 1}]
    assert client.get(path, headers=headers(), params={"from": 1002, "to": 1003}).json()["total"] == 0
    assert client.get(f'/api/employees/{second["id"]}/report', headers=headers()).json()["total"] == 0
    exported = client.get(path, headers=headers(), params={"format": "csv", "from": 1001, "to": 1003})
    assert exported.status_code == 200 and exported.content.startswith(b"\xef\xbb\xbf")
    rows = list(csv.reader(StringIO(exported.content.decode("utf-8-sig")), delimiter=";"))
    assert len(rows) == 2 and rows[0][6] == "Оповещение (МСК)"
    assert all(row[1].startswith("'=") and row[-1].startswith("'@") for row in rows[1:])
    assert {row[9] for row in rows[1:]} == {"Подтверждено"}
    for params in ({"from": "nan"}, {"to": "inf"}, {"from": 1002, "to": 1001}):
        assert client.get(path, headers=headers(), params=params).status_code == 422
    assert client.get('/api/employees/unknown/report', headers=headers()).status_code == 404


def test_no_employee_or_report_limit_and_no_double_counting(env):
    client, journal, _ = env
    main = employee(client)
    other = employee(client, "Второй виновник")
    for index in range(1003):
        journal.employees.create(EmployeeInput(full_name=f"Сотрудник {index}"))
        add_event(journal, f"e{index}", float(index + 1), status="confirmed")
        journal.employees.assign(f"e{index}", [main["id"], other["id"]])
    add_event(journal, "false-positive", 1004.0, status="false")
    journal.employees.assign("false-positive", [main["id"], other["id"]])
    assert len(client.get('/api/employees', headers=headers()).json()) == 1005
    for person in (main, other):
        report = client.get(f'/api/employees/{person["id"]}/report', headers=headers()).json()
        assert report["total"] == len(report["items"]) == report["statuses"]["confirmed"] == 1003
        exported = client.get(f'/api/employees/{person["id"]}/report?format=csv', headers=headers())
        assert len(list(csv.reader(StringIO(exported.content.decode("utf-8-sig")), delimiter=";"))) == 1004
        analytics = client.get('/api/analytics', params={"employee_id": person["id"]}).json()
        assert sum(item["count"] for item in analytics["types"]) == 1003
        assert analytics["statusTypes"]["false"] == []


def test_employee_selection_filters_every_analytics_group_to_confirmed_only(env):
    client, journal, _ = env
    first, second, clean = employee(client), employee(client, "Петров"), employee(client, "Без нарушений")
    events = [
        ("own-confirmed", "confirmed", [first["id"]], "zone_intrusion", "Цех 1"),
        ("own-false", "false", [first["id"]], "ppe_missing", "Исключённый участок"),
        ("own-new", "new", [first["id"]], "fall_immobile", "Исключённый участок"),
        ("own-forwarded", "forwarded", [first["id"]], "vehicle_proximity", "Исключённый участок"),
        ("shared-confirmed", "confirmed", [first["id"], second["id"]], "zone_intrusion", "Цех 2"),
        ("other-confirmed", "confirmed", [second["id"]], "running", "Другой сотрудник"),
        ("unassigned", "confirmed", [], "running", "Без назначений"),
    ]
    for index, (event_id, status, ids, kind, site) in enumerate(events):
        data = dict(id=event_id, camera_id=f"cam{index}", site=site, rule_id="rule", type=kind,
                    severity="critical", status=status, started_at=999.0 + index,
                    confirmed_at=1000.0 + index, bbox=(1, 2, 3, 4))
        journal.add(Event(**data))
        journal.employees.assign(event_id, ids)
    query = {"employee_id": first["id"]}
    response = client.get('/api/analytics', params=query)
    assert response.status_code == 200
    analytics = response.json()
    assert analytics["types"] == [{"key": "zone_intrusion", "count": 2}]
    assert analytics["sites"] == [{"key": "Цех 1", "count": 1}, {"key": "Цех 2", "count": 1}]
    assert {item["key"] for item in analytics["cameras"]} == {"cam0", "cam4"}
    for group in ("types", "sites", "cameras", "days", "weekdayHours", "criticalTypes"):
        assert sum(item["count"] for item in analytics[group]) == 2
    assert analytics["statusTypes"] == {"confirmed": analytics["types"], "new": [], "false": [], "forwarded": []}
    assert sum(item["count"] for item in client.get('/api/analytics').json()["types"]) == len(events)
    period = client.get('/api/analytics', params={**query, "from": 1001, "to": 1004}).json()
    assert period["sites"] == [{"key": "Цех 2", "count": 1}]
    empty = client.get('/api/analytics', params={"employee_id": clean["id"]}).json()
    assert empty["types"] == [] and empty["cameras"] == [] and empty["days"] == []
    assert client.get('/api/analytics', params={"employee_id": "unknown"}).status_code == 404
    client.patch('/api/events/own-confirmed', json={"status": "false"})
    assert sum(item["count"] for item in client.get('/api/analytics', params=query).json()["types"]) == 1
    assert journal.employees.report(first["id"])["total"] == 1
    client.patch('/api/events/own-new', json={"status": "confirmed"})
    updated = client.get('/api/analytics', params=query).json()
    assert {item["key"] for item in updated["types"]} == {"zone_intrusion", "fall_immobile"}
    assert journal.employees.report(first["id"])["total"] == 2


def test_assignments_survive_machine_updates_reopen_and_existing_database(tmp_path):
    path = tmp_path / "guard.db"
    with Journal(path) as journal:
        add_event(journal)
    # Старый журнал без новых таблиц автоматически получает справочник без потери событий.
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE event_employees")
        connection.execute("DROP TABLE employees")
    with Journal(path) as journal:
        person = journal.employees.create(EmployeeInput(full_name="Иванов Иван"))
        journal.employees.assign("ev", [person["id"]])
        journal.update("ev", ended_at=1007.0, details_patch={"reopened_count": 1})
        journal.update("ev", status="confirmed", operator_note="Разобрано")
    with Journal(path) as journal:
        assert journal.employees.list() == [person]
        assert journal.employees.for_event("ev") == [person]
        report = journal.employees.report(person["id"])
        assert report["total"] == 1 and report["statuses"]["confirmed"] == 1
        assert report["items"][0]["operator_note"] == "Разобрано"


def test_gateway_replication_and_replay_preserve_culprits(tmp_path):
    with Journal(tmp_path / "edge.db", replication_edge_id="edge-1") as edge, Journal(tmp_path / "gateway.db") as gateway:
        add_event(edge)
        gateway.apply_remote_event(edge.pending_messages()[0], expected_edge_id="edge-1")
        person = gateway.employees.create(EmployeeInput(full_name="Виновник на gateway"))
        gateway.employees.assign("ev", [person["id"]])
        gateway.update("ev", status="confirmed")
        edge.update("ev", ended_at=1010.0)
        update = edge.pending_messages()[-1]
        gateway.apply_remote_event(update, expected_edge_id="edge-1")
        gateway.apply_remote_event(update, expected_edge_id="edge-1")
        assert gateway.employees.for_event("ev") == [person]
        assert gateway.employees.report(person["id"])["statuses"]["confirmed"] == 1


def test_demo_seed_is_idempotent_and_preserves_existing_staff(env):
    from tools.seed_demo_employees import seed_demo_employees
    client, journal, _ = env
    original = employee(client, "Добавленный администратором сотрудник", "STAFF-001")
    first = seed_demo_employees(journal)
    assert first == {"catalog": 10, "added": 10, "skipped": 0, "total": 11}
    snapshot = journal.employees.list()
    assert seed_demo_employees(journal) == {"catalog": 10, "added": 0, "skipped": 10, "total": 11}
    assert journal.employees.list() == snapshot
    assert next(person for person in snapshot if person["id"] == original["id"]) == original
