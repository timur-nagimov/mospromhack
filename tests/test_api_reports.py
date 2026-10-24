"""Отчёты через реальное приложение и временный журнал SQLite."""

from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from guard.api import reports
from guard.api.app import create_app
from guard.contracts import Event, EventStatus, Severity
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings
from guard.reports.shift_report import ReportData
from guard.rules.geometry import LOCAL_TZ


def _ts(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=LOCAL_TZ)
    return parsed.timestamp()


START = _ts("2026-10-24T08:00:00+03:00")
END = _ts("2026-10-24T20:00:00+03:00")
NOW = datetime.fromisoformat("2026-10-24T12:34:56+03:00")


def _add_event(journal, event_id, timestamp, **extra):
    values = {
        "id": event_id,
        "camera_id": "cam1",
        "site": "Цех 1",
        "zone_id": "z1",
        "rule_id": "s1",
        "type": "zone_intrusion",
        "severity": Severity.high,
        "track_id": 1,
        "started_at": timestamp - 5,
        "confirmed_at": timestamp,
        "bbox": (1, 2, 3, 4),
    }
    values.update(extra)
    event = Event(**values)
    journal.add(event)
    return event


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(reports, "_now", lambda: NOW)
    settings = make_settings(tmp_path)
    engine = PipelineEngine(settings, backend=FakeBackend())
    _add_event(engine.journal, "previous", START - 1)
    _add_event(
        engine.journal,
        "critical-start",
        START,
        severity=Severity.critical,
        status=EventStatus.confirmed,
    )
    _add_event(
        engine.journal,
        "last",
        END - 1,
        status=EventStatus.false,
    )
    _add_event(engine.journal, "excluded-end", END)
    app = create_app(settings, engine=engine)
    with TestClient(app) as client:
        yield client, engine


@pytest.mark.parametrize(
    "format_, media_type",
    [
        ("json", "application/json"),
        ("html", "text/html"),
        ("md", "text/markdown"),
    ],
)
def test_three_formats(env, format_, media_type):
    client, _ = env
    response = client.get(
        "/api/reports/shift",
        params={"from": START, "to": END, "format": format_},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].split(";")[0] == media_type
    if format_ == "json":
        report = ReportData.model_validate(response.json())
        assert report.total == 2
        assert [event.id for event in report.critical_events] == ["critical-start"]
    else:
        assert "charset=utf-8" in response.headers["content-type"]
        assert "Автоматические выводы" in response.text
        assert "Всего событий — 2" in response.text
        assert "critical-start" in response.text
        assert "Europe/Moscow" in response.text
        if format_ == "html":
            assert response.text.startswith("<!doctype html>")
            assert "@media print" in response.text
        else:
            assert response.text.startswith("# Зазеркалье")
            assert "## Итоги" in response.text


def test_json_aggregation_boundaries_and_shared_journal(env):
    client, engine = env
    params = {"from": START, "to": END}
    response = client.get("/api/reports/shift", params=params)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["total"] == 2
    assert data["ts_from"] == START
    assert data["ts_to"] == END
    assert data["previous_ts_from"] == START - 12 * 3600
    assert data["previous_ts_to"] == START
    assert data["tz"] == "Europe/Moscow"
    assert data["by_status"] == {
        "new": 0,
        "confirmed": 1,
        "false": 1,
        "forwarded": 0,
    }
    assert data["by_severity"]["critical"] == 1
    assert data["by_severity"]["high"] == 1
    assert data["reviewed_count"] == 2
    assert data["false_count"] == 1
    assert data["false_positive_rate"] == pytest.approx(0.5)
    assert data["by_camera"] == {"cam1": 2}
    assert data["by_site"] == {"Цех 1": 2}
    assert data["by_shift"] == {"День": 2, "Ночь": 0}
    assert data["by_hour"]["8"] == 1
    assert data["by_hour"]["19"] == 1
    assert data["hotspots"][0]["count"] == 2
    assert data["hotspots"][0]["share"] == pytest.approx(1.0)
    assert data["trend"] == {
        "current_count": 2,
        "previous_count": 1,
        "change_percent": 100.0,
        "direction": "increase",
    }
    assert data["conclusions"]

    # /api/events сохраняет включительную верхнюю границу журнала.
    events_response = client.get("/api/events", params=params)
    assert events_response.status_code == 200, events_response.text
    assert {event["id"] for event in events_response.json()["items"]} == {
        "critical-start",
        "last",
        "excluded-end",
    }

    # Добавление в журнал движка сразу видно обоим маршрутам.
    _add_event(engine.journal, "added", START + 60)
    assert client.get("/api/events/added").status_code == 200
    updated = client.get("/api/reports/shift", params=params)
    assert updated.status_code == 200, updated.text
    assert updated.json()["total"] == 3


@pytest.mark.parametrize(
    "from_, to",
    [
        ("2026-10-24T08:00:00", "2026-10-24T20:00:00"),
        ("2026-10-24T08:00:00+03:00", "2026-10-24T20:00:00+03:00"),
        ("2026-10-24T05:00:00Z", "2026-10-24T17:00:00Z"),
    ],
)
def test_iso_dates(env, from_, to):
    client, _ = env
    response = client.get("/api/reports/shift", params={"from": from_, "to": to})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ts_from"] == START
    assert data["ts_to"] == END
    assert data["total"] == 2


@pytest.mark.parametrize(
    "now, expected_start, expected_end",
    [
        (
            "2026-10-24T00:00:00+03:00",
            "2026-10-23T20:00:00+03:00",
            "2026-10-24T08:00:00+03:00",
        ),
        (
            "2026-10-24T07:59:59.999999+03:00",
            "2026-10-23T20:00:00+03:00",
            "2026-10-24T08:00:00+03:00",
        ),
        (
            "2026-10-24T08:00:00+03:00",
            "2026-10-24T08:00:00+03:00",
            "2026-10-24T20:00:00+03:00",
        ),
        (
            "2026-10-24T19:59:59.999999+03:00",
            "2026-10-24T08:00:00+03:00",
            "2026-10-24T20:00:00+03:00",
        ),
        (
            "2026-10-24T20:00:00+03:00",
            "2026-10-24T20:00:00+03:00",
            "2026-10-25T08:00:00+03:00",
        ),
        (
            "2026-10-24T23:59:59+03:00",
            "2026-10-24T20:00:00+03:00",
            "2026-10-25T08:00:00+03:00",
        ),
        (
            "2026-10-01T07:00:00+03:00",
            "2026-09-30T20:00:00+03:00",
            "2026-10-01T08:00:00+03:00",
        ),
        (
            "2026-10-24T05:00:00+00:00",
            "2026-10-24T08:00:00+03:00",
            "2026-10-24T20:00:00+03:00",
        ),
    ],
)
def test_current_shift_defaults(env, monkeypatch, now, expected_start, expected_end):
    client, _ = env
    frozen = datetime.fromisoformat(now)
    monkeypatch.setattr(reports, "_now", lambda: frozen)
    response = client.get("/api/reports/shift")
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ts_from"] == _ts(expected_start)
    assert data["ts_to"] == _ts(expected_end)
    assert data["ts_to"] - data["ts_from"] == 12 * 3600


@pytest.mark.parametrize(
    "params, expected_start, expected_end, total",
    [
        ({}, START, END, 2),
        ({"from": "", "to": ""}, START, END, 2),
        ({"from": START + 2 * 3600}, START + 2 * 3600, END, 1),
        ({"to": START + 2 * 3600}, START, START + 2 * 3600, 1),
    ],
)
def test_missing_and_empty_bounds(env, params, expected_start, expected_end, total):
    client, _ = env
    response = client.get("/api/reports/shift", params=params)
    assert response.status_code == 200, response.text
    data = response.json()
    assert data["ts_from"] == expected_start
    assert data["ts_to"] == expected_end
    assert data["total"] == total


def test_presets(env, monkeypatch):
    client, _ = env
    frozen = datetime.fromisoformat("2026-10-01T02:15:30+03:00")
    monkeypatch.setattr(reports, "_now", lambda: frozen)
    response = client.get("/api/reports/presets")
    assert response.status_code == 200, response.text
    presets = response.json()
    assert [item["id"] for item in presets] == [
        "current_shift",
        "previous_shift",
        "day",
        "week",
    ]
    assert [item["label"] for item in presets] == [
        "Текущая смена",
        "Прошлая смена",
        "Сутки",
        "7 дней",
    ]
    by_id = {item["id"]: item for item in presets}
    assert by_id["current_shift"]["ts_from"] == _ts("2026-09-30T20:00:00+03:00")
    assert by_id["current_shift"]["ts_to"] == _ts("2026-10-01T08:00:00+03:00")
    assert by_id["previous_shift"]["ts_from"] == _ts("2026-09-30T08:00:00+03:00")
    assert by_id["previous_shift"]["ts_to"] == by_id["current_shift"]["ts_from"]
    assert by_id["day"]["ts_to"] == frozen.timestamp()
    assert by_id["day"]["ts_from"] == frozen.timestamp() - 24 * 3600
    assert by_id["week"]["ts_to"] == frozen.timestamp()
    assert by_id["week"]["ts_from"] == frozen.timestamp() - 7 * 24 * 3600

    default_response = client.get("/api/reports/shift")
    assert default_response.status_code == 200, default_response.text
    default = default_response.json()
    assert default["ts_from"] == by_id["current_shift"]["ts_from"]
    assert default["ts_to"] == by_id["current_shift"]["ts_to"]

    for preset in presets:
        assert preset["tz"] == "Europe/Moscow"
        report_response = client.get(
            "/api/reports/shift",
            params={"from": preset["ts_from"], "to": preset["ts_to"]},
        )
        assert report_response.status_code == 200, report_response.text
        assert report_response.json()["ts_from"] == preset["ts_from"]
        assert report_response.json()["ts_to"] == preset["ts_to"]


@pytest.mark.parametrize("format_", ["json", "html", "md"])
def test_empty_period(env, format_):
    client, _ = env
    response = client.get(
        "/api/reports/shift",
        params={
            "from": "2026-11-10T08:00:00+03:00",
            "to": "2026-11-10T20:00:00+03:00",
            "format": format_,
        },
    )
    assert response.status_code == 200, response.text
    if format_ == "json":
        report = ReportData.model_validate(response.json())
        assert report.total == 0
        assert report.hotspots == []
        assert report.critical_events == []
        assert report.false_positive_rate is None
        assert report.trend.current_count == 0
        assert report.trend.previous_count == 0
        assert report.trend.direction == "unchanged"
        assert report.trend.change_percent == 0
        assert report.conclusions
    else:
        assert "Всего событий — 0" in response.text
        assert "Нет данных" in response.text


@pytest.mark.parametrize(
    "overrides",
    [
        {"from": "yesterday"},
        {"to": "tomorrow"},
        {"from": "2026-02-30T08:00:00"},
        {"from": "nan"},
        {"to": "NaN"},
        {"from": "-inf"},
        {"to": "inf"},
        {"from": "-1e100"},
        {"to": "1e100"},
        {"from": END},
        {"to": START - 1},
        {"format": "pdf"},
        {"from": "0001-01-02T00:00:00+00:00"},
    ],
)
def test_invalid_parameters_return_422(env, overrides):
    client, _ = env
    response = client.get(
        "/api/reports/shift",
        params={"from": START, "to": END, **overrides},
    )
    assert response.status_code == 422, response.text
    assert response.json()["detail"]


def test_openapi_documents_report_and_formats(env):
    client, _ = env
    response = client.get("/openapi.json")
    assert response.status_code == 200, response.text
    schema = response.json()
    operation = schema["paths"]["/api/reports/shift"]["get"]
    content = operation["responses"]["200"]["content"]
    assert {"application/json", "text/html", "text/markdown"} <= set(content)
    assert content["application/json"]["schema"]["$ref"].endswith("/ReportData")
    assert {item["name"] for item in operation["parameters"]} == {
        "from",
        "to",
        "format",
    }
    assert "/api/reports/presets" in schema["paths"]
