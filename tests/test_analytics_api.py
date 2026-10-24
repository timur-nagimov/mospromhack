"""HTTP-статистика на настоящем SQLite, без моделей и внешних сервисов."""

from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from guard.api.app import router
from guard.contracts import Event
import guard.events.journal as journal_module
from guard.events.journal import Journal


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def add_event(journal, event_id, ts, **overrides):
    data = {
        "id": event_id, "camera_id": "cam-1", "site": "Склад",
        "zone_id": "north", "rule_id": "rule-1", "type": "ppe_missing",
        "severity": "high", "status": "new", "started_at": ts - 60,
        "confirmed_at": ts, "bbox": (0, 0, 10, 20),
    }
    data.update(overrides)
    journal.add(Event(**data))


def make_app(journal):
    app = FastAPI()
    app.include_router(router)
    app.state.engine = SimpleNamespace(journal=journal, patch_event=journal.update)
    return app


@pytest.fixture
def api(tmp_path):
    with Journal(tmp_path / "guard.db") as journal:
        with TestClient(make_app(journal)) as client:
            yield client, journal


def items(client, group_by, **filters):
    response = client.get("/api/stats", params={"group_by": group_by, **filters})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body) == {"group_by", "items"}
    assert body["group_by"] == group_by
    return {item["key"]: item["count"] for item in body["items"]}


def counts(items):
    return {item["key"]: item["count"] for item in items}


def analytics(client, **period):
    response = client.get("/api/analytics", params=period)
    assert response.status_code == 200, response.text
    return response.json()


def test_full_period_aggregates_exceed_event_page_limit(api, monkeypatch):
    client, journal = api
    first = timestamp("2026-10-18T21:00:00Z")  # понедельник, 00:00 МСК
    for index in range(5205):
        add_event(
            journal, f"event-{index}", first + index,
            # Чередование типов, статусов и важности создаёт пересечение critical.
            type=("ppe_missing", "fall_immobile", "vehicle_proximity")[index % 3],
            status=("new", "confirmed", "forwarded", "false")[index % 4],
            severity="critical" if index % 5 == 0 else "high",
            site="Склад" if index % 2 == 0 else "Цех",
        )
    add_event(journal, "outside-before", first - 1, type="excluded")
    add_event(journal, "outside-after", first + 5205, type="excluded")
    period = {"from": first, "to": first + 5204}

    page = client.get("/api/events", params={**period, "limit": 1000}).json()
    assert len(page["items"]) == 1000
    assert page["total"] == 5205
    assert client.get("/api/events", params={"limit": 5000}).status_code == 422

    def no_event_loading(*args, **kwargs):
        raise AssertionError("analytics must aggregate in SQLite without loading events")

    monkeypatch.setattr(journal, "query", no_event_loading)
    monkeypatch.setattr(journal_module, "_deserialize", no_event_loading)
    by_type = items(client, "type", **period)
    assert by_type == {"ppe_missing": 1735, "fall_immobile": 1735, "vehicle_proximity": 1735}
    assert items(client, "site", **period) == {"Склад": 2603, "Цех": 2602}
    assert items(client, "weekday_hour", **period) == {"0:0": 3600, "0:1": 1605}
    assert items(client, "day", **period) == {"2026-10-19": 5205}
    assert items(client, "camera", **period) == {"cam-1": 5205}

    statuses = {status: items(client, "type", status=status, **period)
                for status in ("new", "confirmed", "forwarded", "false")}
    assert {kind: sum(counts.get(kind, 0) for counts in statuses.values())
            for kind in by_type} == by_type
    assert sum(statuses["new"].values()) == 1302
    assert sum(statuses["false"].values()) == 1301

    critical_types = items(client, "type", severity="critical", **period)
    critical_count = by_type["fall_immobile"] + by_type["vehicle_proximity"] + critical_types["ppe_missing"]
    expected = sum(index % 3 in (1, 2) or index % 5 == 0 for index in range(5205))
    assert critical_count == expected == 3817
    assert sum(critical_types.values()) == 1041

    # Тот же полный период доступен одним запросом без загрузки событий.
    summary = analytics(client, **period)
    assert counts(summary["types"]) == by_type
    assert counts(summary["cameras"]) == {"cam-1": 5205}
    assert counts(summary["sites"]) == {"Склад": 2603, "Цех": 2602}
    assert counts(summary["days"]) == {"2026-10-19": 5205}
    assert counts(summary["weekdayHours"]) == {"0:0": 3600, "0:1": 1605}
    assert counts(summary["criticalTypes"]) == critical_types
    assert {status: counts(group) for status, group in summary["statusTypes"].items()} == statuses


def test_moscow_midnight_week_boundary_and_inclusive_confirmation_filter(api):
    client, journal = api
    before = timestamp("2026-10-18T20:59:59Z")  # воскресенье, 23:59:59 МСК
    start = timestamp("2026-10-18T21:00:00Z")
    for event_id, ts in (("sunday", before), ("monday", start), ("end", start + 3599)):
        add_event(journal, event_id, ts, started_at=before - 3600)

    assert items(client, "weekday_hour") == {"6:23": 1, "0:0": 2}
    assert items(client, "hour") == {23: 1, 0: 2}
    assert items(client, "day") == {"2026-10-18": 1, "2026-10-19": 2}
    for period in (
        {"from": start, "to": start + 3599},
        {"from": "2026-10-19T00:00:00", "to": "2026-10-19T00:59:59"},
        {"from": "2026-10-18T21:00:00Z", "to": "2026-10-18T21:59:59Z"},
    ):
        assert items(client, "weekday_hour", **period) == {"0:0": 2}
        summary = analytics(client, **period)
        assert counts(summary["weekdayHours"]) == {"0:0": 2}
        assert counts(summary["days"]) == {"2026-10-19": 2}
        page = client.get("/api/events", params=period).json()
        assert page["total"] == 2
        assert {event["id"] for event in page["items"]} == {"monday", "end"}
    assert items(client, "site", **{"from": start, "to": start}) == {"Склад": 1}


def test_stats_filters_match_events_and_preserve_historical_sites(api):
    client, journal = api
    add_event(journal, "a", 100, site="Старый участок", status="confirmed", severity="critical")
    add_event(journal, "b", 200, site="Новый участок", zone_id="south", type="running")
    add_event(journal, "c", 300, camera_id="cam-2", status="false", severity="medium")
    filters = {
        "type": "ppe_missing,running", "camera_id": "cam-1,cam-2",
        "zone_id": "north", "status": "confirmed,false", "severity": "critical,medium",
        "from": 100, "to": 300,
    }
    assert client.get("/api/events", params=filters).json()["total"] == 2
    assert items(client, "type", **filters) == {"ppe_missing": 2}
    assert items(client, "site", **filters) == {"Старый участок": 1, "Склад": 1}
    assert items(client, "site", camera_id="cam-1") == {"Старый участок": 1, "Новый участок": 1}
    assert items(client, "zone", **filters) == {"north": 2}
    assert items(client, "type", status="new,confirmed", severity="medium") == {}
    assert items(client, "site", camera_id="missing") == {}


@pytest.mark.parametrize("value", ["not-a-date", "NaN", "inf", "-inf"])
@pytest.mark.parametrize("boundary", ["from", "to"])
def test_invalid_date_filter_is_visible_in_all_endpoints(api, boundary, value):
    client, _ = api
    for endpoint in ("/api/events", "/api/stats", "/api/analytics"):
        response = client.get(endpoint, params={boundary: value})
        assert response.status_code == 422
        assert boundary in response.json()["detail"]


def test_stats_rejects_unknown_group(api):
    client, _ = api
    assert client.get("/api/stats", params={"group_by": "unknown"}).status_code == 422


@pytest.mark.parametrize(
    ("initial_status", "updated_status"),
    [("new", "confirmed"), ("confirmed", "new")],
)
def test_analytics_keeps_one_snapshot_during_concurrent_patch_and_insert(
    api, tmp_path, monkeypatch, initial_status, updated_status,
):
    client, reader = api
    first = timestamp("2026-10-18T21:00:00Z")
    add_event(reader, "race", first, status=initial_status, severity="critical")
    add_event(reader, "stable", first, type="fall_immobile", status="false", severity="critical")
    original_aggregate = reader._stats_unlocked
    checkpoints = []

    with Journal(tmp_path / "guard.db") as writer, TestClient(make_app(writer)) as writer_client:
        def aggregate_with_external_writes(group_by, ts_from, ts_to, tz, **filters):
            result = original_aggregate(group_by, ts_from, ts_to, tz, **filters)
            if group_by == "type" and not filters and not checkpoints:
                # Типы уже прочитаны. Запись с другого соединения коммитится до
                # подсчёта участков, дат, важности и статусов в этом же ответе.
                add_event(
                    writer, "arrived", first + 3600, type="running", severity="critical",
                    camera_id="cam-2", site="Новый участок", status="confirmed",
                )
                checkpoints.append("insert committed")
            if filters.get("status") == "new" and len(checkpoints) == 1:
                # Точная гонка прежних запросов: new уже прочитан, confirmed — ещё нет.
                response = writer_client.patch("/api/events/race", json={"status": updated_status})
                assert response.status_code == 200, response.text
                assert response.json()["status"] == updated_status
                checkpoints.append("patch committed")
            return result

        monkeypatch.setattr(reader, "_stats_unlocked", aggregate_with_external_writes)
        summary = analytics(client)
        assert checkpoints == ["insert committed", "patch committed"]
        assert counts(summary["types"]) == {"ppe_missing": 1, "fall_immobile": 1}
        assert counts(summary["cameras"]) == {"cam-1": 2}
        assert counts(summary["sites"]) == {"Склад": 2}
        assert counts(summary["days"]) == {"2026-10-19": 2}
        assert counts(summary["weekdayHours"]) == {"0:0": 2}
        assert counts(summary["criticalTypes"]) == counts(summary["types"])
        assert counts(summary["statusTypes"][initial_status]) == {"ppe_missing": 1}
        assert counts(summary["statusTypes"][updated_status]) == {}
        assert counts(summary["statusTypes"]["false"]) == {"fall_immobile": 1}

        by_type = counts(summary["types"])
        status_totals = {
            kind: sum(counts(group).get(kind, 0) for group in summary["statusTypes"].values())
            for kind in by_type
        }
        assert status_totals == by_type  # ни потерь, ни двойного счёта
        assert not reader._conn.in_transaction
        assert writer.get("race").status == updated_status

        # После завершения снимка следующий запрос видит обе закоммиченные записи.
        monkeypatch.setattr(reader, "_stats_unlocked", original_aggregate)
        fresh = analytics(client)
        assert sum(item["count"] for item in fresh["types"]) == 3
        assert counts(fresh["statusTypes"][updated_status])["ppe_missing"] == 1
        assert counts(fresh["sites"])["Новый участок"] == 1


def test_empty_analytics_has_all_groups_and_statuses(api):
    client, _ = api
    assert analytics(client) == {
        "types": [], "cameras": [], "sites": [], "days": [], "weekdayHours": [], "criticalTypes": [],
        "statusTypes": {"new": [], "confirmed": [], "false": [], "forwarded": []},
    }


def test_failed_analytics_rolls_back_read_transaction(api, monkeypatch):
    client, journal = api
    add_event(journal, "existing", 100)
    original_aggregate = journal._stats_unlocked

    def fail_after_first_select(group_by, *args, **kwargs):
        if group_by == "camera":
            raise RuntimeError("interrupted analytics")
        return original_aggregate(group_by, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(journal, "_stats_unlocked", fail_after_first_select)
        with pytest.raises(RuntimeError, match="interrupted analytics"):
            analytics(client)
    assert not journal._conn.in_transaction
    add_event(journal, "after-error", 100)
    assert counts(analytics(client)["types"]) == {"ppe_missing": 2}
