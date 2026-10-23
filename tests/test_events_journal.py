import csv
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from datetime import datetime

import pytest
from pydantic import ValidationError

from guard.contracts import Event, EventStatus, Severity
from guard.events.journal import Journal


def make_event(event_id="event-1", **overrides):
    data = {
        "id": event_id,
        "camera_id": "camera-a",
        "site": "Склад",
        "zone_id": "north",
        "rule_id": "rule-ppe",
        "type": "ppe_missing",
        "severity": Severity.high,
        "track_id": 42,
        "started_at": 1_700_000_000.125,
        "confirmed_at": 1_700_000_002.25,
        "bbox": (10.5, 20.25, 100.0, 200.0),
        "snapshot_path": "media/кадр.jpg",
        "details": {
            "reason": "Без каски",
            "attributes": {"helmet": False},
            "scores": [0.9, None],
        },
        "latency_ms": {"infer": 12.5, "total": 21.25},
    }
    data.update(overrides)
    return Event(**data)


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


@pytest.fixture
def journal(tmp_path):
    with Journal(tmp_path / "guard.db") as instance:
        yield instance


def populate(journal):
    configurations = [
        ("e1", 100, "camera-a", "north", "ppe_missing", "new", "high"),
        ("e2", 200, "camera-a", "south", "zone_intrusion", "confirmed", "critical"),
        ("e3", 300, "camera-b", "north", "ppe_missing", "forwarded", "high"),
        ("e4", 400, "camera-b", None, "fall_immobile", "false", "medium"),
        ("e5", 500, "camera-a", "north", "ppe_missing", "confirmed", "high"),
    ]
    for event_id, ts, camera, zone, kind, status, severity in configurations:
        journal.add(
            make_event(
                event_id,
                started_at=ts - 2,
                confirmed_at=ts,
                camera_id=camera,
                zone_id=zone,
                type=kind,
                status=status,
                severity=severity,
            )
        )


def test_add_get_update_and_reopen(tmp_path):
    db_path = tmp_path / "nested" / "guard.db"
    original = make_event()
    changes = {
        "status": "confirmed",
        "operator_note": "Ответственный уведомлён",
        "ended_at": original.confirmed_at + 15.5,
        "snapshot_path": "media/new.jpg",
        "clip_path": "media/clip.mp4",
        "details": {"review": {"accepted": True}, "comments": ["Проверено"]},
    }
    expected = Event.model_validate({**original.model_dump(), **changes})

    with Journal(db_path) as journal:
        assert journal.get("missing") is None
        journal.add(original)
        assert journal.get(original.id) == original
        assert journal.update(original.id, **changes) == expected
        assert journal.get(original.id) == expected
        assert journal.update(original.id) == expected
        assert original.status == EventStatus.new

    with Journal(str(db_path)) as reopened:
        assert reopened.get(original.id) == expected
        assert reopened.query() == ([expected], 1)
        cleared = reopened.update(
            original.id,
            operator_note=None,
            ended_at=None,
            snapshot_path=None,
            clip_path=None,
        )
        assert cleared.operator_note is None
        assert cleared.ended_at is None
        assert cleared.snapshot_path is None
        assert cleared.clip_path is None


def test_schema_wal_indexes_and_json(tmp_path):
    db_path = tmp_path / "guard.db"
    with Journal(db_path) as journal:
        event = make_event()
        journal.add(event)
        with closing(sqlite3.connect(db_path)) as connection:
            assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
            assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
            columns = {
                row[1] for row in connection.execute("PRAGMA table_info(events)")
            }
            assert columns == set(Event.model_fields)
            indexes = set()
            for row in connection.execute("PRAGMA index_list(events)").fetchall():
                name = row[1].replace('"', '""')
                indexes.add(
                    tuple(
                        item[2]
                        for item in connection.execute(
                            f'PRAGMA index_info("{name}")'
                        )
                    )
                )
            assert {
                ("camera_id", "confirmed_at"),
                ("type",),
                ("status",),
                ("zone_id",),
                ("confirmed_at",),
            } <= indexes
            bbox, details, latency = connection.execute(
                "SELECT bbox, details, latency_ms FROM events"
            ).fetchone()
            assert json.loads(bbox) == list(event.bbox)
            assert json.loads(details) == event.details
            assert json.loads(latency) == event.latency_ms


def test_reject_future_schema_without_changing_version(tmp_path):
    db_path = tmp_path / "future.db"
    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute("PRAGMA user_version = 99")
    with pytest.raises(ValueError, match="schema version"):
        Journal(db_path)
    with closing(sqlite3.connect(db_path)) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 99


def test_duplicate_and_invalid_updates_are_atomic(journal):
    event = make_event()
    journal.add(event)
    with pytest.raises(sqlite3.IntegrityError):
        journal.add(make_event(event.id, operator_note="Дубликат"))
    with pytest.raises(ValueError):
        journal.update(event.id, camera_id="other", operator_note="Изменение")
    with pytest.raises(ValidationError):
        journal.update(event.id, status="invalid", operator_note="Изменение")
    with pytest.raises(ValidationError):
        journal.update(event.id, details=None)
    with pytest.raises(KeyError):
        journal.update("missing", status="confirmed")
    assert journal.get(event.id) == event
    assert journal.query()[1] == 1
    journal.add(make_event("next"))
    assert journal.query()[1] == 2


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        ({}, ["e5", "e4", "e3", "e2", "e1"]),
        ({"type": "ppe_missing"}, ["e5", "e3", "e1"]),
        ({"type": ["zone_intrusion", "fall_immobile"]}, ["e4", "e2"]),
        ({"camera_id": "camera-a"}, ["e5", "e2", "e1"]),
        ({"camera_id": ["camera-b"]}, ["e4", "e3"]),
        ({"zone_id": "north"}, ["e5", "e3", "e1"]),
        ({"zone_id": ["north", "south"]}, ["e5", "e3", "e2", "e1"]),
        ({"status": EventStatus.confirmed}, ["e5", "e2"]),
        ({"status": ["new", EventStatus.false]}, ["e4", "e1"]),
        ({"severity": Severity.high}, ["e5", "e3", "e1"]),
        ({"severity": ["critical", "medium"]}, ["e4", "e2"]),
        ({"ts_from": 200, "ts_to": 400}, ["e4", "e3", "e2"]),
        ({"ts_from": 100, "ts_to": 100}, ["e1"]),
        ({"ts_from": 500}, ["e5"]),
        ({"ts_to": 100}, ["e1"]),
        (
            {
                "type": ["ppe_missing", "zone_intrusion"],
                "camera_id": ["camera-a"],
                "zone_id": ["north"],
                "status": ["new", "confirmed"],
                "severity": ["high"],
                "ts_from": 100,
                "ts_to": 500,
            },
            ["e5", "e1"],
        ),
        ({"type": "ppe_missing' OR 1=1 --"}, []),
        ({"ts_from": 500, "ts_to": 100}, []),
    ],
)
def test_query_filters(journal, filters, expected):
    populate(journal)
    events, total = journal.query(**filters)
    assert [event.id for event in events] == expected
    assert total == len(expected)


@pytest.mark.parametrize("name", ["type", "camera_id", "zone_id", "status", "severity"])
def test_empty_filter_list_matches_nothing(journal, name):
    populate(journal)
    assert journal.query(**{name: []}) == ([], 0)
    assert journal.stats("type", **{name: []}) == []


def test_pagination_total_and_stable_order(journal):
    populate(journal)
    events, total = journal.query(
        type="ppe_missing", limit=1, offset=1, order="asc"
    )
    assert [event.id for event in events] == ["e3"]
    assert total == 3
    assert journal.query(limit=0) == ([], 5)
    assert journal.query(limit=2, offset=10) == ([], 5)
    journal.add(make_event("e6", confirmed_at=500))
    descending, total = journal.query(limit=2)
    assert [event.id for event in descending] == ["e6", "e5"]
    assert total == 6
    ascending, total = journal.query(ts_from=500, order="asc")
    assert [event.id for event in ascending] == ["e5", "e6"]
    assert total == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"limit": -1},
        {"offset": -1},
        {"limit": 1.5},
        {"limit": True},
        {"order": "desc; DROP TABLE events"},
    ],
)
def test_invalid_pagination(journal, kwargs):
    with pytest.raises(ValueError):
        journal.query(**kwargs)


def add_midnight_events(journal):
    configurations = [
        ("a", "2025-05-01T20:59:59Z", "camera-a", "ppe_missing", "new", "high", "north"),
        ("b", "2025-05-01T21:00:00Z", "camera-b", "ppe_missing", "confirmed", "critical", "north"),
        ("c", "2025-05-01T21:30:00Z", "camera-a", "zone_intrusion", "confirmed", "high", None),
        ("d", "2025-05-02T20:59:59Z", "camera-b", "zone_intrusion", "false", "medium", "south"),
    ]
    for event_id, value, camera, kind, status, severity, zone in configurations:
        ts = timestamp(value)
        journal.add(
            make_event(
                event_id,
                started_at=ts - 120,
                confirmed_at=ts,
                camera_id=camera,
                type=kind,
                status=status,
                severity=severity,
                zone_id=zone,
            )
        )


@pytest.mark.parametrize(
    ("group_by", "expected"),
    [
        ("type", [("ppe_missing", 2), ("zone_intrusion", 2)]),
        ("camera", [("camera-a", 2), ("camera-b", 2)]),
        ("zone", [(None, 1), ("north", 2), ("south", 1)]),
        ("severity", [("critical", 1), ("high", 2), ("medium", 1)]),
        ("status", [("confirmed", 2), ("false", 1), ("new", 1)]),
        ("hour", [(0, 2), (23, 2)]),
        ("day", [("2025-05-01", 1), ("2025-05-02", 3)]),
    ],
)
def test_stats_local_midnight(journal, group_by, expected):
    add_midnight_events(journal)
    assert journal.stats(group_by) == [
        {"key": key, "count": count} for key, count in expected
    ]


def test_stats_timezone_and_filters(journal):
    add_midnight_events(journal)
    assert journal.stats("hour", tz="UTC") == [
        {"key": 20, "count": 2},
        {"key": 21, "count": 2},
    ]
    assert journal.stats("day", tz="UTC") == [
        {"key": "2025-05-01", "count": 3},
        {"key": "2025-05-02", "count": 1},
    ]
    assert journal.stats(
        "day",
        ts_from=timestamp("2025-05-01T21:00:00Z"),
        ts_to=timestamp("2025-05-01T21:30:00Z"),
        type=["ppe_missing", "zone_intrusion"],
        status=EventStatus.confirmed,
    ) == [{"key": "2025-05-02", "count": 2}]
    assert journal.stats(
        "camera", camera_id=["camera-a"], severity=Severity.high, zone_id="north"
    ) == [{"key": "camera-a", "count": 1}]
    assert journal.stats("type", camera_id="missing") == []
    with pytest.raises(ValueError):
        journal.stats("site")
    with pytest.raises(TypeError):
        journal.stats("type", unknown="value")


def test_stats_repeated_hour_during_dst_transition(journal):
    for index, value in enumerate(
        ["2025-11-02T05:30:00Z", "2025-11-02T06:30:00Z"]
    ):
        journal.add(make_event(str(index), confirmed_at=timestamp(value)))
    assert journal.stats("hour", tz="America/New_York") == [
        {"key": 1, "count": 2}
    ]
    assert journal.stats("day", tz="America/New_York") == [
        {"key": "2025-11-02", "count": 2}
    ]


def test_parallel_writes_from_four_threads(tmp_path):
    db_path = tmp_path / "parallel.db"
    with Journal(db_path) as journal:
        def write_events(worker):
            for index in range(100):
                journal.add(
                    make_event(
                        f"{worker}-{index}",
                        camera_id=f"camera-{worker}",
                        confirmed_at=1_700_000_000 + worker * 100 + index,
                    )
                )

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(write_events, range(4)))

        events, total = journal.query(limit=1000)
        assert total == 400
        assert len(events) == 400
        assert len({event.id for event in events}) == 400
        assert journal.stats("camera") == [
            {"key": f"camera-{worker}", "count": 100} for worker in range(4)
        ]

    with Journal(db_path) as reopened:
        assert reopened.query(limit=0) == ([], 400)


def test_export_csv_has_russian_headers_and_all_filtered_rows(journal, tmp_path):
    note = 'Ответственный: "Иванов", проверено\nЗамечание устранено'
    for index in range(60):
        journal.add(
            make_event(
                f"export-{index}",
                confirmed_at=1_700_000_002 + index,
                operator_note=note,
            )
        )
    journal.add(make_event("excluded", camera_id="camera-b"))
    path = tmp_path / "exports" / "events.csv"
    journal.export_csv(path, camera_id=["camera-a"], status="new")
    assert path.is_file()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames is not None
        assert len(reader.fieldnames) == len(Event.model_fields)
        assert "Тип нарушения" in reader.fieldnames
        assert "Комментарий ответственного" in reader.fieldnames
    assert len(rows) == 60
    assert rows[0]["Идентификатор события"] == "export-59"
    assert {row["Камера"] for row in rows} == {"camera-a"}
    assert rows[0]["Комментарий ответственного"] == note
    assert rows[0]["Окончание (UTC)"] == ""
    assert json.loads(rows[0]["Подробности"])["reason"] == "Без каски"
    assert datetime.fromisoformat(
        rows[0]["Подтверждение (UTC)"]
    ).timestamp() == 1_700_000_061

    filtered_path = tmp_path / "range.csv"
    journal.export_csv(
        filtered_path,
        camera_id="camera-a",
        ts_from=1_700_000_010,
        ts_to=1_700_000_011,
        order="asc",
    )
    with filtered_path.open(encoding="utf-8-sig", newline="") as stream:
        assert [
            row["Идентификатор события"] for row in csv.DictReader(stream)
        ] == ["export-8", "export-9"]

    empty_path = tmp_path / "empty.csv"
    journal.export_csv(empty_path, type=[])
    with empty_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames
        assert list(reader) == []
