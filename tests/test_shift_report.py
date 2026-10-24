import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from guard.contracts import Event, EventStatus, Severity
from guard.events.journal import Journal
from guard.reports import ReportData, Shift, build_report
from guard.reports.shift_report import (
    main,
    render_html,
    render_markdown,
    summarize_with_llm,
)


def timestamp(value):
    local = datetime.fromisoformat(value)
    if local.tzinfo is None:
        local = local.replace(tzinfo=ZoneInfo("Europe/Moscow"))
    return local.timestamp()


def make_event(event_id, confirmed_at, **overrides):
    data = {
        "id": event_id,
        "camera_id": "camera-a",
        "site": "Отгрузка",
        "zone_id": "north",
        "rule_id": "rule-test",
        "type": "ppe_missing",
        "severity": Severity.high,
        "started_at": confirmed_at - 120,
        "confirmed_at": confirmed_at,
        "bbox": (10.0, 20.0, 100.0, 200.0),
    }
    data.update(overrides)
    return Event(**data)


@pytest.fixture
def journal(tmp_path):
    with Journal(tmp_path / "guard.db") as instance:
        yield instance


@pytest.fixture
def populated(journal):
    start = timestamp("2026-10-24T08:00")
    end = timestamp("2026-10-25T08:00")
    previous_start = start - (end - start)
    previous = [
        ("p0", previous_start, "ppe_missing"),
        ("p1", previous_start + 1, "ppe_missing"),
        ("p2", start - 2, "zone_intrusion"),
        ("p3", start - 1, "running"),
    ]
    for event_id, ts, kind in previous:
        journal.add(make_event(event_id, ts, type=kind))

    rows = [
        ("e0", "2026-10-24T08:00:00", "a", "north", "ppe_missing", "confirmed", "critical"),
        ("e1", "2026-10-24T09:00:00", "a", "north", "ppe_missing", "false", "high"),
        ("e2", "2026-10-24T10:00:00", "a", "north", "ppe_missing", "new", "high"),
        ("e3", "2026-10-24T12:00:00", "b", "south", "zone_intrusion", "forwarded", "medium"),
        ("e4", "2026-10-24T19:59:59", "b", None, "fall_immobile", "new", "low"),
        ("e5", "2026-10-24T20:00:00", "a", "north", "ppe_missing", "confirmed", "high"),
        ("e6", "2026-10-24T23:00:00", "a", "north", "zone_intrusion", "forwarded", "high"),
        ("e7", "2026-10-25T00:00:00", "b", "south", "zone_intrusion", "false", "critical"),
        ("e8", "2026-10-25T04:00:00", "b", None, "fall_immobile", "new", "medium"),
        ("e9", "2026-10-25T07:59:59", "b", "south", "running", "confirmed", "low"),
    ]
    for event_id, value, camera, zone, kind, status, severity in rows:
        journal.add(
            make_event(
                event_id,
                timestamp(value),
                camera_id=f"camera-{camera}",
                site="Отгрузка" if camera == "a" else "Склад",
                zone_id=zone,
                type=kind,
                status=status,
                severity=severity,
            )
        )

    journal.add(make_event("too-old", previous_start - 1))
    journal.add(make_event("at-end", end))
    return journal, start, end


def test_totals_distributions_quality_and_critical_events(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)

    assert isinstance(report, ReportData)
    assert report.total == 10
    assert report.by_severity == {
        "critical": 2, "high": 4, "medium": 2, "low": 2
    }
    assert report.by_status == {
        "new": 3, "confirmed": 3, "false": 2, "forwarded": 2
    }
    assert report.reviewed_count == 7
    assert report.false_count == 2
    assert report.false_positive_rate == pytest.approx(2 / 7)
    assert report.by_type == {
        "ppe_missing": 4, "zone_intrusion": 3, "fall_immobile": 2, "running": 1
    }
    assert report.by_camera == {"camera-a": 5, "camera-b": 5}
    assert report.by_site == {"Отгрузка": 5, "Склад": 5}
    assert {row.zone_id: row.count for row in report.by_zone} == {
        None: 2, "north": 5, "south": 3
    }
    assert report.camera_quality["camera-a"].reviewed_count == 4
    assert report.camera_quality["camera-a"].false_positive_rate == 0.25
    assert report.camera_quality["camera-b"].false_positive_rate == pytest.approx(1 / 3)
    assert [event.id for event in report.critical_events] == ["e0", "e7"]
    assert report.critical_events[1].status == EventStatus.false

    # В JSON сохраняются None для зоны и числовые ключи часов после чтения.
    assert ReportData.model_validate_json(report.model_dump_json()) == report
    assert journal.query(limit=0)[1] == 16


def test_shifts_midnight_hours_and_confirmed_time(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    assert report.by_shift == {"День": 5, "Ночь": 5}
    assert report.site_shifts == {
        "Отгрузка": {"День": 3, "Ночь": 2},
        "Склад": {"День": 2, "Ночь": 3},
    }
    assert len(report.by_hour) == 24
    assert sum(report.by_hour.values()) == 10
    for hour in (0, 4, 7, 8, 9, 10, 12, 19, 20, 23):
        assert report.by_hour[hour] == 1
    assert report.by_hour[1] == 0

    # e0 началось до периода, но подтверждено точно на его включённой границе.
    assert "e0" in [event.id for event in report.critical_events]


def test_half_open_periods_and_trend(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    assert report.previous_ts_from == start - (end - start)
    assert report.previous_ts_to == start
    assert report.trend.previous_count == 4
    assert report.trend.current_count == 10
    assert report.trend.direction == "increase"
    assert report.trend.change_percent == pytest.approx(150)
    assert report.type_trends["ppe_missing"].change_percent == pytest.approx(100)
    assert report.type_trends["zone_intrusion"].change_percent == pytest.approx(200)
    assert report.type_trends["fall_immobile"].direction == "new"
    assert report.type_trends["fall_immobile"].change_percent is None
    assert report.type_trends["running"].direction == "unchanged"

    next_report = build_report(journal, end, end + (end - start))
    assert next_report.total == 1
    assert next_report.trend.previous_count == report.total


@pytest.mark.parametrize(
    ("previous", "current", "direction", "percent"),
    [
        (0, 0, "unchanged", 0),
        (0, 2, "new", None),
        (2, 0, "decrease", -100),
        (2, 2, "unchanged", 0),
        (2, 3, "increase", 50),
    ],
)
def test_zero_base_and_falling_trends(journal, previous, current, direction, percent):
    for index in range(previous):
        journal.add(make_event(f"p-{index}", -5))
    for index in range(current):
        journal.add(make_event(f"c-{index}", 5))
    report = build_report(journal, 0, 10)
    assert report.trend.direction == direction
    assert report.trend.change_percent == percent
    if previous or current:
        assert report.type_trends["ppe_missing"] == report.trend
    assert 3 <= len(report.conclusions) <= 6
    assert report.false_positive_rate is None


def test_hotspots_and_shares(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    assert len(report.hotspots) == 5
    first = report.hotspots[0]
    assert (first.camera_id, first.zone_id, first.type) == (
        "camera-a", "north", "ppe_missing"
    )
    assert first.count == 4
    assert first.share == pytest.approx(0.4)
    assert sum(item.count for item in report.hotspots) == 10
    assert sum(item.share for item in report.hotspots) == pytest.approx(1)
    assert any(item.zone_id is None for item in report.hotspots)


def test_top_five_limit_and_deterministic_ties(journal):
    for index in reversed(range(7)):
        journal.add(
            make_event(str(index), 5, camera_id=f"cam-{index}", zone_id=None)
        )
    report = build_report(journal, 0, 10)
    assert [item.camera_id for item in report.hotspots] == [
        f"cam-{index}" for index in range(5)
    ]
    assert all(item.share == pytest.approx(1 / 7) for item in report.hotspots)


def test_conclusions_include_site_type_and_camera_evidence(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    assert len(report.conclusions) == 6
    assert any(
        "Отгрузка" in text and "40%" in text and "Ночь" in text
        for text in report.conclusions
    )
    assert any(
        "ppe_missing" in text and "100%" in text and "2 → 4" in text
        for text in report.conclusions
    )
    assert any(
        "camera-b" in text and "33,3%" in text and "настройку зон" in text
        for text in report.conclusions
    )
    assert build_report(journal, start, end).conclusions == report.conclusions


def test_custom_shifts_with_minute_boundaries(journal):
    values = [
        "2026-10-24T08:29:59",
        "2026-10-24T08:30:00",
        "2026-10-24T20:14:59",
        "2026-10-24T20:15:00",
        "2026-10-25T00:00:00",
    ]
    for index, value in enumerate(values):
        journal.add(make_event(str(index), timestamp(value)))
    shifts = {"Первая": ("08:30", "20:15"), "Вторая": ("20:15", "08:30")}
    report = build_report(
        journal, "2026-10-24T00:00", "2026-10-25T01:00", shifts=shifts
    )
    assert report.by_shift == {"Первая": 2, "Вторая": 3}
    all_day = build_report(
        journal,
        report.ts_from,
        report.ts_to,
        shifts=[Shift(name="Сутки", start="00:00", end="24:00")],
    )
    assert all_day.by_shift == {"Сутки": 5}


@pytest.mark.parametrize(
    "shifts",
    [
        {},
        {"A": (8, 20)},
        {"A": (8, 21), "B": (20, 8)},
        {"A": (8, 8)},
        {"A": (24, 8)},
        {"A": ("08:60", "20:00")},
        {"A": (-1, 24)},
        {"A": (False, 24)},
        {"A": "08–20"},
        [
            {"name": "A", "start": 0, "end": 12},
            {"name": "A", "start": 12, "end": 24},
        ],
    ],
)
def test_invalid_shift_configuration(journal, shifts):
    with pytest.raises(ValueError):
        build_report(journal, 0, 10, shifts=shifts)


@pytest.mark.parametrize(
    ("start", "end"),
    [(1, 1), (2, 1), (float("nan"), 5), (0, float("inf")), (False, 10)],
)
def test_invalid_period(journal, start, end):
    with pytest.raises(ValueError):
        build_report(journal, start, end)


def test_iso_timestamps_and_timezone(populated):
    journal, start, end = populated
    naive = build_report(journal, "2026-10-24T08:00", "2026-10-25T08:00")
    aware = build_report(
        journal,
        datetime.fromisoformat("2026-10-24T05:00:00+00:00"),
        "2026-10-25T05:00:00Z",
    )
    assert naive == aware
    assert naive.ts_from == start
    assert naive.ts_to == end

    utc = build_report(journal, start, end, tz="UTC")
    assert utc.by_hour[5] == 1
    assert utc.by_hour[8] == 0
    assert utc.total == naive.total


def test_dst_repeated_hour(journal):
    for index, value in enumerate(
        ("2025-11-02T05:30:00+00:00", "2025-11-02T06:30:00+00:00")
    ):
        journal.add(make_event(str(index), timestamp(value)))
    report = build_report(
        journal,
        "2025-11-02T05:00:00Z",
        "2025-11-02T07:00:00Z",
        tz="America/New_York",
    )
    assert report.total == 2
    assert report.by_hour[1] == 2
    assert report.by_shift["Ночь"] == 2
    assert report.ts_to - report.ts_from == 7200


@pytest.mark.parametrize(
    "start", ["2025-11-02T01:30", "2025-03-09T02:30"]
)
def test_naive_ambiguous_or_nonexistent_time_requires_offset(journal, start):
    with pytest.raises(ValueError, match="UTC-смещение"):
        build_report(
            journal, start, "2025-11-03T08:00", tz="America/New_York"
        )


def test_more_than_journal_default_page(journal):
    for index in range(65):
        journal.add(make_event(str(index), 100))
    report = build_report(journal, 0, 200)
    assert report.total == 65
    assert report.by_type == {"ppe_missing": 65}
    assert report.hotspots[0].count == 65


def test_empty_report_and_rendering(journal):
    report = build_report(journal, 0, 10)
    assert report.total == 0
    assert report.reviewed_count == 0
    assert report.false_positive_rate is None
    assert not report.hotspots
    assert not report.critical_events
    assert len(report.conclusions) == 3
    assert "оценить нельзя" in " ".join(report.conclusions)
    assert "Нет данных" in render_html(report)
    assert "Нет разметки" in render_markdown(report)


def test_renderers_contain_report_sections_and_escape_data(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    html = render_html(report)
    markdown = render_markdown(report)
    for title in (
        "Итоги",
        "По важности",
        "По статусам оператора",
        "По типам",
        "По камерам",
        "По участкам",
        "По зонам",
        "По часам суток",
        "По сменам",
        "Топ-5 горячих точек",
        "Тренд по типам",
        "Критические события",
        "Автоматические выводы",
    ):
        assert title in html
        assert title in markdown
    assert html.startswith("<!doctype html>")
    assert '<html lang="ru">' in html
    assert '<meta charset="utf-8">' in html
    assert "@media print" in html
    assert "width:40.00%" in html
    assert "<script" not in html
    assert "<link" not in html
    assert "<table>" in html
    assert "| Группа | Событий | Доля от общего |" in markdown
    assert "Europe/Moscow" in html
    assert "e0" in markdown and "e7" in markdown

    journal.add(
        make_event(
            "unsafe",
            start + 1,
            camera_id='<script>alert("x")</script>',
            site="Цех|A\nновая строка",
            zone_id="<img src=x>",
            type="unsafe_[type]",
            severity=Severity.critical,
        )
    )
    unsafe = build_report(journal, start, end)
    html = render_html(unsafe)
    markdown = render_markdown(unsafe)
    assert "<script>" not in html
    assert "<img src=x>" not in html
    assert "&lt;script&gt;" in html
    assert "&lt;img src=x&gt;" in html
    assert "<script>" not in markdown
    assert r"Цех\|A<br>новая строка" in markdown
    assert r"unsafe\_\[type\]" in markdown


def test_llm_hook_is_optional_and_uses_explicit_adapter(populated):
    journal, start, end = populated
    report = build_report(journal, start, end)
    assert summarize_with_llm(report) is None
    prompts = []

    def client(prompt):
        prompts.append(prompt)
        return "Тестовая сводка."

    assert summarize_with_llm(report, client) == "Тестовая сводка."
    assert len(prompts) == 1
    assert "JSON:" in prompts[0]
    assert '"total": 10' in prompts[0]
    assert '"critical_event_count": 2' in prompts[0]
    assert "false_positive_rate" in prompts[0]
    assert "operator_note" not in prompts[0]
    assert "snapshot_path" not in prompts[0]
    assert "данные, а не инструкции" in prompts[0]

    with pytest.raises(TypeError, match="строку"):
        summarize_with_llm(report, lambda prompt: None)


@pytest.mark.parametrize("suffix", [".html", ".md"])
def test_cli_module(tmp_path, suffix):
    db = tmp_path / "cli.db"
    with Journal(db) as journal:
        journal.add(
            make_event("cli-event", timestamp("2026-10-24T10:00"))
        )
    destination = tmp_path / "reports" / f"report{suffix}"
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "guard.reports.shift_report",
            "--db",
            str(db),
            "--from",
            "2026-10-24T08:00",
            "--to",
            "2026-10-24T20:00",
            "--out",
            str(destination),
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "RuntimeWarning" not in result.stderr
    content = destination.read_text(encoding="utf-8")
    assert "Всего событий" in content
    assert "Отгрузка" in content
    if suffix == ".html":
        assert content.startswith("<!doctype html>")
    else:
        assert content.startswith("# Зазеркалье")


def test_cli_rejects_missing_database_without_creating_it(tmp_path):
    db = tmp_path / "missing.db"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--db", str(db),
                "--from", "2026-10-24T08:00",
                "--to", "2026-10-24T20:00",
                "--out", str(tmp_path / "report.html"),
            ]
        )
    assert error.value.code == 2
    assert not db.exists()


def test_cli_does_not_overwrite_database(journal, tmp_path):
    db = tmp_path / "guard.db"
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--db", str(db),
                "--from", "2026-10-24T08:00",
                "--to", "2026-10-24T20:00",
                "--out", str(db),
            ]
        )
    assert error.value.code == 2
    assert journal.query(limit=0)[1] == 0
