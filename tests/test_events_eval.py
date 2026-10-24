"""Синтетические проверки метрик, форматов и CLI."""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from guard.contracts import Event, EventStatus, Severity
from guard.events.journal import Journal
from ml.eval.events_eval import (
    LabelEvent,
    PredEvent,
    evaluate,
    load_database,
    load_labels,
    load_predictions,
    main,
    predictions_from_events,
)


def label(
    start=10.0, end=20.0, *, video="a.mp4", type="fall_immobile",
    uncertain=False,
):
    return LabelEvent(video, type, start, end, uncertain)


def pred(
    start=10.0, confirmed=12.0, *, video="a.mp4", type="fall_immobile",
    id="",
):
    return PredEvent(video, type, start, confirmed, id)


def make_event(
    event_id="e1", *, camera="cam-a", started=1010.0, confirmed=1012.0,
):
    return Event(
        id=event_id,
        camera_id=camera,
        site="test",
        rule_id="fall-rule",
        type="fall_immobile",
        severity=Severity.high,
        started_at=started,
        confirmed_at=confirmed,
        bbox=(0.0, 0.0, 100.0, 200.0),
        status=EventStatus.false,
        latency_ms={"total": 9000.0},
    )


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_perfect_match():
    result = evaluate([label()], [pred()], durations={"a.mp4": 3600})
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 0, 0)
    assert overall["precision"] == overall["recall"] == overall["f1"] == 1
    assert overall["duplicates"] == 0
    assert overall["latency_median_s"] == overall["latency_p95_s"] == 2
    assert overall["false_alarms_per_hour"] == 0
    assert result["by_type"]["fall_immobile"] == overall


def test_missed_event():
    metrics = evaluate([label()], [])["overall"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 0, 1)
    assert metrics["precision"] == metrics["recall"] == metrics["f1"] == 0
    assert metrics["latency_median_s"] is None
    assert metrics["latency_p95_s"] is None
    assert metrics["false_alarms_per_hour"] is None


def test_false_alarm_on_clean_video():
    metrics = evaluate([], [pred()], durations={"a.mp4": 1800})["overall"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 1, 0)
    assert metrics["false_alarms_per_hour"] == 2


def test_duplicate_is_not_false_positive():
    metrics = evaluate(
        [label()], [pred(), pred(11, 13)], durations={"a.mp4": 3600}
    )["overall"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 0, 0)
    assert metrics["duplicates"] == 1
    assert metrics["precision"] == 1
    assert metrics["false_alarms_per_hour"] == 0
    assert metrics["latency_median_s"] == 2


def test_uncertain_excludes_predictions_and_does_not_create_fn():
    labels = [label(uncertain=True)]
    metrics = evaluate(
        labels, [pred(), pred(11, 13), pred(40, 41)]
    )["overall"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (0, 1, 0)
    assert metrics["ignored_uncertain"] == 2
    assert metrics["duplicates"] == 0
    assert metrics["latency_median_s"] is None
    assert evaluate(labels, [])["overall"]["fn"] == 0


def test_certain_labels_have_priority_over_uncertain():
    metrics = evaluate(
        [label(0, 100, uncertain=True), label()],
        [pred(), pred(11, 13)],
    )["overall"]
    assert metrics["tp"] == 1
    assert metrics["duplicates"] == 1
    assert metrics["ignored_uncertain"] == 0


@pytest.mark.parametrize(
    "start,confirmed,tol,expected_tp",
    [
        (7, 8, 2, 1),
        (7, 7.9999, 2, 0),
        (22, 23, 2, 1),
        (22.0001, 23, 2, 0),
        (20, 20, 0, 1),
        (8, 9, 0, 0),
    ],
)
def test_time_tolerance_and_inclusive_boundaries(
    start, confirmed, tol, expected_tp,
):
    metrics = evaluate(
        [label()], [pred(start, confirmed)], tol_s=tol
    )["overall"]
    assert metrics["tp"] == expected_tp
    assert metrics["fp"] == metrics["fn"] == 1 - expected_tp


def test_negative_delay_is_preserved():
    metrics = evaluate([label()], [pred(8, 9)], tol_s=1)["overall"]
    assert metrics["tp"] == 1
    assert metrics["latency_median_s"] == metrics["latency_p95_s"] == -1


def test_multiple_videos_and_types_and_micro_average():
    labels = [
        label(),
        label(video="b.mp4"),
        label(30, 35, type="running"),
    ]
    preds = [
        pred(),
        pred(30, 31, type="running"),
        pred(video="b.mp4", type="running"),
    ]
    result = evaluate(
        labels,
        preds,
        durations={"a.mp4": 1200, "b.mp4": 1200, "clean.mp4": 1200},
    )
    overall = result["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (2, 1, 1)
    assert overall["precision"] == pytest.approx(2 / 3)
    assert overall["recall"] == pytest.approx(2 / 3)
    assert overall["f1"] == pytest.approx(2 / 3)
    assert overall["false_alarms_per_hour"] == 1
    fall = result["by_type"]["fall_immobile"]
    running = result["by_type"]["running"]
    assert (fall["tp"], fall["fp"], fall["fn"]) == (1, 0, 1)
    assert (running["tp"], running["fp"], running["fn"]) == (1, 1, 0)
    assert result["duration_s"] == 3600
    assert result["videos"] == ["a.mp4", "b.mp4", "clean.mp4"]


def test_one_prediction_cannot_match_two_labels():
    metrics = evaluate(
        [label(10, 10), label(15, 15)], [pred(9, 16)], tol_s=0
    )["overall"]
    assert (metrics["tp"], metrics["fn"], metrics["duplicates"]) == (1, 1, 0)


def test_free_label_precedes_duplicate_and_order_is_deterministic():
    labels = [label(10, 20), label(15, 25)]
    preds = [pred(12, 17, id="second"), pred(10, 16, id="first")]
    labels_before = list(labels)
    preds_before = list(preds)
    result = evaluate(labels, preds, tol_s=0)
    assert result["overall"]["tp"] == 2
    assert result["overall"]["duplicates"] == 0
    assert result["overall"]["latency_median_s"] == 4
    assert evaluate(list(reversed(labels)), list(reversed(preds)), 0) == result
    assert labels == labels_before
    assert preds == preds_before


def test_latency_median_and_p95_use_all_tp_samples():
    result = evaluate(
        [
            label(10, 10),
            label(30, 30),
            label(50, 50),
            label(10, 10, video="b.mp4", type="running"),
        ],
        [
            pred(10, 10),
            pred(30, 32),
            pred(50, 54),
            pred(10, 20, video="b.mp4", type="running"),
        ],
        tol_s=0,
    )
    assert result["overall"]["tp"] == 4
    assert result["overall"]["latency_median_s"] == 3
    assert result["overall"]["latency_p95_s"] == pytest.approx(9.1)
    assert result["by_type"]["fall_immobile"]["latency_p95_s"] == pytest.approx(3.8)
    assert result["by_type"]["running"]["latency_p95_s"] == 10


def test_empty_dataset_and_clean_video_duration():
    result = evaluate([], [], durations={"clean.mp4": 60})
    assert result["by_type"] == {}
    assert result["overall"]["false_alarms_per_hour"] == 0
    assert result["overall"]["latency_p95_s"] is None
    assert result["overall"]["f1"] == 0
    assert evaluate([], [], durations={})["overall"]["false_alarms_per_hour"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: label(-1, 20),
        lambda: label(20, 10),
        lambda: label(float("nan"), 20),
        lambda: label(uncertain="false"),
        lambda: pred(20, 10),
        lambda: pred(10, float("inf")),
        lambda: pred(video=""),
        lambda: evaluate([], [], tol_s=-1),
        lambda: evaluate([], [], tol_s=float("nan")),
        lambda: evaluate([], [], durations={"a.mp4": 0}),
        lambda: evaluate([], [], durations={"a.mp4": -1}),
        lambda: evaluate([], [], durations={"a.mp4": float("inf")}),
        lambda: evaluate([], [], durations={"a.mp4": True}),
        lambda: evaluate([label()], [], durations={"b.mp4": 60}),
    ],
)
def test_invalid_inputs_are_rejected(factory):
    with pytest.raises(ValueError):
        factory()


def test_event_conversion_uses_epoch_mapping_and_ignores_operator_status():
    result = predictions_from_events(
        [make_event()], {"cam-a": {"video": "a.mp4", "t0": 1000}}
    )
    assert result == [pred(id="e1")]
    metrics = evaluate([label()], result)["overall"]
    assert metrics["tp"] == 1
    assert metrics["latency_median_s"] == 2


def test_unknown_camera_is_not_silently_skipped():
    with pytest.raises(ValueError, match="No mapping"):
        predictions_from_events([make_event()], {})


def test_loading_labels_preserves_clean_videos(tmp_path):
    write_json(tmp_path / "a.json", {
        "video": "a.mp4",
        "events": [{
            "type": "fall_immobile", "t_start": 10, "t_end": 20,
            "uncertain": False, "note": "ручная разметка",
        }],
    })
    write_json(tmp_path / "clean.json", {"video": "clean.mp4", "events": []})
    labels, videos = load_labels(tmp_path)
    assert videos == {"a.mp4", "clean.mp4"}
    assert len(labels) == 1
    assert labels[0].note == "ручная разметка"


def test_duplicate_label_video_is_rejected(tmp_path):
    for name in ("one.json", "two.json"):
        write_json(tmp_path / name, {"video": "a.mp4", "events": []})
    with pytest.raises(ValueError, match="duplicate annotation"):
        load_labels(tmp_path)


def test_incomplete_api_page_is_rejected(tmp_path):
    path = tmp_path / "preds.json"
    write_json(path, {"items": [], "total": 100})
    with pytest.raises(ValueError, match="incomplete API page"):
        load_predictions(path, {})


def test_missing_database_is_not_created(tmp_path):
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        load_database(path, {})
    assert not path.exists()


@pytest.mark.parametrize("source", ["list", "items", "db"])
def test_cli_json_and_database_include_all_events(tmp_path, source):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    write_json(labels_dir / "a.json", {
        "video": "a.mp4",
        "events": [{
            "type": "fall_immobile", "t_start": 10, "t_end": 20,
            "uncertain": False,
        }],
    })
    write_json(labels_dir / "clean.json", {
        "video": "clean.mp4", "events": [],
    })
    mapping_path = tmp_path / "mapping.json"
    write_json(mapping_path, {
        "cam-a": {"video": "a.mp4", "t0": 1000},
        "cam-a-2": {"video": "a.mp4", "t0": 1000},
        "cam-clean": {"video": "clean.mp4", "t0": 2000},
    })
    durations_path = tmp_path / "durations.json"
    write_json(durations_path, {"a.mp4": 3600, "clean.mp4": 3600})
    events = [
        make_event(
            f"e{i:03d}",
            camera="cam-a" if i % 2 == 0 else "cam-a-2",
            confirmed=1012 + i / 1000,
        )
        for i in range(61)
    ]
    events.append(make_event(
        "clean-fp", camera="cam-clean", started=2020, confirmed=2021,
    ))

    db_before = None
    if source == "db":
        source_path = tmp_path / "guard.db"
        with Journal(source_path) as journal:
            for event in events:
                journal.add(event)
        db_before = source_path.read_bytes()
        source_args = ["--db", str(source_path)]
    else:
        source_path = tmp_path / "preds.json"
        payload = [event.model_dump(mode="json") for event in events]
        if source == "items":
            payload = {"items": payload, "total": len(payload)}
        write_json(source_path, payload)
        source_args = ["--preds", str(source_path)]

    out = tmp_path / "output" / "metrics.json"
    command = [
        sys.executable, "-m", "ml.eval.events_eval",
        "--labels", str(labels_dir),
        *source_args,
        "--mapping", str(mapping_path),
        "--durations", str(durations_path),
        "--out", str(out),
    ]
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "fall_immobile" in completed.stdout
    assert "TOTAL" in completed.stdout
    assert "RuntimeWarning" not in completed.stderr
    result = json.loads(out.read_text(encoding="utf-8"))
    metrics = result["overall"]
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (1, 1, 0)
    assert metrics["duplicates"] == 60
    assert metrics["precision"] == 0.5
    assert metrics["recall"] == 1
    assert metrics["f1"] == pytest.approx(2 / 3)
    assert metrics["latency_median_s"] == 2
    assert metrics["false_alarms_per_hour"] == 0.5
    assert result["duration_s"] == 7200
    if source == "db":
        assert source_path.read_bytes() == db_before


@pytest.mark.parametrize("case", ["missing_duration", "extra_duration", "unannotated"])
def test_cli_rejects_incomplete_evaluation_coverage(tmp_path, case, capsys):
    labels_dir = tmp_path / "labels"
    labels_dir.mkdir()
    write_json(labels_dir / "a.json", {"video": "a.mp4", "events": []})
    durations = {"a.mp4": 60}
    mapped_video = "a.mp4"
    if case == "missing_duration":
        durations = {}
    elif case == "extra_duration":
        durations["extra.mp4"] = 3600
    else:
        mapped_video = "unannotated.mp4"
    write_json(tmp_path / "durations.json", durations)
    write_json(tmp_path / "mapping.json", {
        "cam-a": {"video": mapped_video, "t0": 1000},
    })
    write_json(
        tmp_path / "preds.json",
        [make_event().model_dump(mode="json")],
    )
    out = tmp_path / "metrics.json"
    with pytest.raises(SystemExit) as exc:
        main([
            "--labels", str(labels_dir),
            "--preds", str(tmp_path / "preds.json"),
            "--mapping", str(tmp_path / "mapping.json"),
            "--durations", str(tmp_path / "durations.json"),
            "--out", str(out),
        ])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
    assert not out.exists()
