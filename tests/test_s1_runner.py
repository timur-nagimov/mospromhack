"""Deterministic coverage and evaluator composition; never runs pretrained inference."""

import json
import math
import signal
import subprocess
import sys

import pytest

from ml.eval import run_s1
from ml.eval.events_eval import LabelEvent, PredEvent, evaluate


@pytest.fixture
def recipe(tmp_path):
    config = json.loads((run_s1.ROOT / "configs/eval-s1.json").read_text())
    manifest = json.loads((run_s1.ROOT / config["manifest"]).read_text())
    manifest["videos"] = [item for item in manifest["videos"] if item["scenario"] == "S1"]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    config["manifest"] = str(manifest_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config_path, config, manifest_path, manifest


def test_recipe_covers_both_videos_all_labels_and_existing_runtime_thresholds(recipe):
    path, _, _, _ = recipe
    config, manifest, clips = run_s1.load_recipe(path)
    assert [clip["name"] for clip in clips] == list(run_s1.VIDEOS)
    labels, records = run_s1.labels_from_clips(clips)
    assert sum(not label.uncertain for label in labels) == 4
    assert sum(label.uncertain for label in labels) == 3
    assert len(set(row["label_id"] for row in records)) == 7
    from guard.pipeline.settings import Settings
    assert config["tracker"] == {
        "high_thresh": Settings.tracker_high_thresh, "low_thresh": Settings.tracker_low_thresh,
        "match_iou": Settings.tracker_match_iou, "new_track_thresh": Settings.tracker_new_track_thresh,
        "track_buffer_s": Settings.tracker_buffer_s, "min_hits": Settings.tracker_min_hits,
    }


@pytest.mark.parametrize("field,value", [("duration_s", 30), ("sample_fps", 4), ("threads", 8),
                                         ("model_id", "pose"), ("tol_s", -1)])
def test_changed_baseline_or_invalid_tolerance_is_rejected(recipe, field, value):
    path, config, _, _ = recipe
    config[field] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        run_s1.load_recipe(path)


@pytest.mark.parametrize("change", ["missing_video", "uncertain_lost", "wrong_segment", "zone_missing"])
def test_partial_or_ambiguous_manifest_is_rejected(recipe, change):
    path, _, manifest_path, manifest = recipe
    if change == "missing_video":
        manifest["videos"].pop()
    elif change == "uncertain_lost":
        manifest["videos"][0]["events"].pop()
    elif change == "wrong_segment":
        manifest["videos"][0]["segments"][0]["ss"] = 1
    else:
        manifest["videos"][0]["zones"] = []
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        run_s1.load_recipe(path)


@pytest.mark.parametrize("fps", [12, 25, 30, 59.94005994005994, 60])
def test_sampling_is_fixed_five_fps_without_drift_or_boundary_frame(fps):
    plan = run_s1.sample_plan(fps)
    assert len(plan) == 300
    assert plan[0]["source_frame_index"] == 0
    assert plan[-1]["target_s"] == 59.8
    assert len({row["source_frame_index"] for row in plan}) == 300
    for row in plan:
        actual = row["source_frame_index"] / fps
        assert row["target_s"] - 1e-8 <= actual < row["target_s"] + 1 / fps + 1e-8
        assert actual < 60
        assert row["source_frame_index"] < math.ceil(60 * fps - 1e-9)


@pytest.mark.parametrize("fps", [0, 4, float("nan"), float("inf")])
def test_invalid_or_too_slow_source_fps_fails(fps):
    with pytest.raises(ValueError):
        run_s1.sample_plan(fps)


def test_label_boundary_is_not_rounded_to_sampling_slot():
    # An annotation starts between samples; evaluator receives the original 15.4.
    labels = [LabelEvent("a.mp4", "zone_intrusion", 15.4, 26.6)]
    records = [{"label_id": "a:0", **vars_for_label(labels[0])}]
    source_time = run_s1.sample_plan(59.94005994005994)[78]["source_frame_index"] / 59.94005994005994
    predictions = [PredEvent("a.mp4", "zone_intrusion", source_time, source_time + 2, "p")]
    report = run_s1.score_predictions(labels, predictions, {"a.mp4": 60}, 2, records)
    assert report["evaluator"]["overall"]["latency_median_s"] == pytest.approx(source_time + 2 - 15.4)


def vars_for_label(label):
    return {name: getattr(label, name) for name in ("video", "type", "t_start", "t_end", "uncertain", "note")}


def test_existing_evaluator_preserved_and_duplicates_cannot_improve_strict_precision():
    labels = [LabelEvent("a.mp4", "zone_intrusion", 10, 12),
              LabelEvent("a.mp4", "zone_intrusion", 30, 35, uncertain=True)]
    records = [{"label_id": f"a:{index}", **vars_for_label(label)} for index, label in enumerate(labels)]
    predictions = [PredEvent("a.mp4", "zone_intrusion", 10, 12, "tp"),
                   PredEvent("a.mp4", "zone_intrusion", 10.5, 12.5, "duplicate"),
                   PredEvent("a.mp4", "zone_intrusion", 31, 33, "uncertain"),
                   PredEvent("a.mp4", "zone_intrusion", 50, 52, "fp")]
    result = run_s1.score_predictions(labels, predictions, {"a.mp4": 60}, 2, records)
    assert result["evaluator"] == evaluate(labels, predictions, 2, durations={"a.mp4": 60})
    original = result["evaluator"]["overall"]
    assert (original["tp"], original["fp"], original["duplicates"], original["ignored_uncertain"]) == (1, 1, 1, 1)
    strict = result["supplemental_strict"]
    assert original["precision"] == .5 and strict["strict_precision"] == pytest.approx(1 / 3)
    assert strict["raw_predictions"] == 4 and strict["precision_denominator_strict"] == 3
    assert strict["alarms_false_or_duplicate_per_hour"] == 120
    assert strict["independent_clean_video_exposure_s"] == 0
    assert result["uncertain_exclusions"] == [{"prediction_id": "uncertain", "video": "a.mp4",
        "classification": "ignored_uncertain", "reason": "existing_events_eval:ignored_uncertain", "uncertain_label_ids": ["a:1"]}]
    assert result == run_s1.score_predictions(labels, list(reversed(predictions)), {"a.mp4": 60}, 2, records)


def test_certain_priority_and_no_predictions_are_reported_honestly():
    labels = [LabelEvent("a.mp4", "zone_intrusion", 10, 12),
              LabelEvent("a.mp4", "zone_intrusion", 10, 12, uncertain=True)]
    records = [{"label_id": f"a:{index}", **vars_for_label(label)} for index, label in enumerate(labels)]
    predictions = [PredEvent("a.mp4", "zone_intrusion", 10, 12, str(index)) for index in range(2)]
    result = run_s1.score_predictions(labels, predictions, {"a.mp4": 60}, 2, records)
    assert [row["classification"] for row in result["prediction_classifications"]] == ["tp", "duplicates"]
    assert not result["uncertain_exclusions"]
    assert len(result["excluded_uncertain_labels"]) == 1
    empty = run_s1.score_predictions(labels, [], {"a.mp4": 60}, 2, records)
    assert empty["evaluator"]["overall"]["fn"] == 1
    assert empty["supplemental_strict"]["strict_precision"] == 0


def test_output_refusal_preserves_existing_results(tmp_path):
    evidence = tmp_path / "report.json"
    evidence.write_text("preserved")
    with pytest.raises(SystemExit) as error:
        run_s1.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 2 and evidence.read_text() == "preserved"


@pytest.mark.parametrize("wrong_video", [False, True])
def test_missing_models_or_changed_video_fail_before_ort_or_detector_import(tmp_path, wrong_video):
    output = tmp_path / "output"
    output.mkdir()
    data, models = tmp_path / "missing-data", tmp_path / "missing-models"
    if wrong_video:
        models.mkdir()
        (models / "yolov8n-640.onnx").write_bytes(b"never loaded")
        recipe = json.loads((run_s1.ROOT / "configs/eval-s1.json").read_text())
        for source in recipe["source_files"].values():
            video = data / source["path"]
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(b"not the fixed original video")
    code = '''
import builtins, sys, time
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name in {"cv2", "onnxruntime", "guard.inference.onnx_cpu", "guard.pipeline.detector"}:
        raise AssertionError("unexpected inference import")
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from ml.eval import run_s1
args = run_s1.parser().parse_args(["--_run", "--_deadline", str(time.monotonic() + 10),
    "--output-dir", sys.argv[1], "--data-dir", sys.argv[2], "--models-dir", sys.argv[3]])
raise SystemExit(run_s1._run(args))
'''
    result = subprocess.run([sys.executable, "-c", code, str(output), str(data), str(models)],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads((output / "report.json").read_text())
    assert not report["ok"] and "unexpected inference" not in report["error"]
    if wrong_video:
        assert "SHA256" in report["error"]
    assert report["coverage"]["videos_completed"] == 0
    assert not (output / "evaluation.json").exists() and not (output / "guard.db").exists()


def test_watchdog_is_bounded_and_json_reports_timeout(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode, calls = 123456789, None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("runner", timeout)
            self.returncode = -signal.SIGTERM

    signals = []
    monkeypatch.setattr(run_s1.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(run_s1.os, "killpg", lambda pid, sig: signals.append(sig))
    output = tmp_path / "new"
    assert run_s1.main(["--output-dir", str(output), "--timeout", "1", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and not report["ok"]
    assert signals == [signal.SIGTERM]


def test_help_does_not_load_runtime():
    result = subprocess.run([sys.executable, "-m", "ml.eval.run_s1", "--help"], text=True, capture_output=True, timeout=10)
    assert result.returncode == 0 and "--timeout" in result.stdout


@pytest.mark.parametrize("closed_on_previous_frame", [False, True])
def test_lifecycle_persistence_coalesces_reopen_and_preserves_operator_origin_and_clip_error(tmp_path, closed_on_previous_frame):
    from guard.contracts import CameraConfig, Violation
    from guard.events.journal import Journal
    from guard.events.manager import EventManager

    camera = CameraConfig.model_validate({"camera_id": "lifecycle", "name": "synthetic", "site": "unit",
        "source": {"type": "file", "uri": "never-opened"}, "rules": [{"rule_id": "r", "type": "zone_intrusion",
            "min_duration_s": 0, "min_ratio": 1, "close_after_s": 2, "cooldown_s": 30}]})
    manager = EventManager()

    def process(ts, track_id=None):
        violations = [] if track_id is None else [Violation(rule_id="r", type="zone_intrusion", severity="high",
            track_id=track_id, zone_id=None, bbox=(0, 0, 100, 200))]
        return manager.process_with_updates(camera, ts, violations)

    journal = Journal(tmp_path / "events.db")
    try:
        created, _, _ = process(0, 1)
        event = created[0]
        event.details["origin"] = {"capture_id": "original", "source_pos_s": 0}
        journal.add(event)
        journal.update(event.id, status="confirmed", operator_note="keep operator", details_patch={"clip_error": True})
        process(1, 1)
        if closed_on_previous_frame:
            _, closed, reopened = process(3.5)
            run_s1.persist_lifecycle_updates(journal, closed, reopened)
            assert journal.get(event.id).ended_at == 1
        new, closed, reopened = process(4, 2)
        assert not new and len(reopened) == 1
        assert len(closed) == (0 if closed_on_previous_frame else 1)
        assert run_s1.persist_lifecycle_updates(journal, closed, reopened) == []
        stored = journal.get(event.id)
        assert stored.ended_at is None and stored.track_id == 1
        assert stored.status == "confirmed" and stored.operator_note == "keep operator"
        assert stored.details == {"origin": event.details["origin"], "clip_error": True,
                                  "reopened_count": 1, "active_duration_s": 1}
        _, closed, reopened = process(7)
        assert not reopened and len(run_s1.persist_lifecycle_updates(journal, closed, reopened)) == 1
        assert journal.get(event.id).ended_at == 4
        assert journal.get(event.id).details == stored.details
        assert journal.query()[1] == 1
    finally:
        journal.close()
