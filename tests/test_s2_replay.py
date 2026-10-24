"""Offline replay invariants using synthetic cached detections and real state."""

from __future__ import annotations

import copy
import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from guard.contracts import CameraConfig, Detection
from guard.pipeline import compat
from guard.rules.base import build_rules, load_plugins
from ml.eval import replay_s2 as replay, run_s2
from ml.eval.events_eval import predictions_from_events


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))


def jsonl(path, rows):
    path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in rows))


@pytest.fixture
def source(tmp_path, monkeypatch):
    """Independent source run uses the original run_s2 component order."""
    directory = tmp_path / "source"
    directory.mkdir()
    recipe = replay.read(replay.ROOT / "configs/eval-s2.json")
    policy = recipe["candidates"]["buildvision"]
    def label(start, end, note, uncertain=False):
        return {"type": "ppe_missing", "t_start": start, "t_end": end, "uncertain": uncertain, "note": note}
    clips = [
        {"name": "s2_01.mp4", "events": [label(1, 3, "A"), label(5, 8, "B"),
          label(5, 8, recipe["primary_scope_exclusions"][0]["expected_note"], True)], "segments": [{"source": "source0"}]},
        {"name": "s2_02.mp4", "events": [label(0, 5, "C"), label(0, 5, "D"), label(0, 5, "E"),
          label(0, 5, recipe["primary_scope_exclusions"][1]["expected_note"], True), label(8.5, 11.9, "F")],
          "segments": [{"source": "sourceA"}, {"source": "sourceB"}]},
        *[{"name": video, "events": [], "segments": [{"source": video}]} for video in run_s2.VIDEOS[2:]]]
    for exclusion in recipe["primary_scope_exclusions"]:
        clip = next(row for row in clips if row["name"] == exclusion["video"])
        exclusion["expected_label_sha256"] = replay.fingerprint(clip["events"][exclusion["index"]])
    manifest = {"videos": clips}
    save(directory / "manifest.snapshot.json", manifest)
    recipe["manifest_sha256"] = run_s2.asset(directory / "manifest.snapshot.json")["sha256"]
    monkeypatch.setattr(run_s2, "MANIFEST_SHA", recipe["manifest_sha256"])
    monkeypatch.setattr(replay, "EXPECTED_SAMPLES", 240)
    save(directory / "effective_config.json", recipe)
    original, primary, exclusions = run_s2.label_scopes(clips, recipe)
    for name, rows in (("original_labels.json", original), ("primary_labels.json", primary), ("scope_exclusions.json", exclusions)):
        save(directory / name, rows)
    classes = ["person", *[f"other{index}" for index in range(79)]]
    models = {"det": {"model_id": "det", "onnx_path": recipe["det"]["onnx_path"], "classes": classes,
                       "conf_threshold": .35, "iou_threshold": .5},
              "ppe": {"model_id": "ppe", "onnx_path": policy["onnx_path"], "classes": policy["classes"],
                       "conf_threshold": .35, "iou_threshold": .7}}
    for model in models.values():
        model.update(input_size=[640, 640], input_dtype="float32", output_format="yolov8", tpu_program=None)
    fake_asset = lambda marker: {"bytes": 100, "sha256": hashlib.sha256(marker.encode()).hexdigest()}
    encoder = {"manifest_encode": {"fps": 15, "width": 1280, "height": 720}, "ffmpeg_version": "synthetic-ffmpeg"}
    prepared = {video: {**fake_asset(video), "frames": 180, "fps": 15, "width": 1280, "height": 720, "duration_s": 12}
                for video in run_s2.VIDEOS}
    assets = {**{video: {key: value[key] for key in ("bytes", "sha256")} for video, value in prepared.items()},
              "manifest": run_s2.asset(directory / "manifest.snapshot.json"), "encode_report": fake_asset("encode"),
              "source:sourceA": fake_asset("A"), "source:sourceB": fake_asset("B"),
              "det_onnx": {"sha256": recipe["det"]["sha256"]}, "ppe_onnx": {"sha256": policy["sha256"]}}
    mapping = {f"eval_{Path(clip['name']).stem}": {"video": clip["name"], "t0": 0, "source_segments": clip["segments"]} for clip in clips}
    durations = dict.fromkeys(run_s2.VIDEOS, 12)
    save(directory / "mapping.json", mapping)
    save(directory / "durations.json", durations)
    report = {"schema_version": 1, "status": "completed", "ok": True, "candidate": "buildvision", "effective_recipe": recipe,
              "effective_models": models, "assets": assets, "prep_provenance": {"encode": {"outputs": prepared, "encoder": encoder}},
              "coverage": {"frames_processed": 240, "selected_frames_expected": 240, "videos_completed": 4}, "per_video": {}}
    predictions = []
    load_plugins()
    class SyntheticDetector:
        def __init__(self, model_id):
            self.name, self.detections = model_id, []
        def detect(self, frame, timings):
            timings.update(infer=1., preprocess=0., postprocess=0., queue=0., transfer=0.)
            return self.detections
    with compat.Journal(directory / "guard.db") as journal:
        for video in run_s2.VIDEOS:
            camera = CameraConfig.model_validate({"camera_id": f"eval_{Path(video).stem}", "name": video, "site": "unit",
                "source": {"type": "file", "uri": "never-opened"}, "sample_fps": 5, "rules": [{"rule_id": "s2", **recipe["rule"]}]})
            save(directory / f"camera_{Path(video).stem}.json", camera.model_dump(mode="json"))
            manager, tracker, rules = compat.EventManager(), compat.Tracker(classes={"person"}, **recipe["tracker"]), build_rules(camera)
            detectors = {name: SyntheticDetector(name) for name in ("det", "ppe")}
            stats = dict(created_events=0, close_notifications=0, reopen_transitions=0, frames_with_violations=0,
                         frames_processed=60, selected_frames_expected=60, selected_frames_decoded=60, source_frames_traversed=180)
            traces, lifecycle = [], []
            for row in run_s2.run_s1.sample_plan(15, 12):
                row["source_time_s"] = row["source_frame_index"] / 15
                ts = row["source_time_s"]
                missing = video == "s2_02.mp4" and (ts < 4 or ts >= 8.8)
                name = "no_helmet" if missing else "helmet"
                detectors["det"].detections = [Detection("det", "person", 0, .9, (0, 0, 100, 200))]
                detectors["ppe"].detections = [Detection("ppe", name, policy["classes"].index(name), .9, (30, 10, 50, 30))]
                result = run_s2.process_frame(camera, np.zeros((720, 1280, 3), np.uint8), row, detectors, tracker, manager, rules)
                for event in result["new"]:
                    event.details["origin"] = {"capture_id": f"s2-quality:{video}", "frame_seq": row["source_frame_index"], "source_pos_s": ts}
                    journal.add(event)
                    predictions.append(event.model_copy(deep=True))
                run_s2.run_s1.persist_lifecycle_updates(journal, result["closed"], result["reopened_events"])
                if result["new"] or result["closed"] or result["reopened"]:
                    lifecycle.append({"source_time_s": ts, "created": [event.id for event in result["new"]],
                                      "closed": [{"id": event.id, "ended_at": event.ended_at} for event in result["closed"]], "reopened": result["reopened"]})
                trace = {**result["trace"], "capture_id": f"s2-quality:{video}", "decoder_pos_s": ts,
                         "raw_outputs": {model_id: [{"shape": [1, 4 + len(model["classes"]), 8400], "dtype": "float32",
                            "bytes": (4 + len(model["classes"])) * 8400 * 4, "finite": True, "sha256": "a" * 64}]
                            for model_id, model in models.items()}}
                traces.append(copy.deepcopy(trace))
                stats["created_events"] += len(result["new"])
                stats["close_notifications"] += len(result["closed"])
                stats["reopen_transitions"] += len(result["reopened"])
                stats["frames_with_violations"] += bool(trace["violations"])
            jsonl(directory / f"samples_{Path(video).stem}.jsonl", traces)
            jsonl(directory / f"lifecycle_{Path(video).stem}.jsonl", lifecycle)
            report["per_video"][video] = {"coverage": stats, "coco_track_identity_sha256": hashlib.sha256(
                "".join(row["identity_sha256"] + "\n" for row in traces).encode()).hexdigest(),
                "right_censored_open_events": [event.model_dump(mode="json") for event in manager.open_events()]}
        stored, _ = journal.query(limit=100)
        save(directory / "journal_events.json", [event.model_dump(mode="json") for event in stored])
    save(directory / "predictions.json", [event.model_dump(mode="json") for event in predictions])
    report["evaluation"] = run_s2.evaluate_scopes(original, primary, predictions_from_events(predictions, mapping), durations, recipe)
    report["output_hashes"] = {path.name: run_s2.asset(path) for path in directory.iterdir() if path.is_file()}
    save(directory / "report.json", report)
    evidence = {"schema_version": 1, "status": "verified", "video": "s2_02.mp4", "boundary_basis": "encoded_filter_frame_count",
                "encoder": encoder, "encoder_canonical_json_sha256": replay.fingerprint(encoder), "ffmpeg_version": "synthetic-ffmpeg",
                "encoded_video": assets["s2_02.mp4"], "source_manifest": assets["manifest"], "encode_report": assets["encode_report"],
                "sources": {key: assets[f"source:{key}"] for key in ("sourceA", "sourceB")},
                "parts": {"part0": {"frame_count": 90, "last": {"showinfo_checksum": "old"}},
                          "part1": {"frame_count": 90, "first": {"showinfo_checksum": "new"}}, "joined": {"frame_count": 180}},
                "joined_boundary_frames": [{"frame_index": 89, "showinfo_checksum": "old"}, {"frame_index": 90, "showinfo_checksum": "new"}],
                "boundary_checks": {"count": True, "checksum": True},
                "boundary": {"indexing": "zero-based", "cut_before_source_frame": 90, "cut_time_s": 6,
                    "last_old_selected_sample": {"sample_index": 29, "source_frame_index": 87, "source_time_s": 5.8},
                    "first_new_selected_sample": {"sample_index": 30, "source_frame_index": 90, "source_time_s": 6}}}
    boundary_path = tmp_path / "boundary.json"
    save(boundary_path, evidence)
    return directory, run_s2.asset(directory / "report.json")["sha256"], boundary_path


def rewrite_source(directory, name, mutate):
    path = directory / name
    value = replay.lines(path) if name.endswith("jsonl") else replay.read(path)
    mutate(value)
    (jsonl if name.endswith("jsonl") else save)(path, value)
    report = replay.read(directory / "report.json")
    report["output_hashes"][name] = run_s2.asset(path)
    save(directory / "report.json", report)
    return run_s2.asset(directory / "report.json")["sha256"]


def test_continuous_replay_exactly_matches_source_and_never_imports_model_runtime(source, tmp_path, monkeypatch):
    import builtins
    directory, sha, _ = source
    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"onnx", "onnxruntime", "torch", "torchvision", "transformers"}:
            raise AssertionError(f"Model import during cached replay: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    output = tmp_path / "control"
    assert replay.run(directory, sha, output) == 0
    report = replay.read(output / "report.json")
    assert report["control_parity"]["status"] == "matched" and report["frames_processed"] == 240
    assert report["inference_executed"] is False and report["quality_primary"] is False
    assert "source_boundary_diagnostic" not in report
    for video in run_s2.VIDEOS:
        assert (output / "source_cache" / f"samples_{Path(video).stem}.jsonl").read_bytes() == (directory / f"samples_{Path(video).stem}.jsonl").read_bytes()


def test_boundary_closes_last_violation_before_reset_then_reconfirms_without_changing_any_cache_or_gt(source, tmp_path):
    directory, sha, evidence = source
    cache_before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    output = tmp_path / "diagnostic"
    assert replay.run(directory, sha, output, evidence, run_s2.asset(evidence)["sha256"]) == 0
    report = replay.read(output / "report.json")
    baseline = report["continuous_control"]["per_video"]["s2_02.mp4"]
    treatment = report["source_boundary_diagnostic"]["per_video"]["s2_02.mp4"]
    assert baseline["coverage"]["created_events"] == 1 and baseline["coverage"]["reopen_transitions"] == 1
    assert treatment["coverage"]["created_events"] == 2 and treatment["coverage"]["reopen_transitions"] == 0
    reset = treatment["boundary_resets"][0]
    assert reset["before_sample_index"] == 30 and len(reset["closed"]) == 1
    old = replay.lines(directory / "samples_s2_02.jsonl")
    measured = max(row["source_time_s"] for row in old if row["source_time_s"] < 6 and row["violations"])
    assert reset["closed"][0]["ended_at"] == measured < 5.8 < 6
    new = replay.lines(output / "source-boundary-diagnostic/samples_s2_02.jsonl")
    assert len(new) == len(old) == 60 and new[30]["segment_index"] == 1
    assert new[30]["person_tracks"] == []  # new tracker needs min_hits again
    assert new[31]["coco_tracks_before_attributes"][0]["first_seen"] == 6
    for original, row in zip(old, new):
        assert {key: row[key] for key in replay.SAMPLE_KEYS} == {key: original[key] for key in replay.SAMPLE_KEYS}
        assert row["raw_detections"] == original["raw_detections"] and row["raw_outputs"] == original["raw_outputs"]
        assert row["detection_sha256"] == replay.fingerprint(original["raw_detections"])
    assert all(path.read_bytes() == cache_before[path.name] for path in directory.iterdir() if path.is_file())
    assert report["continuous_control"]["frames_processed"] == report["source_boundary_diagnostic"]["frames_processed"] == 240


def test_boundary_reused_track_ids_have_distinct_capture_origins_in_predictions_and_journal(source, tmp_path):
    directory, sha, evidence = source
    output = tmp_path / "capture-origin"
    assert replay.run(directory, sha, output, evidence, run_s2.asset(evidence)["sha256"]) == 0
    treatment = output / "source-boundary-diagnostic"
    events = [event for event in replay.read(treatment / "predictions.json") if event["camera_id"] == "eval_s2_02"]
    assert len(events) == 2 and events[0]["track_id"] == events[1]["track_id"]
    origins = [event["details"]["origin"] for event in events]
    assert [origin["segment_index"] for origin in origins] == [0, 1]
    assert origins[0]["capture_id"] != origins[1]["capture_id"]
    assert {origin["source_capture_id"] for origin in origins} == {"s2-quality:s2_02.mp4"}
    cached = replay.lines(directory / "samples_s2_02.jsonl")
    trace = replay.lines(treatment / "samples_s2_02.jsonl")
    for event, origin in zip(events, origins):
        original = next(row for row in cached if row["source_frame_index"] == origin["frame_seq"])
        effective = next(row for row in trace if row["source_frame_index"] == origin["frame_seq"])
        assert origin["source_row_sha256"] == replay.fingerprint(original)
        assert effective["capture_id"] == origin["capture_id"]
        assert effective["source_capture_id"] == original["capture_id"]
    journal = {event["id"]: event for event in replay.read(treatment / "journal_events.json")}
    assert all(journal[event["id"]]["details"]["origin"] == event["details"]["origin"] for event in events)
    baseline = replay.read(output / "continuous-control/predictions.json")
    assert all(event["details"]["origin"]["capture_id"] == "s2-quality:s2_02.mp4" for event in baseline)
    assert all("source_capture_id" not in event["details"]["origin"] for event in baseline)
    assert trace[29]["capture_id"] != trace[30]["capture_id"]


@pytest.mark.parametrize("mutation,error", [
    (lambda rows: rows.pop(), "Incomplete"),
    (lambda rows: rows.reverse(), "coordinate/order"),
    (lambda rows: rows[0]["raw_detections"]["ppe"][0].update(cls="wrong"), "Malformed"),
    (lambda rows: rows[0]["raw_outputs"]["ppe"][0].update(sha256="missing"), "raw-output hash"),
    (lambda rows: rows[0].pop("raw_detections"), "raw_detections"),
    (lambda rows: rows[0].update(identity_sha256="wrong"), "identity fingerprint"),
])
def test_complete_hash_pin_does_not_excuse_malformed_or_partial_cache(source, mutation, error):
    directory, _, _ = source
    sha = rewrite_source(directory, "samples_s2_02.jsonl", mutation)
    with pytest.raises((ValueError, KeyError), match=error):
        replay.load_cache(directory, sha)


def test_pinned_report_or_cache_drift_is_refused_before_replay(source):
    directory, sha, _ = source
    with pytest.raises(ValueError, match="report SHA256"):
        replay.load_cache(directory, "0" * 64)
    with (directory / "samples_s2_01.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="bytes/hash"):
        replay.load_cache(directory, sha)


@pytest.mark.parametrize("mutate", [lambda proof: proof.update(boundary_basis="GT20.1"),
    lambda proof: proof["boundary"].update(cut_before_source_frame=90.5),
    lambda proof: proof["sources"]["sourceA"].update(sha256="0" * 64),
    lambda proof: proof["parts"]["part1"].update(frame_count=91),
    lambda proof: proof["boundary"].update(first_new_selected_sample={}),
    lambda proof: proof["encoder"].update(ffmpeg_version="different")])
def test_boundary_must_come_from_integer_encoder_parts_not_gt_or_quality_selection(source, mutate):
    directory, sha, path = source
    cache = replay.load_cache(directory, sha)
    evidence = replay.read(path)
    mutate(evidence)
    save(path, evidence)
    with pytest.raises(ValueError):
        replay.load_boundaries(path, run_s2.asset(path)["sha256"], cache)


def test_control_semantic_drift_blocks_treatment_even_with_self_consistent_source_pin(source, tmp_path):
    directory, _, boundary = source
    sha = rewrite_source(directory, "journal_events.json", lambda rows: rows[0].update(ended_at=11.123))
    output = tmp_path / "blocked"
    assert replay.run(directory, sha, output, boundary, run_s2.asset(boundary)["sha256"]) == 1
    report = replay.read(output / "report.json")
    assert report["status"] == "failed" and "journal/closure" in report["error"]
    assert not (output / "source-boundary-diagnostic").exists()


def test_existing_output_is_never_overwritten(source, tmp_path):
    directory, sha, _ = source
    output = tmp_path / "existing"
    output.mkdir()
    (output / "preserved").write_bytes(b"keep")
    with pytest.raises(ValueError, match="never overwritten"):
        replay.run(directory, sha, output)
    assert (output / "preserved").read_bytes() == b"keep"


def test_boundary_persistence_failure_does_not_reset_state_or_publish_diagnostic_success(source, tmp_path, monkeypatch):
    directory, sha, boundary = source
    real_journal, locations = compat.Journal, {}
    def remember(path, *args, **kwargs):
        journal = real_journal(path, *args, **kwargs)
        locations[id(journal)] = str(path)
        return journal
    monkeypatch.setattr(compat, "Journal", remember)
    real_persist = run_s2.run_s1.persist_lifecycle_updates
    def failed(journal, closed, reopened):
        if closed and "source-boundary-diagnostic" in locations[id(journal)]:
            raise sqlite3.OperationalError("boundary closure unavailable")
        return real_persist(journal, closed, reopened)
    monkeypatch.setattr(run_s2.run_s1, "persist_lifecycle_updates", failed)
    resets = []
    real_reset = compat.EventManager.reset
    def record_reset(manager, camera_id):
        resets.append(camera_id)
        return real_reset(manager, camera_id)
    monkeypatch.setattr(compat.EventManager, "reset", record_reset)
    output = tmp_path / "failed-boundary"
    assert replay.run(directory, sha, output, boundary, run_s2.asset(boundary)["sha256"]) == 1
    report = replay.read(output / "report.json")
    assert report["status"] == "failed" and report["control_parity"]["status"] == "matched"
    assert "boundary closure unavailable" in report["error"] and "source_boundary_diagnostic" not in report
    assert resets == []
    with real_journal(output / "source-boundary-diagnostic/guard.db") as journal:
        rows, count = journal.query(limit=100)
        assert count == 1 and rows[0].ended_at is None


def test_cached_detector_returns_fresh_objects_without_mutating_source_detections_or_timings(source):
    directory, sha, _ = source
    cache = replay.load_cache(directory, sha)
    original = copy.deepcopy(cache.rows["s2_02.mp4"][0])
    detector = replay.CachedDetector("det")
    detector.row = cache.rows["s2_02.mp4"][0]
    timings = {}
    detections = detector.detect(None, timings)
    detections[0].conf = .01
    detections[0].bbox = (1, 2, 3, 4)
    timings["infer"] = 10000
    assert detector.row == original


@pytest.mark.parametrize("location", ["report", "sample"])
def test_legacy_flat_cache_refuses_roi_ownership_or_unknown_masks(source, location):
    directory, _, _ = source
    if location == "report":
        path = directory / "report.json"
        value = replay.read(path)
        value["ppe_input_mode"] = "person_crop"
        save(path, value)
        sha = run_s2.asset(path)["sha256"]
    else:
        sha = rewrite_source(directory, "samples_s2_02.jsonl",
                             lambda rows: rows[0].update(ppe_input_mode="person_crop"))
    with pytest.raises(ValueError, match="flat replay does not support person_crop"):
        replay.load_cache(directory, sha)
