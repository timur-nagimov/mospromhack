"""S2 input/coverage/scope regressions; never loads a pretrained model."""

import copy
import hashlib
import json
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from types import ModuleType

import pytest

from ml.eval import run_s2
from ml.eval.events_eval import PredEvent


@pytest.fixture
def recipe():
    return run_s2.audit.read_json(run_s2.ROOT / "configs/eval-s2.json")


def test_recipe_freezes_runtime_defaults_without_main_duration_tuning(recipe):
    actual = run_s2.load_recipe(run_s2.ROOT / "configs/eval-s2.json")
    assert actual == recipe
    assert actual["rule"]["params"] == {"required": ["helmet"], "min_box_h_px": 60}
    assert actual["rule"]["min_duration_s"] == 2
    assert actual["duration"] == "full_prepared_clip" and set(actual["negative_videos"]) == run_s2.CLEAN


@pytest.mark.parametrize("change", ["min_duration", "required", "tracker", "det_threshold", "ppe_threshold", "model_hash", "manifest_pin", "missing_clean", "scope"])
def test_changed_fixed_policy_is_rejected(recipe, tmp_path, change):
    if change == "min_duration":
        recipe["rule"]["min_duration_s"] = 1.5
    elif change == "required":
        recipe["rule"]["params"]["required"] = ["helmet", "vest"]
    elif change == "tracker":
        recipe["tracker"]["min_hits"] = 1
    elif change == "det_threshold":
        recipe["det"]["conf_threshold"] = .25
    elif change == "ppe_threshold":
        recipe["ppe_thresholds"]["iou_threshold"] = .5
    elif change == "model_hash":
        recipe["candidates"]["vyra"]["sha256"] = "0" * 64
    elif change == "manifest_pin":
        recipe["manifest_sha256"] = "0" * 64
    elif change == "missing_clean":
        recipe["videos"].remove("clean_02.mp4")
    else:
        recipe["primary_scope_exclusions"].pop()
    path = tmp_path / "recipe.json"
    run_s2.write(path, recipe)
    with pytest.raises(ValueError):
        run_s2.load_recipe(path)


@pytest.fixture
def inputs(tmp_path, recipe):
    """Small synthetic byte assets exercise preflight, never inference or quality."""
    data, models = tmp_path / "data", tmp_path / "models"
    data.mkdir()
    models.mkdir()

    def event(start, end, note, uncertain=False):
        return {"type": "ppe_missing", "t_start": start, "t_end": end, "uncertain": uncertain, "note": note}

    clips = [
        {"name": "s2_01.mp4", "title": "positive1", "scenario": "S2", "segments": [{"source": "source0"}],
         "events": [event(18.0, 25.0, "certain A"), event(37.5, 46.5, "certain B"),
                    event(37.5, 46.5, recipe["primary_scope_exclusions"][0]["expected_note"], True)]},
        {"name": "s2_02.mp4", "title": "positive2", "scenario": "S2", "segments": [{"source": "source1"}, {"source": "source2"}],
         "events": [event(0.0, 20.1, "same time actor A"), event(0.0, 20.1, "same time actor B"), event(0.0, 20.1, "same time actor C"),
                    event(0.0, 20.1, recipe["primary_scope_exclusions"][1]["expected_note"], True), event(20.1, 35.0, "certain D")]},
        *[{"name": f"clean_0{index}.mp4", "title": "explicit clean", "scenario": "clean", "segments": [{"source": f"source{index+2}"}], "events": []} for index in (1, 2)],
    ]
    sources, records = {}, []
    (data / "raw").mkdir()
    for index in range(5):
        path = data / "raw" / f"source{index}.mp4"
        path.write_bytes(f"synthetic source {index}".encode())
        info = run_s2.asset(path)
        sources[f"source{index}"] = {"sha256": info["sha256"], "bytes": info["bytes"]}
        records.append({"id": f"source{index}", **info})
    manifest = {"videos": clips, "sources": sources}
    run_s2.write(data / recipe["manifest"], manifest)
    recipe["manifest_sha256"] = run_s2.run_s1._sha(data / recipe["manifest"])
    download = {"manifest_git": recipe["manifest_git"], "manifest_sha256": recipe["manifest_sha256"], "sources": records}
    run_s2.write(data / recipe["download_report"], download)
    (data / "videos").mkdir()
    (data / "labels").mkdir()
    outputs = {}
    for clip in clips:
        video = data / "videos" / clip["name"]
        video.write_bytes(f"synthetic encoded {clip['name']}".encode())
        info = run_s2.asset(video)
        outputs[clip["name"]] = {"relative_path": f"videos/{clip['name']}", "sha256": info["sha256"], "bytes": info["bytes"],
            "frames": 900, "fps": 15, "duration_s": 60, "width": 1280, "height": 720}
        run_s2.write(data / "labels" / f"{Path(clip['name']).stem}.json", {"video": clip["name"], "events": clip["events"]})
    encoded = {"schema_version": 1, "manifest_sha256": recipe["manifest_sha256"], "source_manifest_git": recipe["manifest_git"],
               "source_report_sha256": run_s2.run_s1._sha(data / recipe["download_report"]), "outputs": outputs,
               "tool_versions": {"opencv": "fixture", "imageio_ffmpeg": "fixture"},
               "encoder": {"manifest_encode": {"fps": 15}, "ffmpeg_version": "fixture"}}
    run_s2.write(data / recipe["encode_report"], encoded)
    coco = ["person", *[f"class{index}" for index in range(79)]]
    det = models / "yolov8n-640.onnx"
    det.write_bytes(b"synthetic det; never executed")
    det_info = run_s2.asset(det)
    recipe["det"]["sha256"] = det_info["sha256"]
    run_s2.write(models / "det.json", {"model_id": "det", "onnx": det.name, "onnx_sha256": det_info["sha256"],
        "onnx_bytes": det_info["bytes"], "classes": coco, "input_size": [640, 640], "output_format": "yolov8"})
    for name, policy in recipe["candidates"].items():
        ppe = models / policy["onnx_path"]
        ppe.parent.mkdir(parents=True)
        ppe.write_bytes(f"synthetic PPE {name}; never executed".encode())
        info = run_s2.asset(ppe)
        policy["sha256"] = info["sha256"]
        run_s2.write(ppe.parent / "ppe.json", {"model_id": "ppe", "onnx": ppe.name, "onnx_sha256": info["sha256"],
            "onnx_bytes": info["bytes"], "classes": policy["classes"], "checker": "passed", "source": policy["source"], "revision": policy["revision"],
            "input_size": [640, 640], "input_dtype": "float32", "output_format": "yolov8",
            "inputs": [{"shape": [1, 3, 640, 640], "dtype": 1}], "outputs": [{"shape": [1, 18, 8400], "dtype": 1}]})
        registry = tmp_path / f"models.{name}.yaml"
        run_s2.write(registry, {"models": [{"model_id": "det", "onnx_path": det.name, "input_size": [640, 640], "conf_threshold": .35, "iou_threshold": .5},
            {"model_id": "ppe", "onnx_path": policy["onnx_path"], "input_size": [640, 640], "classes": policy["classes"], "conf_threshold": .35, "iou_threshold": .7}]})
        policy["registry"] = str(registry)
    return data, models, recipe, clips


@pytest.mark.parametrize("candidate", ["buildvision", "vyra"])
def test_preflight_checks_all_four_positive_and_explicit_negative_inputs_without_ml(inputs, monkeypatch, candidate):
    import builtins

    data, models, recipe, _ = inputs
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"cv2", "onnxruntime", "torch", "guard.pipeline.detector", "guard.pipeline.engine", "guard.inference.onnx_cpu"}:
            raise AssertionError(f"ML/runtime imported in preflight: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    result = run_s2.preflight(data, models, recipe, candidate)
    assert len(result["original_labels"]) == 8 and len(result["primary_labels"]) == 6
    assert len(result["scope_exclusions"]) == 2 and set(result["prepared"]) == set(run_s2.VIDEOS)
    assert len([key for key in result["assets"] if key.startswith("source:")]) == 5


@pytest.mark.parametrize("change", ["missing_clean_GT", "implicit_clean", "clip_bytes", "raw_bytes", "prep_hash", "fps", "frames", "GT_notes", "ppe_mapping", "det_threshold"])
def test_preflight_refuses_partial_tampered_or_incomparable_assets(inputs, change):
    data, models, recipe, _ = inputs
    if change == "missing_clean_GT":
        (data / "labels/clean_02.json").unlink()
    elif change == "implicit_clean":
        run_s2.write(data / "labels/clean_02.json", {"video": "clean_02.mp4"})
    elif change == "clip_bytes":
        (data / "videos/s2_02.mp4").write_bytes(b"changed")
    elif change == "raw_bytes":
        (data / "raw/source0.mp4").write_bytes(b"changed")
    elif change in {"prep_hash", "fps", "frames"}:
        path = data / recipe["encode_report"]
        encoded = run_s2.audit.read_json(path)
        if change == "prep_hash":
            encoded["source_report_sha256"] = "wrong"
        else:
            encoded["outputs"]["s2_01.mp4"][change] = 25 if change == "fps" else 100
        run_s2.write(path, encoded)
    elif change == "GT_notes":
        path = data / "labels/s2_01.json"
        labels = run_s2.audit.read_json(path)
        labels["events"][0]["note"] = "changed actor description"
        run_s2.write(path, labels)
    else:
        path = Path(recipe["candidates"]["buildvision"]["registry"])
        registry = run_s2.audit.read_json(path)
        if change == "ppe_mapping":
            registry["models"][1]["classes"] = list(reversed(registry["models"][1]["classes"]))
        else:
            registry["models"][0]["conf_threshold"] = .25
        run_s2.write(path, registry)
    with pytest.raises((ValueError, FileNotFoundError)):
        run_s2.preflight(data, models, recipe, "buildvision")


def test_scope_is_frozen_by_original_hash_note_and_preserves_original_labels(inputs):
    _, _, recipe, clips = inputs
    original_clips = copy.deepcopy(clips)
    original, primary, exclusions = run_s2.label_scopes(clips, recipe)
    assert clips == original_clips and len(original) == 8 and len(primary) == 6
    assert {row["label_id"] for row in exclusions} == {"s2_01.mp4:2", "s2_02.mp4:3"}
    clips[0]["events"][2]["note"] += " changed"
    with pytest.raises(ValueError, match="hash/note"):
        run_s2.label_scopes(clips, recipe)


class Capture:
    def __init__(self, actual, fail_retrieve=False):
        self.actual, self.next, self.fail_retrieve = actual, 0, fail_retrieve

    def grab(self):
        self.next += 1
        return self.next <= self.actual

    def retrieve(self):
        return not self.fail_retrieve, SimpleNamespace(size=1)


def test_sampling_covers_full_duration_beyond_sixty_seconds_and_last_source_frame():
    stats = {}
    rows = list(run_s2.sampled_frames(Capture(1051), {"frames": 1051, "fps": 15}, stats, time.monotonic() + 10))
    assert len(rows) == 351 and rows[-1][0]["source_frame_index"] == 1050
    assert rows[-1][0]["source_time_s"] == 70
    assert stats["source_frames_traversed"] == stats["source_frames_expected"] == 1051
    assert stats["selected_frames_decoded"] == stats["selected_frames_expected"] == 351


@pytest.mark.parametrize("actual,fail_decode", [(99, False), (101, False), (100, True)])
def test_partial_eof_extra_frames_and_selected_decode_failure_are_errors(actual, fail_decode):
    stats = {}
    with pytest.raises(ValueError):
        list(run_s2.sampled_frames(Capture(actual, fail_decode), {"frames": 100, "fps": 15}, stats, time.monotonic() + 10))
    assert stats["frames_processed"] == 0


def test_inference_failure_propagates_before_tracker_rule_event_or_negative_sample():
    class FailingDetector:
        def detect(self, frame, timings):
            raise RuntimeError("real backend failure")

    class NeverTracker:
        def update(self, *args):
            raise AssertionError("Tracker must not consume a failed inference")

    with pytest.raises(RuntimeError, match="real backend failure"):
        run_s2.process_frame(None, None, {"source_time_s": 0}, {"det": FailingDetector()}, NeverTracker(), None, [])


def test_real_tracker_attributes_s2_and_events_preserve_coco_identity_across_ppe_observations(recipe):
    """Synthetic upstream detections exercise real components, not model quality."""
    import numpy as np
    from guard.contracts import CameraConfig, Detection
    from guard.pipeline import compat
    from guard.rules.base import build_rules, load_plugins

    class Detector:
        def __init__(self, detections):
            self.detections = detections

        def detect(self, frame, timings):
            return self.detections

    camera = CameraConfig.model_validate({"camera_id": "unit_s2", "name": "synthetic", "site": "unit",
        "source": {"type": "file", "uri": "never-opened"}, "rules": [{"rule_id": "s2", **recipe["rule"]}]})
    load_plugins()
    rules = build_rules(camera)
    fingerprints, event_counts = {}, {}
    for helmet in (True, False):
        tracker, manager = compat.Tracker(classes={"person"}, **recipe["tracker"]), compat.EventManager()
        detectors = {"det": Detector([Detection("det", "person", 0, .9, (0, 0, 100, 200))]),
                     "ppe": Detector([Detection("ppe", "helmet" if helmet else "no_helmet", 1, .9, (30, 10, 50, 30))])}
        rows, created = [], []
        for index in range(15):
            result = run_s2.process_frame(camera, np.zeros((220, 120, 3), np.uint8),
                {"sample_index": index, "target_s": index / 5, "source_frame_index": index * 3, "source_time_s": index / 5},
                detectors, tracker, manager, rules)
            rows.append(result["trace"])
            created.extend(result["new"])
        fingerprints[helmet] = [row["identity_sha256"] for row in rows]
        event_counts[helmet] = len(created)
        assert rows[-1]["person_tracks"][0]["attributes"]["helmet"] is helmet
        json.dumps(rows[-1], allow_nan=False)
    assert fingerprints[True] == fingerprints[False]
    assert event_counts == {True: 0, False: 1}


def test_vest_only_uncertain_cannot_mask_helmet_fp_and_clean_hour_has_own_denominator(recipe):
    # Deliberately disjoint minimal labels isolate evaluator scope composition;
    # these synthetic labels are not a claim about the public GT or actors.
    original = [{"label_id": "vest-only", "video": "s2_01.mp4", "type": "ppe_missing", "t_start": 10,
                 "t_end": 15, "uncertain": True, "note": "vest only"}]
    preserved = copy.deepcopy(original)
    predictions = [PredEvent("s2_01.mp4", "ppe_missing", 11, 13, "helmet-FP"),
                   PredEvent("clean_01.mp4", "ppe_missing", 1, 3, "clean-FP1"),
                   PredEvent("clean_02.mp4", "ppe_missing", 1, 3, "clean-FP2")]
    durations = {"s2_01.mp4": 60, "s2_02.mp4": 40, "clean_01.mp4": 30, "clean_02.mp4": 10}
    report = run_s2.evaluate_scopes(original, [], predictions, durations, recipe)
    primary, legacy = report["primary_helmet_only"], report["secondary_original_legacy"]
    assert original == preserved
    assert primary["global"]["evaluator"]["overall"]["fp"] == 3
    assert legacy["global"]["evaluator"]["overall"]["ignored_uncertain"] == 1
    assert primary["clean_fp_count"] == 2 and primary["clean_exposure_s"] == 40
    assert primary["clean_fp_per_hour"] == 180
    assert primary["global"]["supplemental_strict"]["positive_video_exposure_s"] == 100
    assert primary["global"]["supplemental_strict"]["alarms_rate_exposure_s"] == 140
    assert primary["global"]["supplemental_strict"]["annotated_helmet_clean_exposure_s"] == 40


def test_process_frame_reports_same_frame_reopen_without_a_new_alarm():
    import numpy as np
    from guard.contracts import CameraConfig, Violation
    from guard.pipeline import compat

    camera = CameraConfig.model_validate({"camera_id": "s2_reopen", "name": "synthetic", "site": "unit",
        "source": {"type": "file", "uri": "never-opened"}, "rules": [{"rule_id": "s2", "type": "ppe_missing",
            "min_duration_s": 0, "min_ratio": 1, "close_after_s": 2, "cooldown_s": 30}]})

    class NoDetections:
        def detect(self, frame, timings):
            return []

    class ObservedViolation:
        def evaluate(self, context):
            return [Violation(rule_id="s2", type="ppe_missing", severity="high",
                track_id=1 if context.ts < 4 else 2, zone_id=None, bbox=(0, 0, 100, 200))]

    tracker, manager = compat.Tracker(classes={"person"}), compat.EventManager()
    frame = np.zeros((10, 10, 3), np.uint8)
    results = [run_s2.process_frame(camera, frame, {"source_time_s": ts, "source_frame_index": index},
        {"det": NoDetections()}, tracker, manager, [ObservedViolation()]) for index, ts in enumerate((0, 1, 4))]
    first, last = results[0]["new"][0], results[-1]
    assert not last["new"] and not last["closed"]
    assert last["reopened"] == [first.id] and len(last["reopened_events"]) == 1
    reopened = last["reopened_events"][0]
    assert reopened.ended_at is None and reopened.details == {"reopened_count": 1, "active_duration_s": 1}
    assert len(manager.open_events()) == 1 and manager.open_events()[0].id == first.id


def test_partial_video_durations_cannot_be_counted_as_complete_negative_exposure(recipe):
    with pytest.raises(ValueError, match="complete durations"):
        run_s2.evaluate_scopes([], [], [], {"s2_01.mp4": 60, "s2_02.mp4": 40}, recipe)


def test_duplicate_counts_remain_visible_in_both_metric_scopes(recipe):
    records = [{"label_id": "certain", "video": "s2_01.mp4", "type": "ppe_missing", "t_start": 10,
                "t_end": 20, "uncertain": False, "note": "no actor identity"}]
    predictions = [PredEvent("s2_01.mp4", "ppe_missing", 10, 12, "tp"), PredEvent("s2_01.mp4", "ppe_missing", 11, 13, "dup")]
    result = run_s2.evaluate_scopes(records, records, predictions, dict.fromkeys(run_s2.VIDEOS, 60), recipe)
    for scope in result.values():
        assert scope["global"]["evaluator"]["overall"]["precision"] == 1
        assert scope["global"]["supplemental_strict"]["strict_precision"] == .5
        assert scope["global"]["prediction_classifications"][1]["prediction_id"] == "dup"


def test_existing_output_refusal_preserves_evidence(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("preserved")
    with pytest.raises(SystemExit) as exc:
        run_s2.main(["--candidate", "buildvision", "--output-dir", str(tmp_path)])
    assert exc.value.code == 2 and path.read_text() == "preserved"


def test_watchdog_cleans_whole_owned_group_even_after_leader_exits(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode, calls = 123456789, None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("runner", timeout)
            self.returncode = -signal.SIGTERM

    signals = []
    def child(*args, **kwargs):
        run_s2.write(tmp_path / "new/report.json", {"ok": True, "status": "completed", "evaluation": {"stale": True}})
        return Process()
    monkeypatch.setattr(run_s2.subprocess, "Popen", child)
    monkeypatch.setattr(run_s2.audit.os, "killpg", lambda pid, sig: signals.append(sig))
    monkeypatch.setattr(run_s2.audit.time, "sleep", lambda duration: None)
    assert run_s2.main(["--candidate", "vyra", "--output-dir", str(tmp_path / "new"), "--timeout", "3", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and signals == [signal.SIGTERM, signal.SIGKILL]
    assert not report["ok"] and "evaluation" not in report


def test_cli_missing_inputs_is_one_json_failure_before_model_load(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ml.eval.run_s2", "--candidate", "buildvision", "--output-dir", str(tmp_path / "new"),
                             "--data-dir", str(tmp_path / "missing"), "--models-dir", str(tmp_path / "missing-models"), "--json"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "failed" and report["coverage"]["videos_completed"] == 0
    assert not (tmp_path / "new/guard.db").exists() and "evaluation" not in report


@pytest.fixture
def boundary_live(inputs, tmp_path, monkeypatch):
    """Run the real four-video orchestration/state/storage, replacing only I/O/ML.

    The synthetic prepared clips have 900 frames, rather than the real benchmark
    counts. Model bytes are deliberately invalid and can never construct ORT.
    """
    import numpy as np
    from guard.contracts import Detection
    from guard.pipeline import compat
    from guard.rules import base as rule_base

    data, models, recipe, clips = inputs
    prepared = run_s2.preflight(data, models, recipe, "buildvision")
    encoded = prepared["prep_provenance"]["encode"]
    assets = prepared["assets"]
    proof = {"schema_version": 1, "status": "verified", "video": "s2_02.mp4",
        "boundary_basis": "encoded_filter_frame_count", "encoder": encoded["encoder"],
        "encoder_canonical_json_sha256": run_s2.audit.fingerprint(encoded["encoder"]), "ffmpeg_version": "fixture",
        "encoded_video": assets["s2_02.mp4"], "source_manifest": assets["manifest"], "encode_report": assets["encode_report"],
        "sources": {name: assets[f"source:{name}"] for name in ("source1", "source2")},
        "parts": {"part0": {"frame_count": 301, "last": {"showinfo_checksum": "old"}},
                  "part1": {"frame_count": 599, "first": {"showinfo_checksum": "new"}}, "joined": {"frame_count": 900}},
        "joined_boundary_frames": [{"frame_index": 300, "showinfo_checksum": "old"}, {"frame_index": 301, "showinfo_checksum": "new"}],
        "boundary_checks": {"count": True, "checksum": True},
        "boundary": {"indexing": "zero-based", "cut_before_source_frame": 301, "cut_time_s": 301 / 15,
            "last_old_selected_sample": {"sample_index": 100, "source_frame_index": 300, "source_time_s": 20},
            "first_new_selected_sample": {"sample_index": 101, "source_frame_index": 303, "source_time_s": 20.2}}}
    boundary = tmp_path / "boundary.json"
    run_s2.write(boundary, proof)
    config = tmp_path / "synthetic-recipe.json"
    run_s2.write(config, recipe)
    state = SimpleNamespace(video=None, frame_index=-1, actions=[], rules=[], managers=[], backends=[], fault=None)
    # Pin/input validation stays real; only this fixture's synthetic recipe is
    # accepted. Concurrent source edits by another agent are outside this test.
    monkeypatch.setattr(run_s2, "load_recipe", lambda path: copy.deepcopy(recipe))
    monkeypatch.setattr(run_s2.audit, "code_snapshot", lambda: {"synthetic_acceptance": True})
    monkeypatch.setattr(run_s2.audit, "check_code_snapshot", lambda *args: None)

    class FakeCapture:
        def __init__(self, path):
            self.video, self.index = Path(path).name, -1
            self.frame = np.zeros((720, 1280, 3), np.uint8)

        def isOpened(self):
            return True

        def get(self, key):
            return {1: 15, 2: 900, 3: 1280, 4: 720, 5: max(0, self.index) / 15 * 1000}[key]

        def grab(self):
            self.index += 1
            return self.index < 900

        def retrieve(self):
            state.video, state.frame_index = self.video, self.index
            self.frame[0, 0] = (self.index % 256, self.index // 256, run_s2.VIDEOS.index(self.video))
            return True, self.frame

        def release(self):
            pass

    cv = ModuleType("cv2")
    cv.CAP_PROP_FPS, cv.CAP_PROP_FRAME_COUNT, cv.CAP_PROP_FRAME_WIDTH, cv.CAP_PROP_FRAME_HEIGHT, cv.CAP_PROP_POS_MSEC = range(1, 6)
    cv.VideoCapture, cv.setNumThreads = FakeCapture, lambda count: None
    ort = ModuleType("onnxruntime")
    ort.disable_telemetry_events = lambda: None
    # A real session must remain impossible even if orchestration regresses.
    ort.InferenceSession = lambda *a, **kw: pytest.fail("Acceptance must not construct ORT sessions")
    monkeypatch.setitem(sys.modules, "cv2", cv)
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)

    class FakeBackend:
        def __init__(self, **kwargs):
            assert kwargs["intra_op_threads"] == 2 and kwargs["max_batch"] == 1
            self.models, self.calls = {}, {}
            state.backends.append(self)

        def load(self, spec):
            self.models[spec.model_id] = spec

        def loaded_models(self):
            return list(self.models)

        def infer_timed(self, model_id, batch):
            self.calls[model_id] = self.calls.get(model_id, 0) + 1
            return [np.array([[self.calls[model_id]]], np.float32)], {"infer": 1}

        def stats(self):
            return {"models": {name: {"calls": count, "items": count, "errors": 0, "max_batch_seen": 1}
                               for name, count in self.calls.items()}, "providers": ["CPUExecutionProvider"], "intra_op_threads": 2}

        def close(self):
            pass

    class FakeDetector:
        def __init__(self, spec, backend):
            self.spec, self.backend = spec, backend

        def detect(self, frame, timings, *, audit=None, force_batch_one=False):
            name = self.spec.model_id
            state.actions.append(("infer", state.video, state.frame_index, name))
            tensor = np.zeros((1, 1, 1, 1), np.float32)
            outputs, _ = self.backend.infer_timed(name, tensor)
            timings["infer"] = timings.get("infer", 0) + 1
            if audit is not None:
                audit.update(input={"sha256": hashlib.sha256(tensor.tobytes()).hexdigest(), "finite": True},
                    outputs=[{"sha256": hashlib.sha256(value.tobytes()).hexdigest(), "finite": True} for value in outputs], request_batch=1)
            if state.video in run_s2.CLEAN:
                return []
            second = state.video == "s2_02.mp4" and state.frame_index >= 282
            if name == "det":
                boxes = [(100, 100, 200, 300), *([(400, 100, 480, 260)] if second else [])]
                return [Detection(name, "person", 0, .9, box) for box in boxes]
            cls = "no_helmet"
            if frame.shape == (720, 1280, 3):
                boxes = [(140, 110, 160, 130), *([(430, 110, 450, 130)] if second else [])]
            else:
                assert force_batch_one
                boxes = [(38, 26, 58, 46)] if frame.shape[:2] == (192, 96) else [(50, 30, 70, 50)]
            return [Detection(name, cls, self.spec.classes.index(cls), .9, box) for box in boxes]

    backend_module, detector_module = ModuleType("guard.inference.onnx_cpu"), ModuleType("guard.pipeline.detector")
    backend_module.OnnxCpuBackend, detector_module.Detector = FakeBackend, FakeDetector
    monkeypatch.setitem(sys.modules, "guard.inference.onnx_cpu", backend_module)
    monkeypatch.setitem(sys.modules, "guard.pipeline.detector", detector_module)
    real_journal, real_manager, real_build = compat.Journal, compat.EventManager, rule_base.build_rules

    def at_boundary():
        return state.video == "s2_02.mp4" and state.frame_index == 303

    class ObservedJournal(real_journal):
        def update(self, event_id, **kwargs):
            if at_boundary():
                state.actions.append(("persist", event_id, kwargs.get("ended_at")))
                if state.fault == "persist":
                    raise RuntimeError("injected boundary persistence failure")
            return super().update(event_id, **kwargs)

        def get(self, event_id):
            value = super().get(event_id)
            if at_boundary():
                state.actions.append(("readback", event_id, value.ended_at if value else None))
                if state.fault == "readback":
                    return None
            return value

    class ObservedManager(real_manager):
        def __init__(self):
            super().__init__()
            state.managers.append(self)
            state.actions.append(("new_manager", state.video, state.frame_index))

        def reset(self, camera_id=None):
            state.actions.append(("reset", camera_id, state.frame_index))
            return super().reset(camera_id)

    def observed_rules(camera):
        rules = real_build(camera)
        for rule in rules:
            state.rules.append(rule)  # retain real instances so object IDs cannot be reused
            original = rule.evaluate
            seen = []

            def evaluate(context, original=original, seen=seen, rule=rule):
                state.actions.append(("rule", id(rule), context.camera.camera_id, context.frame_idx, len(seen)))
                seen.append(context.frame_idx)
                return original(context)

            rule.evaluate = evaluate
        return rules

    monkeypatch.setattr(compat, "Journal", ObservedJournal)
    monkeypatch.setattr(compat, "EventManager", ObservedManager)
    monkeypatch.setattr(rule_base, "build_rules", observed_rules)

    def execute(mode="full_frame", with_boundary=True, fault=None, sha=None):
        state.video, state.frame_index, state.fault = None, -1, fault
        state.actions, state.managers, state.rules, state.backends = [], [], [], []
        output = tmp_path / f"acceptance-{mode}-{len(list(tmp_path.glob('acceptance-*')))}"
        output.mkdir()
        args = run_s2.parser().parse_args(["--candidate", "buildvision", "--ppe-input-mode", mode,
            "--data-dir", str(data), "--models-dir", str(models), "--config", str(config), "--output-dir", str(output)])
        args._deadline = time.monotonic() + 120
        args.boundary_map = boundary if with_boundary else None
        args.boundary_map_sha256 = (sha or run_s2.asset(boundary)["sha256"]) if with_boundary else None
        code = run_s2.run(args)
        read = lambda name: run_s2.audit.read_json(output / name)
        lines = lambda name: [json.loads(line) for line in (output / name).read_text().splitlines()]
        return SimpleNamespace(code=code, output=output, report=read("report.json"), read=read, lines=lines,
                               actions=list(state.actions), managers=list(state.managers), backends=list(state.backends))

    return SimpleNamespace(run=execute, state=state, boundary=boundary, proof=proof, config=config, data=data)


def _semantic_events(events):
    return sorted(({key: value for key, value in event.items() if key not in {"id", "latency_ms"}} for event in events),
                  key=lambda row: (row["camera_id"], row["started_at"], row["track_id"]))


def _semantic_lifecycle(result, video):
    ids = {event["id"]: (event["camera_id"], event["track_id"], event["started_at"], event["confirmed_at"])
           for event in result.read("journal_events.json")}
    rows = result.lines(f"lifecycle_{Path(video).stem}.jsonl")
    return [{**row, "created": [ids[event] for event in row["created"]],
             "closed": [{"event": ids[event["id"]], "ended_at": event["ended_at"]} for event in row["closed"]],
             "reopened": [ids[event] for event in row["reopened"]]} for row in rows]


def test_live_boundary_full_frame_and_roi_share_source_samples_state_reset_and_durable_lifecycle(boundary_live):
    import numpy as np
    from guard.pipeline import compat

    full, crop = boundary_live.run(), boundary_live.run("person_crop")
    assert full.code == crop.code == 0, (full.report.get("error"), crop.report.get("error"))
    for result in (full, crop):
        assert result.report["coverage"]["frames_processed"] == result.report["coverage"]["selected_frames_expected"] == 1200
        reset = result.report["per_video"]["s2_02.mp4"]["boundary_resets"]
        assert len(reset) == 1 and reset[0]["before_sample_index"] == 101 and reset[0]["before_source_frame_index"] == 303
        assert len(reset[0]["closed"]) == result.report["per_video"]["s2_02.mp4"]["coverage"]["boundary_closures"] == 1
        assert reset[0]["closed"][0]["ended_at"] == 20 < 301 / 15 < 20.2
        rows = result.lines("samples_s2_02.jsonl")
        assert rows[100]["source_frame_index"] == 300 and rows[100]["source_time_s"] == 20
        assert rows[101]["source_frame_index"] == 303 and rows[101]["source_time_s"] == 20.2
        expected_frame = np.zeros((720, 1280, 3), np.uint8)
        expected_frame[0, 0] = (303 % 256, 303 // 256, 1)
        assert rows[101]["source_frame"]["sha256"] == hashlib.sha256(expected_frame.tobytes()).hexdigest()
        assert rows[101]["source_frame"]["layout"] == "decoded BGR, C-order bytes"
        assert rows[101]["person_tracks"] == []  # same positions must pass min_hits again
        assert all(track["first_seen"] == 20.2 for track in rows[102]["coco_tracks_before_attributes"])
        assert {row["capture_id"] for row in rows[:101]} == {"s2-quality:s2_02.mp4:source-segment:0:cut:301"}
        assert {row["capture_id"] for row in rows[101:]} == {"s2-quality:s2_02.mp4:source-segment:1:cut:301"}
        assert all(row["source_capture_id"] == "s2-quality:s2_02.mp4" for row in rows)
        events = [event for event in result.read("journal_events.json") if event["camera_id"] == "eval_s2_02"]
        assert len(events) == 3 and len({event["id"] for event in events}) == 3
        first, new = sorted(events, key=lambda event: (event["started_at"], event["track_id"]))[0], [event for event in events if event["started_at"] >= 20.2]
        assert first["track_id"] == 1 and first["ended_at"] == 20
        assert {event["track_id"] for event in new} == {1, 2} and all(event["started_at"] == pytest.approx(20.8) for event in new)
        assert all(event["confirmed_at"] == pytest.approx(22.8) for event in new)
        # Track2 had an unconfirmed old candidate since 19.4. Neither it nor PPE
        # negative votes may carry across the cut and confirm the new scene early.
        assert any(v["track_id"] == 2 for v in rows[97]["violations"])
        assert all(track["attributes"]["helmet"] is None for track in rows[102]["person_tracks"])
        assert not rows[104]["new_event_ids"]
        first_new_rule = next(action for action in result.actions if action[0] == "rule" and action[2:4] == ("eval_s2_02", 303))
        assert first_new_rule[4] == 0
        old_rule = next(action for action in result.actions if action[0] == "rule" and action[2:4] == ("eval_s2_02", 300))
        assert first_new_rule[1] != old_rule[1]
        persist = next(index for index, action in enumerate(result.actions) if action[0] == "persist")
        readback = next(index for index, action in enumerate(result.actions) if action[0] == "readback")
        reset_index = next(index for index, action in enumerate(result.actions) if action[0] == "reset")
        first_new_infer = next(index for index, action in enumerate(result.actions) if action == ("infer", "s2_02.mp4", 303, "det"))
        assert persist < readback < reset_index < first_new_infer
        lifecycle = result.lines("lifecycle_s2_02.jsonl")
        closure = next(row for row in lifecycle if row.get("kind") == "capture_boundary")
        assert closure["closed"] == [{"id": first["id"], "ended_at": 20}]
        assert not closure["created"] and not closure["reopened"]
        with compat.Journal(result.output / "guard.db") as journal:
            persisted, total = journal.query(limit=100)
        assert total == len(result.read("journal_events.json"))
        assert _semantic_events([event.model_dump(mode="json") for event in persisted]) == _semantic_events(result.read("journal_events.json"))
        assert result.report["evaluation"]["primary_helmet_only"]["clean_fp_count"] == 0
    for video in run_s2.VIDEOS:
        a, b = full.lines(f"samples_{Path(video).stem}.jsonl"), crop.lines(f"samples_{Path(video).stem}.jsonl")
        keys = ("sample_index", "target_s", "source_frame_index", "source_time_s", "source_frame", "identity_sha256", "coco_tracks_before_attributes", "capture_id")
        assert [{key: row[key] for key in keys} for row in a] == [{key: row[key] for key in keys} for row in b]
        assert all(row["source_frame"]["shape"] == [720, 1280, 3] and row["source_frame"]["dtype"] == "uint8"
                   and row["source_frame"]["bytes"] == 720 * 1280 * 3 for row in a)
        assert full.report["per_video"][video]["coco_track_identity_sha256"] == crop.report["per_video"][video]["coco_track_identity_sha256"]
        assert _semantic_lifecycle(full, video) == _semantic_lifecycle(crop, video)
    assert full.report["ppe_observations"]["calls"] == 1200 > crop.report["ppe_observations"]["calls"] > 0
    assert _semantic_events(full.read("journal_events.json")) == _semantic_events(crop.read("journal_events.json"))
    assert full.read("original_labels.json") == crop.read("original_labels.json")
    assert full.read("primary_labels.json") == crop.read("primary_labels.json")


def test_live_no_boundary_preserves_original_capture_and_continuous_temporal_state(boundary_live):
    result = boundary_live.run(with_boundary=False)
    assert result.code == 0, result.report.get("error")
    assert "capture_boundary_protocol" not in result.report
    rows = result.lines("samples_s2_02.jsonl")
    assert len(rows) == 300 and {row["capture_id"] for row in rows} == {"s2-quality:s2_02.mp4"}
    assert all("source_frame" not in row and "source_capture_id" not in row and "segment_index" not in row for row in rows)
    assert rows[102]["coco_tracks_before_attributes"][0]["first_seen"] == 0
    assert "boundary_resets" not in result.report["per_video"]["s2_02.mp4"]
    assert not any(action[0] == "reset" for action in result.actions)
    assert len(result.managers) == 4
    events = [event for event in result.read("journal_events.json") if event["camera_id"] == "eval_s2_02"]
    assert len(events) == 2 and all(event["details"]["origin"]["capture_id"] == "s2-quality:s2_02.mp4" for event in events)


def test_boundary_persist_or_readback_failure_prevents_reset_and_new_frame_inference(boundary_live):
    for fault in ("persist", "readback"):
        result = boundary_live.run(fault=fault)
        assert result.code == 1 and result.report["status"] == "failed" and "evaluation" not in result.report
        assert result.report["coverage"]["videos_completed"] == 1
        assert result.report["per_video"]["s2_02.mp4"]["boundary_resets"] == []
        assert not any(action[0] == "reset" for action in result.actions)
        assert len(result.managers) == 2  # no fresh state allocated for the new segment
        assert not any(action[:3] == ("infer", "s2_02.mp4", 303) for action in result.actions)
        rows = result.lines("samples_s2_02.jsonl")
        assert len(rows) == 101 and rows[-1]["source_frame_index"] == 300
        assert not any(row.get("kind") == "capture_boundary" for row in result.lines("lifecycle_s2_02.jsonl"))


def test_boundary_invalid_pin_or_unpaired_cli_fails_before_ml_imports(boundary_live, monkeypatch, tmp_path):
    import builtins

    real_import = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name in {"cv2", "onnxruntime", "guard.pipeline.detector", "guard.inference.onnx_cpu"}:
            raise AssertionError(f"ML import before rejected boundary proof: {name}")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    result = boundary_live.run(sha="0" * 64)
    assert result.code == 1 and "Boundary-map SHA256 mismatch" in result.report["error"]
    assert not result.backends and not (result.output / "guard.db").exists() and "evaluation" not in result.report
    for option, value in (("--boundary-map", str(boundary_live.boundary)), ("--boundary-map-sha256", "a" * 64)):
        output = tmp_path / option.lstrip("-")
        with pytest.raises(SystemExit) as error:
            run_s2.main(["--candidate", "buildvision", option, value, "--output-dir", str(output)])
        assert error.value.code == 2 and not output.exists()
