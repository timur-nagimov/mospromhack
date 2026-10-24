"""A/B evidence/timeout tests with synthetic traces; no pretrained inference."""

import json
import os
import select
import signal
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

from ml.eval import compare_s1_models as comparison
from ml.eval import run_s1
from ml.eval.events_eval import LabelEvent, PredEvent


@pytest.fixture
def model_files(tmp_path):
    def create(variant="n"):
        directory = tmp_path / variant
        directory.mkdir()
        onnx = directory / f"yolov8{variant}-640.onnx"
        onnx.write_bytes(f"not executable {variant}".encode())
        weights = directory / f"yolov8{variant}.pt"
        weights.write_bytes(f"never deserialize {variant}".encode())
        registry = directory / "models.yaml"
        registry.write_text(json.dumps({"models": [{"model_id": "det", "onnx_path": onnx.name,
            "input_size": [640, 640], "conf_threshold": .25, "iou_threshold": .5}]}))
        metadata = {"model_id": "det", "task": "detect", "onnx": onnx.name, "input_size": [640, 640],
            "classes": ["person", *[f"class_{index}" for index in range(79)]], "output_format": "yolov8",
            "onnx_sha256": run_s1._sha(onnx), "onnx_bytes": onnx.stat().st_size,
            "weights": str(weights), "weights_sha256": run_s1._sha(weights),
            "source": f"https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8{variant}.pt",
            "inputs": {"images": [1, 3, 640, 640]}, "outputs": {"output0": [1, 84, 8400]}, "ort_output_shapes": [[1, 84, 8400]],
            "export_device": "cpu", "problems": []}
        metadata["source_classes"] = metadata["classes"]
        run_s1._write(directory / "det.json", metadata)
        return directory, registry, metadata
    return create


@pytest.mark.parametrize("variant", ["n", "s", "x"])
def test_model_provenance_checks_actual_bytes_without_loading_ml(model_files, monkeypatch, variant):
    import builtins

    directory, registry, _ = model_files(variant)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name in {"torch", "onnx", "onnxruntime", "cv2", "guard.inference.onnx_cpu"}:
            raise AssertionError(f"Unexpected ML import: {name}")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    assets = comparison.validate_model(registry, directory, variant)
    assert assets["metadata"]["sha256"] == run_s1._sha(directory / "det.json")
    assert assets["onnx"]["sha256"] != assets["weights"]["sha256"]


@pytest.mark.parametrize("change", ["onnx_bytes", "weights_bytes", "filename", "sidecar_role", "source", "threshold", "classes"])
def test_mismatched_provenance_or_inference_settings_fail(model_files, change):
    directory, registry, metadata = model_files("s")
    if change == "onnx_bytes":
        (directory / metadata["onnx"]).write_bytes(b"changed")
    elif change == "weights_bytes":
        Path(metadata["weights"]).write_bytes(b"changed")
    elif change == "filename":
        metadata["onnx"] = "yolov8n-640.onnx"
    elif change == "sidecar_role":
        metadata["model_id"] = "det_s640"
    elif change == "source":
        metadata["source"] = "https://untrusted.example/yolov8s.pt"
    elif change == "threshold":
        config = json.loads(registry.read_text())
        config["models"][0]["conf_threshold"] = .3
        run_s1._write(registry, config)
    else:
        metadata["source_classes"] = list(reversed(metadata["classes"]))
    run_s1._write(directory / "det.json", metadata)
    with pytest.raises(ValueError):
        comparison.validate_model(registry, directory, "s")


@pytest.fixture
def paired(tmp_path, model_files, request):
    directories, models = {}, {}
    recipe = json.loads((run_s1.ROOT / "configs/eval-s1.json").read_text())
    candidate_variant = getattr(request, "param", "s")
    for name, variant in (("baseline", "n"), ("candidate", candidate_variant)):
        model_dir, registry, _ = model_files(variant)
        models[name] = comparison.validate_model(registry, model_dir, variant)
        directory = directories[name] = tmp_path / name
        directory.mkdir()
        run_s1._write(directory / "effective_model.json", models[name]["effective_model"])
        for relative in comparison.COMMON_OUTPUTS:
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            run_s1._write(path, [])
        run_s1._write(directory / "labels_with_ids.json", [{"label_id": "s1_01.mp4:0", "video": "s1_01.mp4", "t_start": 10, "t_end": 20}])
        labels = [LabelEvent("s1_01.mp4", "zone_intrusion", 10, 20)]
        records = [{"label_id": "s1_01.mp4:0", "video": "s1_01.mp4", "type": "zone_intrusion", "t_start": 10, "t_end": 20, "uncertain": False}]
        preds = [PredEvent("s1_01.mp4", "zone_intrusion", 10, 12, "tp"),
                 PredEvent("s1_01.mp4", "zone_intrusion", 11, 13, "duplicate")]
        scored = run_s1.score_predictions(labels, preds, {"s1_01.mp4": 120}, 2, records)
        report = {"ok": True, "status": "completed", "coverage": {"videos_expected": 2, "videos_completed": 2,
            "selected_frames_expected": 600, "frames_processed": 600}, "label_counts": {"certain": 4, "uncertain": 3},
            "inference_stats": {"backend": "onnx-cpu", "providers": ["CPUExecutionProvider"], "intra_op_threads": 2,
                                "models": {"det": {"errors": 0, "calls": 600, "items": 600, "max_batch_seen": 1}}},
            "effective_recipe": {**recipe, "model_registry": str(registry)}, "git": {"head": "fixed", "tracked_changes": []},
            "versions": {"python": "3.11"}, "assets": {key: {"sha256": f"same:{key}"} for key in comparison.COMMON_ASSETS},
            "output_hashes": {}, "global": scored, "per_video": {}, "wall_s": 5}
        for key, model_key in (("det_onnx", "onnx"), ("det_metadata", "metadata"), ("model_registry", "registry")):
            report["assets"][key] = models[name][model_key]
        for video in run_s1.VIDEOS:
            coverage = {"frames_processed": 300, "selected_frames_decoded": 300, "source_frames_traversed": 1800,
                "source_frames_expected": 1800, "source_fps": 30, "wall_s": 2}
            report["per_video"][video] = {"coverage": coverage, **scored}
            rows = [{**row, "source_time_s": row["source_frame_index"] / 30, "decoder_pos_s": row["source_frame_index"] / 30,
                "capture_id": f"s1-quality:{video}", "detections": 0, "raw_detections": [], "person_tracks": [],
                "violations": [], "new_event_ids": [], "timings_ms": {"preprocess": 1, "infer": 1 if video == run_s1.VIDEOS[0] else 3,
                                                                         "postprocess": 1}} for row in run_s1.sample_plan(30)]
            path = directory / f"samples_{Path(video).stem}.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        run_s1._write(directory / "report.json", report)
    output = tmp_path / "comparison"
    output.mkdir()
    return directories, models, output, recipe


def test_fresh_paired_metrics_keep_duplicates_and_full_600_inference_samples(paired):
    result = comparison.compare_runs(*paired)
    for name in ("baseline", "candidate"):
        timing = result["runs"][name]["timing"]
        assert timing["stages_ms"]["infer"] == {"n": 600, "sum_ms": 1200, "mean_ms": 2, "p50_ms": 2, "p95_ms": 3, "max_ms": 3}
        assert timing["amortized_video_loop_ms_per_frame"] == pytest.approx(4000 / 600)
        assert "tracking" in timing["unmeasured_individual_stages"]
        metrics = result["runs"][name]["global"]
        assert metrics["evaluator"]["overall"]["precision"] == 1
        assert metrics["supplemental_strict"]["strict_precision"] == .5
        assert metrics["prediction_classifications"][1]["prediction_id"] == "duplicate"
    assert len(result["per_video"]["s1_01.mp4"]["manual_review_landmarks"]) == 7
    assert result["per_video"]["s1_01.mp4"]["paired_samples"] == 300


@pytest.mark.parametrize("change", ["coverage", "config", "paired_config_drift", "model", "GT", "sample_grid", "timestamp", "trace_missing", "inference_count", "hash"])
def test_incomparable_or_partial_runs_fail_instead_of_producing_comparison(paired, change):
    directories, _, _, _ = paired
    directory = directories["candidate"]
    path = directory / "report.json"
    report = json.loads(path.read_text())
    if change == "coverage":
        report["coverage"]["frames_processed"] = 599
    elif change == "config":
        report["effective_recipe"]["tol_s"] = 3
    elif change == "paired_config_drift":
        report["effective_recipe"]["tol_s"] = 3
        base = json.loads((directories["baseline"] / "report.json").read_text())
        base["effective_recipe"]["tol_s"] = 3
        run_s1._write(directories["baseline"] / "report.json", base)
    elif change == "model":
        model = json.loads((directory / "effective_model.json").read_text())
        model["conf_threshold"] = .3
        run_s1._write(directory / "effective_model.json", model)
    elif change == "GT":
        run_s1._write(directory / "labels_with_ids.json", ["changed"])
    elif change == "inference_count":
        report["inference_stats"]["models"]["det"]["calls"] = 599
    elif change == "hash":
        report["output_hashes"]["mapping.json"] = {"sha256": "wrong"}
    else:
        trace = directory / "samples_s1_01.jsonl"
        rows = [json.loads(line) for line in trace.read_text().splitlines()]
        if change == "sample_grid":
            rows[1]["source_frame_index"] += 1
        elif change == "timestamp":
            rows[1]["decoder_pos_s"] += .1
        else:
            rows.pop()
        trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    run_s1._write(path, report)
    with pytest.raises(ValueError):
        comparison.compare_runs(*paired)


def test_raw_attribution_is_descriptive_and_preserves_run_local_tracks(paired):
    directory = paired[0]["candidate"]
    path = directory / "samples_s1_01.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[50].update(detections=1, raw_detections=[{"cls": "person", "conf": .5, "bbox": [0, 0, 10, 10]}],
                    person_tracks=[{"id": 17, "bbox": [0, 0, 10, 10]}],
                    violations=[{"track_id": 17, "zone_id": "zone", "score": .5}])
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    result = comparison.compare_runs(*paired)
    video = result["per_video"]["s1_01.mp4"]
    assert video["frames_track_count_changed"] == video["frames_violation_presence_changed"] == 1
    assert video["boundary_examples"][0]["person_tracks"]["candidate"][0]["id"] == 17
    assert "not actor GT" in result["attribution_limit"]


def test_output_refusal_preserves_evidence(tmp_path):
    path = tmp_path / "report.json"
    path.write_text("preserved")
    with pytest.raises(SystemExit) as error:
        comparison.main(["--output-dir", str(tmp_path)])
    assert error.value.code == 2 and path.read_text() == "preserved"


@pytest.mark.parametrize("value", ["nan", "inf", "-1", "0", "1801"])
def test_timeout_bounds(value):
    with pytest.raises(SystemExit) as error:
        comparison.parser().parse_args(["--timeout", value])
    assert error.value.code == 2


def test_outer_watchdog_kills_subtree_and_returns_json(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode, calls = 123456789, None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("comparison", timeout)
            self.returncode = -signal.SIGTERM

    signals = []
    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(comparison.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    monkeypatch.setattr(comparison.time, "sleep", lambda seconds: None)
    assert comparison.main(["--output-dir", str(tmp_path / "new"), "--timeout", "3", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and not report["ok"]
    assert signals == [(123456789, signal.SIGTERM), (123456789, signal.SIGKILL)]
    assert report["process_group_cleanup"]["leader_reaped"]


def test_per_run_timeout_terminates_child_without_detached_session(monkeypatch, tmp_path):
    class Process:
        returncode, calls = None, 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("run", timeout)
            self.returncode = -15

        def terminate(self):
            self.terminated = True

    process = Process()
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    monkeypatch.setattr(comparison.subprocess, "Popen", popen)
    result = comparison.run_child(tmp_path / "config", tmp_path, tmp_path, tmp_path / "output", 3, float("inf"))
    assert result["timed_out"] and process.terminated
    assert not calls[0][1].get("start_new_session", False)
    assert "--_run" in calls[0][0] and calls[0][1]["env"]["OMP_NUM_THREADS"] == "2"


@pytest.mark.parametrize("paired", ["s", "x"], indirect=True)
def test_execute_starts_two_fresh_runs_in_order_without_reusing_baseline(paired, monkeypatch):
    directories, models, output, _ = paired
    calls = []
    variant = "x" if models["candidate"]["metadata_content"]["onnx"] == "yolov8x-640.onnx" else "s"
    validations = []

    def validate(registry, path, variant):
        validations.append((registry.name, variant))
        return models["baseline" if variant == "n" else "candidate"]

    def run(config, model_dir, data_dir, destination, timeout, deadline):
        name = destination.name
        assert not destination.exists()
        assert json.loads(config.read_text())["sample_fps"] == 5
        calls.append(name)
        shutil.copytree(directories[name], destination)
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(comparison, "validate_model", validate)
    monkeypatch.setattr(comparison, "run_child", run)
    args = comparison.parser().parse_args(["--output-dir", str(output), "--candidate-models-dir", "/unused", "--candidate-variant", variant,
                                          "--_deadline", "1e100"])
    assert comparison.execute(args) == 0
    assert calls == ["baseline", "candidate"]
    report = json.loads((output / "report.json").read_text())
    assert report["ok"] and report["comparison"]["runs"]["baseline"]["timing"]["sample_denominator"] == 600
    assert report["candidate_variant"] == variant
    assert validations == [("models.cpu-baseline.yaml", "n"), (f"models.yolov8{variant}-640.yaml", variant)]
    assert [row["stage"] for row in report["provenance"]["code_checks"]] == [
        "before_baseline", "after_baseline", "before_candidate", "after_candidate", "final"]
    assert all(row["matches_frozen"] for row in report["provenance"]["code_checks"])
    assert report["comparison"]["executable_code_sha256"] == report["provenance"]["executable_code"]["sha256"]


def test_missing_assets_fail_before_ml_import_in_worker(tmp_path):
    code = '''
import builtins, json, sys, time
from pathlib import Path
original = builtins.__import__
def guard(name, *args, **kwargs):
    if name in {"torch", "cv2", "onnxruntime", "guard.inference.onnx_cpu"}:
        raise AssertionError("ML imported before preflight")
    return original(name, *args, **kwargs)
builtins.__import__ = guard
from ml.eval import compare_s1_models
output = Path(sys.argv[1])
output.mkdir()
args = compare_s1_models.parser().parse_args(["--output-dir", str(output), "--baseline-models-dir", sys.argv[2],
    "--candidate-models-dir", sys.argv[2], "--_deadline", str(time.monotonic()+5)])
code = compare_s1_models.execute(args)
print((output / "report.json").read_text())
raise SystemExit(code)
'''
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "new"), str(tmp_path / "missing")],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "failed" and "ML imported" not in report["error"]
    assert not (tmp_path / "new/baseline/guard.db").exists()


def test_public_cli_missing_assets_returns_one_json_object(tmp_path):
    result = subprocess.run([sys.executable, "-m", "ml.eval.compare_s1_models", "--output-dir", str(tmp_path / "new"),
                             "--baseline-models-dir", str(tmp_path / "missing"), "--json"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 1
    assert json.loads(result.stdout)["status"] == "failed"


@pytest.mark.parametrize("change", ["same_filename_new_content", "module_added", "module_removed"])
def test_code_snapshot_detects_bytes_and_untracked_module_set_changes(tmp_path, monkeypatch, change):
    guard, ml = tmp_path / "guard", tmp_path / "ml"
    guard.mkdir()
    ml.mkdir()
    module = guard / "compat.py"
    module.write_text("value = 1\n")
    (ml / "untracked.py").write_text("value = 2\n")
    frozen = comparison.code_snapshot(tmp_path)
    assert frozen["files"].keys() == {"guard/compat.py", "ml/untracked.py"}
    if change == "same_filename_new_content":
        module.write_text("value = 3\n")
        changed_path = "guard/compat.py"
    elif change == "module_added":
        (guard / "new.py").write_text("value = 4\n")
        changed_path = "guard/new.py"
    else:
        module.unlink()
        changed_path = "guard/compat.py"
    snapshot = comparison.code_snapshot
    monkeypatch.setattr(comparison, "code_snapshot", lambda: snapshot(tmp_path))
    report = {"provenance": {"code_checks": []}}
    with pytest.raises(ValueError, match="Executable source changed"):
        comparison.check_code_snapshot(frozen, report, "after_baseline")
    check = report["provenance"]["code_checks"][0]
    assert check["changed_paths"] == [changed_path] and not check["matches_frozen"]


def test_code_drift_after_first_run_aborts_comparison_and_records_failed_check(paired, monkeypatch, tmp_path):
    directories, models, output, _ = paired
    source = tmp_path / "executable"
    (source / "guard").mkdir(parents=True)
    code = source / "guard/compat.py"
    code.write_text("value = 1\n")
    snapshot = comparison.code_snapshot
    monkeypatch.setattr(comparison, "code_snapshot", lambda: snapshot(source))
    monkeypatch.setattr(comparison, "validate_model", lambda registry, path, variant: models["baseline" if variant == "n" else "candidate"])
    calls = []

    def run(config, model_dir, data_dir, destination, timeout, deadline):
        calls.append(destination.name)
        shutil.copytree(directories[destination.name], destination)
        code.write_text("value = 2\n")  # Same dirty filename, different executed content.
        return {"returncode": 0, "timed_out": False}

    monkeypatch.setattr(comparison, "run_child", run)
    args = comparison.parser().parse_args(["--output-dir", str(output), "--candidate-models-dir", "/unused", "--_deadline", "1e100"])
    assert comparison.execute(args) == 1 and calls == ["baseline"]
    report = json.loads((output / "report.json").read_text())
    assert report["status"] == "failed" and "comparison" not in report
    assert report["provenance"]["code_checks"][-1]["changed_paths"] == ["guard/compat.py"]


@pytest.mark.parametrize("leader_already_exited", [False, True])
def test_owned_group_kills_term_ignoring_descendant_after_leader_exits(tmp_path, leader_already_exited):
    """Real disposable processes: the original early-break bug leaves child alive."""
    ready, term = tmp_path / "ready", tmp_path / "term"
    read_fd, write_fd = os.pipe()
    child_code = f'''
import os, signal, time
from pathlib import Path
signal.signal(signal.SIGTERM, lambda *args: Path({str(term)!r}).write_text("ignored"))
os.fstat({write_fd})
Path({str(ready)!r}).write_text(str(os.getpid()))
while True:
    time.sleep(.02)
'''
    leader_code = f'''
import subprocess, sys, time
from pathlib import Path
subprocess.Popen([sys.executable, "-c", {child_code!r}], pass_fds=({write_fd},))
while not Path({str(ready)!r}).exists():
    time.sleep(.01)
if {leader_already_exited!r}:
    raise SystemExit(0)
while True:
    time.sleep(.02)
'''
    process = subprocess.Popen([sys.executable, "-c", leader_code], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, pass_fds=(write_fd,))
    os.close(write_fd)
    fully_stopped = False
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists(), "Own descendant did not start"
        assert int(ready.read_text()) > 0
        if leader_already_exited:
            assert process.wait(timeout=2) == 0
        assert not select.select([read_fd], [], [], 0)[0], "Descendant must still hold its inherited pipe open"
        cleanup = comparison.terminate_owned_group(process, grace_s=.15)
        assert cleanup == {"term_sent": True, "kill_sent": True, "leader_reaped": True}
        assert term.read_text() == "ignored"
        assert process.returncode == (0 if leader_already_exited else -signal.SIGTERM)
        # EOF requires every inherited writer (leader AND descendant) to exit.
        # This avoids OS-wide process inspection and zombie/reaping races.
        assert select.select([read_fd], [], [], 2)[0], "Own descendant survived group KILL"
        assert os.read(read_fd, 1) == b""
        fully_stopped = True
    finally:
        if not fully_stopped:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=2)
        os.close(read_fd)


def test_outer_cleanup_runs_after_normal_worker_exit_not_only_global_timeout(monkeypatch, tmp_path, capsys):
    class Process:
        pid, returncode = 123456789, 1

        def wait(self, timeout):
            return self.returncode

    process = Process()
    cleaned = []
    monkeypatch.setattr(comparison.subprocess, "Popen", lambda *args, **kwargs: process)

    def cleanup(worker):
        cleaned.append(worker.pid)
        return {"term_sent": True, "kill_sent": True, "leader_reaped": True}

    monkeypatch.setattr(comparison, "terminate_owned_group", cleanup)
    assert comparison.main(["--output-dir", str(tmp_path / "new"), "--json"]) == 1
    assert cleaned == [process.pid]
    assert json.loads(capsys.readouterr().out)["process_group_cleanup"]["kill_sent"]


@pytest.mark.parametrize("variant,override", [("s", False), ("x", False), ("x", True)])
def test_candidate_paths_and_variant_reach_worker_with_heavy_deadlines(monkeypatch, tmp_path, capsys, variant, override):
    class Process:
        pid, returncode = 123456789, 1

        def wait(self, timeout):
            assert 1790 < timeout <= 1800
            return self.returncode

    calls = []

    def popen(command, **kwargs):
        calls.append(command)
        return Process()

    monkeypatch.setattr(comparison.subprocess, "Popen", popen)
    monkeypatch.setattr(comparison, "terminate_owned_group", lambda process: {"leader_reaped": True})
    baseline = tmp_path / "models"
    arguments = ["--output-dir", str(tmp_path / "new"), "--baseline-models-dir", str(baseline),
                 "--candidate-variant", variant, "--timeout", "1800", "--run-timeout", "1500", "--json"]
    if override:
        arguments.extend(["--candidate-models-dir", str(tmp_path / "explicit")])
    assert comparison.main(arguments) == 1
    json.loads(capsys.readouterr().out)
    command = calls[0]
    expected = tmp_path / "explicit" if override else baseline / f"candidates/yolov8{variant}-640"
    assert command[command.index("--candidate-models-dir") + 1] == str(expected)
    assert command[command.index("--candidate-variant") + 1] == variant
    assert command[command.index("--timeout") + 1] == "1800.0"
    assert command[command.index("--run-timeout") + 1] == "1500.0"


def test_default_s_and_timeouts_are_backward_compatible():
    args = comparison.parser().parse_args([])
    assert args.candidate_variant == "s" and args.timeout == 420 and args.run_timeout == 180
    assert comparison.bounded_timeout("1800") == 1800


@pytest.mark.parametrize("variant", ["n", "m", "../../outside"])
def test_candidate_variant_cli_is_explicit_and_restricted(variant):
    with pytest.raises(SystemExit) as error:
        comparison.parser().parse_args(["--candidate-variant", variant])
    assert error.value.code == 2


@pytest.mark.parametrize("bad_provenance", ["source", "weights_filename", "onnx_hash", "unknown_variant"])
def test_x_requires_correct_variant_provenance_and_actual_bytes(model_files, bad_provenance):
    directory, registry, metadata = model_files("x")
    if bad_provenance == "source":
        metadata["source"] = metadata["source"].replace("yolov8x.pt", "yolov8s.pt")
    elif bad_provenance == "weights_filename":
        metadata["weights"] = str(directory / "yolov8s.pt")
    elif bad_provenance == "onnx_hash":
        metadata["onnx_sha256"] = "0" * 64
    run_s1._write(directory / "det.json", metadata)
    with pytest.raises(ValueError):
        comparison.validate_model(registry, directory, "m" if bad_provenance == "unknown_variant" else "x")


def test_x_registry_preserves_s_computation_settings():
    from guard.pipeline.config import load_models

    s = load_models(comparison.ROOT / "configs/models.yolov8s-640.yaml")["det"].model_dump(mode="json")
    x = load_models(comparison.ROOT / "configs/models.yolov8x-640.yaml")["det"].model_dump(mode="json")
    assert comparison.normalized_model(s) == comparison.normalized_model(x)
    assert x["onnx_path"] == "yolov8x-640.onnx" and x["model_id"] == "det"
