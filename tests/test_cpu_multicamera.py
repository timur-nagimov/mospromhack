"""The acceptance runner's failure paths, observation contract and full-run maths.

No test starts the heavy video/model acceptance workload.
"""
from __future__ import annotations

import json
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tools import cpu_multicamera as multi


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "181", "oops"])
def test_timeout_cli_refuses_invalid_values_as_json(value, capsys):
    assert multi.main(["--timeout", value, "--json"]) == 2
    report = json.loads(capsys.readouterr().out)
    assert not report["ok"] and report["runtime_started"] is False


def test_existing_output_is_never_modified(tmp_path, capsys):
    evidence = tmp_path / "report.json"
    evidence.write_text("preserved", encoding="utf-8")
    assert multi.main(["--output-dir", str(tmp_path), "--json"]) == 2
    assert evidence.read_text() == "preserved"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["report.json"]
    assert json.loads(capsys.readouterr().out)["runtime_started"] is False


def test_internal_child_cannot_overwrite_existing_evidence(tmp_path, capsys):
    evidence = tmp_path / "report.json"
    evidence.write_text("preserved", encoding="utf-8")
    assert multi.main(["--_run", "--_deadline", "10", "--output-dir", str(tmp_path), "--json"]) == 2
    assert evidence.read_text() == "preserved"
    assert not json.loads(capsys.readouterr().out)["ok"]


def _local_inputs(tmp_path):
    data, models = tmp_path / "data", tmp_path / "models"
    for _, relative, _ in multi.CAMERAS:
        path = data / "videos" / relative
        path.parent.mkdir(parents=True)
        path.write_bytes(b"test existence only; not a video")
    models.mkdir()
    (models / "yolov8n-640.onnx").write_bytes(b"existence only; not ONNX")
    return SimpleNamespace(data_dir=data, models_dir=models)


def test_config_uses_manifest_zones_and_explicit_full_frame_load_zone(tmp_path):
    args = _local_inputs(tmp_path)
    result = multi._preflight(args)
    manifest = json.loads((multi.ROOT / "tools/data_manifest.json").read_text())
    entries = {item["name"]: item for item in manifest["videos"]}
    for camera, (_, _, annotation) in zip(result["cameras"], multi.CAMERAS):
        assert camera["sample_fps"] == 5
        assert camera["source"]["type"] == "file"
        assert camera["source"]["realtime"] is True and camera["source"]["loop"] is False
        assert Path(camera["source"]["uri"]).is_file()
        assert camera["rules"][0]["type"] == "zone_intrusion"
        if annotation:
            assert camera["zones"][0]["polygon"] == entries[annotation]["zones"][0]["polygon"]
        else:
            assert camera["zones"][0]["zone_id"] == "full_frame_load_test"
            assert camera["zones"][0]["polygon"] == [[0, 0], [1, 0], [1, 1], [0, 1]]


def test_explicit_settings_do_not_read_dotenv(tmp_path, monkeypatch):
    from guard.pipeline.config import load_cameras, load_models
    from guard.pipeline.settings import Settings
    monkeypatch.setattr(Settings, "from_env", lambda: pytest.fail("must not load .env"))
    args = _local_inputs(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    settings = multi._settings(args, output, multi._preflight(args)["cameras"])
    assert settings.mode == "local" and settings.autostart is False
    assert settings.db_path == output / "guard.db" and settings.media_dir == output / "media"
    assert settings.ort_threads == 2 and settings.batching is False
    assert settings.rabbit_url is None
    assert json.loads(settings.models_config.read_text())["models"][0]["model_id"] == "det"
    cameras = load_cameras(settings.cameras_config)
    assert len(cameras) == 3 and all(camera.enabled and camera.rules and camera.zones for camera in cameras)
    assert sorted(load_models(settings.models_config, args.models_dir)) == ["det"]


@pytest.mark.parametrize("mode", ["--help", "bad_timeout", "missing", "child_missing"])
def test_no_runtime_imports_in_help_or_preflight_failure(tmp_path, mode):
    code = '''
import builtins, sys
original = builtins.__import__
def guarded(name, *args, **kwargs):
    blocked = ("cv2", "onnxruntime", "guard.pipeline.engine", "guard.sources", "fastapi")
    if any(name == item or name.startswith(item + ".") for item in blocked):
        raise AssertionError("runtime import before preflight: " + name)
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from tools import cpu_multicamera as multi
mode, output = sys.argv[1:]
args = ["--json", "--data-dir", output + "/missing-data", "--models-dir", output + "/missing-models", "--output-dir", output]
if mode == "--help":
    args = ["--help"]
elif mode == "bad_timeout":
    args += ["--timeout", "nan"]
elif mode == "child_missing":
    from pathlib import Path
    Path(output).mkdir()
    args += ["--_run", "--_deadline", str(multi.time.monotonic() + 10)]
raise SystemExit(multi.main(args))
'''
    output = tmp_path / "output"
    result = subprocess.run([sys.executable, "-c", code, mode, str(output)], text=True, capture_output=True, timeout=10)
    assert "runtime import before" not in result.stderr + result.stdout
    if mode == "--help":
        assert result.returncode == 0 and "--timeout" in result.stdout
    elif mode == "child_missing":
        assert result.returncode == 1, result.stderr
        report = json.loads((output / "report.json").read_text())
        assert report["preflight"]["ok"] is False and not report["ok"]
    else:
        assert result.returncode == (2 if mode == "bad_timeout" else 1), result.stderr
        report = json.loads(result.stdout)
        assert not report["ok"] and report["runtime_started"] is False
    assert not (output / "guard.db").exists() and not (output / "media").exists()


def test_script_help_works_without_runtime():
    result = subprocess.run([sys.executable, str(multi.ROOT / "tools/cpu_multicamera.py"), "--help"],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode == 0 and "--models-dir" in result.stdout


def test_global_watchdog_kills_own_process_group_including_descendants(monkeypatch, tmp_path, capsys):
    class Process:
        pid = 123456789
        returncode = None
        calls = 0

        def wait(self, timeout):
            self.calls += 1
            if self.calls == 1:
                raise subprocess.TimeoutExpired("acceptance", timeout)
            # Parent exits after TERM. KILL must still target its group because
            # a remaining ffmpeg descendant can ignore TERM.
            self.returncode = -signal.SIGTERM

    process, signals, invocations = Process(), [], []
    def popen(command, **kwargs):
        invocations.append((command, kwargs))
        return process
    monkeypatch.setattr(multi, "_preflight", lambda args: {})
    monkeypatch.setattr(multi.subprocess, "Popen", popen)
    monkeypatch.setattr(multi.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    output = tmp_path / "new-output"
    assert multi.main(["--output-dir", str(output), "--timeout", "1", "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "timeout" and not report["ok"]
    assert signals == [(process.pid, signal.SIGTERM), (process.pid, signal.SIGKILL)]
    assert invocations[0][1]["start_new_session"] is True
    assert invocations[0][1]["env"]["OMP_NUM_THREADS"] == "2"
    assert json.loads((output / "report.json").read_text()) == report


def test_nonzero_runtime_cannot_report_pass(monkeypatch, tmp_path, capsys):
    class Process:
        returncode = 1
        def wait(self, timeout):
            pass
    output = tmp_path / "new-output"
    def popen(*args, **kwargs):
        multi._write(output / "report.json", {"ok": True, "status": "passed"})
        return Process()
    monkeypatch.setattr(multi, "_preflight", lambda args: {})
    monkeypatch.setattr(multi.subprocess, "Popen", popen)
    assert multi.main(["--output-dir", str(output), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["status"] == "failed"


def _row(seq, total, ts):
    return {"ok": True, "capture_id": "capture", "seq": seq, "capture_ts": ts,
            "source_pos_s": ts, "done_wall": ts + .1, "timings_ms": {"total": total, "infer": total - 1},
            "detection_counts": {"det": 1}, "rules": ["s1_zone"]}


def test_full_run_percentiles_use_all_observations_and_explicit_denominators():
    # More than the 300-frame rolling window: the first slow 60 frames must
    # influence full-run p95 even when the last 300 are fast.
    rows = [_row(index, 1000 if index < 60 else 10, index / 5) for index in range(360)]
    rows.append({"ok": False, "seq": 360})
    report = multi._summarize(rows, 1800, {"frames": 1800, "fps": 25, "duration_s": 72})
    assert report["timings_ms"]["total"] == {"n": 360, "p50": 10, "p95": 1000, "max": 1000}
    assert report["configured_sample_target"] == 360
    assert report["sample_target_coverage"] == 1
    assert report["effective_processed_fps"] == 5
    assert report["skipped_by_sampling_or_overload"] == 1440
    assert report["processing_failures"] == 1
    assert report["capture_gap_s"]["n"] == 359
    assert report["analysis_age_at_completion_ms"]["n"] == 360
    assert report["det_observations"] == report["rule_observations"] == 360
    assert "dropped" not in report


def test_empty_observations_do_not_fake_fps_or_percentiles():
    report = multi._summarize([], 100, {"frames": 100, "fps": 25, "duration_s": 4})
    assert report["sample_target_coverage"] == report["effective_processed_fps"] == 0
    assert report["analysis_age_at_completion_ms"] == {"n": 0, "p50": None, "p95": None, "max": None}
    assert report["first"] is None and report["last"] is None


def test_observer_calls_original_once_retains_result_and_restores_method(tmp_path):
    cid = multi.CAMERAS[0][0]
    context = SimpleNamespace(timings={"total": 12}, detections={"det": [1]})
    class Worker:
        cfg = SimpleNamespace(camera_id=cid)
        rules = [SimpleNamespace(cfg=SimpleNamespace(rule_id="zone"))]
        calls = []
        def process(self, packet):
            self.calls.append(packet)
            return context
    original = Worker.process
    packet = SimpleNamespace(capture_id="cap", seq=7, ts=5.0, pos_s=1.2)
    observations = multi.Observations(tmp_path)
    try:
        with multi._observe_process(Worker, observations):
            assert Worker().process(packet) is context
        assert Worker.process is original and Worker.calls == [packet]
        row = observations.snapshot()[cid][0]
        assert row["capture_id"] == "cap" and row["seq"] == 7 and row["source_pos_s"] == 1.2
        assert row["capture_ts"] == 5.0 and row["done_wall"] > 5.0
        assert row["timings_ms"] == {"total": 12} and row["detection_counts"] == {"det": 1}
        assert row["rules"] == ["zone"] and row["ok"] is True
        assert json.loads((tmp_path / f"frames-{cid}.jsonl").read_text()) == row
    finally:
        observations.close()


def test_observer_preserves_original_exception_and_records_failure(tmp_path):
    cid = multi.CAMERAS[0][0]
    failure = ValueError("the original error")
    class Worker:
        cfg = SimpleNamespace(camera_id=cid)
        def process(self, packet):
            raise failure
    original = Worker.process
    observations = multi.Observations(tmp_path)
    try:
        with pytest.raises(ValueError) as caught:
            with multi._observe_process(Worker, observations):
                Worker().process(SimpleNamespace(capture_id="cap", seq=1, ts=2, pos_s=3))
        assert caught.value is failure and Worker.process is original
        assert observations.snapshot()[cid][0]["ok"] is False
    finally:
        observations.close()


def test_settled_requires_every_eof_last_analysis_and_all_clip_completions():
    workers = {}
    for cid, _, _ in multi.CAMERAS:
        source = SimpleNamespace(eof=True, running=False, latest=lambda: SimpleNamespace(seq=20))
        workers[cid] = SimpleNamespace(source=source, last=SimpleNamespace(seq=20))
    media = {"pending": 0, "clips_written": 2, "clip_errors": 0}
    assert multi._settled(workers, 2, media)
    worker = next(iter(workers.values()))
    worker.last.seq = 19
    assert not multi._settled(workers, 2, media)
    worker.last.seq = 20
    worker.source.eof = False
    assert not multi._settled(workers, 2, media)
    worker.source.eof = True
    assert not multi._settled(workers, 3, media)  # media.pending excludes its internal pending list
    assert not multi._settled(workers, 2, {**media, "clip_errors": 1})
    assert not multi._settled(dict(list(workers.items())[:2]), 2, media)


def test_sqlite_readback_does_not_create_missing_database(tmp_path):
    import sqlite3
    path = tmp_path / "missing.db"
    with pytest.raises(sqlite3.OperationalError):
        multi._sqlite_events(path)
    assert not path.exists()
