"""The preflight must stay useful on a checkout with no data or inference runtime."""

from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from guard.doctor import main, run_checks
from guard.pipeline.config import resolve_source_uri
from guard.pipeline.settings import Settings


def camera(**updates):
    item = {
        "camera_id": "cam", "name": "Camera", "site": "Shop",
        "source": {"type": "file", "uri": "videos/clip.mp4"},
        "rules": [{"rule_id": "r", "type": "zone_intrusion"}],
    }
    item.update(updates)
    return item


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data"
    models = tmp_path / "models"
    configs = tmp_path / "configs"
    (data / "videos").mkdir(parents=True)
    models.mkdir()
    configs.mkdir()
    (data / "videos" / "clip.mp4").write_bytes(b"not decoded")
    (models / "det.onnx").write_bytes(b"not loaded")
    (configs / "cameras.yaml").write_text(yaml.safe_dump({"cameras": [camera()]}))
    (configs / "models.yaml").write_text(yaml.safe_dump({"models": [
        {"model_id": "det", "onnx_path": "det.onnx"},
        {"model_id": "pose", "onnx_path": "missing-pose.onnx"},
    ]}))
    return Settings(
        data_dir=data, models_dir=models,
        cameras_config=configs / "cameras.yaml", models_config=configs / "models.yaml",
        db_path=data / "guard.db", media_dir=data / "media", frontend_dist=tmp_path / "dist",
    )


def write_cameras(settings, items):
    settings.cameras_config.write_text(yaml.safe_dump({"cameras": items}))


def errors(report):
    return [check for check in report["checks"] if check["status"] == "error"]


def test_happy_path_does_not_create_outputs_or_load_files(settings, monkeypatch):
    before = sorted(str(path) for path in settings.root.rglob("*"))
    real_open = builtins.open

    def readonly_open(file, mode="r", *args, **kwargs):
        assert not any(flag in mode for flag in "wax+")
        assert str(file) not in {str(settings.db_path), str(settings.models_dir / "det.onnx")}
        assert not str(file).endswith(".mp4")
        return real_open(file, mode, *args, **kwargs)

    def forbidden(*args, **kwargs):
        raise AssertionError("doctor attempted a filesystem mutation")

    monkeypatch.setattr(builtins, "open", readonly_open)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    monkeypatch.setattr(Path, "touch", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    monkeypatch.setattr(Path, "write_bytes", forbidden)
    report = run_checks(settings)
    assert report["ok"], errors(report)
    assert report["scope"] == "offline"
    assert report["required_models"] == ["det"]
    assert report["summary"]["skipped"] >= 1
    assert before == sorted(str(path) for path in settings.root.rglob("*"))


def test_only_active_rules_and_cameras_require_models(settings):
    write_cameras(settings, [
        camera(rules=[{"rule_id": "off", "type": "fall_immobile", "enabled": False}]),
        camera(camera_id="disabled", enabled=False, source={"type": "file", "uri": "missing"},
               rules=[{"rule_id": "pose", "type": "fall_immobile"}]),
    ])
    report = run_checks(settings)
    assert report["ok"], errors(report)
    assert report["required_models"] == []
    assert any(check["check"] == "camera:disabled:source" and check["status"] == "skipped"
               for check in report["checks"])


def test_enabled_pose_rule_requires_missing_pose(settings):
    write_cameras(settings, [camera(rules=[{"rule_id": "pose", "type": "fall_immobile"}])])
    report = run_checks(settings)
    assert not report["ok"]
    assert report["required_models"] == ["det", "pose"]
    assert any(check["check"] == "model:pose" for check in errors(report))


@pytest.mark.parametrize("contents", ["[", "42", "cameras: wrong", "cameras:\n- source: {}"])
def test_invalid_camera_yaml_is_a_report_not_a_traceback(settings, contents):
    settings.cameras_config.write_text(contents)
    report = run_checks(settings)
    assert not report["ok"]
    assert errors(report)


@pytest.mark.parametrize("contents", ["", "cameras: []", "[]", "{}"])
def test_empty_camera_configs_warn(settings, contents):
    settings.cameras_config.write_text(contents)
    report = run_checks(settings)
    assert report["ok"], errors(report)
    assert report["summary"]["warning"] > 0


@pytest.mark.parametrize("change", [
    {"sample_fps": 0}, {"sample_fps": float("nan")}, {"camera_id": " "},
    {"rules": [{"rule_id": "r", "type": "unknown"}]},
    {"rules": [{"rule_id": "r", "type": "running", "zones": ["missing"]}]},
    {"rules": [{"rule_id": "r", "type": "running", "min_ratio": float("nan")}]},
    {"rules": [{"rule_id": "r", "type": "running", "close_after_s": -1}]},
    {"rules": [{"rule_id": "r", "type": "running", "params": {"min_speed_bh": "bad"}}]},
    {"zones": [{"zone_id": "z", "name": "zone", "polygon": [[0, 0], [1, 1]]}]},
    {"zones": [{"zone_id": "z", "name": "zone", "polygon": [[0, 0], [1, 0], [0, 1]],
                "active_hours": ["bad"]}]},
])
def test_camera_semantics(settings, change):
    write_cameras(settings, [camera(**change)])
    assert not run_checks(settings)["ok"]


def test_duplicate_ids_and_registry_entries(settings):
    write_cameras(settings, [camera(), camera()])
    settings.models_config.write_text("models:\n- {model_id: det, onnx: det.onnx}\n- {model_id: det, onnx: det.onnx}\n")
    report = run_checks(settings)
    assert sum("уникаль" in check["message"] for check in errors(report)) == 2


@pytest.mark.parametrize("contents", ["[", "models: wrong", "models: {det: no}", "models: [{}]", "models: []"])
def test_invalid_or_empty_registry_does_not_hide_required_models(settings, contents):
    settings.models_config.write_text(contents)
    report = run_checks(settings)
    assert not report["ok"]
    assert any(check["check"] == "model:det" for check in errors(report))


def test_missing_registry_is_error_only_when_model_needed(settings):
    settings.models_config.unlink()
    assert not run_checks(settings)["ok"]
    write_cameras(settings, [])
    assert run_checks(settings)["ok"]


def test_missing_config_file_is_error(settings):
    settings.cameras_config.unlink()
    assert not run_checks(settings)["ok"]


def test_absolute_paths_and_source_resolution_match_runtime(settings, tmp_path):
    clip = tmp_path / "outside.mp4"
    clip.write_bytes(b"video")
    model = tmp_path / "outside.onnx"
    model.write_bytes(b"model")
    write_cameras(settings, [camera(source={"type": "file", "uri": str(clip)})])
    settings.models_config.write_text(yaml.safe_dump({"det": {"onnx": str(model)}}))
    assert run_checks(settings)["ok"]
    # Existing runtime fallback via data_dir.parent also applies to doctor.
    write_cameras(settings, [camera(source={"type": "file", "uri": "outside.mp4"})])
    assert resolve_source_uri("outside.mp4", settings.data_dir) == str(clip)
    assert run_checks(settings)["ok"]


def test_images_empty_directory_files_and_glob(settings):
    images = settings.data_dir / "images"
    images.mkdir()
    write_cameras(settings, [camera(source={"type": "images", "uri": "images"})])
    assert not run_checks(settings)["ok"]
    (images / "directory.jpg").mkdir()
    assert not run_checks(settings)["ok"]
    (images / "frame.JPG").write_bytes(b"not decoded")
    assert run_checks(settings)["ok"]
    write_cameras(settings, [camera(source={"type": "images", "uri": "images/*.JPG"})])
    assert run_checks(settings)["ok"]


def test_remote_sources_and_tpu_are_explicitly_skipped(settings):
    write_cameras(settings, [camera(source={"type": "rtsp", "uri": "rtsp://user:secret@host/stream"})])
    settings.models_config.write_text("models:\n- {model_id: det, onnx_path: missing.onnx, tpu_program: 'file:///worker/model.tpu'}\n")
    report = run_checks(replace(settings, backend="linq-tpu"))
    assert report["ok"], errors(report)
    serialized = json.dumps(report)
    assert "secret" not in serialized and "rtsp://" not in serialized
    assert any(check["check"] == "model:det" and check["status"] == "skipped" for check in report["checks"])


def test_local_tpu_program_checked_without_loading_it(settings):
    settings.models_config.write_text("models:\n- {model_id: det, onnx_path: det.onnx, tpu_program: det.tpu}\n")
    assert not run_checks(replace(settings, backend="linq-tpu"))["ok"]
    (settings.models_dir / "det.tpu").write_bytes(b"not loaded")
    assert run_checks(replace(settings, backend="linq-tpu"))["ok"]


@pytest.mark.parametrize("path_field,kind", [("data_dir", "file"), ("media_dir", "file"), ("db_path", "dir")])
def test_output_wrong_types(settings, tmp_path, path_field, kind):
    path = tmp_path / "wrong"
    path.write_bytes(b"x") if kind == "file" else path.mkdir()
    assert not run_checks(replace(settings, **{path_field: path}))["ok"]


def test_missing_output_directories_use_existing_parent_without_creation(settings, tmp_path):
    data = tmp_path / "fresh" / "data"
    report = run_checks(replace(settings, data_dir=data, db_path=data / "guard.db", media_dir=data / "media"))
    # Input camera uses original data file through an absolute path.
    assert not report["ok"]  # source no longer resolves, output checks themselves pass
    assert all(check["status"] == "ok" for check in report["checks"] if check["check"].startswith("output:"))
    assert not data.exists()


def test_output_parent_cannot_be_file(settings, tmp_path):
    parent = tmp_path / "file-parent"
    parent.write_bytes(b"x")
    report = run_checks(replace(settings, db_path=parent / "db"))
    assert any(check["check"] == "output:db" for check in errors(report))


def test_permissions_checked_without_writing(settings, monkeypatch):
    real_access = os.access
    monkeypatch.setattr(os, "access", lambda path, mode: False if Path(path) == settings.media_dir.parent
                        and mode & os.W_OK else real_access(path, mode))
    report = run_checks(settings)
    assert any(check["check"] == "output:media" for check in errors(report))


@pytest.mark.parametrize("invalid", [False, True])
def test_credentials_never_appear_on_validation_errors(settings, invalid):
    uri = "rtsp://user:very-secret-password@host/stream?token=private-token"
    item = camera(source={"type": "rtsp", "uri": uri}, sample_fps="invalid" if invalid else 5)
    item["camera_id"] = uri
    write_cameras(settings, [item])
    output = json.dumps(run_checks(settings))
    assert "very-secret-password" not in output
    assert "private-token" not in output


@pytest.mark.parametrize("password", ["hidden-password", "hidden password\nremainder"])
def test_mistyped_uri_as_local_file_does_not_leak_after_path_normalization(settings, password):
    uri = f"rtsp://user:{password}@host/stream?token=private-token"
    write_cameras(settings, [camera(source={"type": "file", "uri": uri})])
    settings.models_config.write_text(yaml.safe_dump({"models": [{"model_id": "det", "onnx_path": uri}]}))
    output = json.dumps(run_checks(settings))
    assert "hidden" not in output and "remainder" not in output and "private-token" not in output


def cli_env(settings):
    return {
        **os.environ, "PYTHONDONTWRITEBYTECODE": "1",
        "GUARD_DATA_DIR": str(settings.data_dir), "GUARD_MODELS_DIR": str(settings.models_dir),
        "GUARD_CAMERAS_CONFIG": str(settings.cameras_config), "GUARD_MODELS_CONFIG": str(settings.models_config),
        "GUARD_DB_PATH": str(settings.db_path), "GUARD_MEDIA_DIR": str(settings.media_dir),
        "GUARD_BACKEND": "onnx-cpu", "GUARD_MAX_BATCH": "8",
    }


def test_cli_json_stdout_exit_codes_and_env(settings):
    env = cli_env(settings)
    result = subprocess.run([sys.executable, "-m", "guard.doctor", "--json"], env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True
    assert result.stderr == ""
    settings.models_config.unlink()
    result = subprocess.run([sys.executable, "-m", "guard.doctor", "--json"], env=env, text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False
    result = subprocess.run([sys.executable, "-m", "guard.doctor", "--invalid"], env=env, text=True, capture_output=True)
    assert result.returncode == 2


def test_cli_bad_env_is_sanitized(settings):
    env = {**cli_env(settings), "GUARD_MAX_BATCH": "rtsp://u:hidden-password@host"}
    result = subprocess.run([sys.executable, "-m", "guard.doctor", "--json"], env=env, text=True, capture_output=True)
    assert result.returncode == 1
    assert json.loads(result.stdout)["ok"] is False
    assert "hidden-password" not in result.stdout + result.stderr


def test_doctor_import_and_run_never_loads_runtime_or_network(settings):
    code = r'''
import builtins, json, runpy, socket, sys
original_import = builtins.__import__
blocked = ("guard.pipeline.engine", "guard.pipeline.worker", "guard.sources", "guard.inference.onnx_cpu",
           "guard.inference.linq_tpu", "onnxruntime", "cv2", "pika", "aio_pika")
def no_runtime(name, *args, **kwargs):
    if any(name == module or name.startswith(module + ".") for module in blocked):
        raise AssertionError("runtime import: " + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = no_runtime
def forbidden(*args, **kwargs):
    raise AssertionError("network call")
socket.create_connection = forbidden
socket.getaddrinfo = forbidden
socket.socket = forbidden
def audit(event, args):
    if event in {"os.mkdir", "os.remove", "os.rename", "socket.connect", "subprocess.Popen"}:
        raise AssertionError("mutation/network: " + event)
    if event == "open" and isinstance(args[2], int) and args[2] & (1 | 2 | 64 | 512 | 1024):
        raise AssertionError("file opened for writing")
sys.addaudithook(audit)
sys.argv = ["guard.doctor", "--json"]
runpy.run_module("guard.doctor", run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", code], env=cli_env(settings), text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["ok"] is True


def test_text_cli_has_scope_and_summary(settings, monkeypatch, capsys):
    monkeypatch.setattr("guard.doctor.Settings.from_env", lambda: settings)
    assert main([]) == 0
    text = capsys.readouterr().out
    assert "scope=offline" in text and "Итог:" in text and "skipped" in text
