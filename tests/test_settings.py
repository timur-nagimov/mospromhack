"""Merged service settings and portable, separate LINQ broker configuration."""

import os
from pathlib import Path

import pytest

from guard.pipeline import settings as configuration
from guard.pipeline.settings import Settings, find_dotenv
from guard.inference import linq_tpu as lt


@pytest.fixture(autouse=True)
def isolated_environment(tmp_path, monkeypatch):
    for key in list(os.environ):
        if key.startswith(("GUARD_", "LINQ_")) or key == "JUPYTER_LOGIN":
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GUARD_ROOT", str(tmp_path))
    # Tests may read only their own explicitly-created temporary .env.
    monkeypatch.setattr(configuration, "find_dotenv", lambda start=None: (
        tmp_path / ".env" if (tmp_path / ".env").exists() else None))
    monkeypatch.setattr(lt, "find_dotenv", configuration.find_dotenv)


def test_merged_settings_keep_identity_secrets_and_upstream_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARD_MODE", " EDGE ")
    monkeypatch.setenv("GUARD_EDGE_ID", "edge-7")
    monkeypatch.setenv("GUARD_BACKEND", "tpu")
    monkeypatch.setenv("GUARD_RABBIT_URL", "amqp://event-user:private-marker@event-broker/")
    monkeypatch.setenv("GUARD_TRANSPORT", "memory")
    monkeypatch.setenv("GUARD_EDGE_SNAPSHOT_WIDTH", "320")
    result = Settings.from_env()
    assert result.mode == "edge" and result.edge_id == "edge-7" and result.backend == "linq-tpu"
    assert result.transport == "memory" and result.clips_enabled is False
    assert result.publish_frame_meta is True and result.edge_snapshot_width == 320
    assert result.data_dir == tmp_path / "data" and result.models_dir == tmp_path / "models"
    assert result.cameras_config == configuration.CODE_ROOT / "configs/cameras.yaml"
    assert "private-marker" not in repr(result) and "event-broker" not in repr(result)


def test_missing_or_invalid_nonlocal_identity_is_rejected(monkeypatch):
    monkeypatch.setenv("GUARD_MODE", "gateway")
    with pytest.raises(ValueError, match="GUARD_EDGE_ID"):
        Settings.from_env()
    monkeypatch.setenv("GUARD_EDGE_ID", "invalid/node")
    with pytest.raises(ValueError, match="Invalid GUARD_EDGE_ID"):
        Settings.from_env()


def test_ppe_input_mode_is_opt_in_and_validated(monkeypatch):
    assert Settings.from_env().ppe_input_mode == "full_frame"
    monkeypatch.setenv("GUARD_PPE_INPUT_MODE", " PERSON_CROP ")
    assert Settings.from_env().ppe_input_mode == "person_crop"
    monkeypatch.setenv("GUARD_PPE_INPUT_MODE", "implicit-fallback")
    with pytest.raises(ValueError, match="GUARD_PPE_INPUT_MODE"):
        Settings.from_env()


def test_guard_and_linq_dotenv_values_are_separate_and_environment_wins(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "GUARD_RABBIT_URL=amqp://event-broker/\n"
        "export LINQ_RABBIT_HOST=chip-broker\n"
        "LINQ_RABBIT_PORT=5673\n"
        "LINQ_RABBIT_USER=fixture-user\n"
        "LINQ_RABBIT_PASSWORD=fixture-password\n"
        "LINQ_TIMEOUT_S=5\nLINQ_RAW=1\n"
        "UNRELATED_AUTH=must-not-load\n")
    monkeypatch.setenv("LINQ_RABBIT_HOST", "explicit-chip-host")
    result = Settings.from_env()
    assert result.rabbit_url == "amqp://event-broker/"
    assert "LINQ_RABBIT_PASSWORD" not in os.environ and "UNRELATED_AUTH" not in os.environ
    backend = lt.LinqTpuBackend()
    try:
        assert backend.host == "explicit-chip-host" and backend.port == 5673
        assert backend.user == "fixture-user" and backend.password == "fixture-password"
        assert backend.timeout_s == 5 and backend.raw is True
        assert backend.host not in result.rabbit_url
        state = backend.stats()
        assert "fixture-password" not in repr(state) and "fixture-user" not in repr(state)
    finally:
        backend.close()


def test_portable_root_is_dynamic_and_manifest_lookup_uses_it(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARD_ROOT", str(tmp_path / "deployment"))
    root = configuration.project_root()
    assert root == tmp_path / "deployment"
    directory = root / "models/tpu/model"
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text('{"batch": 4}')
    from guard.contracts import ModelSpec
    manifest = lt._find_manifest(ModelSpec(model_id="model", onnx_path="model.onnx"), None, None)
    assert manifest["batch"] == 4
    monkeypatch.delenv("GUARD_ROOT")
    assert configuration.project_root() == configuration.CODE_ROOT
    assert configuration.CODE_ROOT == Path(configuration.__file__).resolve().parents[2]


def test_guard_dotenv_does_not_override_explicit_values(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text("GUARD_BACKEND=tpu\nGUARD_MODE=local\n")
    monkeypatch.setenv("GUARD_BACKEND", "cpu")
    assert Settings.from_env().backend == "onnx-cpu"


def test_real_dotenv_finder_supports_nested_worktree_without_reading_parent_secrets(tmp_path):
    (tmp_path / ".env").write_text("# synthetic fixture only\n")
    nested = tmp_path / ".worktrees/feature"
    nested.mkdir(parents=True)
    assert find_dotenv(nested) == tmp_path / ".env"


def test_settings_from_env_does_not_import_inference_or_ml(tmp_path):
    import subprocess
    import sys

    script = '''
import importlib.abc
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("guard.inference", "cv2", "onnxruntime", "numpy")):
            raise AssertionError("forbidden settings import: " + fullname)
sys.meta_path.insert(0, Block())
from guard.pipeline.settings import Settings
result = Settings.from_env()
assert result.backend == "linq-tpu"
print("settings-lightweight-ok")
'''
    env = {key: value for key, value in os.environ.items() if not key.startswith(("GUARD_", "LINQ_"))}
    env.update(GUARD_ROOT=str(tmp_path), GUARD_BACKEND="TPU")
    process = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, timeout=10)
    assert process.returncode == 0, process.stdout + process.stderr
    assert "settings-lightweight-ok" in process.stdout
