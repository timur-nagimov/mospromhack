"""The module CLI must use durable edge replication, without TPU/network access."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from guard import edge
from guard.contracts import Event
from guard.edge.runner import EdgeEngine
from guard.pipeline.engine import PipelineEngine
from guard.pipeline.testing import FakeBackend, make_settings
from guard.transport.protocol import decode


def build_fixture(tmp_path, monkeypatch, extra=()):
    cameras = tmp_path / "cameras.yaml"
    cameras.write_text(yaml.safe_dump({
        "edge": {"node_id": "yaml-edge", "backend": "linq-tpu", "linq": {"raw": True, "timeout_s": 5}},
        "cameras": [],
    }))
    settings = make_settings(tmp_path)
    observed = []

    def from_env():
        observed.append((os.environ.get("GUARD_MODE"), os.environ.get("GUARD_EDGE_ID")))
        return settings

    monkeypatch.setattr(edge.Settings, "from_env", from_env)
    monkeypatch.setattr(edge, "load_dotenv_guard", lambda: None)
    monkeypatch.setattr(edge, "disable_ort_spinning", lambda: False)
    args = edge.parse_args(["--config", str(cameras), "--transport", "memory", *extra])
    result, engine = edge.build(args, backend=FakeBackend())
    return args, result, engine, observed


def test_build_uses_durable_outbox_and_versioned_wire(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARD_BACKEND_FALLBACK", "onnx-cpu")
    monkeypatch.setenv("LINQ_RABBIT_HOST", "external-tpu.invalid")
    monkeypatch.delenv("LINQ_RAW", raising=False)
    monkeypatch.delenv("LINQ_TIMEOUT_S", raising=False)
    monkeypatch.delenv("GUARD_EDGE_ID", raising=False)
    monkeypatch.delenv("GUARD_NODE_ID", raising=False)
    args, settings, engine, observed = build_fixture(tmp_path, monkeypatch, [
        "--backend", "linq-tpu", "--node-id", "cli-edge", "--url", "amqp://event-user:test-secret@event-bus.invalid/",
        "--models-config", str(tmp_path / "models.yaml"),
    ])
    try:
        assert isinstance(engine, EdgeEngine)
        assert settings.mode == "edge" and settings.edge_id == "cli-edge"
        assert observed == [("edge", "cli-edge")]
        assert settings.backend == "linq-tpu" and not settings.clips_enabled
        assert settings.models_config == tmp_path / "models.yaml"
        assert os.environ["GUARD_BACKEND_FALLBACK"] == "none"
        assert os.environ["LINQ_RABBIT_HOST"] == "external-tpu.invalid"
        assert os.environ["LINQ_RAW"] == "1" and os.environ["LINQ_TIMEOUT_S"] == "5.0"
        event = Event(id="durable-event", camera_id="camera", site="site", rule_id="rule", type="zone_intrusion",
                      severity="high", started_at=100.0, confirmed_at=102.0, bbox=(0, 0, 10, 20))
        engine.journal.add(event)
        assert len(engine.journal.pending_messages()) == 1
        assert engine.replication.publish_pending_once() == 1
        wire = engine.replication.publisher.events[0]
        envelope = decode(wire, expected_edge_id="cli-edge")
        assert envelope.kind == "event" and envelope.payload["event"]["id"] == event.id
        assert engine.journal.pending_messages() == []
        assert settings.db_path.is_file()
    finally:
        engine.stop()


def test_run_is_bounded_and_stops_real_durable_engine_with_fake_backend(tmp_path, monkeypatch):
    args, _, engine, _ = build_fixture(tmp_path, monkeypatch, ["--backend", "onnx-cpu", "--duration", "0.02", "--stats-period", "0.005"])
    monkeypatch.setattr(edge, "build", lambda args: (engine.settings, engine))
    assert edge.run(args) == 0
    assert engine._stopped


@pytest.mark.parametrize("explicit_cli", [False, True])
def test_dotenv_identity_and_config_are_loaded_before_edge_overrides(tmp_path, monkeypatch, explicit_cli):
    fixture = make_settings(tmp_path)
    env_config = tmp_path / "from-env.yaml"
    cli_config = tmp_path / "from-cli.yaml"
    for config in (env_config, cli_config):
        config.write_text(yaml.safe_dump({"edge": {"node_id": "section-edge"}, "cameras": []}))
    dotenv = tmp_path / ".env"
    dotenv.write_text("\n".join([
        f"GUARD_CAMERAS_CONFIG={env_config}", "GUARD_EDGE_ID=dotenv-edge", "GUARD_BACKEND=onnx-cpu",
        f"GUARD_DATA_DIR={fixture.data_dir}", f"GUARD_MODELS_DIR={fixture.models_dir}",
        f"GUARD_MODELS_CONFIG={fixture.models_config}",
    ]))
    for key in ("GUARD_CAMERAS_CONFIG", "GUARD_EDGE_ID", "GUARD_NODE_ID", "GUARD_BACKEND",
                "GUARD_DATA_DIR", "GUARD_MODELS_DIR", "GUARD_MODELS_CONFIG"):
        # Register absent keys too: the dotenv loader will add them directly.
        monkeypatch.setenv(key, os.environ.get(key, ""))
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("GUARD_ROOT", str(tmp_path))
    monkeypatch.setattr(edge, "disable_ort_spinning", lambda: False)
    read_text = Path.read_text

    def guarded(path, *args, **kwargs):
        if path.name == ".env" and path != dotenv:
            raise AssertionError("test must not read a real dotenv")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded)
    argv = ["--transport", "memory"]
    if explicit_cli:
        argv += ["--config", str(cli_config), "--node-id", "cli-edge"]
    settings, engine = edge.build(edge.parse_args(argv), backend=FakeBackend())
    try:
        assert settings.cameras_config == (cli_config if explicit_cli else env_config)
        assert settings.edge_id == ("cli-edge" if explicit_cli else "dotenv-edge")
        assert os.environ["GUARD_EDGE_ID"] == "dotenv-edge"
        assert engine.journal.make_message("metrics", {"data": {}}).edge_id == settings.edge_id
    finally:
        engine.stop()


@pytest.mark.parametrize("status", ["ok", "degraded", "error"])
def test_edge_health_preserves_parent_backend_status(monkeypatch, status):
    engine = object.__new__(EdgeEngine)
    engine.persistence_status = lambda: {"pending": 0, "last_error": None}
    engine.journal = SimpleNamespace(replication_stats=lambda: {})
    engine.replication = SimpleNamespace(status=lambda: {})
    monkeypatch.setattr(PipelineEngine, "health", lambda self: {"status": status, "backend": "linq-tpu"})
    assert engine.health()["status"] == status
    engine.persistence_status = lambda: {"pending": 1, "last_error": "OperationalError"}
    assert engine.health()["status"] == "error"


@pytest.mark.parametrize("args", [["--duration", "nan"], ["--duration", "-1"], ["--stats-period", "0"]])
def test_cli_rejects_unbounded_or_invalid_intervals(args):
    with pytest.raises(SystemExit) as error:
        edge.parse_args(args)
    assert error.value.code == 2


def test_module_help_does_not_construct_backend_or_read_dotenv():
    code = '''
import pathlib, runpy
original = pathlib.Path.read_text
def guarded(self, *args, **kwargs):
    if self.name == ".env":
        raise AssertionError("help read dotenv")
    return original(self, *args, **kwargs)
pathlib.Path.read_text = guarded
import guard.inference
def forbidden(*args, **kwargs):
    raise AssertionError("help created backend")
guard.inference.get_backend = forbidden
import sys
sys.argv = ["guard.edge", "--help"]
runpy.run_module("guard.edge", run_name="__main__")
'''
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr
    for flag in ("--config", "--models-config", "--backend", "--transport", "--url", "--duration", "--node-id"):
        assert flag in result.stdout
