"""Ownership and failure boundaries of the opt-in local broker launcher."""
import json
from pathlib import Path
import signal
import subprocess

import pytest

from tools import local_rabbit as rabbit


def test_existing_unowned_directory_is_preserved(tmp_path, monkeypatch, capsys):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("untouched")
    monkeypatch.setattr(rabbit, "_binary", lambda _: Path("/unused/rabbitmq-server"))
    assert rabbit.main(["start", "--state-dir", str(tmp_path)]) == 1
    assert sentinel.read_text() == "untouched"
    assert not (tmp_path / "state.json").exists()
    assert json.loads(capsys.readouterr().out)["error"] == "ValueError"


def test_port_conflict_does_not_create_owned_state(tmp_path, monkeypatch):
    base = tmp_path / "new"
    args = rabbit.parser().parse_args(["start", "--state-dir", str(base)])
    monkeypatch.setattr(rabbit, "_binary", lambda _: Path("/unused/rabbitmq-server"))
    monkeypatch.setattr(rabbit, "_open", lambda port: True)
    with pytest.raises(ValueError, match="occupied"):
        rabbit._prepare(args, base)
    assert not base.exists()


@pytest.mark.parametrize("change", [{"host": "0.0.0.0"}, {"node": "other@localhost"}, {"port": True}, {"dist_port": 56783}])
def test_stop_refuses_invalid_owned_node_identity(tmp_path, change, capsys):
    state = {"schema": "guard-local-rabbit-v1", "state_dir": str(tmp_path),
             "host": "127.0.0.1", "node": "guard_local_example@localhost",
             "port": 56783, "dist_port": 25683, "epmd_port": 43690}
    state.update(change)
    (tmp_path / "state.json").write_text(json.dumps(state))
    assert rabbit.main(["stop", "--state-dir", str(tmp_path)]) == 1
    assert json.loads(capsys.readouterr().out)["error"] == "ValueError"


def test_failed_start_terminates_its_process_group(tmp_path, monkeypatch):
    class Child:
        pid = 12345678
        def poll(self):
            return None
        def wait(self, timeout):
            raise subprocess.TimeoutExpired("broker", timeout)
    child = Child()
    signals = []
    monkeypatch.setattr(rabbit.subprocess, "Popen", lambda *a, **kw: child)
    monkeypatch.setattr(rabbit, "_open", lambda port: False)
    monkeypatch.setattr(rabbit.os, "killpg", lambda pid, sig: signals.append((pid, sig)))
    # Even cleanup timeout must preserve the primary startup failure.
    with pytest.raises(TimeoutError, match="handshake"):
        rabbit._start(tmp_path, {"port": 56783, "dist_port": 25683, "server": "/unused"}, {}, 0)
    assert signals == [(child.pid, signal.SIGTERM), (child.pid, signal.SIGKILL)]


def test_generated_node_binds_only_loopback_and_keeps_cookie_private(tmp_path, monkeypatch):
    base = tmp_path / "new"
    args = rabbit.parser().parse_args(["start", "--state-dir", str(base)])
    monkeypatch.setattr(rabbit, "_binary", lambda _: Path("/unused/rabbitmq-server"))
    monkeypatch.setattr(rabbit, "_open", lambda port: False)
    state = rabbit._prepare(args, base)
    conf = (base / "rabbitmq.conf").read_text()
    assert "listeners.tcp.1 = 127.0.0.1:" in conf
    assert "distribution.listener.interface = 127.0.0.1" in conf
    assert (base / "private-env.json").stat().st_mode & 0o777 == 0o600
    assert base.stat().st_mode & 0o777 == 0o700
    assert state["node"].endswith("@localhost")
    control = json.loads((base / "control.json").read_text())
    assert control["scope"] == "owned_loopback_broker" and "setcookie" not in json.dumps(control)


@pytest.mark.parametrize("timeout", ["nan", "inf", "0", "121"])
def test_invalid_timeout_never_creates_state(tmp_path, timeout, capsys):
    base = tmp_path / "new"
    assert rabbit.main(["start", "--state-dir", str(base), "--timeout", timeout]) == 1
    assert not base.exists()
    assert json.loads(capsys.readouterr().out)["error"] == "ValueError"
